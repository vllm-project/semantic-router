import {
  TRACE_LEARNING,
  TRACE_LOOPER,
  TRACE_MODALITY,
  TRACE_ON_UNKNOWN,
  TRACE_PROMPT_CACHE,
  TRACE_RANKING,
  TRACE_RETENTION,
  TRACE_SIGNALS,
  TRACE_TOOLS,
  X_VSR_TRACE,
  X_VSR_WARNINGS,
} from '../generated/vsrHeaders'
import { matchedSignalKey, traceKey } from './routingMetadataDisplay'

// One RFC 8941 item: its value, with strings and tokens unquoted and numbers
// as written, and its parameters.
export interface StructuredItem {
  value: string
  params: Record<string, string>
}

// One member of a dictionary or list: an item, or an inner list of items
// with the list's parameters.
export interface StructuredMember extends StructuredItem {
  items: StructuredItem[]
}

class StructuredFieldParser {
  private at = 0

  constructor(private readonly input: string) {}

  done(): boolean {
    return this.at >= this.input.length
  }

  consume(char: string): boolean {
    if (this.input[this.at] !== char) return false
    this.at += 1
    return true
  }

  skipSpaces(): void {
    while (this.input[this.at] === ' ' || this.input[this.at] === '\t') this.at += 1
  }

  key(): string {
    const start = this.at
    while (!this.done() && /[a-z0-9_.*-]/.test(this.input[this.at])) this.at += 1
    return this.input.slice(start, this.at)
  }

  // A string is unquoted and percent-decoded: the Router percent-encodes "%"
  // and every byte outside printable ASCII.
  bareItem(): string {
    if (!this.consume('"')) {
      const start = this.at
      while (!this.done() && !' ,;()'.includes(this.input[this.at])) this.at += 1
      return this.input.slice(start, this.at)
    }
    let raw = ''
    while (!this.done()) {
      const char = this.input[this.at]
      this.at += 1
      if (char === '\\' && !this.done()) {
        raw += this.input[this.at]
        this.at += 1
      } else if (char === '"') {
        break
      } else {
        raw += char
      }
    }
    try {
      return decodeURIComponent(raw)
    } catch {
      return raw
    }
  }

  params(): Record<string, string> {
    const params: Record<string, string> = {}
    while (this.consume(';')) {
      this.skipSpaces()
      const key = this.key()
      params[key] = this.consume('=') ? this.bareItem() : '?1'
    }
    return params
  }

  member(): StructuredMember {
    const items: StructuredItem[] = []
    let value = ''
    if (this.consume('(')) {
      for (this.skipSpaces(); !this.done() && !this.consume(')'); this.skipSpaces()) {
        items.push({ value: this.bareItem(), params: this.params() })
      }
    } else {
      value = this.bareItem()
    }
    return { value, items, params: this.params() }
  }

  // next moves past the comma between members, reporting whether one follows.
  next(): boolean {
    this.skipSpaces()
    if (!this.consume(',')) return false
    this.skipSpaces()
    return true
  }
}

// parseStructuredDictionary parses an RFC 8941 dictionary such as x-vsr-trace;
// a malformed value parses as far as it is well formed.
export function parseStructuredDictionary(value: string): Map<string, StructuredMember> {
  const parser = new StructuredFieldParser(value)
  const members = new Map<string, StructuredMember>()
  parser.skipSpaces()
  while (!parser.done()) {
    const key = parser.key()
    if (!key) break
    members.set(
      key,
      parser.consume('=') ? parser.member() : { value: '?1', items: [], params: parser.params() },
    )
    if (!parser.next()) break
  }
  return members
}

// parseStructuredList parses an RFC 8941 list such as x-vsr-warnings.
export function parseStructuredList(value: string): StructuredMember[] {
  const parser = new StructuredFieldParser(value)
  const members: StructuredMember[] = []
  parser.skipSpaces()
  while (!parser.done()) {
    members.push(parser.member())
    if (!parser.next()) break
  }
  return members
}

const isTrue = (value: string): string => String(value === '?1')

// expandRoutingHeaders turns a response's headers into display entries: the
// public headers as they are, x-vsr-warnings as its codes, and each member of
// x-vsr-trace under traceKey, with lists comma-separated and parameters as
// entries of their own; the trace's matched signals go under
// matchedSignalKey.
export function expandRoutingHeaders(headers: Record<string, string>): Record<string, string> {
  const entries: Record<string, string> = {}
  Object.entries(headers).forEach(([key, value]) => {
    if (key !== X_VSR_TRACE && key !== X_VSR_WARNINGS) entries[key] = value
  })
  if (headers[X_VSR_WARNINGS]) {
    const codes = parseStructuredList(headers[X_VSR_WARNINGS]).map((warning) => warning.value)
    entries[X_VSR_WARNINGS] = [...new Set(codes)].join(',')
  }
  parseStructuredDictionary(headers[X_VSR_TRACE] ?? '').forEach((member, key) => {
    const values = member.items.map((item) => item.value)
    const addParams = (convert: (param: string, value: string) => string = (_, value) => value) =>
      Object.entries(member.params).forEach(([param, value]) => {
        entries[traceKey(`${key}.${param}`)] = convert(param, value)
      })
    switch (key) {
      case TRACE_SIGNALS: {
        const byType = new Map<string, string[]>()
        values.forEach((signal) => {
          const separator = signal.indexOf(':')
          if (separator <= 0) return
          const type = signal.slice(0, separator)
          byType.set(type, [...(byType.get(type) ?? []), signal.slice(separator + 1)])
        })
        byType.forEach((names, type) => {
          entries[matchedSignalKey(type)] = names.join(',')
        })
        return
      }
      case TRACE_LEARNING:
        entries[traceKey(`${key}.methods`)] = values.join(',')
        ;['action', 'scope', 'reason'].forEach((field) => {
          const pairs = member.items
            .filter((item) => item.params[field])
            .map((item) => `${item.value}=${item.params[field]}`)
          if (pairs.length > 0) entries[traceKey(`${key}.${field}s`)] = pairs.join(',')
        })
        return
      case TRACE_LOOPER:
        entries[traceKey(`${key}.models`)] = values.join(',')
        addParams()
        return
      case TRACE_MODALITY:
        entries[traceKey(key)] = values.join(',')
        return
      case TRACE_RETENTION:
        addParams((param, value) => (param === 'ttl_turns' ? value : isTrue(value)))
        return
      case TRACE_PROMPT_CACHE:
      case TRACE_TOOLS:
        entries[traceKey(key)] = member.value
        addParams()
        return
      case TRACE_ON_UNKNOWN:
        entries[traceKey(key)] = member.items
          .map((item) => `${item.value}=${item.params.policy ?? ''}`)
          .join(',')
        return
      case TRACE_RANKING:
        entries[traceKey(key)] = member.params.reason
          ? `${member.value}: ${member.params.reason}`
          : member.value
        return
      default:
        entries[traceKey(key)] = member.value === '?1' ? 'true' : member.value
    }
  })
  return entries
}
