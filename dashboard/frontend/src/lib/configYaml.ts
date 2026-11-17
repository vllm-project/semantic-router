import { concreteRouterConfigSchema, type JSONSchemaNode } from './routerConfigSchema'

// Renders configuration fragments the way the Dashboard backend writes them:
// typed sections keep the Router schema's field order (Go struct order) and
// free-form maps, such as plugin configurations, sort their keys.

type Schema = JSONSchemaNode | string | undefined

const RESERVED_SCALAR = /^(?:true|false|yes|no|on|off|y|n|null|~)$/i
const NUMERIC_SCALAR =
  /^[-+]?(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$|^0x[0-9a-fA-F]+$|^0o[0-7]+$|^[-+]?\.inf$|^\.nan$/i
const INDICATOR_START = /^[-?:,[\]{}#&*!|>'"%@`]/

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function concrete(schema: Schema): JSONSchemaNode | undefined {
  if (!schema) return undefined
  try {
    return concreteRouterConfigSchema(schema)
  } catch {
    return undefined
  }
}

function needsQuotes(text: string): boolean {
  return (
    text === '' ||
    text !== text.trim() ||
    INDICATOR_START.test(text) ||
    RESERVED_SCALAR.test(text) ||
    NUMERIC_SCALAR.test(text) ||
    /: |\s#|:$/.test(text) ||
    hasControlCharacter(text)
  )
}

function hasControlCharacter(text: string): boolean {
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index)
    if (code < 0x20 || code === 0x7f) return true
  }
  return false
}

function scalar(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return '.nan'
    if (!Number.isFinite(value)) return value > 0 ? '.inf' : '-.inf'
    return String(value)
  }
  const text = String(value)
  return needsQuotes(text) ? JSON.stringify(text) : text
}

function blockLiteral(text: string, indent: number): string[] | null {
  if (!text.includes('\n') || text.includes('\r') || /^\s/.test(text)) return null
  const body = text.endsWith('\n') ? text.slice(0, -1) : text
  if (body.includes('\n\n\n') || /[ \t]$/m.test(body)) return null
  const pad = ' '.repeat(indent)
  return [
    text.endsWith('\n') ? '|' : '|-',
    ...body.split('\n').map((line) => (line ? pad + line : '')),
  ]
}

function key(name: string): string {
  return needsQuotes(name) ? JSON.stringify(name) : name
}

function orderedKeys(value: Record<string, unknown>, schema: JSONSchemaNode | undefined): string[] {
  const present = Object.keys(value).filter((name) => value[name] !== undefined)
  const declared = schema?.properties ? Object.keys(schema.properties) : []
  const known = declared.filter((name) => present.includes(name))
  const rest = present.filter((name) => !declared.includes(name)).sort()
  return [...known, ...rest]
}

function childSchema(schema: JSONSchemaNode | undefined, name: string): JSONSchemaNode | undefined {
  if (!schema) return undefined
  const property = schema.properties?.[name]
  if (property) return property
  return typeof schema.additionalProperties === 'object' ? schema.additionalProperties : undefined
}

function emitValue(prefix: string, value: unknown, schema: Schema, indent: number): string[] {
  if (Array.isArray(value)) {
    return value.length === 0
      ? [`${prefix} []`]
      : [prefix, ...emitSequence(value, concrete(schema)?.items, indent)]
  }
  if (isRecord(value)) {
    return Object.keys(value).some((name) => value[name] !== undefined)
      ? [prefix, ...emitMapping(value, schema, indent)]
      : [`${prefix} {}`]
  }
  if (typeof value === 'string') {
    const block = blockLiteral(value, indent)
    if (block) return [`${prefix} ${block[0]}`, ...block.slice(1)]
  }
  return [`${prefix} ${scalar(value)}`]
}

function emitMapping(value: Record<string, unknown>, schema: Schema, indent: number): string[] {
  const resolved = concrete(schema)
  const pad = ' '.repeat(indent)
  return orderedKeys(value, resolved).flatMap((name) =>
    emitValue(`${pad}${key(name)}:`, value[name], childSchema(resolved, name), indent + 2),
  )
}

function emitSequence(items: unknown[], schema: Schema, indent: number): string[] {
  const pad = ' '.repeat(indent)
  return items.flatMap((item) => {
    if (isRecord(item) && Object.keys(item).some((name) => item[name] !== undefined)) {
      const lines = emitMapping(item, schema, indent + 2)
      return [`${pad}- ${lines[0].trimStart()}`, ...lines.slice(1)]
    }
    if (Array.isArray(item) && item.length > 0) {
      return [`${pad}-`, ...emitSequence(item, concrete(schema)?.items, indent + 2)]
    }
    return emitValue(`${pad}-`, item, schema, indent + 2)
  })
}

/** YAML for a mapping, ordered by the Router schema when one is given. */
export function configYaml(value: Record<string, unknown>, schema?: Schema): string {
  return `${emitMapping(value, schema, 0).join('\n')}\n`
}

/**
 * YAML for one entry of a configuration collection, nested under its path,
 * for example `['signals', 'keywords']` with the entry's schema reference.
 */
export function configCollectionItemYaml(
  path: readonly string[],
  item: unknown,
  itemSchema?: Schema,
): string {
  const lines = path.map((segment, index) => `${' '.repeat(index * 2)}${key(segment)}:`)
  return `${[...lines, ...emitSequence([item], itemSchema, path.length * 2)].join('\n')}\n`
}

export interface LineChange {
  kind: 'same' | 'add' | 'remove'
  text: string
}

/** A line diff between two short documents, in reading order. */
export function diffLines(before: string, after: string): LineChange[] {
  const a = before.replace(/\n$/, '').split('\n')
  const b = after.replace(/\n$/, '').split('\n')
  const lengths = Array.from({ length: a.length + 1 }, () =>
    new Array<number>(b.length + 1).fill(0),
  )
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      lengths[i][j] =
        a[i] === b[j] ? lengths[i + 1][j + 1] + 1 : Math.max(lengths[i + 1][j], lengths[i][j + 1])
    }
  }
  const changes: LineChange[] = []
  let i = 0
  let j = 0
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      changes.push({ kind: 'same', text: a[i] })
      i += 1
      j += 1
    } else if (lengths[i + 1][j] >= lengths[i][j + 1]) {
      changes.push({ kind: 'remove', text: a[i] })
      i += 1
    } else {
      changes.push({ kind: 'add', text: b[j] })
      j += 1
    }
  }
  while (i < a.length) changes.push({ kind: 'remove', text: a[i++] })
  while (j < b.length) changes.push({ kind: 'add', text: b[j++] })
  return changes
}
