import { ROUTER_CONFIG_SCHEMA } from '../generated/routerConfigContract'
import { concreteRouterConfigSchema, type JSONSchemaNode } from './routerConfigSchema'

// Renders configuration fragments the way the Dashboard backend writes them:
// typed sections keep the Router schema's field order (Go struct order) and
// free-form maps, such as plugin configurations, sort their keys.

type Schema = JSONSchemaNode | string | undefined

/** The schema of a whole Router configuration document, for YAML that starts at its root. */
export const ROUTER_CONFIG_DOCUMENT_SCHEMA = ROUTER_CONFIG_SCHEMA as unknown as JSONSchemaNode

const RESERVED_SCALAR = /^(?:true|false|yes|no|on|off|y|n|null|~)$/i
const NUMERIC_SCALAR =
  /^[-+]?(?:\d[\d_]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$|^0x[0-9a-fA-F]+$|^0o[0-7]+$|^[-+]?\.inf$|^\.nan$/i
const INDICATOR_START = /^[-?:,[\]{}#&*!|>'"%@`]/

/** One line of rendered YAML. */
export interface ConfigYamlLine {
  text: string
  /** Keys and list indexes from the document root to the value the line writes. */
  path: string[]
}

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

function emitValue(
  prefix: string,
  value: unknown,
  schema: Schema,
  indent: number,
  path: string[],
): ConfigYamlLine[] {
  const line = (text: string): ConfigYamlLine => ({ text, path })
  if (Array.isArray(value)) {
    return value.length === 0
      ? [line(`${prefix} []`)]
      : [line(prefix), ...emitSequence(value, concrete(schema)?.items, indent, path)]
  }
  if (isRecord(value)) {
    return Object.keys(value).some((name) => value[name] !== undefined)
      ? [line(prefix), ...emitMapping(value, schema, indent, path)]
      : [line(`${prefix} {}`)]
  }
  if (typeof value === 'string') {
    const block = blockLiteral(value, indent)
    if (block) return [line(`${prefix} ${block[0]}`), ...block.slice(1).map(line)]
  }
  return [line(`${prefix} ${scalar(value)}`)]
}

function emitMapping(
  value: Record<string, unknown>,
  schema: Schema,
  indent: number,
  path: string[],
): ConfigYamlLine[] {
  const resolved = concrete(schema)
  const pad = ' '.repeat(indent)
  return orderedKeys(value, resolved).flatMap((name) =>
    emitValue(`${pad}${key(name)}:`, value[name], childSchema(resolved, name), indent + 2, [
      ...path,
      name,
    ]),
  )
}

function emitSequence(
  items: unknown[],
  schema: Schema,
  indent: number,
  path: string[],
): ConfigYamlLine[] {
  const pad = ' '.repeat(indent)
  return items.flatMap((item, index) => {
    const itemPath = [...path, String(index)]
    if (isRecord(item) && Object.keys(item).some((name) => item[name] !== undefined)) {
      const [first, ...rest] = emitMapping(item, schema, indent + 2, itemPath)
      return [{ ...first, text: `${pad}- ${first.text.trimStart()}` }, ...rest]
    }
    if (Array.isArray(item) && item.length > 0) {
      return [
        { text: `${pad}-`, path: itemPath },
        ...emitSequence(item, concrete(schema)?.items, indent + 2, itemPath),
      ]
    }
    return emitValue(`${pad}-`, item, schema, indent + 2, itemPath)
  })
}

/** YAML lines for a mapping, each with the path of the value it writes. */
export function configYamlLines(value: Record<string, unknown>, schema?: Schema): ConfigYamlLine[] {
  return emitMapping(value, schema, 0, [])
}

/** YAML for a mapping, ordered by the Router schema when one is given. */
export function configYaml(value: Record<string, unknown>, schema?: Schema): string {
  return `${configYamlLines(value, schema)
    .map((line) => line.text)
    .join('\n')}\n`
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
  const entry = emitSequence([item], itemSchema, path.length * 2, [...path])
  return `${[...lines, ...entry.map((line) => line.text)].join('\n')}\n`
}

export interface LineChange {
  kind: 'same' | 'add' | 'remove'
  text: string
}

interface DiffStep {
  kind: LineChange['kind']
  /** Index in the first document, or -1 for an added line. */
  before: number
  /** Index in the second document, or -1 for a removed line. */
  after: number
}

// A longest-common-subsequence diff. Lines both documents start or end with
// are matched first, so an edit to a long document only compares the middle.
function diffSteps(a: readonly string[], b: readonly string[]): DiffStep[] {
  let start = 0
  while (start < a.length && start < b.length && a[start] === b[start]) start += 1
  let endA = a.length
  let endB = b.length
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) {
    endA -= 1
    endB -= 1
  }
  const rows = endA - start
  const width = endB - start + 1
  const lengths = new Uint32Array((rows + 1) * width)
  for (let i = rows - 1; i >= 0; i -= 1) {
    for (let j = width - 2; j >= 0; j -= 1) {
      lengths[i * width + j] =
        a[start + i] === b[start + j]
          ? lengths[(i + 1) * width + j + 1] + 1
          : Math.max(lengths[(i + 1) * width + j], lengths[i * width + j + 1])
    }
  }

  const steps: DiffStep[] = []
  for (let index = 0; index < start; index += 1) {
    steps.push({ kind: 'same', before: index, after: index })
  }
  let i = 0
  let j = 0
  while (i < rows && j < width - 1) {
    if (a[start + i] === b[start + j]) {
      steps.push({ kind: 'same', before: start + i, after: start + j })
      i += 1
      j += 1
    } else if (lengths[(i + 1) * width + j] >= lengths[i * width + j + 1]) {
      steps.push({ kind: 'remove', before: start + i, after: -1 })
      i += 1
    } else {
      steps.push({ kind: 'add', before: -1, after: start + j })
      j += 1
    }
  }
  while (i < rows) steps.push({ kind: 'remove', before: start + i++, after: -1 })
  while (j < width - 1) steps.push({ kind: 'add', before: -1, after: start + j++ })
  for (let offset = 0; endA + offset < a.length; offset += 1) {
    steps.push({ kind: 'same', before: endA + offset, after: endB + offset })
  }
  return steps
}

/** A line diff between two short documents, in reading order. */
export function diffLines(before: string, after: string): LineChange[] {
  const a = before.replace(/\n$/, '').split('\n')
  const b = after.replace(/\n$/, '').split('\n')
  return diffSteps(a, b).map((step) => ({
    kind: step.kind,
    text: step.kind === 'add' ? b[step.after] : a[step.before],
  }))
}

/** A changed or context line of a YAML diff. */
export interface ConfigYamlChange extends LineChange {
  path: string[]
  /** The line's number before the change, for same and removed lines. */
  oldNumber?: number
  /** The line's number after the change, for same and added lines. */
  newNumber?: number
}

export interface ConfigYamlHunk {
  /** Dotted path of the mapping the hunk's first change is in; empty at the document root. */
  header: string
  lines: ConfigYamlChange[]
  /** Unchanged lines left out since the previous hunk, or since the start. */
  skippedBefore: number
}

/**
 * The line diff of two documents rendered as YAML, cut into hunks with
 * `context` unchanged lines around each change. `Infinity` keeps one hunk
 * with every line.
 */
export function configYamlDiff(
  before: Record<string, unknown>,
  after: Record<string, unknown>,
  { schema, context = 2 }: { schema?: Schema; context?: number } = {},
): ConfigYamlHunk[] {
  const a = configYamlLines(before, schema)
  const b = configYamlLines(after, schema)
  const changes: ConfigYamlChange[] = diffSteps(
    a.map((line) => line.text),
    b.map((line) => line.text),
  ).map((step) => {
    const line = step.kind === 'remove' ? a[step.before] : b[step.after]
    return {
      kind: step.kind,
      text: line.text,
      path: line.path,
      oldNumber: step.before >= 0 ? step.before + 1 : undefined,
      newNumber: step.after >= 0 ? step.after + 1 : undefined,
    }
  })

  const ranges: Array<[number, number]> = []
  changes.forEach((change, index) => {
    if (change.kind === 'same') return
    const start = Math.max(0, index - context)
    const end = Math.min(changes.length - 1, index + context)
    const last = ranges[ranges.length - 1]
    if (last && start <= last[1] + 1) last[1] = Math.max(last[1], end)
    else ranges.push([start, end])
  })

  let previousEnd = -1
  return ranges.map(([start, end]) => {
    const lines = changes.slice(start, end + 1)
    const firstChange = lines.find((line) => line.kind !== 'same') ?? lines[0]
    const hunk: ConfigYamlHunk = {
      header: firstChange.path
        .slice(0, -1)
        .filter((segment) => !/^\d+$/.test(segment))
        .join('.'),
      lines,
      skippedBefore: start - previousEnd - 1,
    }
    previousEnd = end
    return hunk
  })
}
