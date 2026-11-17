import { isRecord } from './configPageRouterSettingsDraft'

// Canonical YAML of the global override block, for the review diff and the
// read-only YAML view. Strings that YAML would read as another type are
// double-quoted, which keeps the output a valid document.

export interface YamlLine {
  text: string
  /** The key path the line belongs to, below the document root. */
  path: string[]
}

const RESERVED_SCALAR = /^(true|false|yes|no|on|off|null|~|y|n)$/i
const NUMBER_LIKE =
  /^[-+]?(\d[\d_]*(\.\d*)?|\.\d+)([eE][-+]?\d+)?$|^0x[0-9a-f]+$|^[-+]?\.(inf|nan)$/i
const PLAIN_KEY = /^[A-Za-z_$][\w.$/-]*$/

export function yamlScalar(value: unknown): string {
  if (value === null || value === undefined) return 'null'
  if (typeof value === 'number')
    return Number.isFinite(value) ? String(value) : JSON.stringify(String(value))
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  const text = String(value)
  const needsQuotes =
    text === '' ||
    RESERVED_SCALAR.test(text) ||
    NUMBER_LIKE.test(text) ||
    /^[\s\-?:,[\]{}#&*!|>'"%@`]/.test(text) ||
    /\s$/.test(text) ||
    /: |\s#|\n|\t/.test(text) ||
    text.endsWith(':')
  return needsQuotes ? JSON.stringify(text) : text
}

function yamlKey(key: string): string {
  return PLAIN_KEY.test(key) && !RESERVED_SCALAR.test(key) ? key : JSON.stringify(key)
}

function isEmptyCollection(value: unknown): boolean {
  return (
    (Array.isArray(value) && value.length === 0) ||
    (isRecord(value) && Object.keys(value).length === 0)
  )
}

function inlineValue(value: unknown): string {
  if (Array.isArray(value)) return '[]'
  if (isRecord(value)) return '{}'
  return yamlScalar(value)
}

function emitMapping(
  value: Record<string, unknown>,
  indent: number,
  path: string[],
  out: YamlLine[],
) {
  const pad = ' '.repeat(indent)
  for (const [key, child] of Object.entries(value)) {
    if (child === undefined) continue
    const childPath = [...path, key]
    if ((Array.isArray(child) || isRecord(child)) && !isEmptyCollection(child)) {
      out.push({ text: `${pad}${yamlKey(key)}:`, path: childPath })
      if (Array.isArray(child)) emitSequence(child, indent + 2, childPath, out)
      else emitMapping(child, indent + 2, childPath, out)
    } else {
      out.push({ text: `${pad}${yamlKey(key)}: ${inlineValue(child)}`, path: childPath })
    }
  }
}

function emitSequence(value: unknown[], indent: number, path: string[], out: YamlLine[]) {
  const pad = ' '.repeat(indent)
  value.forEach((item, index) => {
    const itemPath = [...path, String(index)]
    if (isRecord(item) && !isEmptyCollection(item)) {
      const nested: YamlLine[] = []
      emitMapping(item, indent + 2, itemPath, nested)
      nested.forEach((line, lineIndex) => {
        out.push(
          lineIndex === 0 ? { ...line, text: `${pad}- ${line.text.slice(indent + 2)}` } : line,
        )
      })
    } else if (Array.isArray(item) && item.length > 0) {
      out.push({ text: `${pad}-`, path: itemPath })
      emitSequence(item, indent + 2, itemPath, out)
    } else {
      out.push({ text: `${pad}- ${inlineValue(item)}`, path: itemPath })
    }
  })
}

/** YAML lines for a document whose root is a mapping. */
export function yamlLines(root: Record<string, unknown>): YamlLine[] {
  const out: YamlLine[] = []
  emitMapping(root, 0, [], out)
  return out
}

export function toYaml(root: Record<string, unknown>): string {
  return yamlLines(root)
    .map((line) => line.text)
    .join('\n')
}

export type DiffKind = 'equal' | 'add' | 'remove'

export interface DiffLine {
  kind: DiffKind
  text: string
  path: string[]
  oldNumber?: number
  newNumber?: number
}

export interface DiffHunk {
  /** Dotted path of the block the hunk changes. */
  header: string
  lines: DiffLine[]
  /** Unchanged lines skipped before this hunk. */
  skippedBefore: number
}

function lineOps(before: YamlLine[], after: YamlLine[]): DiffLine[] {
  const n = before.length
  const m = after.length
  const table = new Uint32Array((n + 1) * (m + 1))
  const at = (i: number, j: number) => i * (m + 1) + j
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      table[at(i, j)] =
        before[i].text === after[j].text
          ? table[at(i + 1, j + 1)] + 1
          : Math.max(table[at(i + 1, j)], table[at(i, j + 1)])
    }
  }
  const ops: DiffLine[] = []
  let i = 0
  let j = 0
  while (i < n || j < m) {
    if (i < n && j < m && before[i].text === after[j].text) {
      ops.push({
        kind: 'equal',
        text: after[j].text,
        path: after[j].path,
        oldNumber: i + 1,
        newNumber: j + 1,
      })
      i += 1
      j += 1
    } else if (i < n && (j === m || table[at(i + 1, j)] >= table[at(i, j + 1)])) {
      ops.push({ kind: 'remove', text: before[i].text, path: before[i].path, oldNumber: i + 1 })
      i += 1
    } else {
      ops.push({ kind: 'add', text: after[j].text, path: after[j].path, newNumber: j + 1 })
      j += 1
    }
  }
  return ops
}

/** A line diff of two YAML documents, cut into hunks with a few lines of context. */
export function yamlDiff(
  before: Record<string, unknown>,
  after: Record<string, unknown>,
  context = 2,
): DiffHunk[] {
  const ops = lineOps(yamlLines(before), yamlLines(after))
  const changed = ops
    .map((op, index) => (op.kind === 'equal' ? -1 : index))
    .filter((index) => index >= 0)
  if (changed.length === 0) return []

  const ranges: Array<[number, number]> = []
  for (const index of changed) {
    const start = Math.max(0, index - context)
    const end = Math.min(ops.length - 1, index + context)
    const last = ranges[ranges.length - 1]
    if (last && start <= last[1] + 1) last[1] = Math.max(last[1], end)
    else ranges.push([start, end])
  }

  let previousEnd = -1
  return ranges.map(([start, end]) => {
    const lines = ops.slice(start, end + 1)
    const firstChange = lines.find((line) => line.kind !== 'equal') ?? lines[0]
    const parent = firstChange.path.slice(0, -1).filter((segment) => !/^\d+$/.test(segment))
    const hunk: DiffHunk = {
      header: parent.join('.') || 'global',
      lines,
      skippedBefore: start - previousEnd - 1,
    }
    previousEnd = end
    return hunk
  })
}
