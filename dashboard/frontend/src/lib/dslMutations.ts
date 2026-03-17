/**
 * DSL Text-Level Mutations
 *
 * Pure functions that operate on DSL source text to add, update, and delete
 * entities (models, signals, routes, plugins). These work by finding
 * the relevant block in the DSL text using regex-based matching and brace counting,
 * then performing surgical string replacements.
 *
 * After each mutation, the caller should call parseAST() to refresh the AST.
 */

import type { BoolExprNode, DSLFieldObject, DSLFieldValue } from '@/types/dsl'

// ---------- Block finding ----------

interface BlockSpan {
  start: number  // char index of block start (e.g., "SIGNAL keyword ...")
  end: number    // char index after closing brace + newline
  body: string   // the full block text
}

/**
 * Find a top-level DSL block by construct + name.
 * E.g., findBlock(src, 'SIGNAL', 'keyword', 'urgent_request')
 *       findBlock(src, 'ROUTE', null, 'math_decision')
 */
export function findBlock(
  src: string,
  construct: 'MODEL' | 'SIGNAL' | 'ROUTE' | 'PLUGIN',
  subType: string | null,
  name: string | null,
): BlockSpan | null {
  // Build a regex to find the block header
  let pattern: string
  if (construct === 'MODEL') {
    pattern = `^MODEL\\s+${dslNamePattern(name!)}\\s*\\{`
  } else if (construct === 'SIGNAL' && subType) {
    pattern = name
      ? `^SIGNAL\\s+${escRe(subType)}\\s+${escRe(name)}\\s*\\{`
      : `^SIGNAL\\s+${escRe(subType)}\\s+\\S+\\s*\\{`
  } else if (construct === 'ROUTE') {
    // ROUTE name or ROUTE name (description = "...")
    pattern = `^ROUTE\\s+${escRe(name!)}\\s*(?:\\([^)]*\\))?\\s*\\{`
  } else if (construct === 'PLUGIN' && subType) {
    pattern = `^PLUGIN\\s+${escRe(name!)}\\s+${escRe(subType)}\\s*\\{`
  } else {
    // Generic fallback
    const parts: string[] = [construct]
    if (subType) parts.push(escRe(subType))
    if (name) parts.push(escRe(name))
    pattern = `^${parts.join('\\s+')}\\s*\\{`
  }

  const regex = new RegExp(pattern, 'm')
  const match = regex.exec(src)
  if (!match) return null

  // Now find the matching closing brace via brace counting
  const blockStart = match.index
  let braceCount = 0
  let blockEnd = blockStart
  let foundOpenBrace = false

  for (let i = blockStart; i < src.length; i++) {
    if (src[i] === '{') {
      braceCount++
      foundOpenBrace = true
    } else if (src[i] === '}') {
      braceCount--
      if (foundOpenBrace && braceCount === 0) {
        blockEnd = i + 1
        // Include trailing newline(s)
        while (blockEnd < src.length && (src[blockEnd] === '\n' || src[blockEnd] === '\r')) {
          blockEnd++
        }
        break
      }
    }
  }

  if (!foundOpenBrace || braceCount !== 0) return null

  return {
    start: blockStart,
    end: blockEnd,
    body: src.slice(blockStart, blockEnd),
  }
}

// ---------- Model mutations ----------

/**
 * Update a top-level MODEL declaration's fields.
 */
export function updateModel(
  src: string,
  name: string,
  fields: DSLFieldObject,
): string {
  const block = findBlock(src, 'MODEL', null, name)
  if (!block) return src

  const body = serializeFields(fields)
  const newBlock = `MODEL ${formatDslName(name)} {\n${body}\n}\n`
  return src.slice(0, block.start) + newBlock + src.slice(block.end)
}

/**
 * Add a new top-level MODEL declaration.
 */
export function addModel(
  src: string,
  name: string,
  fields: DSLFieldObject,
): string {
  const body = serializeFields(fields)
  const newBlock = `MODEL ${formatDslName(name)} {\n${body}\n}\n`

  const modelPattern = /^MODEL\s+(?:"[^"]+"|\S+)\s*\{/gm
  let lastMatch: RegExpExecArray | null = null
  let m: RegExpExecArray | null
  while ((m = modelPattern.exec(src)) !== null) {
    lastMatch = m
  }

  if (lastMatch) {
    const lastBlock = findBlockFromIndex(src, lastMatch.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  const signalPattern = /^SIGNAL\s+\S+\s+\S+\s*\{/gm
  let lastSignal: RegExpExecArray | null = null
  while ((m = signalPattern.exec(src)) !== null) {
    lastSignal = m
  }
  if (lastSignal) {
    const lastBlock = findBlockFromIndex(src, lastSignal.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  const firstNonCommentLine = findFirstNonCommentLine(src)
  return src.slice(0, firstNonCommentLine) + newBlock + '\n' + src.slice(firstNonCommentLine)
}

/**
 * Delete a top-level MODEL declaration.
 */
export function deleteModel(src: string, name: string): string {
  const block = findBlock(src, 'MODEL', null, name)
  if (!block) return src

  let result = src.slice(0, block.start) + src.slice(block.end)
  result = result.replace(/\n{3,}/g, '\n\n')
  return result
}

// ---------- Signal mutations ----------

/**
 * Serialize a signal's fields to DSL block body text.
 * Supports recursive indentation to match Go decompiler output format.
 */
export function serializeFields(fields: DSLFieldObject, indent = '  ', opts?: { blankLineBefore?: boolean }): string {
  const lines: string[] = []
  const entries = Object.entries(fields).filter(([, v]) => v !== undefined && v !== null) as Array<[string, DSLFieldValue]>
  for (const [key, value] of entries) {
    const serialized = serializeValue(value, indent)
    // Add blank line before nested object blocks (matches Go decompiler)
    if (opts?.blankLineBefore && typeof value === 'object' && !Array.isArray(value) && value !== null) {
      lines.push('')
    }
    lines.push(`${indent}${key}: ${serialized}`)
  }
  return lines.join('\n')
}

/**
 * Count the number of leaf (non-object) fields in an object, recursively.
 */
function countLeafFields(obj: DSLFieldObject): number {
  let count = 0
  for (const v of Object.values(obj)) {
    if (v === undefined || v === null) continue
    if (isDSLFieldObject(v)) {
      count += countLeafFields(v)
    } else {
      count++
    }
  }
  return count
}

function serializeValue(value: DSLFieldValue, currentIndent = '  '): string {
  if (typeof value === 'string') return `"${value}"`
  if (typeof value === 'number') return String(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (value === null) return 'null'
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    const simple = value.every(v => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean')
    if (simple) {
      return `[${value.map(v => serializeValue(v, currentIndent)).join(', ')}]`
    }
    const childIndent = currentIndent + '  '
    const items = value.map(v => `${childIndent}${serializeValue(v, childIndent)}`).join(',\n')
    return `[\n${items}\n${currentIndent}]`
  }
  if (isDSLFieldObject(value)) {
    const entries = Object.entries(value).filter(([, v]) => v !== undefined && v !== null) as Array<[string, DSLFieldValue]>
    if (entries.length === 0) return '{}'
    // Inline small flat objects (≤3 leaf fields, all primitive) — matches Go decompiler style
    const leafCount = countLeafFields(value)
    const allPrimitive = entries.every(([, v]) => typeof v !== 'object' || v === null)
    if (allPrimitive && leafCount <= 3) {
      const parts = entries.map(([k, v]) => `${k}: ${serializeValue(v, currentIndent)}`)
      return `{ ${parts.join(', ')} }`
    }
    // Multi-line nested object
    const childIndent = currentIndent + '  '
    const inner = serializeFields(value, childIndent)
    if (!inner.trim()) return '{}'
    return `{\n${inner}\n${currentIndent}}`
  }
  return String(value)
}

/**
 * Update a signal's fields in DSL source.
 * Replaces the entire block body with new field values.
 */
export function updateSignal(
  src: string,
  signalType: string,
  name: string,
  fields: DSLFieldObject,
): string {
  const block = findBlock(src, 'SIGNAL', signalType, name)
  if (!block) return src

  const body = serializeFields(fields)
  const newBlock = `SIGNAL ${signalType} ${name} {\n${body}\n}\n`

  return src.slice(0, block.start) + newBlock + src.slice(block.end)
}

/**
 * Add a new signal to the DSL source.
 * Inserts after the last existing SIGNAL block, or at the top if none exist.
 */
export function addSignal(
  src: string,
  signalType: string,
  name: string,
  fields: DSLFieldObject,
): string {
  const body = serializeFields(fields)
  const newBlock = `SIGNAL ${signalType} ${name} {\n${body}\n}\n`

  // Find last SIGNAL block
  const signalPattern = /^SIGNAL\s+\S+\s+\S+\s*\{/gm
  let lastMatch: RegExpExecArray | null = null
  let m: RegExpExecArray | null
  while ((m = signalPattern.exec(src)) !== null) {
    lastMatch = m
  }

  if (lastMatch) {
    // Find the end of the last signal block
    const lastBlock = findBlockFromIndex(src, lastMatch.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  // No existing signals — insert at top (after any comments)
  const firstNonCommentLine = findFirstNonCommentLine(src)
  return src.slice(0, firstNonCommentLine) + newBlock + '\n' + src.slice(firstNonCommentLine)
}

/**
 * Delete a signal from the DSL source.
 */
export function deleteSignal(src: string, signalType: string, name: string): string {
  const block = findBlock(src, 'SIGNAL', signalType, name)
  if (!block) return src

  // Remove extra blank lines left behind
  let result = src.slice(0, block.start) + src.slice(block.end)
  result = result.replace(/\n{3,}/g, '\n\n')
  return result
}

/**
 * Update a plugin declaration's fields.
 */
export function updatePlugin(
  src: string,
  name: string,
  pluginType: string,
  fields: DSLFieldObject,
): string {
  const block = findBlock(src, 'PLUGIN', pluginType, name)
  if (!block) return src

  const body = serializeFields(fields)
  const newBlock = `PLUGIN ${name} ${pluginType} {\n${body}\n}\n`
  return src.slice(0, block.start) + newBlock + src.slice(block.end)
}

/**
 * Add a new plugin declaration.
 */
export function addPlugin(
  src: string,
  name: string,
  pluginType: string,
  fields: DSLFieldObject,
): string {
  const body = serializeFields(fields)
  const newBlock = `PLUGIN ${name} ${pluginType} {\n${body}\n}\n`

  // Find insertion point — after last PLUGIN or after last SIGNAL
  const pluginPattern = /^PLUGIN\s+\S+\s+\S+\s*\{/gm
  let lastMatch: RegExpExecArray | null = null
  let m: RegExpExecArray | null
  while ((m = pluginPattern.exec(src)) !== null) {
    lastMatch = m
  }

  if (lastMatch) {
    const lastBlock = findBlockFromIndex(src, lastMatch.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  // No plugins — insert after signals
  const signalPattern2 = /^SIGNAL\s+\S+\s+\S+\s*\{/gm
  let lastSig: RegExpExecArray | null = null
  while ((m = signalPattern2.exec(src)) !== null) {
    lastSig = m
  }
  if (lastSig) {
    const lastBlock = findBlockFromIndex(src, lastSig.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  return src + '\n' + newBlock
}

/**
 * Delete a plugin declaration.
 */
export function deletePlugin(src: string, name: string, pluginType: string): string {
  const block = findBlock(src, 'PLUGIN', pluginType, name)
  if (!block) return src
  let result = src.slice(0, block.start) + src.slice(block.end)
  result = result.replace(/\n{3,}/g, '\n\n')
  return result
}

/**
 * Delete a route declaration.
 */
export function deleteRoute(src: string, name: string): string {
  const block = findBlock(src, 'ROUTE', null, name)
  if (!block) return src
  let result = src.slice(0, block.start) + src.slice(block.end)
  result = result.replace(/\n{3,}/g, '\n\n')
  return result
}

// ---------- Route mutations ----------

export interface RouteModelInput {
  model: string
  reasoning?: boolean
  effort?: string
  lora?: string
  paramSize?: string
  weight?: number
  reasoningFamily?: string
}

export interface RouteAlgoInput {
  algoType: string
  fields: DSLFieldObject
}

export interface RoutePluginInput {
  name: string
  fields?: DSLFieldObject
}

export interface RouteInput {
  description?: string
  priority: number
  when?: string            // raw WHEN expression text, e.g. "domain(\"math\") AND complexity(\"hard\")"
  models: RouteModelInput[]
  algorithm?: RouteAlgoInput
  plugins: RoutePluginInput[]
}

function serializeRouteBody(input: RouteInput): string {
  const lines: string[] = []

  // Priority
  lines.push(`  PRIORITY ${input.priority}`)
  lines.push('')

  // WHEN
  if (input.when && input.when.trim()) {
    lines.push(`  WHEN ${input.when.trim()}`)
    lines.push('')
  }

  // Models
  if (input.models.length > 0) {
    const modelParts = input.models.map((m) => {
      const attrs: string[] = []
      if (m.reasoning !== undefined) attrs.push(`reasoning = ${m.reasoning}`)
      if (m.effort) attrs.push(`effort = "${m.effort}"`)
      if (m.lora) attrs.push(`lora = "${m.lora}"`)
      if (m.paramSize) attrs.push(`param_size = "${m.paramSize}"`)
      if (m.weight !== undefined) attrs.push(`weight = ${m.weight}`)
      if (m.reasoningFamily) attrs.push(`reasoning_family = "${m.reasoningFamily}"`)
      const attrStr = attrs.length > 0 ? ` (${attrs.join(', ')})` : ''
      return `"${m.model}"${attrStr}`
    })
    if (modelParts.length === 1) {
      lines.push(`  MODEL ${modelParts[0]}`)
    } else {
      lines.push(`  MODEL ${modelParts.join(',\n        ')}`)
    }
    lines.push('')
  }

  // Algorithm
  if (input.algorithm && input.algorithm.algoType) {
    const algoFields = serializeFields(input.algorithm.fields, '    ')
    if (algoFields.trim()) {
      lines.push(`  ALGORITHM ${input.algorithm.algoType} {`)
      lines.push(algoFields)
      lines.push(`  }`)
    } else {
      lines.push(`  ALGORITHM ${input.algorithm.algoType} {}`)
    }
    lines.push('')
  }

  // Plugins
  for (const p of input.plugins) {
    if (p.fields && Object.keys(p.fields).length > 0) {
      const pluginFields = serializeFields(p.fields, '    ')
      lines.push(`  PLUGIN ${p.name} {`)
      lines.push(pluginFields)
      lines.push(`  }`)
    } else {
      lines.push(`  PLUGIN ${p.name}`)
    }
  }

  return lines.join('\n')
}

/**
 * Update a route's content in DSL source.
 */
export function updateRoute(
  src: string,
  name: string,
  input: RouteInput,
): string {
  const block = findBlock(src, 'ROUTE', null, name)
  if (!block) return src

  const descPart = input.description ? ` (description = "${input.description}")` : ''
  const body = serializeRouteBody(input)
  const newBlock = `ROUTE ${name}${descPart} {\n${body}\n}\n`
  return src.slice(0, block.start) + newBlock + src.slice(block.end)
}

/**
 * Add a new route to the DSL source.
 * Inserts after the last existing ROUTE block, or after signals if none exist.
 */
export function addRoute(
  src: string,
  name: string,
  input: RouteInput,
): string {
  const descPart = input.description ? ` (description = "${input.description}")` : ''
  const body = serializeRouteBody(input)
  const newBlock = `ROUTE ${name}${descPart} {\n${body}\n}\n`

  // Find last ROUTE block
  const routePattern = /^ROUTE\s+\S+/gm
  let lastMatch: RegExpExecArray | null = null
  let m: RegExpExecArray | null
  while ((m = routePattern.exec(src)) !== null) {
    lastMatch = m
  }

  if (lastMatch) {
    const lastBlock = findBlockFromIndex(src, lastMatch.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  // No routes — insert after signals
  const signalPattern2 = /^SIGNAL\s+\S+\s+\S+\s*\{/gm
  let lastSig: RegExpExecArray | null = null
  while ((m = signalPattern2.exec(src)) !== null) {
    lastSig = m
  }
  if (lastSig) {
    const lastBlock = findBlockFromIndex(src, lastSig.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  return src + '\n' + newBlock
}

/**
 * Serialize a BoolExprNode back to DSL text.
 */
export function serializeBoolExpr(expr: BoolExprNode | null): string {
  if (!expr) return ''
  const type = expr.type
  switch (type) {
    case 'signal_ref':
      return `${expr.signalType}("${expr.signalName}")`
    case 'and':
      return `${serializeBoolExpr(expr.left)} AND ${serializeBoolExpr(expr.right)}`
    case 'or':
      return `(${serializeBoolExpr(expr.left)} OR ${serializeBoolExpr(expr.right)})`
    case 'not':
      return `NOT ${serializeBoolExpr(expr.expr)}`
    default:
      return ''
  }
}

function isDSLFieldObject(value: DSLFieldValue): value is DSLFieldObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Rename an entity (signal, plugin, backend).
 * Finds the block and replaces the old name with the new name in the header.
 */
export function renameSignal(
  src: string,
  signalType: string,
  oldName: string,
  newName: string,
): string {
  const block = findBlock(src, 'SIGNAL', signalType, oldName)
  if (!block) return src

  const newHeader = block.body.replace(
    new RegExp(`^SIGNAL\\s+${escRe(signalType)}\\s+${escRe(oldName)}`),
    `SIGNAL ${signalType} ${newName}`,
  )
  return src.slice(0, block.start) + newHeader + src.slice(block.end)
}

// ---------- Helpers ----------

function escRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function dslNamePattern(name: string): string {
  return `(?:${escRe(name)}|"${escRe(name)}")`
}

function formatDslName(name: string): string {
  return /^[_A-Za-z][\w]*$/.test(name) ? name : `"${name}"`
}

function findBlockFromIndex(src: string, startIndex: number): BlockSpan | null {
  let braceCount = 0
  let blockEnd = startIndex
  let foundOpenBrace = false

  for (let i = startIndex; i < src.length; i++) {
    if (src[i] === '{') {
      braceCount++
      foundOpenBrace = true
    } else if (src[i] === '}') {
      braceCount--
      if (foundOpenBrace && braceCount === 0) {
        blockEnd = i + 1
        while (blockEnd < src.length && (src[blockEnd] === '\n' || src[blockEnd] === '\r')) {
          blockEnd++
        }
        return { start: startIndex, end: blockEnd, body: src.slice(startIndex, blockEnd) }
      }
    }
  }
  return null
}

function findFirstNonCommentLine(src: string): number {
  const lines = src.split('\n')
  let offset = 0
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed === '' || trimmed.startsWith('#')) {
      offset += line.length + 1
    } else {
      break
    }
  }
  return offset
}

export type {
  AlgorithmType,
  FieldSchema,
  SignalType,
} from './dslSchemas'
export {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  BACKEND_TYPES,
  getAlgorithmFieldSchema,
  getPluginFieldSchema,
  getSignalFieldSchema,
  PLUGIN_DESCRIPTIONS,
  PLUGIN_TYPES,
  SIGNAL_TYPES,
} from './dslSchemas'
