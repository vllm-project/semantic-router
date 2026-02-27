/**
 * DSL Text-Level Mutations
 *
 * Pure functions that operate on DSL source text to add, update, and delete
 * entities (signals, routes, plugins, backends, global). These work by finding
 * the relevant block in the DSL text using regex-based matching and brace counting,
 * then performing surgical string replacements.
 *
 * After each mutation, the caller should call parseAST() to refresh the AST.
 */

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
 *       findBlock(src, 'GLOBAL', null, null)
 */
export function findBlock(
  src: string,
  construct: 'SIGNAL' | 'ROUTE' | 'PLUGIN' | 'BACKEND' | 'GLOBAL',
  subType: string | null,
  name: string | null,
): BlockSpan | null {
  // Build a regex to find the block header
  let pattern: string
  if (construct === 'GLOBAL') {
    pattern = `^GLOBAL\\s*\\{`
  } else if (construct === 'SIGNAL' && subType) {
    pattern = name
      ? `^SIGNAL\\s+${escRe(subType)}\\s+${escRe(name)}\\s*\\{`
      : `^SIGNAL\\s+${escRe(subType)}\\s+\\S+\\s*\\{`
  } else if (construct === 'ROUTE') {
    // ROUTE name or ROUTE name (description = "...")
    pattern = `^ROUTE\\s+${escRe(name!)}\\s*(?:\\([^)]*\\))?\\s*\\{`
  } else if (construct === 'PLUGIN' && subType) {
    pattern = `^PLUGIN\\s+${escRe(name!)}\\s+${escRe(subType)}\\s*\\{`
  } else if (construct === 'BACKEND' && subType) {
    pattern = `^BACKEND\\s+${escRe(subType)}\\s+${escRe(name!)}\\s*\\{`
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

// ---------- Signal mutations ----------

/**
 * Serialize a signal's fields to DSL block body text.
 * Supports recursive indentation to match Go decompiler output format.
 */
export function serializeFields(fields: Record<string, unknown>, indent = '  ', opts?: { blankLineBefore?: boolean }): string {
  const lines: string[] = []
  const entries = Object.entries(fields).filter(([, v]) => v !== undefined && v !== null)
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
function countLeafFields(obj: Record<string, unknown>): number {
  let count = 0
  for (const v of Object.values(obj)) {
    if (v === undefined || v === null) continue
    if (typeof v === 'object' && !Array.isArray(v)) {
      count += countLeafFields(v as Record<string, unknown>)
    } else {
      count++
    }
  }
  return count
}

function serializeValue(value: unknown, currentIndent = '  '): string {
  if (typeof value === 'string') return `"${value}"`
  if (typeof value === 'number') return String(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
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
  if (typeof value === 'object' && value !== null) {
    const obj = value as Record<string, unknown>
    const entries = Object.entries(obj).filter(([, v]) => v !== undefined && v !== null)
    if (entries.length === 0) return '{}'
    // Inline small flat objects (≤3 leaf fields, all primitive) — matches Go decompiler style
    const leafCount = countLeafFields(obj)
    const allPrimitive = entries.every(([, v]) => typeof v !== 'object' || v === null)
    if (allPrimitive && leafCount <= 3) {
      const parts = entries.map(([k, v]) => `${k}: ${serializeValue(v, currentIndent)}`)
      return `{ ${parts.join(', ')} }`
    }
    // Multi-line nested object
    const childIndent = currentIndent + '  '
    const inner = serializeFields(obj, childIndent)
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
  fields: Record<string, unknown>,
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
  fields: Record<string, unknown>,
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
  fields: Record<string, unknown>,
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
  fields: Record<string, unknown>,
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
 * Update a backend declaration's fields.
 */
export function updateBackend(
  src: string,
  backendType: string,
  name: string,
  fields: Record<string, unknown>,
): string {
  const block = findBlock(src, 'BACKEND', backendType, name)
  if (!block) return src

  const body = serializeFields(fields)
  const newBlock = `BACKEND ${backendType} ${name} {\n${body}\n}\n`
  return src.slice(0, block.start) + newBlock + src.slice(block.end)
}

/**
 * Add a new backend declaration.
 */
export function addBackend(
  src: string,
  backendType: string,
  name: string,
  fields: Record<string, unknown>,
): string {
  const body = serializeFields(fields)
  const newBlock = `BACKEND ${backendType} ${name} {\n${body}\n}\n`

  const backendPattern = /^BACKEND\s+\S+\s+\S+\s*\{/gm
  let lastMatch: RegExpExecArray | null = null
  let m: RegExpExecArray | null
  while ((m = backendPattern.exec(src)) !== null) {
    lastMatch = m
  }

  if (lastMatch) {
    const lastBlock = findBlockFromIndex(src, lastMatch.index)
    if (lastBlock) {
      return src.slice(0, lastBlock.end) + '\n' + newBlock + src.slice(lastBlock.end)
    }
  }

  // Append before GLOBAL if exists
  const globalBlock = findBlock(src, 'GLOBAL', null, null)
  if (globalBlock) {
    return src.slice(0, globalBlock.start) + newBlock + '\n' + src.slice(globalBlock.start)
  }

  return src + '\n' + newBlock
}

/**
 * Delete a backend declaration.
 */
export function deleteBackend(src: string, backendType: string, name: string): string {
  const block = findBlock(src, 'BACKEND', backendType, name)
  if (!block) return src
  let result = src.slice(0, block.start) + src.slice(block.end)
  result = result.replace(/\n{3,}/g, '\n\n')
  return result
}

/**
 * Update the GLOBAL block's fields.
 * Uses blankLineBefore to match Go decompiler formatting.
 */
export function updateGlobal(
  src: string,
  fields: Record<string, unknown>,
): string {
  const block = findBlock(src, 'GLOBAL', null, null)
  if (!block) return src

  const body = serializeFields(fields, '  ', { blankLineBefore: true })
  const newBlock = `GLOBAL {\n${body}\n}\n`
  return src.slice(0, block.start) + newBlock + src.slice(block.end)
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
  fields: Record<string, unknown>
}

export interface RoutePluginInput {
  name: string
  fields?: Record<string, unknown>
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
export function serializeBoolExpr(expr: Record<string, unknown>): string {
  if (!expr || typeof expr !== 'object') return ''
  const type = expr.type as string
  switch (type) {
    case 'signal_ref':
      return `${expr.signalType}("${expr.signalName}")`
    case 'and':
      return `${serializeBoolExpr(expr.left as Record<string, unknown>)} AND ${serializeBoolExpr(expr.right as Record<string, unknown>)}`
    case 'or':
      return `(${serializeBoolExpr(expr.left as Record<string, unknown>)} OR ${serializeBoolExpr(expr.right as Record<string, unknown>)})`
    case 'not':
      return `NOT ${serializeBoolExpr(expr.expr as Record<string, unknown>)}`
    default:
      return ''
  }
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

// ---------- Signal type field schemas ----------

export interface FieldSchema {
  key: string
  label: string
  type: 'string' | 'number' | 'boolean' | 'string[]' | 'number[]' | 'select' | 'json'
  options?: string[]
  required?: boolean
  placeholder?: string
  description?: string
}

export const SIGNAL_TYPES = [
  'keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
  'preference', 'language', 'context', 'complexity', 'modality', 'authz',
  'jailbreak', 'pii',
] as const

export type SignalType = typeof SIGNAL_TYPES[number]

/**
 * Returns the field schema for a given signal type.
 */
export function getSignalFieldSchema(signalType: string): FieldSchema[] {
  switch (signalType) {
    case 'keyword':
      return [
        { key: 'operator', label: 'Operator', type: 'select', options: ['any', 'all', 'OR', 'AND'], required: true },
        { key: 'keywords', label: 'Keywords', type: 'string[]', required: true, placeholder: 'Add keyword...' },
        { key: 'method', label: 'Method', type: 'select', options: ['regex', 'bm25', 'ngram'] },
        { key: 'case_sensitive', label: 'Case Sensitive', type: 'boolean' },
        { key: 'fuzzy_match', label: 'Fuzzy Match', type: 'boolean' },
        { key: 'fuzzy_threshold', label: 'Fuzzy Threshold', type: 'number', placeholder: '2' },
        { key: 'bm25_threshold', label: 'BM25 Threshold', type: 'number' },
        { key: 'ngram_threshold', label: 'N-gram Threshold', type: 'number' },
        { key: 'ngram_arity', label: 'N-gram Arity', type: 'number' },
      ]
    case 'embedding':
      return [
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.75' },
        { key: 'candidates', label: 'Candidates', type: 'string[]', required: true, placeholder: 'Add candidate...' },
        { key: 'aggregation_method', label: 'Aggregation', type: 'select', options: ['mean', 'max', 'any'] },
      ]
    case 'domain':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
        { key: 'mmlu_categories', label: 'MMLU Categories', type: 'string[]', placeholder: 'Add category...' },
        { key: 'model_scores', label: 'Model Scores', type: 'json' },
      ]
    case 'fact_check':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
      ]
    case 'user_feedback':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
      ]
    case 'preference':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
      ]
    case 'language':
      return [
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'context':
      return [
        { key: 'min_tokens', label: 'Min Tokens', type: 'string', required: true, placeholder: '4K' },
        { key: 'max_tokens', label: 'Max Tokens', type: 'string', required: true, placeholder: '32K' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'complexity':
      return [
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.1' },
        { key: 'hard', label: 'Hard Examples', type: 'json', description: 'e.g. { candidates: ["..."] }' },
        { key: 'easy', label: 'Easy Examples', type: 'json', description: 'e.g. { candidates: ["..."] }' },
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'composer', label: 'Composer', type: 'string' },
      ]
    case 'modality':
      return [
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'authz':
      return [
        { key: 'subjects', label: 'Subjects', type: 'json', required: true, description: '[{ kind: "Group", name: "..." }]' },
        { key: 'role', label: 'Role', type: 'string', required: true, placeholder: 'premium_tier' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'jailbreak':
      return [
        { key: 'method', label: 'Method', type: 'select', options: ['classifier', 'contrastive'], description: 'Detection algorithm' },
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.9', description: 'Minimum score to trigger (0.0-1.0)' },
        { key: 'include_history', label: 'Include History', type: 'boolean', description: 'Include conversation history in detection' },
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'jailbreak_patterns', label: 'Jailbreak Patterns', type: 'string[]', placeholder: 'Add jailbreak example...', description: 'Contrastive mode: example jailbreak prompts' },
        { key: 'benign_patterns', label: 'Benign Patterns', type: 'string[]', placeholder: 'Add benign example...', description: 'Contrastive mode: example benign prompts' },
      ]
    case 'pii':
      return [
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.8', description: 'Minimum confidence for PII detection (0.0-1.0)' },
        { key: 'pii_types_allowed', label: 'PII Types Allowed', type: 'string[]', placeholder: 'e.g. EMAIL_ADDRESS', description: 'PII types to allow through (others trigger signal)' },
        { key: 'include_history', label: 'Include History', type: 'boolean', description: 'Include conversation history in detection' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    default:
      return [
        { key: 'description', label: 'Description', type: 'string' },
      ]
  }
}

export const PLUGIN_TYPES = [
  'semantic_cache', 'memory', 'system_prompt',
  'header_mutation', 'hallucination', 'router_replay', 'rag', 'image_gen',
  'fast_response',
] as const

/** Description for each plugin type shown in the UI */
export const PLUGIN_DESCRIPTIONS: Record<string, string> = {
  semantic_cache: 'Cache semantically similar queries to reduce latency and cost',
  memory: 'Persistent conversation memory with vector retrieval',
  system_prompt: 'Inject or replace system prompts for the model',
  header_mutation: 'Add, update, or remove HTTP headers on requests/responses',
  hallucination: 'Detect hallucinated content using NLI or other methods',
  router_replay: 'Record request/response pairs for replay and debugging',
  rag: 'Retrieval-Augmented Generation — inject retrieved context into prompts',
  image_gen: 'Route to image generation backends',
  fast_response: 'Short-circuit and return a fixed response without calling upstream models',
}

/** Get typed field schema for a given plugin type */
export function getPluginFieldSchema(pluginType: string): FieldSchema[] {
  switch (pluginType) {
    case 'semantic_cache':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'similarity_threshold', label: 'Similarity Threshold', type: 'number', placeholder: '0.95', description: 'Minimum similarity for cache hit (0-1)' },
      ]
    case 'memory':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'retrieval_limit', label: 'Retrieval Limit', type: 'number', placeholder: '5', description: 'Max memories to retrieve' },
        { key: 'similarity_threshold', label: 'Similarity Threshold', type: 'number', placeholder: '0.7' },
        { key: 'auto_store', label: 'Auto Store', type: 'boolean', description: 'Automatically store conversation turns' },
      ]
    case 'system_prompt':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'system_prompt', label: 'System Prompt', type: 'string', required: true, placeholder: 'You are a helpful assistant...' },
        { key: 'mode', label: 'Mode', type: 'select', options: ['', 'replace', 'insert'], description: 'Replace or insert before existing prompt' },
      ]
    case 'hallucination':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'use_nli', label: 'Use NLI', type: 'boolean', description: 'Use Natural Language Inference for detection' },
        { key: 'hallucination_action', label: 'Action', type: 'select', options: ['', 'header', 'body', 'none'], description: 'What to do when hallucination is detected' },
      ]
    case 'router_replay':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'max_records', label: 'Max Records', type: 'number', placeholder: '10000' },
        { key: 'capture_request_body', label: 'Capture Request Body', type: 'boolean' },
        { key: 'capture_response_body', label: 'Capture Response Body', type: 'boolean' },
        { key: 'max_body_bytes', label: 'Max Body Bytes', type: 'number', placeholder: '4096' },
      ]
    case 'rag':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'backend', label: 'Backend', type: 'string', required: true, placeholder: 'my_vector_store', description: 'Backend name for retrieval' },
        { key: 'top_k', label: 'Top K', type: 'number', placeholder: '5', description: 'Number of documents to retrieve' },
        { key: 'similarity_threshold', label: 'Similarity Threshold', type: 'number', placeholder: '0.7' },
        { key: 'injection_mode', label: 'Injection Mode', type: 'select', options: ['', 'system', 'user', 'context'] },
        { key: 'on_failure', label: 'On Failure', type: 'select', options: ['', 'skip', 'fail'] },
      ]
    case 'header_mutation':
      return [
        { key: 'add', label: 'Add Headers', type: 'json', description: '[{ "name": "X-Custom", "value": "..." }]' },
        { key: 'update', label: 'Update Headers', type: 'json', description: '[{ "name": "X-Custom", "value": "..." }]' },
        { key: 'delete', label: 'Delete Headers', type: 'string[]', placeholder: 'Header name to delete' },
      ]
    case 'image_gen':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'backend', label: 'Backend', type: 'string', required: true, placeholder: 'my_image_gen_backend' },
      ]
    case 'fast_response':
      return [
        { key: 'message', label: 'Message', type: 'string', required: true, placeholder: 'I cannot help with that request.', description: 'The response message returned directly to the client' },
      ]
    default:
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
      ]
  }
}

export const BACKEND_TYPES = [
  'vllm_endpoint', 'provider_profile', 'embedding_model', 'semantic_cache',
  'memory', 'response_api', 'vector_store', 'image_gen_backend',
] as const

export const ALGORITHM_TYPES = [
  'confidence', 'ratings', 'remom', 'static', 'elo', 'router_dc', 'automix',
  'hybrid', 'rl_driven', 'gmtrouter', 'latency_aware', 'knn', 'kmeans', 'svm',
] as const

export type AlgorithmType = typeof ALGORITHM_TYPES[number]

/** Description for each algorithm type shown in the UI */
export const ALGORITHM_DESCRIPTIONS: Record<string, string> = {
  confidence: 'Try smaller models first, escalate to larger models if confidence is low',
  ratings: 'Execute all models concurrently and return multiple choices for comparison',
  remom: 'Multi-round parallel reasoning with intelligent synthesis (ReMoM)',
  static: 'Use static scores from configuration (no extra fields)',
  elo: 'Elo rating system with Bradley-Terry model for model selection',
  router_dc: 'Dual-contrastive learning for query-model matching',
  automix: 'POMDP-based cost-quality optimization (arXiv:2310.12963)',
  hybrid: 'Combine multiple selection methods with configurable weights',
  rl_driven: 'Reinforcement learning with Thompson Sampling (arXiv:2506.09033)',
  gmtrouter: 'Heterogeneous graph learning for personalized routing',
  latency_aware: 'TPOT/TTFT percentile thresholds for latency-aware model selection',
  knn: 'K-Nearest Neighbors for query-based model selection (no extra fields)',
  kmeans: 'KMeans clustering for model selection (no extra fields)',
  svm: 'Support Vector Machine for model classification (no extra fields)',
}

/** Get typed field schema for a given algorithm type */
export function getAlgorithmFieldSchema(algoType: string): FieldSchema[] {
  switch (algoType) {
    case 'confidence':
      return [
        { key: 'confidence_method', label: 'Confidence Method', type: 'select', options: ['avg_logprob', 'margin', 'hybrid', 'self_verify'], description: 'How to evaluate model confidence' },
        { key: 'threshold', label: 'Threshold', type: 'number', placeholder: '-1.0', description: 'Confidence threshold for escalation' },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
        { key: 'escalation_order', label: 'Escalation Order', type: 'select', options: ['', 'size', 'cost', 'automix'], description: 'How models are ordered for cascade' },
        { key: 'cost_quality_tradeoff', label: 'Cost/Quality Tradeoff', type: 'number', placeholder: '0.3', description: '0.0=quality, 1.0=cost (for automix order)' },
      ]
    case 'ratings':
      return [
        { key: 'max_concurrent', label: 'Max Concurrent', type: 'number', placeholder: '0 (no limit)', description: 'Limit concurrent model calls' },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
      ]
    case 'remom':
      return [
        { key: 'breadth_schedule', label: 'Breadth Schedule', type: 'number[]', required: true, placeholder: 'e.g. 32', description: 'Parallel calls per round, e.g. [4], [16], [32, 4]' },
        { key: 'model_distribution', label: 'Model Distribution', type: 'select', options: ['', 'weighted', 'equal', 'first_only'] },
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '1.0' },
        { key: 'include_reasoning', label: 'Include Reasoning', type: 'boolean', description: 'Include reasoning content in synthesis' },
        { key: 'compaction_strategy', label: 'Compaction Strategy', type: 'select', options: ['', 'full', 'last_n_tokens'] },
        { key: 'compaction_tokens', label: 'Compaction Tokens', type: 'number', placeholder: '1000', description: 'Tokens to keep (last_n_tokens strategy)' },
        { key: 'max_concurrent', label: 'Max Concurrent', type: 'number', placeholder: '0 (no limit)' },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
        { key: 'include_intermediate_responses', label: 'Include Intermediate', type: 'boolean', description: 'Save intermediate responses for dashboard' },
      ]
    case 'elo':
      return [
        { key: 'initial_rating', label: 'Initial Rating', type: 'number', placeholder: '1500' },
        { key: 'k_factor', label: 'K Factor', type: 'number', placeholder: '32', description: 'Rating volatility' },
        { key: 'category_weighted', label: 'Category Weighted', type: 'boolean', description: 'Per-category Elo ratings' },
        { key: 'decay_factor', label: 'Decay Factor', type: 'number', placeholder: '0 (no decay)', description: 'Time decay 0-1' },
        { key: 'min_comparisons', label: 'Min Comparisons', type: 'number', placeholder: '5' },
        { key: 'cost_scaling_factor', label: 'Cost Scaling', type: 'number', placeholder: '0', description: '0 = ignore cost' },
        { key: 'storage_path', label: 'Storage Path', type: 'string', placeholder: '/tmp/elo' },
      ]
    case 'router_dc':
      return [
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '0.07', description: 'Softmax scaling' },
        { key: 'dimension_size', label: 'Dimension Size', type: 'number', placeholder: '768' },
        { key: 'min_similarity', label: 'Min Similarity', type: 'number', placeholder: '0.3' },
        { key: 'use_query_contrastive', label: 'Query Contrastive', type: 'boolean' },
        { key: 'use_model_contrastive', label: 'Model Contrastive', type: 'boolean' },
      ]
    case 'automix':
      return [
        { key: 'verification_threshold', label: 'Verification Threshold', type: 'number', placeholder: '0.7' },
        { key: 'max_escalations', label: 'Max Escalations', type: 'number', placeholder: '2' },
        { key: 'cost_aware_routing', label: 'Cost-Aware Routing', type: 'boolean' },
        { key: 'cost_quality_tradeoff', label: 'Cost/Quality Tradeoff', type: 'number', placeholder: '0.3' },
        { key: 'discount_factor', label: 'Discount Factor', type: 'number', placeholder: '0.95', description: 'POMDP value iteration' },
      ]
    case 'hybrid':
      return [
        { key: 'elo_weight', label: 'Elo Weight', type: 'number', placeholder: '0.3' },
        { key: 'router_dc_weight', label: 'RouterDC Weight', type: 'number', placeholder: '0.3' },
        { key: 'automix_weight', label: 'AutoMix Weight', type: 'number', placeholder: '0.2' },
        { key: 'cost_weight', label: 'Cost Weight', type: 'number', placeholder: '0.2' },
        { key: 'quality_gap_threshold', label: 'Quality Gap Threshold', type: 'number', placeholder: '0.1' },
        { key: 'normalize_scores', label: 'Normalize Scores', type: 'boolean' },
      ]
    case 'rl_driven':
      return [
        { key: 'exploration_rate', label: 'Exploration Rate', type: 'number', placeholder: '0.3', description: '0-1' },
        { key: 'use_thompson_sampling', label: 'Thompson Sampling', type: 'boolean' },
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        { key: 'personalization_blend', label: 'Personalization Blend', type: 'number', placeholder: '0.3', description: 'Global vs user-specific (0-1)' },
        { key: 'cost_awareness', label: 'Cost Awareness', type: 'boolean' },
      ]
    case 'gmtrouter':
      return [
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        { key: 'history_sample_size', label: 'History Sample Size', type: 'number', placeholder: '5' },
        { key: 'min_interactions_for_personalization', label: 'Min Interactions', type: 'number' },
        { key: 'max_interactions_per_user', label: 'Max Interactions/User', type: 'number', placeholder: '100' },
        { key: 'model_path', label: 'Model Path', type: 'string', placeholder: '/models/gmt' },
      ]
    case 'latency_aware':
      return [
        { key: 'tpot_percentile', label: 'TPOT Percentile', type: 'number', required: true, placeholder: '20', description: 'Time Per Output Token (1-100)' },
        { key: 'ttft_percentile', label: 'TTFT Percentile', type: 'number', required: true, placeholder: '20', description: 'Time To First Token (1-100)' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    // static, knn, kmeans, svm — no configurable fields
    default:
      return []
  }
}
