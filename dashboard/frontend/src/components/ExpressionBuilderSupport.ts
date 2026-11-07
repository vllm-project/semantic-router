import { formatSignalRef } from '@/lib/dslMutations'
import type { DSLFieldObject, DSLFieldValue } from '@/types/dsl'

export type SignalDescriptor = { signalType: string; name: string }

export type SignalRuleNode = { signalType: string; signalName: string; fields?: DSLFieldObject }

export type RuleNode =
  | { operator: 'AND' | 'OR'; conditions: RuleNode[] }
  | { operator: 'NOT'; conditions: [RuleNode] }
  | SignalRuleNode

export type NodePath = number[]

export interface DragDataSignal {
  kind: 'signal'
  signalType: string
  signalName: string
}

export interface DragDataOperator {
  kind: 'operator'
  operator: 'AND' | 'OR' | 'NOT'
}

export interface DragDataTreeNode {
  kind: 'tree-node'
  path: NodePath
}

export type DragData = DragDataSignal | DragDataOperator | DragDataTreeNode

interface ParseCtx {
  pos: number
}

export const DRAG_MIME = 'application/x-expr-builder'

export function isLeaf(n: RuleNode): n is SignalRuleNode {
  return 'signalType' in n
}

export function isOperator(n: RuleNode): n is Exclude<RuleNode, { signalType: string }> {
  return 'operator' in n
}

export function serializeNode(n: RuleNode): string {
  if (isLeaf(n)) return formatSignalRef(n.signalType, n.signalName, n.fields)
  if (n.operator === 'NOT') {
    const child = n.conditions[0]
    if (!child) return 'NOT (?)'
    const serializedChild = serializeNode(child)
    return isOperator(child) && child.operator !== 'NOT'
      ? `NOT (${serializedChild})`
      : `NOT ${serializedChild}`
  }
  if (n.conditions.length === 0) return `(? ${n.operator} ?)`
  if (n.conditions.length === 1) return serializeNode(n.conditions[0])
  const parts = n.conditions.map((child) => {
    const serializedChild = serializeNode(child)
    if (
      isOperator(child) &&
      (child.operator === 'AND' || child.operator === 'OR') &&
      child.operator !== n.operator
    ) {
      return `(${serializedChild})`
    }
    return serializedChild
  })
  return parts.join(` ${n.operator} `)
}

export function parseExprText(text: string): RuleNode | null {
  const trimmed = text.trim()
  if (!trimmed) return null
  try {
    return parseOr(trimmed, { pos: 0 })
  } catch {
    return null
  }
}

function skipWhitespace(src: string, ctx: ParseCtx) {
  while (ctx.pos < src.length && src[ctx.pos] === ' ') ctx.pos++
}

function parseOr(src: string, ctx: ParseCtx): RuleNode {
  let left = parseAnd(src, ctx)
  while (true) {
    skipWhitespace(src, ctx)
    if (src.slice(ctx.pos, ctx.pos + 2).toUpperCase() === 'OR' && /\s/.test(src[ctx.pos + 2] ?? '')) {
      ctx.pos += 2
      skipWhitespace(src, ctx)
      const right = parseAnd(src, ctx)
      if (isOperator(left) && left.operator === 'OR') {
        left = { operator: 'OR', conditions: [...left.conditions, right] }
      } else {
        left = { operator: 'OR', conditions: [left, right] }
      }
    } else {
      break
    }
  }
  return left
}

function parseAnd(src: string, ctx: ParseCtx): RuleNode {
  let left = parseNot(src, ctx)
  while (true) {
    skipWhitespace(src, ctx)
    if (src.slice(ctx.pos, ctx.pos + 3).toUpperCase() === 'AND' && /[\s(]/.test(src[ctx.pos + 3] ?? '')) {
      ctx.pos += 3
      skipWhitespace(src, ctx)
      const right = parseNot(src, ctx)
      if (isOperator(left) && left.operator === 'AND') {
        left = { operator: 'AND', conditions: [...left.conditions, right] }
      } else {
        left = { operator: 'AND', conditions: [left, right] }
      }
    } else {
      break
    }
  }
  return left
}

function parseNot(src: string, ctx: ParseCtx): RuleNode {
  skipWhitespace(src, ctx)
  if (src.slice(ctx.pos, ctx.pos + 3).toUpperCase() === 'NOT' && /[\s(]/.test(src[ctx.pos + 3] ?? '')) {
    ctx.pos += 3
    skipWhitespace(src, ctx)
    const child = parseNot(src, ctx)
    return { operator: 'NOT', conditions: [child] }
  }
  return parseAtom(src, ctx)
}

function parseAtom(src: string, ctx: ParseCtx): RuleNode {
  skipWhitespace(src, ctx)
  if (src[ctx.pos] === '(') {
    ctx.pos++
    skipWhitespace(src, ctx)
    const inner = parseOr(src, ctx)
    skipWhitespace(src, ctx)
    if (src[ctx.pos] === ')') ctx.pos++
    return inner
  }
  const signal = parseSignalRef(src, ctx)
  if (signal) return signal
  const wordMatch = src.slice(ctx.pos).match(/^\w+/)
  if (wordMatch) {
    ctx.pos += wordMatch[0].length
    return { signalType: wordMatch[0], signalName: '' }
  }
  throw new Error('Unexpected token')
}

const SIGNAL_TYPE = /\w+/y
const FIELD_KEY = /[A-Za-z_][\w\-./]*/y
const NUMBER = /[+-]?\d+(?:\.\d+)?/y

function readPattern(src: string, ctx: ParseCtx, pattern: RegExp): string | null {
  skipWhitespace(src, ctx)
  pattern.lastIndex = ctx.pos
  const match = pattern.exec(src)
  if (!match) return null
  ctx.pos = pattern.lastIndex
  return match[0]
}

function readChar(src: string, ctx: ParseCtx, char: string): boolean {
  skipWhitespace(src, ctx)
  if (src[ctx.pos] !== char) return false
  ctx.pos++
  return true
}

function readString(src: string, ctx: ParseCtx): string | null {
  skipWhitespace(src, ctx)
  if (src[ctx.pos] !== '"') return null
  let end = ctx.pos + 1
  while (end < src.length && src[end] !== '"') end += src[end] === '\\' ? 2 : 1
  if (end >= src.length) throw new Error('Unterminated string')
  const literal = src.slice(ctx.pos, end + 1)
  ctx.pos = end + 1
  return JSON.parse(literal) as string
}

function readFields(src: string, ctx: ParseCtx, close: string): DSLFieldObject {
  const fields: DSLFieldObject = {}
  while (!readChar(src, ctx, close)) {
    readChar(src, ctx, ',')
    const key = readPattern(src, ctx, FIELD_KEY)
    if (key === null || !readChar(src, ctx, ':')) throw new Error('Expected a field')
    fields[key] = readValue(src, ctx)
  }
  return fields
}

function readValue(src: string, ctx: ParseCtx): DSLFieldValue {
  const text = readString(src, ctx)
  if (text !== null) return text
  if (readChar(src, ctx, '{')) return readFields(src, ctx, '}')
  if (readChar(src, ctx, '[')) {
    const items: DSLFieldValue[] = []
    while (!readChar(src, ctx, ']')) {
      items.push(readValue(src, ctx))
      readChar(src, ctx, ',')
    }
    return items
  }
  const number = readPattern(src, ctx, NUMBER)
  if (number !== null) return Number(number)
  const word = readPattern(src, ctx, FIELD_KEY)
  if (word === null) throw new Error('Expected a value')
  return word === 'true' || word === 'false' ? word === 'true' : word
}

// Reads classifier("risk", label: "unsafe", predicate: { gte: 0.5 }); null leaves ctx untouched.
function parseSignalRef(src: string, ctx: ParseCtx): SignalRuleNode | null {
  const start = ctx.pos
  try {
    const signalType = readPattern(src, ctx, SIGNAL_TYPE)
    if (signalType === null || !readChar(src, ctx, '(')) throw new Error('Expected a signal')
    const signalName = readString(src, ctx)
    if (signalName === null) throw new Error('Expected a signal name')
    const fields = readFields(src, ctx, ')')
    if (Object.keys(fields).length === 0) return { signalType, signalName }
    return { signalType, signalName, fields }
  } catch {
    ctx.pos = start
    return null
  }
}

export function boolExprToRuleNode(expr: Record<string, unknown> | null): RuleNode | null {
  if (!expr) return null
  const type = expr.type as string
  switch (type) {
    case 'signal_ref': {
      const leaf = { signalType: expr.signalType as string, signalName: expr.signalName as string }
      return expr.fields ? { ...leaf, fields: expr.fields as DSLFieldObject } : leaf
    }
    case 'and': {
      const left = boolExprToRuleNode(expr.left as Record<string, unknown>)
      const right = boolExprToRuleNode(expr.right as Record<string, unknown>)
      if (!left || !right) return left || right
      return { operator: 'AND', conditions: [left, right] }
    }
    case 'or': {
      const left = boolExprToRuleNode(expr.left as Record<string, unknown>)
      const right = boolExprToRuleNode(expr.right as Record<string, unknown>)
      if (!left || !right) return left || right
      return { operator: 'OR', conditions: [left, right] }
    }
    case 'not': {
      const child = boolExprToRuleNode(expr.expr as Record<string, unknown>)
      if (!child) return null
      return { operator: 'NOT', conditions: [child] }
    }
    default:
      return null
  }
}

export function pathEq(a: NodePath, b: NodePath): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index])
}

export function pathStartsWith(path: NodePath, prefix: NodePath): boolean {
  if (prefix.length > path.length) return false
  return prefix.every((value, index) => value === path[index])
}

export function getNodeAtPath(root: RuleNode, path: NodePath): RuleNode | null {
  if (path.length === 0) return root
  if (!isOperator(root)) return null
  const [head, ...tail] = path
  if (head < 0 || head >= root.conditions.length) return null
  return getNodeAtPath(root.conditions[head], tail)
}

export function replaceAtPath(root: RuleNode, path: NodePath, replacement: RuleNode): RuleNode {
  if (path.length === 0) return replacement
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  const newChild = replaceAtPath(root.conditions[head], tail, replacement)
  const newConditions = root.conditions.map((child, index) => (index === head ? newChild : child))
  return { ...root, conditions: newConditions } as RuleNode
}

export function removeAtPath(root: RuleNode, path: NodePath): RuleNode | null {
  if (path.length === 0) return null
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  if (tail.length === 0) {
    const newConditions = root.conditions.filter((_, index) => index !== head)
    if (newConditions.length === 0) return null
    if (root.operator !== 'NOT' && newConditions.length === 1) return newConditions[0]
    return { ...root, conditions: newConditions } as RuleNode
  }
  const newChild = removeAtPath(root.conditions[head], tail)
  if (!newChild) {
    const newConditions = root.conditions.filter((_, index) => index !== head)
    if (newConditions.length === 0) return null
    if (root.operator !== 'NOT' && newConditions.length === 1) return newConditions[0]
    return { ...root, conditions: newConditions } as RuleNode
  }
  const newConditions = root.conditions.map((child, index) => (index === head ? newChild : child))
  return { ...root, conditions: newConditions } as RuleNode
}

export function insertAtPath(root: RuleNode, path: NodePath, insertIdx: number, node: RuleNode): RuleNode {
  if (path.length === 0) {
    if (isOperator(root)) {
      const conditions = [...root.conditions]
      conditions.splice(insertIdx, 0, node)
      return { ...root, conditions } as RuleNode
    }
    return { operator: 'AND', conditions: [root, node] }
  }
  if (!isOperator(root)) return root
  const [head, ...tail] = path
  const newChild = insertAtPath(root.conditions[head], tail, insertIdx, node)
  const newConditions = root.conditions.map((child, index) => (index === head ? newChild : child))
  return { ...root, conditions: newConditions } as RuleNode
}

export function addChildAtPath(root: RuleNode, path: NodePath, child: RuleNode): RuleNode {
  const target = getNodeAtPath(root, path)
  if (!target) return root
  if (isOperator(target)) {
    const newTarget = { ...target, conditions: [...target.conditions, child] } as RuleNode
    return replaceAtPath(root, path, newTarget)
  }
  const wrapped: RuleNode = { operator: 'AND', conditions: [target, child] }
  return replaceAtPath(root, path, wrapped)
}

export function encodeDrag(data: DragData): string {
  return JSON.stringify(data)
}

export function decodeDrag(raw: string): DragData | null {
  try {
    return JSON.parse(raw) as DragData
  } catch {
    return null
  }
}

export function makeDragNode(data: DragData): RuleNode | null {
  if (data.kind === 'signal') return { signalType: data.signalType, signalName: data.signalName }
  if (data.kind === 'operator') {
    return data.operator === 'NOT'
      ? { operator: 'NOT', conditions: [] as unknown as [RuleNode] }
      : { operator: data.operator, conditions: [] }
  }
  return null
}

export function validateTree(node: RuleNode | null, signals: SignalDescriptor[]): string[] {
  const warnings: string[] = []
  if (!node) return warnings
  if (isLeaf(node)) {
    if (!signals.some((signal) => signal.signalType === node.signalType && signal.name === node.signalName)) {
      warnings.push(`Signal ${node.signalType}("${node.signalName}") is not defined`)
    }
  } else {
    if (node.operator === 'NOT' && node.conditions.length !== 1) {
      warnings.push('NOT must have exactly one child')
    }
    if ((node.operator === 'AND' || node.operator === 'OR') && node.conditions.length < 2) {
      warnings.push(`${node.operator} needs at least 2 children`)
    }
    for (const child of node.conditions) warnings.push(...validateTree(child, signals))
  }
  return warnings
}
