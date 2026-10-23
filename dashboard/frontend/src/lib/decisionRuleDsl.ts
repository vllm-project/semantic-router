/**
 * The single serializer and parser for DSL WHEN expressions.
 *
 * The grammar mirrors pkg/dsl/ast.go (BoolExprTop / BoolAndTerm / BoolFactor /
 * rawSignalRefExpr) and the token rules in pkg/dsl/parser.go. Parsing is
 * all-or-nothing: an incomplete expression is an error, never a valid prefix.
 */

import type { BoolExprNode } from '@/types/dsl'

import {
  isRuleGroup,
  isRuleLeaf,
  newRuleNodeId,
  RULE_PREDICATE_KEYS,
  type NumericPredicate,
  type RuleIssue,
  type RuleLeaf,
  type RuleNode,
  type RuleOnError,
  type RuleOperator,
} from './decisionRuleAst'

// JSON string escapes are a subset of the Go escapes the DSL parser unquotes.
export function quoteDSLString(value: string): string {
  return JSON.stringify(value)
}

const PREDICATE_KEYS: readonly string[] = RULE_PREDICATE_KEYS
type PredicateKey = keyof NumericPredicate

const PRECEDENCE: Record<RuleOperator, number> = { OR: 1, AND: 2, NOT: 3 }

// ---------- Serializer ----------

/** Go formats predicate numbers with strconv 'f' -1: never in exponent form. */
// String() already gives the shortest round-trip digits; an exponent form is
// expanded by shifting the decimal point, which never rounds.
function formatNumber(value: number): string {
  const text = String(value)
  const match = /^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/i.exec(text)
  if (!match) return text
  const [, sign, lead, fraction = '', exponent] = match
  const digits = lead + fraction
  const point = 1 + Number(exponent)
  if (point <= 0) return `${sign}0.${'0'.repeat(-point)}${digits}`
  if (point >= digits.length) return `${sign}${digits}${'0'.repeat(point - digits.length)}`
  return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`
}

function formatPredicate(predicate: NumericPredicate): string {
  const parts: string[] = []
  for (const key of RULE_PREDICATE_KEYS) {
    const value = predicate[key]
    if (value !== undefined) parts.push(`${key}: ${formatNumber(value)}`)
  }
  return `{ ${parts.join(', ')} }`
}

/** Skips one-child AND/OR groups, which the DSL cannot express. */
function effectiveNode(node: RuleNode, path: number[]): [RuleNode, number[]] {
  let current = node
  let currentPath = path
  while (isRuleGroup(current) && current.operator !== 'NOT' && current.conditions.length === 1 && !current.extra) {
    current = current.conditions[0]
    currentPath = [...currentPath, 0]
  }
  return [current, currentPath]
}

function needsGroupParens(node: RuleNode): boolean {
  return isRuleGroup(node) && node.operator !== 'NOT' && node.conditions.length >= 2
}

interface SerializeCtx {
  errors: RuleIssue[]
}

function issue(ctx: SerializeCtx, node: RuleNode, path: number[], message: string) {
  ctx.errors.push({ nodeId: node.id, path, severity: 'error', message })
}

function reportExtra(ctx: SerializeCtx, node: RuleNode, path: number[]) {
  for (const key of Object.keys(node.extra ?? {})) {
    issue(ctx, node, path, `field ${key} cannot be expressed in the DSL`)
  }
}

function serializeLeaf(leaf: RuleLeaf, path: number[], ctx: SerializeCtx): string {
  reportExtra(ctx, leaf, path)
  if (!leaf.signalType) issue(ctx, leaf, path, 'leaf condition requires a type')
  if (!leaf.signalName) issue(ctx, leaf, path, `condition ${leaf.signalType || '?'} requires a name`)
  const args = [quoteDSLString(leaf.signalName)]
  if (leaf.label !== undefined && leaf.label !== '') args.push(`label: ${quoteDSLString(leaf.label)}`)
  if (leaf.predicate !== undefined) {
    for (const [key, value] of Object.entries(leaf.predicate)) {
      if (value === undefined) continue
      if (!PREDICATE_KEYS.includes(key)) {
        issue(ctx, leaf, path, `predicate field ${key} is not supported`)
      } else if (typeof value !== 'number' || !Number.isFinite(value)) {
        issue(ctx, leaf, path, 'predicate values must be finite')
      }
    }
    args.push(`predicate: ${formatPredicate(leaf.predicate)}`)
  }
  if (leaf.onError !== undefined) args.push(`on_error: ${quoteDSLString(leaf.onError)}`)
  return `${leaf.signalType || '?'}(${args.join(', ')})`
}

function serializeAt(node: RuleNode, path: number[], root: boolean, ctx: SerializeCtx): string {
  if (isRuleLeaf(node)) return serializeLeaf(node, path, ctx)
  reportExtra(ctx, node, path)
  if (node.operator === 'NOT') {
    if (node.conditions.length !== 1) {
      issue(ctx, node, path, `NOT requires exactly one child condition, got ${node.conditions.length}`)
      return 'NOT (?)'
    }
    const [child, childPath] = effectiveNode(node.conditions[0], [...path, 0])
    const text = serializeAt(child, childPath, false, ctx)
    return needsGroupParens(child) ? `NOT (${text})` : `NOT ${text}`
  }
  if (!(node.operator in PRECEDENCE)) {
    issue(ctx, node, path, `unsupported operator "${node.operator}" (expected AND, OR, NOT)`)
  }
  if (node.conditions.length === 0) {
    // A childless root AND is the catch-all decision: the DSL omits WHEN.
    if (root && node.operator === 'AND') return ''
    issue(ctx, node, path, `${node.operator} combination requires at least one child condition`)
    return `(? ${node.operator} ?)`
  }
  if (node.conditions.length === 1) {
    const [child, childPath] = effectiveNode(node.conditions[0], [...path, 0])
    const text = serializeAt(child, childPath, root, ctx)
    return !root && needsGroupParens(child) ? `(${text})` : text
  }
  return node.conditions
    .map((rawChild, index) => {
      const [child, childPath] = effectiveNode(rawChild, [...path, index])
      const text = serializeAt(child, childPath, false, ctx)
      // Explicit nesting of either operator keeps its parentheses so a
      // re-parse yields the same tree; NOT and leaves bind tighter than both.
      return needsGroupParens(child) ? `(${text})` : text
    })
    .join(` ${node.operator} `)
}

export function serializeRuleExpr(tree: RuleNode | null): { text: string; errors: RuleIssue[] } {
  if (!tree) return { text: '', errors: [] }
  const ctx: SerializeCtx = { errors: [] }
  const text = serializeAt(tree, [], true, ctx)
  return { text, errors: ctx.errors }
}

// ---------- Lexer ----------

type TokenKind =
  | 'Float' | 'Int' | 'String' | 'Arrow' | 'Ident'
  | '{' | '}' | '(' | ')' | '[' | ']' | ':' | ',' | '>' | '=' | 'EOF'

interface Token {
  kind: TokenKind
  text: string
  offset: number
}

class RuleParseError extends Error {
  constructor(readonly offset: number, message: string) {
    super(message)
  }
}

// Same order as dslLexer: the first matching rule wins. RE2's \s is ASCII.
const LEX_RULES: Array<[TokenKind | 'skip', RegExp]> = [
  ['skip', /#[^\n]*/y],
  ['skip', /[\t\n\f\r ]+/y],
  ['Float', /[+-]?[0-9]+\.[0-9]+/y],
  ['Int', /[+-]?[0-9]+/y],
  ['String', /"(?:[^"\\]|\\.)*"/y],
  ['Arrow', /->|→/y],
  ['Ident', /[a-zA-Z_][a-zA-Z0-9_\-./]*/y],
]
const PUNCTUATION = new Set(['{', '}', '(', ')', '[', ']', ':', ',', '>', '='])

function tokenize(text: string): Token[] {
  const tokens: Token[] = []
  let pos = 0
  outer: while (pos < text.length) {
    for (const [kind, pattern] of LEX_RULES) {
      pattern.lastIndex = pos
      const match = pattern.exec(text)
      if (!match || match[0].length === 0) continue
      if (kind !== 'skip') tokens.push({ kind, text: match[0], offset: pos })
      pos += match[0].length
      continue outer
    }
    const char = text[pos]
    if (PUNCTUATION.has(char)) {
      tokens.push({ kind: char as TokenKind, text: char, offset: pos })
      pos += 1
      continue
    }
    if (char === '"') throw new RuleParseError(pos, 'unterminated string')
    throw new RuleParseError(pos, `unexpected character ${JSON.stringify(char)}`)
  }
  tokens.push({ kind: 'EOF', text: '', offset: text.length })
  return tokens
}

const SIMPLE_ESCAPES: Record<string, string> = {
  a: '\x07', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t', v: '\v', '\\': '\\', '"': '"',
}

/** Go strconv.Unquote for a double-quoted literal; null when Go would reject it. */
function goUnquote(literal: string): string | null {
  const body = literal.slice(1, -1)
  let out = ''
  for (let i = 0; i < body.length; i++) {
    const char = body[i]
    if (char === '\n') return null
    if (char !== '\\') { out += char; continue }
    const next = body[++i]
    if (next in SIMPLE_ESCAPES) { out += SIMPLE_ESCAPES[next]; continue }
    const hexLength = next === 'x' ? 2 : next === 'u' ? 4 : next === 'U' ? 8 : 0
    if (hexLength) {
      const hex = body.slice(i + 1, i + 1 + hexLength)
      if (!new RegExp(`^[0-9a-fA-F]{${hexLength}}$`).test(hex)) return null
      const code = parseInt(hex, 16)
      if (code > 0x10ffff || (code >= 0xd800 && code < 0xe000 && next !== 'x')) return null
      out += next === 'x' ? String.fromCharCode(code) : String.fromCodePoint(code)
      i += hexLength
      continue
    }
    const octal = body.slice(i, i + 3)
    if (/^[0-7]{3}$/.test(octal) && parseInt(octal, 8) <= 0xff) {
      out += String.fromCharCode(parseInt(octal, 8))
      i += 2
      continue
    }
    return null
  }
  return out
}

/** Mirrors pkg/dsl unquote: a literal Go cannot unquote keeps its raw body. */
function unquote(literal: string): string {
  return goUnquote(literal) ?? literal.slice(1, -1)
}

// ---------- Parser ----------

const KEYWORDS = new Set(['AND', 'OR', 'NOT'])

type FieldValue =
  | { kind: 'string'; value: string }
  | { kind: 'number'; value: number }
  | { kind: 'bool'; value: boolean }
  | { kind: 'array' }
  | { kind: 'object'; fields: Array<{ key: string; value: FieldValue; offset: number }> }

class RuleExprParser {
  private index = 0

  constructor(private readonly tokens: Token[]) {}

  private peek(): Token {
    return this.tokens[this.index]
  }

  private next(): Token {
    return this.tokens[this.index++]
  }

  private isKeyword(word: string): boolean {
    const token = this.peek()
    return token.kind === 'Ident' && token.text === word
  }

  private describe(token: Token): string {
    return token.kind === 'EOF' ? 'end of input' : JSON.stringify(token.text)
  }

  private fail(token: Token, message: string): never {
    throw new RuleParseError(token.offset, message)
  }

  private expect(kind: TokenKind, what: string): Token {
    const token = this.peek()
    if (token.kind !== kind) this.fail(token, `expected ${what}, found ${this.describe(token)}`)
    return this.next()
  }

  parseExpression(): RuleNode {
    const tree = this.parseOr()
    const rest = this.peek()
    if (rest.kind !== 'EOF') this.fail(rest, `unexpected input ${this.describe(rest)}`)
    return tree
  }

  private parseOr(): RuleNode {
    const terms = [this.parseAnd()]
    while (this.isKeyword('OR')) {
      this.next()
      terms.push(this.parseAnd())
    }
    return terms.length === 1 ? terms[0] : { id: newRuleNodeId(), operator: 'OR', conditions: terms }
  }

  private parseAnd(): RuleNode {
    const factors = [this.parseFactor()]
    while (this.isKeyword('AND')) {
      this.next()
      factors.push(this.parseFactor())
    }
    return factors.length === 1 ? factors[0] : { id: newRuleNodeId(), operator: 'AND', conditions: factors }
  }

  private parseFactor(): RuleNode {
    const token = this.peek()
    if (this.isKeyword('NOT')) {
      this.next()
      return { id: newRuleNodeId(), operator: 'NOT', conditions: [this.parseFactor()] }
    }
    if (token.kind === '(') {
      this.next()
      const inner = this.parseOr()
      this.expect(')', '")"')
      return inner
    }
    if (token.kind === 'Ident' && !KEYWORDS.has(token.text)) return this.parseSignalRef()
    return this.fail(token, `expected a condition, found ${this.describe(token)}`)
  }

  private parseSignalRef(): RuleLeaf {
    const typeToken = this.next()
    this.expect('(', `"(" after ${typeToken.text}`)
    const nameToken = this.peek()
    if (nameToken.kind !== 'String' && nameToken.kind !== 'Ident') {
      this.fail(nameToken, `expected a signal name, found ${this.describe(nameToken)}`)
    }
    this.next()
    const name = nameToken.kind === 'String' ? unquote(nameToken.text) : nameToken.text
    if (name === '') this.fail(nameToken, 'signal name must not be empty')
    const leaf: RuleLeaf = { id: newRuleNodeId(), signalType: typeToken.text, signalName: name }
    const seen = new Set<string>()
    while (this.peek().kind !== ')') {
      const { key, value, offset } = this.parseField()
      if (seen.has(key)) throw new RuleParseError(offset, `duplicate field ${key}`)
      seen.add(key)
      this.applyField(leaf, key, value, offset)
    }
    this.next()
    return leaf
  }

  private parseField(): { key: string; value: FieldValue; offset: number } {
    if (this.peek().kind === ',') this.next()
    const keyToken = this.peek()
    if (keyToken.kind !== 'Ident') {
      this.fail(keyToken, `expected ")" or a field name, found ${this.describe(keyToken)}`)
    }
    this.next()
    this.expect(':', `":" after ${keyToken.text}`)
    return { key: keyToken.text, value: this.parseValue(), offset: keyToken.offset }
  }

  private parseValue(): FieldValue {
    const token = this.peek()
    switch (token.kind) {
      case 'String':
        this.next()
        return { kind: 'string', value: unquote(token.text) }
      case 'Float':
      case 'Int':
        this.next()
        return { kind: 'number', value: Number(token.text) }
      case 'Ident':
        this.next()
        if (token.text === 'true' || token.text === 'false') return { kind: 'bool', value: token.text === 'true' }
        return { kind: 'string', value: token.text }
      case '[': {
        this.next()
        if (this.peek().kind !== ']') {
          this.parseValue()
          while (this.peek().kind === ',') {
            this.next()
            if (this.peek().kind === ']') break
            this.parseValue()
          }
        }
        this.expect(']', '"]"')
        return { kind: 'array' }
      }
      case '{': {
        this.next()
        const fields: Array<{ key: string; value: FieldValue; offset: number }> = []
        while (this.peek().kind !== '}') fields.push(this.parseField())
        this.next()
        return { kind: 'object', fields }
      }
      default:
        return this.fail(token, `expected a value, found ${this.describe(token)}`)
    }
  }

  // Only the fields the Go compiler reads are accepted; it would silently
  // drop any other key, so accepting one here would lose what was written.
  private applyField(leaf: RuleLeaf, key: string, value: FieldValue, offset: number) {
    if (key === 'label' || key === 'on_error') {
      if (value.kind !== 'string') throw new RuleParseError(offset, `${key} must be a string`)
      if (key === 'label') leaf.label = value.value
      else leaf.onError = value.value as RuleOnError
      return
    }
    if (key === 'predicate') {
      if (value.kind !== 'object') throw new RuleParseError(offset, 'predicate must be an object such as { gte: 0.7 }')
      const predicate: NumericPredicate = {}
      for (const field of value.fields) {
        if (!PREDICATE_KEYS.includes(field.key)) {
          throw new RuleParseError(field.offset, `unknown predicate field ${field.key} (expected gt, gte, lt, lte)`)
        }
        if (field.value.kind !== 'number' || !Number.isFinite(field.value.value)) {
          throw new RuleParseError(field.offset, `predicate ${field.key} must be a finite number`)
        }
        if (predicate[field.key as PredicateKey] !== undefined) {
          throw new RuleParseError(field.offset, `duplicate predicate field ${field.key}`)
        }
        predicate[field.key as PredicateKey] = field.value.value
      }
      leaf.predicate = predicate
      return
    }
    throw new RuleParseError(offset, `unknown field ${key} (expected label, predicate, on_error)`)
  }
}

export interface RuleExprParseError {
  offset: number
  message: string
}

/**
 * Parses a complete WHEN expression. Empty text is the unconditional rule.
 * The error message names the 1-based column of the offending input.
 */
export function parseRuleExpr(text: string): { tree: RuleNode | null; error?: RuleExprParseError } {
  if (text.trim() === '') return { tree: null }
  try {
    return { tree: new RuleExprParser(tokenize(text)).parseExpression() }
  } catch (error) {
    if (error instanceof RuleParseError) {
      return {
        tree: null,
        error: { offset: error.offset, message: `${error.message} at column ${error.offset + 1}` },
      }
    }
    throw error
  }
}

// ---------- DSL AST (WASM JSON) conversion ----------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function signalRefToLeaf(expr: Record<string, unknown>): RuleLeaf | null {
  if (typeof expr.signalType !== 'string' || typeof expr.signalName !== 'string') return null
  const leaf: RuleLeaf = { id: newRuleNodeId(), signalType: expr.signalType, signalName: expr.signalName }
  const fields = isRecord(expr.fields) ? expr.fields : {}
  const extra: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined) continue
    if (key === 'label' && typeof value === 'string') leaf.label = value
    else if (key === 'on_error' && typeof value === 'string') leaf.onError = value as RuleOnError
    else if (key === 'predicate' && isRecord(value)) leaf.predicate = { ...value } as NumericPredicate
    else extra[key] = value
  }
  if (Object.keys(extra).length > 0) leaf.extra = extra
  return leaf
}

/**
 * Converts a WASM BoolExpr into a rule tree. Same-operator binary chains are
 * flattened, matching the Go compiler. Returns null, rather than silently
 * dropping a child, when any part of the expression cannot be converted.
 */
export function boolExprToRuleNode(expr: BoolExprNode | Record<string, unknown> | null | undefined): RuleNode | null {
  if (!isRecord(expr)) return null
  switch (expr.type) {
    case 'signal_ref':
      return signalRefToLeaf(expr)
    case 'and':
    case 'or': {
      const operands: RuleNode[] = []
      const collect = (node: unknown): boolean => {
        if (isRecord(node) && node.type === expr.type) return collect(node.left) && collect(node.right)
        const converted = boolExprToRuleNode(node as Record<string, unknown>)
        if (!converted) return false
        operands.push(converted)
        return true
      }
      if (!collect(expr)) return null
      const operator = expr.type === 'and' ? 'AND' : 'OR'
      return { id: newRuleNodeId(), operator, conditions: operands }
    }
    case 'not': {
      const child = boolExprToRuleNode(expr.expr as Record<string, unknown>)
      return child ? { id: newRuleNodeId(), operator: 'NOT', conditions: [child] } : null
    }
    default:
      return null
  }
}
