// When the Builder route form rewrites a ROUTE block, the header options and statements it
// does not edit (on_unknown, TIER, ACTION, FOR, EMIT, ...) are carried over as written.
// If the existing block does not follow the route grammar, the form's block is written alone.

// What updateRoute writes from the route form.
const FORM_STATEMENTS = new Set(['PRIORITY', 'WHEN', 'MODEL', 'ALGORITHM', 'PLUGIN', 'DESCRIPTION'])
const FORM_OPTIONS = new Set(['description'])

const OPENERS = new Set(['(', '[', '{'])
const CLOSERS = new Set([')', ']', '}'])
// dslLexer rules (pkg/dsl/parser.go): comment or whitespace, number, string, ident, symbol.
const TOKEN = /(#[^\n]*|\s+)|([+-]?\d+(?:\.\d+)?)|("(?:[^"\\]|\\.)*")|([A-Za-z_][\w\-./]*)|./y

type TokenKind = 'number' | 'string' | 'ident' | 'symbol'

interface Token {
  kind: TokenKind
  text: string
  start: number
  end: number
}

// A header option or body statement as written, keyed by its option name or keyword.
interface Part {
  key: string
  text: string
}

interface RouteBlockLayout {
  nameEnd: number
  options: Part[]
  optionsClose: number | null
  // null when the body does not follow the route grammar
  statements: Part[] | null
  bodyClose: number
}

function tokenize(source: string): Token[] {
  const tokens: Token[] = []
  TOKEN.lastIndex = 0
  for (let match = TOKEN.exec(source); match; match = TOKEN.exec(source)) {
    if (match[1]) continue
    const kind = match[2] ? 'number' : match[3] ? 'string' : match[4] ? 'ident' : 'symbol'
    tokens.push({ kind, text: match[0], start: match.index, end: TOKEN.lastIndex })
  }
  return tokens
}

function matchingClose(tokens: Token[], open: number): number {
  let depth = 0
  for (let index = open; index < tokens.length; index += 1) {
    if (OPENERS.has(tokens[index].text)) depth += 1
    else if (CLOSERS.has(tokens[index].text) && --depth === 0) return index
  }
  return -1
}

// A grammar rule takes the index of its first token and returns the index after its last one,
// or -1 when the tokens do not match. Given -1, it returns -1.
type Rule = (tokens: Token[], at: number) => number

function token(tokens: Token[], at: number, ...kinds: TokenKind[]): number {
  const next = tokens[at]
  return next && kinds.includes(next.kind) ? at + 1 : -1
}

function literal(tokens: Token[], at: number, text: string): number {
  return tokens[at]?.text === text ? at + 1 : -1
}

function group(tokens: Token[], at: number, open: string): number {
  if (tokens[at]?.text !== open) return -1
  const close = matchingClose(tokens, at)
  return close < 0 ? -1 : close + 1
}

function optionalGroup(tokens: Token[], at: number, open: string): number {
  return tokens[at]?.text === open ? group(tokens, at, open) : at
}

// Val: a number, string, bare word, [array] or {object}.
function value(tokens: Token[], at: number): number {
  const open = tokens[at]?.text
  if (open === '[' || open === '{') return group(tokens, at, open)
  return token(tokens, at, 'number', 'string', 'ident')
}

// rawModelList: "a" (reasoning = true), b, with an optional trailing comma.
function modelList(tokens: Token[], at: number): number {
  let next = optionalGroup(tokens, token(tokens, at, 'string', 'ident'), '(')
  while (literal(tokens, next, ',') >= 0) {
    // A name after the comma is the next model, even when it is spelled like a keyword.
    if (token(tokens, next + 1, 'string', 'ident') < 0) return next + 1
    next = optionalGroup(tokens, next + 2, '(')
  }
  return next
}

// BoolFactor: NOT factor, (expression), or a signal reference such as domain("math").
function boolFactor(tokens: Token[], at: number): number {
  if (literal(tokens, at, 'NOT') >= 0) return boolFactor(tokens, at + 1)
  if (tokens[at]?.text === '(') return group(tokens, at, '(')
  return group(tokens, token(tokens, at, 'ident'), '(')
}

// BoolExprTop: factors joined by AND and OR.
function boolExpr(tokens: Token[], at: number): number {
  let next = boolFactor(tokens, at)
  while (literal(tokens, next, 'AND') >= 0 || literal(tokens, next, 'OR') >= 0) {
    next = boolFactor(tokens, next + 1)
  }
  return next
}

// rawCandidateForDecl: FOR candidate IN decision.candidates or [models] { MODEL candidate }.
function candidateFor(tokens: Token[], at: number): number {
  const source = literal(tokens, token(tokens, at, 'ident'), 'IN')
  const body =
    tokens[source]?.text === '['
      ? group(tokens, source, '[')
      : token(tokens, source, 'string', 'ident')
  return group(tokens, body, '{')
}

// The operands of each route body statement (rawRouteItem in pkg/dsl/ast.go), so that an
// operand spelled like a keyword, as in MODEL TIER, stays inside its statement.
const STATEMENTS = new Map<string, Rule>([
  ['PRIORITY', (tokens, at) => token(tokens, at, 'number')],
  ['TIER', (tokens, at) => token(tokens, at, 'number')],
  ['WHEN', boolExpr],
  ['MODEL', modelList],
  ['ALGORITHM', (tokens, at) => optionalGroup(tokens, token(tokens, at, 'ident'), '{')],
  ['PLUGIN', (tokens, at) => optionalGroup(tokens, token(tokens, at, 'string', 'ident'), '{')],
  ['DESCRIPTION', (tokens, at) => token(tokens, at, 'string')],
  ['ACTION', (tokens, at) => token(tokens, token(tokens, at, 'ident'), 'string', 'ident')],
  ['FOR', candidateFor],
  ['EMIT', (tokens, at) => group(tokens, token(tokens, at, 'ident'), '{')],
])

function sourceOf(block: string, tokens: Token[], from: number, to: number): string {
  return block.slice(tokens[from].start, tokens[to - 1].end)
}

// RouteOpt: key = value, with an optional comma.
function splitOptions(block: string, tokens: Token[]): Part[] | null {
  const options: Part[] = []
  for (let at = 0; at < tokens.length; ) {
    const end = value(tokens, literal(tokens, token(tokens, at, 'ident'), '='))
    if (end < 0) return null
    options.push({ key: tokens[at].text, text: sourceOf(block, tokens, at, end) })
    at = literal(tokens, end, ',') >= 0 ? end + 1 : end
  }
  return options
}

function splitStatements(block: string, tokens: Token[]): Part[] | null {
  const statements: Part[] = []
  for (let at = 0; at < tokens.length; ) {
    const rule = STATEMENTS.get(tokens[at].text)
    const end = rule ? rule(tokens, at + 1) : -1
    if (end < 0) return null
    statements.push({ key: tokens[at].text, text: sourceOf(block, tokens, at, end) })
    at = end
  }
  return statements
}

function layoutRouteBlock(block: string): RouteBlockLayout | null {
  const tokens = tokenize(block)
  if (tokens[0]?.text !== 'ROUTE' || !tokens[1]) return null

  let index = 2
  let options: Part[] | null = []
  let optionsClose: number | null = null
  if (tokens[index]?.text === '(') {
    const close = matchingClose(tokens, index)
    options = close < 0 ? null : splitOptions(block, tokens.slice(index + 1, close))
    if (!options) return null
    optionsClose = tokens[close].start
    index = close + 1
  }
  if (tokens[index]?.text !== '{') return null
  const bodyClose = matchingClose(tokens, index)
  if (bodyClose < 0) return null

  return {
    nameEnd: tokens[1].end,
    options,
    optionsClose,
    statements: splitStatements(block, tokens.slice(index + 1, bodyClose)),
    bodyClose: tokens[bodyClose].start,
  }
}

export function keepRouteSettingsOutsideForm(
  existingBlock: string,
  rewrittenBlock: string,
): string {
  const existing = layoutRouteBlock(existingBlock)
  const rewritten = layoutRouteBlock(rewrittenBlock)
  if (!existing?.statements || !rewritten) return rewrittenBlock

  const statements = existing.statements
    .filter((statement) => !FORM_STATEMENTS.has(statement.key))
    .map((statement) => `  ${statement.text}\n`)
    .join('')
  const options = existing.options
    .filter((option) => !FORM_OPTIONS.has(option.key))
    .map((option) => option.text)
    .join(', ')

  // Insert the statements first: the header offsets come before the body.
  const block =
    rewrittenBlock.slice(0, rewritten.bodyClose) +
    statements +
    rewrittenBlock.slice(rewritten.bodyClose)
  if (!options) return block
  if (rewritten.optionsClose === null) {
    return `${block.slice(0, rewritten.nameEnd)} (${options})${block.slice(rewritten.nameEnd)}`
  }
  const separator = rewritten.options.length > 0 ? ', ' : ''
  return (
    block.slice(0, rewritten.optionsClose) +
    separator +
    options +
    block.slice(rewritten.optionsClose)
  )
}
