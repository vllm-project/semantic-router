// When the Builder route form rewrites a ROUTE block, the header options and statements it
// does not edit (on_unknown, TIER, ACTION, FOR, EMIT, ...) are carried over as written.

// Route body statements of the DSL grammar (rawRouteItem in pkg/dsl/ast.go).
const ROUTE_STATEMENTS = new Set([
  'PRIORITY',
  'TIER',
  'WHEN',
  'MODEL',
  'ALGORITHM',
  'PLUGIN',
  'DESCRIPTION',
  'ACTION',
  'FOR',
  'EMIT',
])
// What updateRoute writes from the route form.
const FORM_STATEMENTS = new Set(['PRIORITY', 'WHEN', 'MODEL', 'ALGORITHM', 'PLUGIN', 'DESCRIPTION'])
const FORM_OPTIONS = new Set(['description'])

const OPENERS = new Set(['(', '[', '{'])
const CLOSERS = new Set([')', ']', '}'])
const IDENT = /[A-Za-z_][\w\-./]*/y

interface Token {
  text: string
  start: number
  end: number
  ident: boolean
}

interface RouteBlockLayout {
  nameEnd: number
  options: Array<{ key: string; text: string }>
  optionsClose: number | null
  statements: Array<{ keyword: string; text: string }>
  bodyClose: number
}

// Follows the DSL lexer closely enough to skip comments and string contents.
function tokenize(source: string): Token[] {
  const tokens: Token[] = []
  let index = 0
  while (index < source.length) {
    const char = source[index]
    if (/\s/.test(char)) {
      index += 1
    } else if (char === '#') {
      const lineEnd = source.indexOf('\n', index)
      index = lineEnd === -1 ? source.length : lineEnd
    } else if (char === '"') {
      let end = index + 1
      while (end < source.length && source[end] !== '"') end += source[end] === '\\' ? 2 : 1
      end = Math.min(end + 1, source.length)
      tokens.push({ text: source.slice(index, end), start: index, end, ident: false })
      index = end
    } else {
      IDENT.lastIndex = index
      const ident = IDENT.exec(source) !== null
      const end = ident ? IDENT.lastIndex : index + 1
      tokens.push({ text: source.slice(index, end), start: index, end, ident })
      index = end
    }
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

function sourceOf(block: string, run: Token[]): string {
  return block.slice(run[0].start, run[run.length - 1].end)
}

function splitOptions(block: string, tokens: Token[]): RouteBlockLayout['options'] {
  const runs: Token[][] = [[]]
  let depth = 0
  for (const token of tokens) {
    if (depth === 0 && token.text === ',') {
      runs.push([])
      continue
    }
    runs[runs.length - 1].push(token)
    if (OPENERS.has(token.text)) depth += 1
    else if (CLOSERS.has(token.text)) depth -= 1
  }
  return runs
    .filter((run) => run.length > 0)
    .map((run) => ({ key: run[0].text, text: sourceOf(block, run) }))
}

function splitStatements(block: string, tokens: Token[]): RouteBlockLayout['statements'] {
  const runs: Token[][] = []
  let depth = 0
  for (const token of tokens) {
    if (depth === 0 && token.ident && ROUTE_STATEMENTS.has(token.text)) runs.push([])
    runs[runs.length - 1]?.push(token)
    if (OPENERS.has(token.text)) depth += 1
    else if (CLOSERS.has(token.text)) depth -= 1
  }
  return runs.map((run) => ({ keyword: run[0].text, text: sourceOf(block, run) }))
}

function layoutRouteBlock(block: string): RouteBlockLayout | null {
  const tokens = tokenize(block)
  if (tokens[0]?.text !== 'ROUTE' || !tokens[1]) return null

  let index = 2
  let options: RouteBlockLayout['options'] = []
  let optionsClose: number | null = null
  if (tokens[index]?.text === '(') {
    const close = matchingClose(tokens, index)
    if (close < 0) return null
    options = splitOptions(block, tokens.slice(index + 1, close))
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
  if (!existing || !rewritten) return rewrittenBlock

  const statements = existing.statements
    .filter((statement) => !FORM_STATEMENTS.has(statement.keyword))
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
