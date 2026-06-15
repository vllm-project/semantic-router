import { createTool } from '../registry'
import type { CalculateArgs, CalculateResult, ToolExecutionContext } from '../types'

export type { CalculateArgs, CalculateResult }

type Operator = '+' | '-' | '*' | '/' | '%' | '^'

type Token =
  | { type: 'number'; value: number }
  | { type: 'identifier'; value: string }
  | { type: 'operator'; value: Operator }
  | { type: 'paren'; value: '(' | ')' }
  | { type: 'comma' }

class ExpressionParser {
  private readonly tokens: Token[]
  private index = 0

  constructor(tokens: Token[]) {
    this.tokens = tokens
  }

  parse(): number {
    const value = this.parseExpression()
    if (!this.isAtEnd()) {
      throw new Error('Unexpected token at end of expression')
    }
    return value
  }

  private parseExpression(): number {
    return this.parseAdditive()
  }

  private parseAdditive(): number {
    let value = this.parseMultiplicative()
    while (this.matchOperator('+') || this.matchOperator('-')) {
      const operator = this.previousOperator().value
      const right = this.parseMultiplicative()
      value = operator === '+' ? value + right : value - right
    }
    return value
  }

  private parseMultiplicative(): number {
    let value = this.parsePower()
    while (this.matchOperator('*') || this.matchOperator('/') || this.matchOperator('%')) {
      const operator = this.previousOperator().value
      const right = this.parsePower()
      if ((operator === '/' || operator === '%') && right === 0) {
        throw new Error('Division by zero is not allowed')
      }
      if (operator === '*') value *= right
      if (operator === '/') value /= right
      if (operator === '%') value %= right
    }
    return value
  }

  private parsePower(): number {
    let value = this.parseUnary()
    if (this.matchOperator('^')) {
      value = Math.pow(value, this.parsePower())
    }
    return value
  }

  private parseUnary(): number {
    if (this.matchOperator('+')) {
      return this.parseUnary()
    }
    if (this.matchOperator('-')) {
      return -this.parseUnary()
    }
    return this.parsePrimary()
  }

  private parsePrimary(): number {
    if (this.matchNumber()) {
      return this.previousNumber().value
    }

    if (this.matchIdentifier()) {
      const identifier = this.previousIdentifier().value.toLowerCase()
      if (this.matchParen('(')) {
        const args = this.parseArguments()
        this.consumeParen(')', 'Expected closing parenthesis after function arguments')
        return evaluateFunction(identifier, args)
      }
      return evaluateConstant(identifier)
    }

    if (this.matchParen('(')) {
      const value = this.parseExpression()
      this.consumeParen(')', 'Expected closing parenthesis')
      return value
    }

    throw new Error('Expected a number, constant, or sub-expression')
  }

  private parseArguments(): number[] {
    const args: number[] = []
    if (this.checkParen(')')) {
      return args
    }

    do {
      args.push(this.parseExpression())
    } while (this.matchComma())

    return args
  }

  private matchOperator(operator: Operator) {
    if (this.checkOperator(operator)) {
      this.index += 1
      return true
    }
    return false
  }

  private matchParen(value: '(' | ')') {
    if (this.checkParen(value)) {
      this.index += 1
      return true
    }
    return false
  }

  private consumeParen(value: '(' | ')', message: string) {
    if (!this.matchParen(value)) {
      throw new Error(message)
    }
  }

  private matchNumber() {
    if (this.peek()?.type === 'number') {
      this.index += 1
      return true
    }
    return false
  }

  private matchIdentifier() {
    if (this.peek()?.type === 'identifier') {
      this.index += 1
      return true
    }
    return false
  }

  private matchComma() {
    if (this.peek()?.type === 'comma') {
      this.index += 1
      return true
    }
    return false
  }

  private checkOperator(operator: Operator) {
    const token = this.peek()
    return token?.type === 'operator' && token.value === operator
  }

  private checkParen(value: '(' | ')') {
    const token = this.peek()
    return token?.type === 'paren' && token.value === value
  }

  private previous() {
    return this.tokens[this.index - 1]
  }

  private previousNumber() {
    const token = this.previous()
    if (token?.type !== 'number') {
      throw new Error('Expected previous token to be a number')
    }
    return token
  }

  private previousIdentifier() {
    const token = this.previous()
    if (token?.type !== 'identifier') {
      throw new Error('Expected previous token to be an identifier')
    }
    return token
  }

  private previousOperator() {
    const token = this.previous()
    if (token?.type !== 'operator') {
      throw new Error('Expected previous token to be an operator')
    }
    return token
  }

  private peek() {
    return this.tokens[this.index]
  }

  private isAtEnd() {
    return this.index >= this.tokens.length
  }
}

function tokenizeExpression(expression: string): Token[] {
  const tokens: Token[] = []
  let cursor = 0

  while (cursor < expression.length) {
    const char = expression[cursor]

    if (/\s/.test(char)) {
      cursor += 1
      continue
    }

    const numberMatch = expression.slice(cursor).match(/^\d+(\.\d+)?/)
    if (numberMatch) {
      tokens.push({ type: 'number', value: Number(numberMatch[0]) })
      cursor += numberMatch[0].length
      continue
    }

    const identifierMatch = expression.slice(cursor).match(/^[A-Za-z_][A-Za-z0-9_]*/)
    if (identifierMatch) {
      tokens.push({ type: 'identifier', value: identifierMatch[0] })
      cursor += identifierMatch[0].length
      continue
    }

    if (char === ',') {
      tokens.push({ type: 'comma' })
      cursor += 1
      continue
    }

    if (char === '(' || char === ')') {
      tokens.push({ type: 'paren', value: char })
      cursor += 1
      continue
    }

    if ('+-*/%^'.includes(char)) {
      tokens.push({ type: 'operator', value: char as Operator })
      cursor += 1
      continue
    }

    throw new Error(`Unsupported character: ${char}`)
  }

  return tokens
}

function evaluateConstant(identifier: string) {
  switch (identifier) {
    case 'pi':
      return Math.PI
    case 'e':
      return Math.E
    default:
      throw new Error(`Unknown constant: ${identifier}`)
  }
}

function expectArity(name: string, args: number[], min: number, max = min) {
  if (args.length < min || args.length > max) {
    const expected = min === max ? `${min}` : `${min}-${max}`
    throw new Error(`${name} expects ${expected} argument(s)`)
  }
}

function evaluateFunction(identifier: string, args: number[]) {
  switch (identifier) {
    case 'abs':
      expectArity('abs', args, 1)
      return Math.abs(args[0])
    case 'sqrt':
      expectArity('sqrt', args, 1)
      return Math.sqrt(args[0])
    case 'sin':
      expectArity('sin', args, 1)
      return Math.sin(args[0])
    case 'cos':
      expectArity('cos', args, 1)
      return Math.cos(args[0])
    case 'tan':
      expectArity('tan', args, 1)
      return Math.tan(args[0])
    case 'asin':
      expectArity('asin', args, 1)
      return Math.asin(args[0])
    case 'acos':
      expectArity('acos', args, 1)
      return Math.acos(args[0])
    case 'atan':
      expectArity('atan', args, 1)
      return Math.atan(args[0])
    case 'ln':
      expectArity('ln', args, 1)
      return Math.log(args[0])
    case 'log':
      expectArity('log', args, 1, 2)
      return args.length === 1 ? Math.log10(args[0]) : Math.log(args[1]) / Math.log(args[0])
    case 'exp':
      expectArity('exp', args, 1)
      return Math.exp(args[0])
    case 'round':
      expectArity('round', args, 1)
      return Math.round(args[0])
    case 'floor':
      expectArity('floor', args, 1)
      return Math.floor(args[0])
    case 'ceil':
      expectArity('ceil', args, 1)
      return Math.ceil(args[0])
    case 'min':
      expectArity('min', args, 1, Number.POSITIVE_INFINITY)
      return Math.min(...args)
    case 'max':
      expectArity('max', args, 1, Number.POSITIVE_INFINITY)
      return Math.max(...args)
    case 'pow':
      expectArity('pow', args, 2)
      return Math.pow(args[0], args[1])
    default:
      throw new Error(`Unknown function: ${identifier}`)
  }
}

function formatNumber(value: number) {
  if (!Number.isFinite(value)) {
    throw new Error('Result is not a finite number')
  }
  if (Number.isInteger(value)) {
    return `${value}`
  }
  return Number(value.toPrecision(12)).toString()
}

function validateCalculateArgs(args: unknown): CalculateArgs {
  if (typeof args !== 'object' || args === null || Array.isArray(args)) {
    throw new Error('Arguments must be an object')
  }

  const rawExpression = (args as Record<string, unknown>).expression
  const expression = typeof rawExpression === 'string' ? rawExpression.trim() : ''

  if (!expression) {
    throw new Error('expression is required and must be a non-empty string')
  }

  return { expression }
}

function evaluateExpression(expression: string) {
  const parser = new ExpressionParser(tokenizeExpression(expression))
  return parser.parse()
}

async function executeCalculate(
  args: CalculateArgs,
  context: ToolExecutionContext,
): Promise<CalculateResult> {
  context.onProgress?.(20)
  const result = evaluateExpression(args.expression)
  context.onProgress?.(100)
  return {
    expression: args.expression,
    result,
    formatted_result: formatNumber(result),
  }
}

function formatCalculateResult(result: CalculateResult) {
  return `${result.expression} = ${result.formatted_result}`
}

export const calculateTool = createTool<CalculateArgs, CalculateResult>({
  metadata: {
    id: 'calculate',
    displayName: 'Calculator',
    category: 'custom',
    icon: 'calculator',
    enabled: true,
    version: '1.0.0',
  },
  definition: {
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Evaluate mathematical expressions using arithmetic, parentheses, constants like pi and e, and common functions such as sqrt, abs, sin, cos, log, min, and max.',
      parameters: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'The mathematical expression to evaluate, for example (15 * 4) + sqrt(81) or max(3, 7, 10).',
          },
        },
        required: ['expression'],
      },
    },
  },
  validateArgs: validateCalculateArgs,
  executor: executeCalculate,
  formatResult: formatCalculateResult,
})
