import type { ParsedToolCallChunk } from './chatResponseParsing'

export interface TextToolCallExtraction {
  content: string
  toolCalls: ParsedToolCallChunk[]
}

const parseArgumentValue = (value: string): unknown => {
  const trimmed = value.trim()
  if (!trimmed) return ''

  try {
    return JSON.parse(trimmed)
  } catch {
    return trimmed
  }
}

const parseJsonToolCall = (body: string, index: number): ParsedToolCallChunk | null => {
  try {
    const parsed = JSON.parse(body.trim()) as {
      id?: unknown
      name?: unknown
      arguments?: unknown
      function?: { name?: unknown; arguments?: unknown }
    }
    const name = parsed.function?.name ?? parsed.name
    const args = parsed.function?.arguments ?? parsed.arguments ?? {}
    if (typeof name !== 'string' || !name.trim()) return null

    return {
      id: typeof parsed.id === 'string' ? parsed.id : undefined,
      index,
      functionName: name.trim(),
      functionArguments: typeof args === 'string' ? args : JSON.stringify(args),
    }
  } catch {
    return null
  }
}

const parseTaggedToolCall = (body: string, index: number): ParsedToolCallChunk | null => {
  const functionMatch = body.match(
    /<function\s*=\s*["']?([^"'\s>]+)["']?\s*>([\s\S]*?)<\/function\s*>/i,
  )
  if (!functionMatch) return null

  const args: Record<string, unknown> = {}
  const parameterPattern = /<parameter\s*=\s*["']?([^"'\s>]+)["']?\s*>([\s\S]*?)<\/parameter\s*>/gi
  let parameterMatch: RegExpExecArray | null
  while ((parameterMatch = parameterPattern.exec(functionMatch[2])) !== null) {
    args[parameterMatch[1]] = parseArgumentValue(parameterMatch[2])
  }

  return {
    index,
    functionName: functionMatch[1],
    functionArguments: JSON.stringify(args),
  }
}

export const extractTextToolCalls = (content: string): TextToolCallExtraction => {
  const toolCalls: ParsedToolCallChunk[] = []
  const toolCallPattern = /<tool_call\s*>([\s\S]*?)<\/tool_call\s*>/gi
  const cleaned = content.replace(toolCallPattern, (_fullMatch, body: string) => {
    const index = toolCalls.length
    const parsed = parseJsonToolCall(body, index) ?? parseTaggedToolCall(body, index)
    if (!parsed) return _fullMatch
    toolCalls.push(parsed)
    return ''
  })

  return {
    content: cleaned.replace(/\n[ \t]*\n[ \t]*\n+/g, '\n\n').trim(),
    toolCalls,
  }
}

export const mergeToolCallArgumentChunk = (current: string, incoming: string): string => {
  if (!incoming) return current
  if (!current) return incoming
  if (incoming === current || current.endsWith(incoming)) return current
  if (incoming.startsWith(current)) return incoming
  return current + incoming
}

export const normalizeToolCallArguments = (value: string): string => {
  const trimmed = value.trim()
  if (!trimmed) return '{}'

  try {
    return JSON.stringify(JSON.parse(trimmed))
  } catch {
    // Older Playground streams could concatenate a cumulative JSON object more
    // than once. Recover the first complete object before replaying that history.
  }

  let depth = 0
  let inString = false
  let escaped = false
  for (let index = 0; index < trimmed.length; index += 1) {
    const character = trimmed[index]
    if (inString) {
      if (escaped) {
        escaped = false
      } else if (character === '\\') {
        escaped = true
      } else if (character === '"') {
        inString = false
      }
      continue
    }

    if (character === '"') {
      inString = true
    } else if (character === '{' || character === '[') {
      depth += 1
    } else if (character === '}' || character === ']') {
      depth -= 1
      if (depth === 0) {
        const candidate = trimmed.slice(0, index + 1)
        try {
          return JSON.stringify(JSON.parse(candidate))
        } catch {
          return '{}'
        }
      }
    }
  }

  return '{}'
}
