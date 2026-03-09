import type { Choice, ReMoMRoundResponse } from './ChatComponentTypes'

type JsonRecord = Record<string, unknown>

export interface ChoiceAccumulator {
  content: string
  reasoningContent: string
  model?: string
}

export interface ParsedToolCallChunk {
  id?: string
  index: number
  functionName?: string
  functionArguments?: string
}

export interface ParsedChatChoice {
  index: number
  content: string
  reasoningContent: string
  model?: string
  finishReason?: string
  toolCalls: ParsedToolCallChunk[]
}

export interface ParsedChatCompletion {
  choices: ParsedChatChoice[]
  errorMessage?: string
  reasoningMomResponses?: ReMoMRoundResponse[]
}

const asRecord = (value: unknown): JsonRecord | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  return value as JsonRecord
}

const asString = (value: unknown): string => (typeof value === 'string' ? value : '')

const extractTextContent = (value: unknown): string => {
  if (typeof value === 'string') {
    return value
  }

  if (Array.isArray(value)) {
    return value.map(extractTextContent).filter(Boolean).join('')
  }

  const record = asRecord(value)
  if (!record) {
    return ''
  }

  return (
    asString(record.text) ||
    asString(record.content) ||
    asString(record.output_text)
  )
}

const extractReasoningContent = (...sources: Array<JsonRecord | null>): string => {
  for (const source of sources) {
    if (!source) continue

    const reasoningContent = asString(source.reasoning_content)
    if (reasoningContent) {
      return reasoningContent
    }

    const reasoning = asString(source.reasoning)
    if (reasoning) {
      return reasoning
    }
  }

  return ''
}

const extractToolCalls = (source: JsonRecord | null): ParsedToolCallChunk[] => {
  if (!source || !Array.isArray(source.tool_calls)) {
    return []
  }

  return source.tool_calls.flatMap((value, rawIndex) => {
    const toolCall = asRecord(value)
    if (!toolCall) {
      return []
    }

    const fn = asRecord(toolCall.function)
    return [{
      id: asString(toolCall.id) || undefined,
      index: typeof toolCall.index === 'number' ? toolCall.index : rawIndex,
      functionName: asString(fn?.name) || undefined,
      functionArguments: asString(fn?.arguments) || undefined,
    }]
  })
}

export const isEventStreamContentType = (contentType: string | null): boolean =>
  typeof contentType === 'string' && contentType.includes('text/event-stream')

export const mergeParsedChoices = (
  choiceContents: Map<number, ChoiceAccumulator>,
  parsedChoices: ParsedChatChoice[]
) => {
  for (const parsedChoice of parsedChoices) {
    if (!choiceContents.has(parsedChoice.index)) {
      choiceContents.set(parsedChoice.index, {
        content: '',
        reasoningContent: '',
        model: parsedChoice.model,
      })
    }

    const current = choiceContents.get(parsedChoice.index)
    if (!current) {
      continue
    }

    if (parsedChoice.content) {
      current.content += parsedChoice.content
    }

    if (parsedChoice.reasoningContent) {
      current.reasoningContent += parsedChoice.reasoningContent
    }

    if (parsedChoice.model && !current.model) {
      current.model = parsedChoice.model
    }
  }
}

export const buildChoicesArray = (choiceContents: Map<number, ChoiceAccumulator>): Choice[] =>
  Array.from(choiceContents.entries())
    .sort(([left], [right]) => left - right)
    .map(([, value]) => ({ content: value.content, model: value.model }))

export const getFirstChoice = (choiceContents: Map<number, ChoiceAccumulator>): ChoiceAccumulator | undefined =>
  choiceContents.get(0)

export const parseChatCompletionPayload = (payload: string): ParsedChatCompletion | null => {
  let parsed: unknown

  try {
    parsed = JSON.parse(payload)
  } catch {
    return null
  }

  return parseChatCompletionObject(parsed)
}

export const parseChatCompletionObject = (value: unknown): ParsedChatCompletion | null => {
  const root = asRecord(value)
  if (!root) {
    return null
  }

  const error = asRecord(root.error)
  const topLevelModel = asString(root.model)
  const rawChoices = Array.isArray(root.choices) ? root.choices : []

  const choices = rawChoices.flatMap((value, rawIndex) => {
    const choice = asRecord(value)
    if (!choice) {
      return []
    }

    const delta = asRecord(choice.delta)
    const message = asRecord(choice.message)
    const toolCalls = [
      ...extractToolCalls(delta),
      ...extractToolCalls(message),
    ]

    return [{
      index: typeof choice.index === 'number' ? choice.index : rawIndex,
      content:
        extractTextContent(delta?.content) ||
        extractTextContent(message?.content) ||
        extractTextContent(choice.content),
      reasoningContent: extractReasoningContent(delta, choice, message),
      model: asString(choice.model) || topLevelModel || undefined,
      finishReason: asString(choice.finish_reason) || undefined,
      toolCalls,
    }]
  })

  return {
    choices,
    errorMessage: asString(error?.message) || undefined,
    reasoningMomResponses: Array.isArray(root.reasoning_mom_responses)
      ? root.reasoning_mom_responses as ReMoMRoundResponse[]
      : undefined,
  }
}
