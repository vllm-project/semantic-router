import type { OpenAIChatContent, OpenAIChatUsage } from './openAIChatCompletions'

export interface PlaygroundInferenceMetadata {
  finishReason?: string
  headers: Record<string, string>
  latencyMilliseconds: number
  model?: string
  requestId?: string
  responseId?: string
  ttftMilliseconds?: number
  usage?: OpenAIChatUsage
}

export interface PlaygroundInferenceMessage {
  content: string
  createdAt: string
  id: string
  metadata?: PlaygroundInferenceMetadata
  requestContent?: OpenAIChatContent
  role: 'user' | 'assistant'
  status: 'complete' | 'streaming' | 'cancelled' | 'failed'
}

export function applyPlaygroundInferenceDelta(
  messages: readonly PlaygroundInferenceMessage[],
  assistantId: string,
  delta: string,
): PlaygroundInferenceMessage[] {
  if (!delta) return [...messages]
  return messages.map((message) =>
    message.id === assistantId ? { ...message, content: `${message.content}${delta}` } : message,
  )
}

export function completePlaygroundInferenceMessage(
  messages: readonly PlaygroundInferenceMessage[],
  assistantId: string,
  metadata: PlaygroundInferenceMetadata,
): PlaygroundInferenceMessage[] {
  return messages.map((message) =>
    message.id === assistantId ? { ...message, status: 'complete', metadata } : message,
  )
}

export function assertPlaygroundAssistantText(content: string): void {
  if (!content.trim()) throw new Error('Router completed the stream without assistant text.')
}
