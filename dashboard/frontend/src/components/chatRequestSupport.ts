import { CLAW_MODE_SYSTEM_PROMPT, type Message } from './ChatComponentTypes'
import { buildPlaygroundUserContent, type PlaygroundAttachment } from './playgroundFileAttachments'
import { extractTextToolCalls, normalizeToolCallArguments } from './chatToolCallSupport'
import { serializeToolResultForModel } from '../tools/toolResultSupport'

export interface OutboundChatMessage {
  role: string
  content: unknown
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: {
      name: string
      arguments: string
    }
  }>
  tool_call_id?: string
}

export const PLAYGROUND_DEFAULT_MAX_COMPLETION_TOKENS = 2048
export const PLAYGROUND_REQUEST_TIMEOUT_MS = 120_000
export const PLAYGROUND_MAX_REQUEST_BYTES = 10 * 1024 * 1024

export const assertPlaygroundRequestSize = (request: Record<string, unknown>): void => {
  const bytes = new TextEncoder().encode(JSON.stringify(request)).byteLength
  if (bytes > PLAYGROUND_MAX_REQUEST_BYTES) {
    throw new Error('Playground request exceeds the 10 MB request limit.')
  }
}

export const buildPlaygroundRequestHeaders = (conversationId: string): Record<string, string> => ({
  Accept: 'text/event-stream',
  'Content-Type': 'application/json',
  'x-session-id': conversationId,
  'x-vsr-debug': 'true',
})

const withDefaultCompletionBudget = (request: Record<string, unknown>): Record<string, unknown> => {
  if (request.max_tokens !== undefined || request.max_completion_tokens !== undefined) {
    return request
  }
  return { ...request, max_completion_tokens: PLAYGROUND_DEFAULT_MAX_COMPLETION_TOKENS }
}

const RESPONSE_HEADER_KEYS = [
  // v0.4 keystone headers (#2203)
  'x-vsr-schema-version',
  'x-vsr-response-path',
  'x-vsr-selected-model',
  'x-vsr-selected-algorithm',
  'x-vsr-selected-decision',
  'x-vsr-selected-modality',
  'x-vsr-replay-id',
  'x-vsr-cache-hit',
  'x-vsr-selected-reasoning',
  'x-vsr-learning-methods',
  'x-vsr-learning-actions',
  'x-vsr-learning-scopes',
  'x-vsr-learning-reasons',
  'x-vsr-fast-response',
  'x-vsr-response-warnings',
  'x-vsr-matched-keywords',
  'x-vsr-matched-embeddings',
  'x-vsr-matched-domains',
  'x-vsr-matched-fact-check',
  'x-vsr-matched-user-feedback',
  'x-vsr-matched-reask',
  'x-vsr-matched-preference',
  'x-vsr-matched-language',
  'x-vsr-matched-context',
  'x-vsr-matched-structure',
  'x-vsr-context-token-count',
  'x-vsr-matched-complexity',
  'x-vsr-matched-modality',
  'x-vsr-matched-authz',
  'x-vsr-matched-jailbreak',
  'x-vsr-matched-pii',
  'x-vsr-matched-kb',
  'x-vsr-matched-conversation',
  'x-vsr-matched-event',
  'x-vsr-matched-projections',
  'x-vsr-looper-model',
  'x-vsr-looper-models-used',
  'x-vsr-looper-iterations',
  'x-vsr-looper-algorithm',
  'x-vsr-looper-latency-ms',
  'x-vsr-looper-prompt-tokens',
  'x-vsr-looper-completion-tokens',
  'x-vsr-looper-total-tokens',
  'x-vsr-retention-drop',
  'x-vsr-retention-ttl-turns',
  'x-vsr-retention-keep-current-model',
  'x-vsr-retention-prefer-prefix',
] as const

export const buildChatMessages = (
  messages: Message[],
  nextUserMessage: string,
  enableClawMode: boolean,
  nextUserAttachments: PlaygroundAttachment[] = [],
): OutboundChatMessage[] => {
  const chatMessages: OutboundChatMessage[] = []

  for (const message of messages) {
    if (message.role === 'user') {
      chatMessages.push({
        role: 'user',
        content:
          message.requestContent ??
          buildPlaygroundUserContent(message.content, message.playgroundAttachments ?? []),
      })
      continue
    }

    if (message.role === 'system') {
      chatMessages.push({
        role: 'system',
        content: message.requestContent ?? message.content,
      })
      continue
    }

    if (message.role === 'assistant') {
      const resultsByCallId = new Map(
        (message.toolResults ?? []).map((result) => [result.callId, result]),
      )
      const completedToolCalls = (message.toolCalls ?? []).filter(
        (toolCall) =>
          resultsByCallId.has(toolCall.id) &&
          Boolean(toolCall.function.name) &&
          Boolean(toolCall.function.arguments),
      )

      if (completedToolCalls.length > 0) {
        chatMessages.push({
          role: 'assistant',
          content: null,
          tool_calls: completedToolCalls.map((toolCall) => ({
            id: toolCall.id,
            type: 'function',
            function: {
              name: toolCall.function.name,
              arguments: normalizeToolCallArguments(toolCall.function.arguments),
            },
          })),
        })
        completedToolCalls.forEach((toolCall) => {
          const result = resultsByCallId.get(toolCall.id)
          if (!result) return
          chatMessages.push({
            role: 'tool',
            tool_call_id: toolCall.id,
            content: serializeToolResultForModel(result),
          })
        })
      }

      const assistantContent =
        message.requestContent ??
        (/<tool_call/i.test(message.content)
          ? extractTextToolCalls(message.content).content
          : message.content)
      if (
        (typeof assistantContent === 'string' && assistantContent.length > 0) ||
        (Array.isArray(assistantContent) && assistantContent.length > 0)
      ) {
        chatMessages.push({ role: 'assistant', content: assistantContent })
      }
    }
  }

  if (enableClawMode) {
    chatMessages.unshift({ role: 'system', content: CLAW_MODE_SYSTEM_PROMPT })
  }

  chatMessages.push({
    role: 'user',
    content: buildPlaygroundUserContent(nextUserMessage, nextUserAttachments),
  })
  return chatMessages
}

export const buildChatRequestBody = (
  model: string,
  messages: OutboundChatMessage[],
  activeTools: unknown[],
): Record<string, unknown> => {
  const requestBody: Record<string, unknown> = {
    model,
    messages,
    stream: true,
    max_completion_tokens: PLAYGROUND_DEFAULT_MAX_COMPLETION_TOKENS,
  }

  if (activeTools.length > 0) {
    requestBody.tools = activeTools
    requestBody.tool_choice = 'auto'
  }

  assertPlaygroundRequestSize(requestBody)
  return requestBody
}

export const buildExactChatRequestBody = (
  request: Record<string, unknown>,
  fallbackModel: string,
): Record<string, unknown> => {
  const messages = Array.isArray(request.messages) ? request.messages : []
  const requestModel = typeof request.model === 'string' ? request.model.trim() : ''

  const result = withDefaultCompletionBudget({
    ...request,
    model: requestModel || fallbackModel,
    messages,
    stream: true,
  })
  assertPlaygroundRequestSize(result)
  return result
}

export const collectResponseHeaders = (response: Response): Record<string, string> => {
  const headers: Record<string, string> = {}

  RESPONSE_HEADER_KEYS.forEach((key) => {
    const value = response.headers.get(key)
    if (value) {
      headers[key] = value
    }
  })

  return headers
}
