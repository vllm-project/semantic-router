import {
  X_CONVERSATION_ID,
  X_SESSION_ID,
  X_VSR_CACHE_HIT,
  X_VSR_CONTEXT_TOKEN_COUNT,
  X_VSR_DEBUG,
  X_VSR_EFFECTIVE_INPUT_TOKENS,
  X_VSR_EFFECTIVE_MAX_OUTPUT_TOKENS,
  X_VSR_FAST_RESPONSE,
  X_VSR_LEARNING_ACTIONS,
  X_VSR_LEARNING_METHODS,
  X_VSR_LEARNING_REASONS,
  X_VSR_LEARNING_SCOPES,
  X_VSR_LOOPER_ALGORITHM,
  X_VSR_LOOPER_COMPLETION_TOKENS,
  X_VSR_LOOPER_ITERATIONS,
  X_VSR_LOOPER_LATENCY_MS,
  X_VSR_LOOPER_MODEL,
  X_VSR_LOOPER_MODELS_USED,
  X_VSR_LOOPER_PROMPT_TOKENS,
  X_VSR_LOOPER_TOTAL_TOKENS,
  X_VSR_MATCHED_AUTHZ,
  X_VSR_MATCHED_COMPLEXITY,
  X_VSR_MATCHED_CONTEXT,
  X_VSR_MATCHED_CONVERSATION,
  X_VSR_MATCHED_DECISION_MODEL,
  X_VSR_MATCHED_DOMAINS,
  X_VSR_MATCHED_EVENT,
  X_VSR_MATCHED_FACT_CHECK,
  X_VSR_MATCHED_HALLUCINATION,
  X_VSR_MATCHED_INPUT_MODALITY,
  X_VSR_MATCHED_JAILBREAK,
  X_VSR_MATCHED_KEYWORDS,
  X_VSR_MATCHED_LANGUAGE,
  X_VSR_MATCHED_MODALITY,
  X_VSR_MATCHED_PII,
  X_VSR_MATCHED_PROJECTIONS,
  X_VSR_MATCHED_TURN_RELATION,
  X_VSR_MATCHED_SAFETY,
  X_VSR_MATCHED_STRUCTURE,
  X_VSR_MATCHED_USER_FEEDBACK,
  X_VSR_REPLAY_ID,
  X_VSR_RESPONSE_PATH,
  X_VSR_RESPONSE_WARNINGS,
  X_VSR_RETENTION_DROP,
  X_VSR_RETENTION_KEEP_CURRENT_MODEL,
  X_VSR_RETENTION_PREFER_PREFIX,
  X_VSR_RETENTION_TTL_TURNS,
  X_VSR_ROUTING_LATENCY_MS,
  X_VSR_SCHEMA_VERSION,
  X_VSR_SELECTED_ALGORITHM,
  X_VSR_SELECTED_CONFIDENCE,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODALITY,
  X_VSR_SELECTED_MODEL,
  X_VSR_SELECTED_REASONING,
  X_VSR_SELECTED_RECIPE,
} from '../generated/vsrHeaders'
import type { Message } from './ChatComponentTypes'
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

export const PLAYGROUND_MAX_REQUEST_BYTES = 10 * 1024 * 1024

export const assertPlaygroundRequestSize = (request: Record<string, unknown>): void => {
  const bytes = new TextEncoder().encode(JSON.stringify(request)).byteLength
  if (bytes > PLAYGROUND_MAX_REQUEST_BYTES) {
    throw new Error('Playground request exceeds the 10 MB request limit.')
  }
}

export const buildPlaygroundRequestHeaders = (conversationId: string): Record<string, string> => ({
  'Content-Type': 'application/json',
  [X_SESSION_ID]: conversationId,
  [X_CONVERSATION_ID]: conversationId,
  [X_VSR_DEBUG]: 'true',
})

const RESPONSE_HEADER_KEYS = [
  // v0.4 keystone headers (#2203)
  X_VSR_SCHEMA_VERSION,
  X_VSR_RESPONSE_PATH,
  X_VSR_SELECTED_MODEL,
  X_VSR_EFFECTIVE_INPUT_TOKENS,
  X_VSR_EFFECTIVE_MAX_OUTPUT_TOKENS,
  X_VSR_SELECTED_ALGORITHM,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_RECIPE,
  X_VSR_SELECTED_CONFIDENCE,
  X_VSR_SELECTED_MODALITY,
  X_VSR_REPLAY_ID,
  X_VSR_CACHE_HIT,
  X_VSR_SELECTED_REASONING,
  X_VSR_LEARNING_METHODS,
  X_VSR_LEARNING_ACTIONS,
  X_VSR_LEARNING_SCOPES,
  X_VSR_LEARNING_REASONS,
  X_VSR_FAST_RESPONSE,
  X_VSR_RESPONSE_WARNINGS,
  X_VSR_MATCHED_KEYWORDS,
  X_VSR_MATCHED_DOMAINS,
  X_VSR_MATCHED_FACT_CHECK,
  X_VSR_MATCHED_USER_FEEDBACK,
  X_VSR_MATCHED_TURN_RELATION,
  X_VSR_MATCHED_LANGUAGE,
  X_VSR_MATCHED_CONTEXT,
  X_VSR_MATCHED_STRUCTURE,
  X_VSR_CONTEXT_TOKEN_COUNT,
  X_VSR_MATCHED_COMPLEXITY,
  X_VSR_MATCHED_MODALITY,
  X_VSR_MATCHED_AUTHZ,
  X_VSR_MATCHED_JAILBREAK,
  X_VSR_MATCHED_SAFETY,
  X_VSR_MATCHED_HALLUCINATION,
  X_VSR_MATCHED_PII,
  X_VSR_MATCHED_CONVERSATION,
  X_VSR_MATCHED_EVENT,
  X_VSR_MATCHED_INPUT_MODALITY,
  X_VSR_MATCHED_DECISION_MODEL,
  X_VSR_MATCHED_PROJECTIONS,
  X_VSR_LOOPER_MODEL,
  X_VSR_LOOPER_MODELS_USED,
  X_VSR_LOOPER_ITERATIONS,
  X_VSR_LOOPER_ALGORITHM,
  X_VSR_LOOPER_LATENCY_MS,
  X_VSR_LOOPER_PROMPT_TOKENS,
  X_VSR_LOOPER_COMPLETION_TOKENS,
  X_VSR_LOOPER_TOTAL_TOKENS,
  X_VSR_ROUTING_LATENCY_MS,
  X_VSR_RETENTION_DROP,
  X_VSR_RETENTION_TTL_TURNS,
  X_VSR_RETENTION_KEEP_CURRENT_MODEL,
  X_VSR_RETENTION_PREFER_PREFIX,
] as const

export const buildChatMessages = (
  messages: Message[],
  nextUserMessage: string,
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

  const result = {
    ...request,
    model: requestModel || fallbackModel,
    messages,
    stream: true,
  }
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
