import {
  X_CONVERSATION_ID,
  X_VSR_COST,
  X_VSR_COST_CURRENCY,
  X_VSR_DEBUG,
  X_VSR_REPLAY_ID,
  X_VSR_RESPONSE_PATH,
  X_VSR_SCHEMA_VERSION,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODEL,
  X_VSR_SELECTED_RECIPE,
  X_VSR_SESSION_ID,
  X_VSR_TRACE,
  X_VSR_WARNINGS,
} from '../generated/vsrHeaders'
import type { Message } from './ChatComponentTypes'
import { buildPlaygroundUserContent, type PlaygroundAttachment } from './playgroundFileAttachments'
import { extractTextToolCalls, normalizeToolCallArguments } from './chatToolCallSupport'
import { serializeToolResultForModel } from '../tools/toolResultSupport'
import { expandRoutingHeaders } from './vsrTrace'

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
  [X_VSR_SESSION_ID]: conversationId,
  [X_CONVERSATION_ID]: conversationId,
  [X_VSR_DEBUG]: 'true',
})

// The public set of a routed response, and the trace of a caller the Router
// allows the debug headers.
const RESPONSE_HEADER_KEYS = [
  X_VSR_SCHEMA_VERSION,
  X_VSR_RESPONSE_PATH,
  X_VSR_SELECTED_RECIPE,
  X_VSR_SELECTED_DECISION,
  X_VSR_SELECTED_MODEL,
  X_VSR_REPLAY_ID,
  X_VSR_WARNINGS,
  X_VSR_COST,
  X_VSR_COST_CURRENCY,
  X_VSR_TRACE,
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

// collectResponseHeaders returns the display entries of a routed response:
// its public headers, and the details of its trace.
export const collectResponseHeaders = (response: Response): Record<string, string> => {
  const headers: Record<string, string> = {}

  RESPONSE_HEADER_KEYS.forEach((key) => {
    const value = response.headers.get(key)
    if (value) {
      headers[key] = value
    }
  })

  return expandRoutingHeaders(headers)
}
