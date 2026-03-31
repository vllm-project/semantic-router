import { CLAW_MODE_SYSTEM_PROMPT, type Message } from './ChatComponentTypes'

export interface OutboundChatMessage {
  role: string
  content: string | null
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

const RESPONSE_HEADER_KEYS = [
  'x-vsr-selected-model',
  'x-vsr-selected-decision',
  'x-vsr-selected-modality',
  'x-vsr-cache-hit',
  'x-vsr-selected-reasoning',
  'x-vsr-fast-response',
  'x-vsr-jailbreak-blocked',
  'x-vsr-pii-violation',
  'x-vsr-hallucination-detected',
  'x-vsr-fact-check-needed',
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
  'x-vsr-matched-projection',
  'x-vsr-looper-model',
  'x-vsr-looper-models-used',
  'x-vsr-looper-iterations',
  'x-vsr-looper-algorithm',
] as const

export const buildChatMessages = (
  messages: Message[],
  nextUserMessage: string,
  enableClawMode: boolean
): OutboundChatMessage[] => {
  const chatMessages: OutboundChatMessage[] = []

  for (const message of messages) {
    if (message.role === 'user') {
      chatMessages.push({ role: 'user', content: message.content })
      continue
    }

    if (message.role === 'assistant' && message.content) {
      chatMessages.push({ role: 'assistant', content: message.content })
    }
  }

  if (enableClawMode) {
    chatMessages.unshift({ role: 'system', content: CLAW_MODE_SYSTEM_PROMPT })
  }

  chatMessages.push({ role: 'user', content: nextUserMessage })
  return chatMessages
}

export const buildChatRequestBody = (
  model: string,
  messages: OutboundChatMessage[],
  activeTools: unknown[]
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

  return requestBody
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
