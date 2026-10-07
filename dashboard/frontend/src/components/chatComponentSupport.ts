import type { StoredConversation } from '../hooks'
import type { PlaygroundInvocation } from '../types/playgroundInvocation'
import {
  PLAYGROUND_ACTIVE_CONVERSATION_STORAGE_KEY,
  type ConversationPreview,
  type Message,
} from './ChatComponentTypes'
import type { PlaygroundErrorPresentation } from './playgroundErrorPresentation'

export interface ChatComponentProps {
  endpoint?: string
  feedbackInsightsBasePath?: string
  invocation?: PlaygroundInvocation | null
  isFullscreenMode?: boolean
  onInvocationConsumed?: () => void
}

export function buildFeedbackInsightsHref(basePath: string | undefined, replayId: string) {
  const normalizedBasePath = basePath?.replace(/\/+$/, '')
  return normalizedBasePath ? `${normalizedBasePath}/${encodeURIComponent(replayId)}` : undefined
}

export const resolveActiveConversationPreference = (
  savedConversationId: string | null | undefined,
  conversations: readonly StoredConversation<Message[]>[],
): string | null => {
  const conversationId = savedConversationId?.trim()
  if (!conversationId) return null
  return conversations.some((conversation) => conversation.id === conversationId)
    ? conversationId
    : null
}

export const readActiveConversationPreference = (
  conversations: readonly StoredConversation<Message[]>[],
): string | null => {
  if (typeof window === 'undefined') return null
  return resolveActiveConversationPreference(
    window.localStorage.getItem(PLAYGROUND_ACTIVE_CONVERSATION_STORAGE_KEY),
    conversations,
  )
}

export const writeActiveConversationPreference = (conversationId: string): void => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(PLAYGROUND_ACTIVE_CONVERSATION_STORAGE_KEY, conversationId)
}

export const buildConversationPreviews = (
  conversations: readonly StoredConversation<Message[]>[],
): ConversationPreview[] =>
  [...conversations]
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .map((conversation) => {
      const firstUserMessage = Array.isArray(conversation.payload)
        ? conversation.payload.find((message) => message.role === 'user')
        : undefined
      const title = (conversation.title || firstUserMessage?.content || 'New conversation').trim()
      const preview = title.length > 60 ? `${title.slice(0, 60)}…` : title || 'New conversation'

      return {
        id: conversation.id,
        updatedAt: conversation.updatedAt || conversation.createdAt,
        preview,
      }
    })

export const getLiveThinkingProcess = (messages: readonly Message[]): string =>
  messages.reduceRight(
    (thinking, message) =>
      thinking ||
      (message.role === 'assistant' && message.isStreaming ? message.thinkingProcess || '' : ''),
    '',
  )

export const findQueuedErrorConversationId = (
  queues: Record<string, readonly unknown[] | undefined>,
  conversationErrors: Record<string, PlaygroundErrorPresentation>,
): string | undefined =>
  Object.keys(queues).find(
    (conversationId) =>
      (queues[conversationId]?.length ?? 0) > 0 && Boolean(conversationErrors[conversationId]),
  )
