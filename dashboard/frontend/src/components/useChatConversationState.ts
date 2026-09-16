import { useCallback, useState } from 'react'

import {
  normalizePlaygroundError,
  type PlaygroundErrorInput,
  type PlaygroundErrorPresentation,
} from './playgroundErrorPresentation'

export const useChatConversationState = () => {
  const [conversationErrors, setConversationErrors] = useState<
    Record<string, PlaygroundErrorPresentation>
  >({})
  const [conversationThinking, setConversationThinking] = useState<Record<string, boolean>>({})

  const setConversationError = useCallback(
    (targetConversationId: string, error: PlaygroundErrorInput) => {
      setConversationErrors((prev) => {
        if (!error) {
          if (!(targetConversationId in prev)) return prev
          const next = { ...prev }
          delete next[targetConversationId]
          return next
        }
        const nextError = normalizePlaygroundError(error)
        const current = prev[targetConversationId]
        if (
          current?.message === nextError.message &&
          current.technicalDetails === nextError.technicalDetails
        ) {
          return prev
        }
        return { ...prev, [targetConversationId]: nextError }
      })
    },
    [],
  )

  const setConversationThinkingState = useCallback(
    (targetConversationId: string, visible: boolean) => {
      setConversationThinking((prev) => {
        const current = prev[targetConversationId] ?? false
        if (current === visible) {
          return prev
        }
        if (!visible) {
          if (!(targetConversationId in prev)) {
            return prev
          }
          const next = { ...prev }
          delete next[targetConversationId]
          return next
        }
        return {
          ...prev,
          [targetConversationId]: true,
        }
      })
    },
    [],
  )

  return {
    conversationErrors,
    conversationThinking,
    setConversationError,
    setConversationThinkingState,
  }
}
