import { useCallback, useState } from 'react'

export const useChatConversationState = () => {
  const [conversationErrors, setConversationErrors] = useState<Record<string, string>>({})
  const [conversationThinking, setConversationThinking] = useState<Record<string, boolean>>({})

  const setConversationError = useCallback((targetConversationId: string, error: string | null) => {
    setConversationErrors((prev) => {
      if (!error) {
        if (!(targetConversationId in prev)) {
          return prev
        }
        const next = { ...prev }
        delete next[targetConversationId]
        return next
      }
      if (prev[targetConversationId] === error) {
        return prev
      }
      return {
        ...prev,
        [targetConversationId]: error,
      }
    })
  }, [])

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
