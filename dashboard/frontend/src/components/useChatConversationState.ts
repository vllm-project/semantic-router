import { useCallback, useState } from 'react'

export interface ConversationHeaderRevealState {
  headers: Record<string, string>
  visible: boolean
}

export const useChatConversationState = () => {
  const [conversationErrors, setConversationErrors] = useState<Record<string, string>>({})
  const [conversationThinking, setConversationThinking] = useState<Record<string, boolean>>({})
  const [headerRevealStates, setHeaderRevealStates] = useState<
    Record<string, ConversationHeaderRevealState>
  >({})

  const setConversationError = useCallback((targetConversationId: string, error: string | null) => {
    setConversationErrors(prev => {
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

  const setConversationThinkingState = useCallback((targetConversationId: string, visible: boolean) => {
    setConversationThinking(prev => {
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
  }, [])

  const setConversationHeaderReveal = useCallback((
    targetConversationId: string,
    headers: Record<string, string> | null,
    visible = false
  ) => {
    setHeaderRevealStates(prev => {
      if (!headers || Object.keys(headers).length === 0) {
        if (!(targetConversationId in prev)) {
          return prev
        }
        const next = { ...prev }
        delete next[targetConversationId]
        return next
      }
      const current = prev[targetConversationId]
      if (current && current.visible === visible && current.headers === headers) {
        return prev
      }
      return {
        ...prev,
        [targetConversationId]: {
          headers,
          visible,
        },
      }
    })
  }, [])

  return {
    conversationErrors,
    conversationThinking,
    headerRevealStates,
    setConversationError,
    setConversationHeaderReveal,
    setConversationThinkingState,
  }
}
