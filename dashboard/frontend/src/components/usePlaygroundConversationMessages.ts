import { useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from 'react'

import type { Message } from './ChatComponentTypes'

interface StoredConversation<T> {
  id: string
  payload?: T
}

interface UsePlaygroundConversationMessagesOptions<T extends Message[]> {
  conversationMessages: Record<string, T>
  getConversation: (id?: string) => StoredConversation<T> | undefined
  setConversationMessages: Dispatch<SetStateAction<Record<string, T>>>
}

export const usePlaygroundConversationMessages = <T extends Message[]>({
  conversationMessages,
  getConversation,
  setConversationMessages,
}: UsePlaygroundConversationMessagesOptions<T>) => {
  const conversationMessagesRef = useRef<Record<string, T>>({})

  const restoreMessages = useCallback((payload: T) => {
    return payload.map(message => ({
      ...message,
      timestamp: new Date(message.timestamp),
    })) as T
  }, [])

  const getStoredMessagesForConversation = useCallback((id: string): T => {
    const storedConversation = getConversation(id)
    if (!storedConversation?.payload || !Array.isArray(storedConversation.payload)) {
      return [] as unknown as T
    }
    return restoreMessages(storedConversation.payload)
  }, [getConversation, restoreMessages])

  const updateConversationMessages = useCallback(
    (targetConversationId: string, updater: (prev: T) => T) => {
      setConversationMessages(prev => {
        const baseMessages = prev[targetConversationId] ?? getStoredMessagesForConversation(targetConversationId)
        const nextMessages = updater(baseMessages)
        if (nextMessages === baseMessages) {
          return prev
        }
        return {
          ...prev,
          [targetConversationId]: nextMessages,
        }
      })
    },
    [getStoredMessagesForConversation, setConversationMessages]
  )

  const removeConversationMessages = useCallback((targetConversationId: string) => {
    setConversationMessages(prev => {
      if (!(targetConversationId in prev)) {
        return prev
      }
      const next = { ...prev }
      delete next[targetConversationId]
      return next
    })
  }, [setConversationMessages])

  const getConversationMessagesSnapshot = useCallback((targetConversationId: string) => (
    conversationMessagesRef.current[targetConversationId] ?? getStoredMessagesForConversation(targetConversationId)
  ), [getStoredMessagesForConversation])

  useEffect(() => {
    conversationMessagesRef.current = conversationMessages
  }, [conversationMessages])

  return {
    getConversationMessagesSnapshot,
    getStoredMessagesForConversation,
    removeConversationMessages,
    restoreMessages,
    updateConversationMessages,
  }
}
