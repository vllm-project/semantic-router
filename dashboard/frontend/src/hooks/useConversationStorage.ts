// Browser-local conversation list for playground-style UX only — not server-owned
// or restart-safe. Supported multi-operator chat history lives in OpenClaw room APIs.
import { useCallback, useEffect, useRef, useState } from 'react'

import {
  DEFAULT_MAX_CONVERSATIONS,
  normalizeStoredConversations,
  prepareStoredConversationsForPersistence,
  pruneStoredConversations,
  type StoredConversation,
} from './conversationStorage'

export type { StoredConversation } from './conversationStorage'

interface UseConversationStorageOptions<T> {
  storageKey?: string
  maxConversations?: number
  maxStorageBytes?: number
  preparePayloadForPersistence?: (payload: T) => T | null
}

const DEFAULT_STORAGE_KEY = 'sr:chat:conversations'
const PERSIST_DEBOUNCE_MS = 350

export const useConversationStorage = <T>({
  storageKey = DEFAULT_STORAGE_KEY,
  maxConversations = DEFAULT_MAX_CONVERSATIONS,
  maxStorageBytes,
  preparePayloadForPersistence,
}: UseConversationStorageOptions<T> = {}) => {
  const [conversations, setConversations] = useState<StoredConversation<T>[]>([])
  const [isHydrated, setIsHydrated] = useState(false)
  const pendingPersistenceRef = useRef<StoredConversation<T>[] | null>(null)
  const persistenceTimerRef = useRef<number | null>(null)

  const persistConversations = useCallback(
    (next: StoredConversation<T>[]) => {
      if (typeof window === 'undefined') return

      try {
        const persisted = prepareStoredConversationsForPersistence(next, {
          maxBytes: maxStorageBytes,
          preparePayload: preparePayloadForPersistence,
        })
        if (persisted.length === 0) {
          window.localStorage.removeItem(storageKey)
        } else {
          window.localStorage.setItem(storageKey, JSON.stringify(persisted))
        }
      } catch (err) {
        console.error('Failed to save conversations to localStorage', err)
      }
    },
    [maxStorageBytes, preparePayloadForPersistence, storageKey],
  )

  const flushPendingPersistence = useCallback(() => {
    if (typeof window !== 'undefined' && persistenceTimerRef.current !== null) {
      window.clearTimeout(persistenceTimerRef.current)
    }
    persistenceTimerRef.current = null

    const pending = pendingPersistenceRef.current
    if (!pending) return

    pendingPersistenceRef.current = null
    persistConversations(pending)
  }, [persistConversations])

  const schedulePersistence = useCallback(
    (next: StoredConversation<T>[]) => {
      pendingPersistenceRef.current = next
      if (typeof window === 'undefined') return

      if (persistenceTimerRef.current !== null) {
        window.clearTimeout(persistenceTimerRef.current)
      }
      persistenceTimerRef.current = window.setTimeout(flushPendingPersistence, PERSIST_DEBOUNCE_MS)
    },
    [flushPendingPersistence],
  )

  useEffect(() => {
    if (typeof window === 'undefined') return

    try {
      const raw = window.localStorage.getItem(storageKey)
      if (!raw) return

      const restored = normalizeStoredConversations<T>(JSON.parse(raw), {
        maxConversations,
      })

      if (restored.length === 0) {
        window.localStorage.removeItem(storageKey)
      } else {
        window.localStorage.setItem(storageKey, JSON.stringify(restored))
      }

      setConversations(restored)
    } catch (err) {
      console.error('Failed to load conversations from localStorage', err)
    } finally {
      setIsHydrated(true)
    }
  }, [maxConversations, storageKey])

  useEffect(() => {
    if (typeof window === 'undefined') return

    const flushWhenHidden = () => {
      if (document.visibilityState === 'hidden') {
        flushPendingPersistence()
      }
    }

    window.addEventListener('pagehide', flushPendingPersistence)
    document.addEventListener('visibilitychange', flushWhenHidden)

    return () => {
      window.removeEventListener('pagehide', flushPendingPersistence)
      document.removeEventListener('visibilitychange', flushWhenHidden)
      flushPendingPersistence()
    }
  }, [flushPendingPersistence])

  const updateAndPersist = useCallback(
    (updater: (prev: StoredConversation<T>[]) => StoredConversation<T>[]) => {
      setConversations((prev) => {
        const next = pruneStoredConversations(updater(prev), {
          maxConversations,
        })

        schedulePersistence(next)

        return next
      })
    },
    [maxConversations, schedulePersistence],
  )

  const saveConversation = useCallback(
    (id: string, payload: T) => {
      const now = Date.now()

      updateAndPersist((prev) => {
        const existingIndex = prev.findIndex((conv) => conv.id === id)
        let next: StoredConversation<T>[]

        if (existingIndex >= 0) {
          const updated = { ...prev[existingIndex], payload, updatedAt: now }
          const withoutCurrent = prev.filter((conv) => conv.id !== id)
          next = [updated, ...withoutCurrent]
        } else {
          next = [{ id, payload, createdAt: now, updatedAt: now }, ...prev]
        }

        return next
      })
    },
    [updateAndPersist],
  )

  const deleteConversation = useCallback(
    (id: string) => {
      updateAndPersist((prev) => prev.filter((conv) => conv.id !== id))
    },
    [updateAndPersist],
  )

  const renameConversation = useCallback(
    (id: string, title: string) => {
      const normalizedTitle = title.trim().slice(0, 80)
      if (!normalizedTitle) return
      updateAndPersist((prev) =>
        prev.map((conversation) =>
          conversation.id === id ? { ...conversation, title: normalizedTitle } : conversation,
        ),
      )
    },
    [updateAndPersist],
  )

  const clearAll = useCallback(() => {
    updateAndPersist(() => [])
  }, [updateAndPersist])

  const getConversation = useCallback(
    (id?: string) => {
      if (id) {
        return conversations.find((conv) => conv.id === id)
      }
      return conversations[0]
    },
    [conversations],
  )

  return {
    conversations,
    isHydrated,
    saveConversation,
    renameConversation,
    deleteConversation,
    clearAll,
    getConversation,
    flushPendingPersistence,
  }
}
