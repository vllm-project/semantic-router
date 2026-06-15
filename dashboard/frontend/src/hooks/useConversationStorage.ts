// Browser-local conversation list for playground-style UX only — not server-owned
// or restart-safe. Supported multi-operator chat history lives in OpenClaw room APIs.
import { useCallback, useEffect, useState } from 'react'

import {
    DEFAULT_MAX_CONVERSATIONS,
    normalizeStoredConversations,
    pruneStoredConversations,
    type StoredConversation,
} from './conversationStorage'

export type { StoredConversation } from './conversationStorage'

interface UseConversationStorageOptions {
    storageKey?: string
    maxConversations?: number
}

const DEFAULT_STORAGE_KEY = 'sr:chat:conversations'

export const useConversationStorage = <T,>({
    storageKey = DEFAULT_STORAGE_KEY,
    maxConversations = DEFAULT_MAX_CONVERSATIONS,
}: UseConversationStorageOptions = {}) => {
    const [conversations, setConversations] = useState<StoredConversation<T>[]>([])

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
        }
    }, [maxConversations, storageKey])

    const updateAndPersist = useCallback(
        (updater: (prev: StoredConversation<T>[]) => StoredConversation<T>[]) => {
            setConversations(prev => {
                const next = pruneStoredConversations(updater(prev), {
                    maxConversations,
                })

                if (typeof window !== 'undefined') {
                    try {
                        if (next.length === 0) {
                            window.localStorage.removeItem(storageKey)
                        } else {
                            window.localStorage.setItem(storageKey, JSON.stringify(next))
                        }
                    } catch (err) {
                        console.error('Failed to save conversations to localStorage', err)
                    }
                }

                return next
            })
        },
        [maxConversations, storageKey]
    )

    const saveConversation = useCallback(
        (id: string, payload: T) => {
            const now = Date.now()

            updateAndPersist(prev => {
                const existingIndex = prev.findIndex(conv => conv.id === id)
                let next: StoredConversation<T>[]

                if (existingIndex >= 0) {
                    const updated = { ...prev[existingIndex], payload, updatedAt: now }
                    const withoutCurrent = prev.filter(conv => conv.id !== id)
                    next = [updated, ...withoutCurrent]
                } else {
                    next = [{ id, payload, createdAt: now, updatedAt: now }, ...prev]
                }

                return next
            })
        },
        [updateAndPersist]
    )

    const deleteConversation = useCallback(
        (id: string) => {
            updateAndPersist(prev => prev.filter(conv => conv.id !== id))
        },
        [updateAndPersist]
    )

    const clearAll = useCallback(() => {
        updateAndPersist(() => [])
    }, [updateAndPersist])

    const getConversation = useCallback(
        (id?: string) => {
            if (id) {
                return conversations.find(conv => conv.id === id)
            }
            return conversations[0]
        },
        [conversations]
    )

    return {
        conversations,
        saveConversation,
        deleteConversation,
        clearAll,
        getConversation,
    }
}
