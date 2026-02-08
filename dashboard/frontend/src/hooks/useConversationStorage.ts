import { useCallback, useEffect, useState } from 'react'

export interface StoredConversation<T> {
    id: string
    createdAt: number
    updatedAt: number
    payload: T
}

interface UseConversationStorageOptions {
    storageKey?: string
    maxConversations?: number
}

const DEFAULT_STORAGE_KEY = 'sr:chat:conversations'
const DEFAULT_MAX_CONVERSATIONS = 20

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

            const parsed = JSON.parse(raw)
            if (Array.isArray(parsed)) {
                setConversations(parsed)
            }
        } catch (err) {
            console.error('Failed to load conversations from localStorage', err)
        }
    }, [storageKey])

    const updateAndPersist = useCallback(
        (updater: (prev: StoredConversation<T>[]) => StoredConversation<T>[]) => {
            setConversations(prev => {
                const next = updater(prev)

                if (typeof window !== 'undefined') {
                    try {
                        window.localStorage.setItem(storageKey, JSON.stringify(next))
                    } catch (err) {
                        console.error('Failed to save conversations to localStorage', err)
                    }
                }

                return next
            })
        },
        [storageKey]
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

                if (next.length > maxConversations) {
                    next = next.slice(0, maxConversations)
                }

                return next
            })
        },
        [maxConversations, updateAndPersist]
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
