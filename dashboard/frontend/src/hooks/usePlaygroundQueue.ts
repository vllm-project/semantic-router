import { useCallback, useEffect, useState } from 'react'

import type { PlaygroundTask } from '../components/ChatComponentTypes'
import { PLAYGROUND_QUEUE_STORAGE_KEY } from '../components/ChatComponentTypes'
import {
  DEFAULT_MAX_QUEUE_CONVERSATIONS,
  DEFAULT_MAX_TASKS_PER_CONVERSATION,
  normalizePlaygroundQueues,
  prunePlaygroundQueues,
  type PlaygroundTaskQueues,
} from './playgroundQueueStorage'

interface UsePlaygroundQueueOptions {
  storageKey?: string
  maxConversations?: number
  maxTasksPerConversation?: number
}

const removeConversationQueue = (
  prev: PlaygroundTaskQueues,
  conversationId: string
): PlaygroundTaskQueues => {
  if (!(conversationId in prev)) {
    return prev
  }

  const next = { ...prev }
  delete next[conversationId]
  return next
}

export const usePlaygroundQueue = ({
  storageKey = PLAYGROUND_QUEUE_STORAGE_KEY,
  maxConversations = DEFAULT_MAX_QUEUE_CONVERSATIONS,
  maxTasksPerConversation = DEFAULT_MAX_TASKS_PER_CONVERSATION,
}: UsePlaygroundQueueOptions = {}) => {
  const [queues, setQueues] = useState<PlaygroundTaskQueues>({})

  useEffect(() => {
    if (typeof window === 'undefined') return

    try {
      const raw = window.localStorage.getItem(storageKey)
      if (!raw) return

      const restored = normalizePlaygroundQueues(JSON.parse(raw), {
        maxConversations,
        maxTasksPerConversation,
      })

      if (Object.keys(restored).length === 0) {
        window.localStorage.removeItem(storageKey)
      } else {
        window.localStorage.setItem(storageKey, JSON.stringify(restored))
      }

      setQueues(restored)
    } catch (err) {
      console.error('Failed to load playground queue from localStorage', err)
    }
  }, [maxConversations, maxTasksPerConversation, storageKey])

  const updateAndPersist = useCallback(
    (updater: (prev: PlaygroundTaskQueues) => PlaygroundTaskQueues) => {
      setQueues(prev => {
        const next = prunePlaygroundQueues(updater(prev), {
          maxConversations,
          maxTasksPerConversation,
        })

        if (typeof window !== 'undefined') {
          try {
            if (Object.keys(next).length === 0) {
              window.localStorage.removeItem(storageKey)
            } else {
              window.localStorage.setItem(storageKey, JSON.stringify(next))
            }
          } catch (err) {
            console.error('Failed to save playground queue to localStorage', err)
          }
        }

        return next
      })
    },
    [maxConversations, maxTasksPerConversation, storageKey]
  )

  const getQueue = useCallback(
    (conversationId: string) => queues[conversationId] ?? [],
    [queues]
  )

  const enqueueTask = useCallback(
    (task: PlaygroundTask) => {
      updateAndPersist(prev => ({
        ...prev,
        [task.conversationId]: [...(prev[task.conversationId] ?? []), task],
      }))
    },
    [updateAndPersist]
  )

  const removeTask = useCallback(
    (conversationId: string, taskId: string) => {
      updateAndPersist(prev => {
        const queue = prev[conversationId] ?? []
        const nextQueue = queue.filter(task => task.id !== taskId)

        if (nextQueue.length === queue.length) {
          return prev
        }

        if (nextQueue.length === 0) {
          return removeConversationQueue(prev, conversationId)
        }

        return {
          ...prev,
          [conversationId]: nextQueue,
        }
      })
    },
    [updateAndPersist]
  )

  const reorderTasks = useCallback(
    (conversationId: string, sourceTaskId: string, targetTaskId: string) => {
      updateAndPersist(prev => {
        const queue = prev[conversationId] ?? []
        const sourceIndex = queue.findIndex(task => task.id === sourceTaskId)
        const targetIndex = queue.findIndex(task => task.id === targetTaskId)

        if (sourceIndex < 0 || targetIndex < 0 || sourceIndex === targetIndex) {
          return prev
        }

        const nextQueue = [...queue]
        const [movedTask] = nextQueue.splice(sourceIndex, 1)
        nextQueue.splice(targetIndex, 0, movedTask)

        return {
          ...prev,
          [conversationId]: nextQueue,
        }
      })
    },
    [updateAndPersist]
  )

  const clearConversationQueue = useCallback(
    (conversationId: string) => {
      updateAndPersist(prev => removeConversationQueue(prev, conversationId))
    },
    [updateAndPersist]
  )

  return {
    clearConversationQueue,
    enqueueTask,
    getQueue,
    queues,
    removeTask,
    reorderTasks,
  }
}
