import type { PlaygroundTask } from '../components/ChatComponentTypes'

export type PlaygroundTaskQueues = Record<string, PlaygroundTask[]>

export interface PlaygroundQueueLimits {
  maxConversations?: number
  maxTasksPerConversation?: number
}

export const DEFAULT_MAX_QUEUE_CONVERSATIONS = 20
export const DEFAULT_MAX_TASKS_PER_CONVERSATION = 20

const positiveLimit = (value: number | undefined, fallback: number) => {
  if (!Number.isFinite(value) || value === undefined || value < 1) {
    return fallback
  }
  return Math.floor(value)
}

const taskCreatedAt = (task: PlaygroundTask) => (
  Number.isFinite(task.createdAt) ? task.createdAt : 0
)

const newestTaskCreatedAt = (tasks: PlaygroundTask[]) => (
  tasks.reduce((max, task) => Math.max(max, taskCreatedAt(task)), 0)
)

const isPlaygroundTask = (value: unknown, conversationId: string): value is PlaygroundTask => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const task = value as Partial<PlaygroundTask>
  return (
    typeof task.id === 'string' &&
    task.id.length > 0 &&
    task.conversationId === conversationId &&
    typeof task.prompt === 'string' &&
    typeof task.createdAt === 'number' &&
    Boolean(task.requestOptions) &&
    typeof task.requestOptions === 'object'
  )
}

export const prunePlaygroundQueues = (
  queues: PlaygroundTaskQueues,
  limits: PlaygroundQueueLimits = {}
): PlaygroundTaskQueues => {
  const maxConversations = positiveLimit(limits.maxConversations, DEFAULT_MAX_QUEUE_CONVERSATIONS)
  const maxTasksPerConversation = positiveLimit(
    limits.maxTasksPerConversation,
    DEFAULT_MAX_TASKS_PER_CONVERSATION
  )

  return Object.fromEntries(
    Object.entries(queues)
      .map(([conversationId, tasks]) => [
        conversationId,
        tasks.slice(-maxTasksPerConversation),
      ] as const)
      .filter(([, tasks]) => tasks.length > 0)
      .sort(([, left], [, right]) => newestTaskCreatedAt(right) - newestTaskCreatedAt(left))
      .slice(0, maxConversations)
  )
}

export const normalizePlaygroundQueues = (
  value: unknown,
  limits: PlaygroundQueueLimits = {}
): PlaygroundTaskQueues => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }

  const queues = Object.entries(value).reduce<PlaygroundTaskQueues>((acc, [conversationId, tasks]) => {
    if (!Array.isArray(tasks)) {
      return acc
    }

    const validTasks = tasks.filter(task => isPlaygroundTask(task, conversationId))
    if (validTasks.length > 0) {
      acc[conversationId] = validTasks
    }

    return acc
  }, {})

  return prunePlaygroundQueues(queues, limits)
}
