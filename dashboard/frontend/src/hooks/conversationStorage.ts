export interface StoredConversation<T> {
  id: string
  createdAt: number
  updatedAt: number
  payload: T
}

export interface ConversationStorageLimits {
  maxConversations?: number
}

export const DEFAULT_MAX_CONVERSATIONS = 20

const positiveLimit = (value: number | undefined, fallback: number) => {
  if (!Number.isFinite(value) || value === undefined || value < 1) {
    return fallback
  }
  return Math.floor(value)
}

const isRecord = (value: unknown): value is Record<string, unknown> => (
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)
)

const hasPayload = (value: Record<string, unknown>) => (
  Object.prototype.hasOwnProperty.call(value, 'payload')
)

const acceptsAnyPayload = <T>(_payload: unknown): _payload is T => true

const isStoredConversation = <T>(
  value: unknown,
  isValidPayload: (payload: unknown) => payload is T
): value is StoredConversation<T> => {
  if (!isRecord(value) || !hasPayload(value)) {
    return false
  }

  return (
    typeof value.id === 'string' &&
    value.id.trim().length > 0 &&
    typeof value.createdAt === 'number' &&
    Number.isFinite(value.createdAt) &&
    typeof value.updatedAt === 'number' &&
    Number.isFinite(value.updatedAt) &&
    isValidPayload(value.payload)
  )
}

export const pruneStoredConversations = <T>(
  conversations: StoredConversation<T>[],
  limits: ConversationStorageLimits = {}
): StoredConversation<T>[] => {
  const maxConversations = positiveLimit(limits.maxConversations, DEFAULT_MAX_CONVERSATIONS)
  const seen = new Set<string>()

  return [...conversations]
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .filter(conversation => {
      const id = conversation.id.trim()
      if (id.length === 0 || seen.has(id)) {
        return false
      }

      seen.add(id)
      return true
    })
    .slice(0, maxConversations)
}

export const normalizeStoredConversations = <T = unknown>(
  value: unknown,
  limits: ConversationStorageLimits = {},
  isValidPayload: (payload: unknown) => payload is T = acceptsAnyPayload
): StoredConversation<T>[] => {
  if (!Array.isArray(value)) {
    return []
  }

  const conversations = value.reduce<StoredConversation<T>[]>((acc, item) => {
    if (!isStoredConversation(item, isValidPayload)) {
      return acc
    }

    const id = item.id.trim()
    acc.push({
      ...item,
      id,
    })
    return acc
  }, [])

  return pruneStoredConversations(conversations, limits)
}
