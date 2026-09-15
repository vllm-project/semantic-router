export interface StoredConversation<T> {
  id: string
  title?: string
  createdAt: number
  updatedAt: number
  payload: T
}

export interface ConversationStorageLimits {
  maxConversations?: number
}

export interface ConversationPersistenceOptions<T> {
  maxBytes?: number
  preparePayload?: (payload: T) => T | null
}

export const DEFAULT_MAX_CONVERSATIONS = 20
export const DEFAULT_MAX_CONVERSATION_STORAGE_BYTES = 2 * 1024 * 1024

const positiveLimit = (value: number | undefined, fallback: number) => {
  if (!Number.isFinite(value) || value === undefined || value < 1) {
    return fallback
  }
  return Math.floor(value)
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)

const hasPayload = (value: Record<string, unknown>) =>
  Object.prototype.hasOwnProperty.call(value, 'payload')

const acceptsAnyPayload = <T>(_payload: unknown): _payload is T => true

const isStoredConversation = <T>(
  value: unknown,
  isValidPayload: (payload: unknown) => payload is T,
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
  limits: ConversationStorageLimits = {},
): StoredConversation<T>[] => {
  const maxConversations = positiveLimit(limits.maxConversations, DEFAULT_MAX_CONVERSATIONS)
  const seen = new Set<string>()

  return [...conversations]
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .filter((conversation) => {
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
  isValidPayload: (payload: unknown) => payload is T = acceptsAnyPayload,
): StoredConversation<T>[] => {
  if (!Array.isArray(value)) {
    return []
  }

  const conversations = value.reduce<StoredConversation<T>[]>((acc, item) => {
    if (!isStoredConversation(item, isValidPayload)) {
      return acc
    }

    const id = item.id.trim()
    const restored: StoredConversation<T> = {
      ...item,
      id,
    }
    if (typeof item.title === 'string') {
      const title = item.title.trim().slice(0, 80)
      if (title) restored.title = title
      else delete restored.title
    } else {
      delete restored.title
    }
    acc.push(restored)
    return acc
  }, [])

  return pruneStoredConversations(conversations, limits)
}

export const prepareStoredConversationsForPersistence = <T>(
  conversations: StoredConversation<T>[],
  options: ConversationPersistenceOptions<T> = {},
): StoredConversation<T>[] => {
  const maxBytes = positiveLimit(options.maxBytes, DEFAULT_MAX_CONVERSATION_STORAGE_BYTES)
  const result: StoredConversation<T>[] = []
  for (const conversation of conversations) {
    const payload = options.preparePayload
      ? options.preparePayload(conversation.payload)
      : conversation.payload
    if (payload === null) continue
    const candidate = [...result, { ...conversation, payload }]
    if (new TextEncoder().encode(JSON.stringify(candidate)).byteLength <= maxBytes) {
      result.push({ ...conversation, payload })
    }
  }
  return result
}
