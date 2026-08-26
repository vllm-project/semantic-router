import { useCallback, useEffect, useRef, useState } from 'react'

import {
  normalizeStoredConversations,
  prepareStoredConversationsForPersistence,
  pruneStoredConversations,
  type StoredConversation,
} from './conversationStorage'
import {
  streamOpenAIChatCompletion,
  type OpenAIChatContent,
  type OpenAIChatMessage,
} from '../utils/openAIChatCompletions'
import {
  applyPlaygroundInferenceDelta,
  assertPlaygroundAssistantText,
  completePlaygroundInferenceMessage,
  type PlaygroundInferenceMessage,
  type PlaygroundInferenceMetadata,
} from '../utils/playgroundInferenceMessages'

const STORAGE_KEY = 'vsr:playground:openai-chat:v2'
const MAX_STORED_SESSIONS = 20
const MAX_STORED_MESSAGES = 100

interface SendOpenAIPlaygroundTurn {
  content: OpenAIChatContent
  model: string
  sessionId: string
  tools?: unknown[]
}

interface CreateOpenAIPlaygroundSession {
  keyId: string
  model: string
  title: string
}

export interface OpenAIPlaygroundSession {
  createdAt: string
  id: string
  keyId: string
  mode: 'chat'
  model: string
  title: string
  updatedAt: string
}

interface StoredOpenAIPlaygroundConversation {
  keyId: string
  messages: PlaygroundInferenceMessage[]
  model: string
  title: string
}

interface OpenAIPlaygroundState {
  messagesBySession: Record<string, PlaygroundInferenceMessage[]>
  sessions: OpenAIPlaygroundSession[]
}

interface UseOpenAIPlaygroundRuntimeOptions {
  endpoint: string
  getAccessToken: () => Promise<string>
  storageScope: string
}

function generatedId(prefix: string): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function contentLabel(content: OpenAIChatContent): string {
  if (typeof content === 'string') return content
  const text = content
    .filter((part): part is Extract<(typeof content)[number], { type: 'text' }> =>
      Boolean(part.type === 'text'),
    )
    .map((part) => part.text)
    .join('\n\n')
  const images = content.filter((part) => part.type === 'image_url').length
  if (!images) return text
  const label = `${images} image${images === 1 ? '' : 's'}`
  return text ? `${text}\n\n${label}` : label
}

function isPersistedMessage(value: unknown): value is PlaygroundInferenceMessage {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const message = value as Partial<PlaygroundInferenceMessage>
  return (
    typeof message.id === 'string' &&
    (message.role === 'user' || message.role === 'assistant') &&
    typeof message.content === 'string' &&
    typeof message.createdAt === 'string' &&
    message.metadata === undefined &&
    message.requestContent === undefined &&
    (message.status === 'complete' ||
      message.status === 'streaming' ||
      message.status === 'cancelled' ||
      message.status === 'failed')
  )
}

function isPersistedConversation(value: unknown): value is StoredOpenAIPlaygroundConversation {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const conversation = value as Partial<StoredOpenAIPlaygroundConversation>
  return (
    typeof conversation.keyId === 'string' &&
    Boolean(conversation.keyId) &&
    typeof conversation.model === 'string' &&
    Boolean(conversation.model) &&
    typeof conversation.title === 'string' &&
    Boolean(conversation.title.trim()) &&
    Array.isArray(conversation.messages) &&
    conversation.messages.every(isPersistedMessage)
  )
}

function transcriptStorageKey(scope: string): string {
  return `${STORAGE_KEY}:${encodeURIComponent(scope || 'unscoped')}`
}

function loadState(storageKey: string): OpenAIPlaygroundState {
  if (typeof window === 'undefined') return { messagesBySession: {}, sessions: [] }
  try {
    const parsed = JSON.parse(window.localStorage.getItem(storageKey) ?? '[]') as unknown
    const conversations = normalizeStoredConversations<StoredOpenAIPlaygroundConversation>(
      parsed,
      { maxConversations: MAX_STORED_SESSIONS },
      isPersistedConversation,
    )
    return {
      messagesBySession: Object.fromEntries(
        conversations.map((conversation) => [
          conversation.id,
          conversation.payload.messages
            .filter((message) => message.status !== 'streaming')
            .slice(-MAX_STORED_MESSAGES),
        ]),
      ),
      sessions: conversations.map((conversation) => ({
        id: conversation.id,
        keyId: conversation.payload.keyId,
        mode: 'chat',
        model: conversation.payload.model,
        title: conversation.payload.title,
        createdAt: new Date(conversation.createdAt).toISOString(),
        updatedAt: new Date(conversation.updatedAt).toISOString(),
      })),
    }
  } catch {
    return { messagesBySession: {}, sessions: [] }
  }
}

function safePersistentMessage(message: PlaygroundInferenceMessage): PlaygroundInferenceMessage {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: message.createdAt,
    status: message.status,
  }
}

function requestHistory(messages: readonly PlaygroundInferenceMessage[]): OpenAIChatMessage[] {
  return messages.flatMap((message) => {
    if (message.status === 'failed' || message.status === 'cancelled') return []
    if (message.role === 'assistant' && !message.content) return []
    return [
      {
        role: message.role,
        content: message.requestContent ?? message.content,
      } satisfies OpenAIChatMessage,
    ]
  })
}

function requestId(headers: Record<string, string>, responseId?: string): string | undefined {
  return (
    headers['x-request-id'] || headers['request-id'] || headers['openai-request-id'] || responseId
  )
}

export function useOpenAIPlaygroundRuntime({
  endpoint,
  getAccessToken,
  storageScope,
}: UseOpenAIPlaygroundRuntimeOptions) {
  const storageKey = transcriptStorageKey(storageScope)
  const [state, setState] = useState<OpenAIPlaygroundState>(() => loadState(storageKey))
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [runningSessionId, setRunningSessionId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const stateRef = useRef(state)
  const loadedStorageKeyRef = useRef(storageKey)
  const activeRequestRef = useRef<{ controller: AbortController; sessionId: string } | null>(null)

  const updateMessages = useCallback(
    (
      sessionId: string,
      update: (messages: PlaygroundInferenceMessage[]) => PlaygroundInferenceMessage[],
    ) => {
      setState((current) => {
        const updatedAt = new Date().toISOString()
        const next = {
          messagesBySession: {
            ...current.messagesBySession,
            [sessionId]: update(current.messagesBySession[sessionId] ?? []),
          },
          sessions: current.sessions.map((session) =>
            session.id === sessionId ? { ...session, updatedAt } : session,
          ),
        }
        stateRef.current = next
        return next
      })
    },
    [],
  )

  useEffect(() => {
    stateRef.current = state
    if (typeof window === 'undefined') return
    if (loadedStorageKeyRef.current !== storageKey) return
    const timer = window.setTimeout(() => {
      const stored: StoredConversation<StoredOpenAIPlaygroundConversation>[] = state.sessions.map(
        (session) => ({
          id: session.id,
          createdAt: Date.parse(session.createdAt),
          updatedAt: Date.parse(session.updatedAt),
          payload: {
            keyId: session.keyId,
            model: session.model,
            title: session.title,
            messages: state.messagesBySession[session.id] ?? [],
          },
        }),
      )
      const prepared = prepareStoredConversationsForPersistence(
        pruneStoredConversations(stored, { maxConversations: MAX_STORED_SESSIONS }),
        {
          preparePayload: (conversation) => ({
            ...conversation,
            messages: conversation.messages
              .filter((message) => message.status !== 'streaming')
              .slice(-MAX_STORED_MESSAGES)
              .map(safePersistentMessage),
          }),
        },
      )
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(prepared))
      } catch {
        // An in-memory transcript remains usable when browser storage is full or disabled.
      }
    }, 350)
    return () => window.clearTimeout(timer)
  }, [state, storageKey])

  useEffect(() => {
    activeRequestRef.current?.controller.abort()
    const next = loadState(storageKey)
    loadedStorageKeyRef.current = storageKey
    stateRef.current = next
    setState(next)
    setActiveSessionId(null)
    setRunningSessionId(null)
    setError(null)
  }, [storageKey])

  useEffect(() => () => activeRequestRef.current?.controller.abort(), [])

  const createSession = useCallback(
    ({ keyId, model, title }: CreateOpenAIPlaygroundSession): OpenAIPlaygroundSession => {
      const now = new Date().toISOString()
      const session: OpenAIPlaygroundSession = {
        id: generatedId('chat'),
        keyId,
        mode: 'chat',
        model,
        title: title.trim() || 'New conversation',
        createdAt: now,
        updatedAt: now,
      }
      setState((current) => {
        const next = {
          messagesBySession: { ...current.messagesBySession, [session.id]: [] },
          sessions: [session, ...current.sessions.filter((item) => item.id !== session.id)],
        }
        stateRef.current = next
        return next
      })
      setActiveSessionId(session.id)
      return session
    },
    [],
  )

  const send = useCallback(
    async ({ content, model, sessionId, tools = [] }: SendOpenAIPlaygroundTurn) => {
      if (activeRequestRef.current) throw new Error('Wait for the current response to finish.')
      const controller = new AbortController()
      activeRequestRef.current = { controller, sessionId }
      setRunningSessionId(sessionId)
      setError(null)

      const startedAt = performance.now()
      let firstTokenAt: number | undefined
      const userMessage: PlaygroundInferenceMessage = {
        id: generatedId('user'),
        role: 'user',
        content: contentLabel(content),
        requestContent: content,
        createdAt: new Date().toISOString(),
        status: 'complete',
      }
      const assistantId = generatedId('assistant')
      const assistantMessage: PlaygroundInferenceMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        createdAt: new Date().toISOString(),
        status: 'streaming',
      }
      const existing = stateRef.current.messagesBySession[sessionId] ?? []
      const history = requestHistory(existing)
      updateMessages(sessionId, (messages) => [...messages, userMessage, assistantMessage])

      let pendingDelta = ''
      let receivedText = ''
      let frame: number | null = null
      const flushDelta = () => {
        if (!pendingDelta) return
        const delta = pendingDelta
        pendingDelta = ''
        updateMessages(sessionId, (messages) =>
          applyPlaygroundInferenceDelta(messages, assistantId, delta),
        )
      }
      const scheduleDelta = (delta: string) => {
        if (firstTokenAt === undefined) firstTokenAt = performance.now()
        receivedText += delta
        pendingDelta += delta
        if (frame !== null) return
        frame = window.requestAnimationFrame(() => {
          frame = null
          flushDelta()
        })
      }

      try {
        const result = await streamOpenAIChatCompletion({
          accessToken: await getAccessToken(),
          endpoint,
          messages: [...history, { role: 'user', content }],
          model,
          onDelta: scheduleDelta,
          sessionId,
          signal: controller.signal,
          tools,
        })
        if (frame !== null) window.cancelAnimationFrame(frame)
        frame = null
        flushDelta()
        assertPlaygroundAssistantText(receivedText)
        const completedAt = performance.now()
        const responseRequestId = requestId(result.headers, result.responseId)
        const metadata: PlaygroundInferenceMetadata = {
          headers: result.headers,
          latencyMilliseconds: Math.max(0, Math.round(completedAt - startedAt)),
          ...(firstTokenAt !== undefined
            ? { ttftMilliseconds: Math.max(0, Math.round(firstTokenAt - startedAt)) }
            : {}),
          ...(result.finishReason ? { finishReason: result.finishReason } : {}),
          ...(result.model ? { model: result.model } : {}),
          ...(result.responseId ? { responseId: result.responseId } : {}),
          ...(responseRequestId ? { requestId: responseRequestId } : {}),
          ...(result.usage ? { usage: result.usage } : {}),
        }
        updateMessages(sessionId, (messages) =>
          completePlaygroundInferenceMessage(messages, assistantId, metadata),
        )
      } catch (cause) {
        if (frame !== null) window.cancelAnimationFrame(frame)
        frame = null
        flushDelta()
        if (controller.signal.aborted) {
          updateMessages(sessionId, (messages) =>
            messages.map((message) =>
              message.id === assistantId ? { ...message, status: 'cancelled' } : message,
            ),
          )
          return
        }
        const message = cause instanceof Error ? cause.message : 'The message could not be sent.'
        setError(message)
        updateMessages(sessionId, (messages) =>
          messages.map((item) => (item.id === assistantId ? { ...item, status: 'failed' } : item)),
        )
        throw cause
      } finally {
        if (activeRequestRef.current?.controller === controller) activeRequestRef.current = null
        setRunningSessionId((current) => (current === sessionId ? null : current))
      }
    },
    [endpoint, getAccessToken, updateMessages],
  )

  const cancel = useCallback(() => activeRequestRef.current?.controller.abort(), [])
  const clearError = useCallback(() => setError(null), [])
  const deleteSession = useCallback((sessionId: string) => {
    setState((current) => {
      if (!current.sessions.some((session) => session.id === sessionId)) return current
      const messagesBySession = { ...current.messagesBySession }
      delete messagesBySession[sessionId]
      const next = {
        messagesBySession,
        sessions: current.sessions.filter((session) => session.id !== sessionId),
      }
      stateRef.current = next
      return next
    })
    setActiveSessionId((current) => (current === sessionId ? null : current))
  }, [])
  const selectSession = useCallback((sessionId: string | null) => {
    if (sessionId && !stateRef.current.sessions.some((session) => session.id === sessionId)) return
    setActiveSessionId(sessionId)
    setError(null)
  }, [])

  const activeSession = state.sessions.find((session) => session.id === activeSessionId) ?? null

  return {
    activeSession,
    activeSessionId,
    cancel,
    clearError,
    createSession,
    deleteSession,
    error,
    messagesForSession: (sessionId: string | null) =>
      sessionId ? (state.messagesBySession[sessionId] ?? []) : [],
    running: runningSessionId !== null,
    runningSessionId,
    selectSession,
    send,
    sessions: state.sessions,
  }
}
