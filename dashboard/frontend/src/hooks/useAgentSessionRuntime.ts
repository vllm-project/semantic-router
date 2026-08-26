import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  isAgentLiveModelStepEvent,
  latestAgentSequence,
  mergeAgentEvents,
  parseAgentEventStream,
  reconcileAgentLiveModelSteps,
} from '../utils/agentEventStream'
import { activeAgentTurnId, agentTurnIsTerminal } from '../utils/agentEventProjection'
import { agentManagementApi, type AgentManagementApi } from '../utils/agentManagementApi'
import { shouldStreamAgentSessionEvents } from '../utils/agentSessionEventPolicy'
import { ManagementApiError } from '../utils/managementApiContract'
import type {
  AgentApprovalRequestPayload,
  AgentContentBlock,
  AgentEvent,
  AgentLiveModelStepEvent,
  AgentSession,
  AgentSessionInput,
} from '../generated/managementApiContract'

export type AgentStreamStatus = 'idle' | 'connecting' | 'live' | 'reconnecting' | 'error'

interface UseAgentSessionRuntimeOptions {
  api?: AgentManagementApi
  builderEventsOnly?: boolean
  enabled?: boolean
  search?: string
}

function describeError(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback
}

function shouldReconnect(error: unknown): boolean {
  if (!(error instanceof ManagementApiError)) return error instanceof TypeError
  return error.status === 408 || error.status === 429 || error.status >= 500
}

class AgentEventGapError extends Error {
  constructor(expected: number, received: number) {
    super(`Agent event sequence skipped from ${expected - 1} to ${received}.`)
    this.name = 'AgentEventGapError'
  }
}

function waitForRetry(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }
    const onAbort = () => {
      window.clearTimeout(timer)
      reject(new DOMException('Aborted', 'AbortError'))
    }
    const timer = window.setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, milliseconds)
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

function sortSessions(sessions: readonly AgentSession[]): AgentSession[] {
  return [...sessions].sort(
    (left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt),
  )
}

export function useAgentSessionRuntime({
  api = agentManagementApi,
  builderEventsOnly = false,
  enabled = true,
  search = '',
}: UseAgentSessionRuntimeOptions = {}) {
  const [sessions, setSessions] = useState<AgentSession[]>([])
  const [sessionsCursor, setSessionsCursor] = useState<string | undefined>()
  const [sessionsHaveMore, setSessionsHaveMore] = useState(false)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [activeSessionSnapshot, setActiveSessionSnapshot] = useState<AgentSession | null>(null)
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [liveModelSteps, setLiveModelSteps] = useState<AgentLiveModelStepEvent[]>([])
  const [eventsCursor, setEventsCursor] = useState<string | undefined>()
  const [eventsHaveMore, setEventsHaveMore] = useState(false)
  const [streamStatus, setStreamStatus] = useState<AgentStreamStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [loadingSessions, setLoadingSessions] = useState(enabled)
  const [loadingEvents, setLoadingEvents] = useState(false)
  const [mutating, setMutating] = useState(false)
  const [submittedTurnId, setSubmittedTurnId] = useState<string | null>(null)
  const [streamAttempt, setStreamAttempt] = useState(0)
  const eventSequenceRef = useRef(0)
  const activeSessionIdRef = useRef<string | null>(null)
  const searchRef = useRef(search)
  const sessionsGenerationRef = useRef(0)
  const sessionsRequestRef = useRef<AbortController | null>(null)
  const eventsPageGenerationRef = useRef(0)
  const eventsPageRequestRef = useRef<AbortController | null>(null)
  const noticeTimerRef = useRef<number | null>(null)

  useEffect(
    () => () => {
      if (noticeTimerRef.current !== null) window.clearTimeout(noticeTimerRef.current)
    },
    [],
  )

  const showNotice = useCallback((notice: string) => {
    if (noticeTimerRef.current !== null) window.clearTimeout(noticeTimerRef.current)
    setError(notice)
    noticeTimerRef.current = window.setTimeout(() => {
      setError((current) => (current === notice ? null : current))
      noticeTimerRef.current = null
    }, 5_000)
  }, [])

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId
  }, [activeSessionId])

  useEffect(() => {
    searchRef.current = search
  }, [search])

  const activeSession =
    sessions.find((session) => session.id === activeSessionId) ??
    (activeSessionSnapshot?.id === activeSessionId ? activeSessionSnapshot : null)
  const shouldStreamActiveSession = shouldStreamAgentSessionEvents({
    activeSessionId,
    activeSessionMode: activeSession?.mode,
    builderEventsOnly,
  })

  const refreshSessions = useCallback(async () => {
    if (!enabled) return
    const generation = ++sessionsGenerationRef.current
    sessionsRequestRef.current?.abort()
    const controller = new AbortController()
    sessionsRequestRef.current = controller
    setLoadingSessions(true)
    try {
      const page = await api.listSessions(search, undefined, 50, controller.signal)
      if (controller.signal.aborted || generation !== sessionsGenerationRef.current) return
      setSessions(sortSessions(page.data))
      setSessionsCursor(page.page.nextCursor)
      setSessionsHaveMore(page.page.hasMore)
      // Searching recent sessions must never navigate away from the session the
      // user is reading (or from an intentional new-chat canvas).
      const active = activeSessionIdRef.current
      if (active) {
        const refreshed = page.data.find((session) => session.id === active)
        if (refreshed) setActiveSessionSnapshot(refreshed)
      }
      setError(null)
    } catch (cause) {
      if (controller.signal.aborted || generation !== sessionsGenerationRef.current) return
      setError(describeError(cause, 'Sessions are unavailable.'))
    } finally {
      if (!controller.signal.aborted && generation === sessionsGenerationRef.current) {
        setLoadingSessions(false)
      }
      if (sessionsRequestRef.current === controller) sessionsRequestRef.current = null
    }
  }, [api, enabled, search])

  useEffect(() => {
    if (!enabled) {
      setSessions([])
      setActiveSessionId(null)
      setLoadingSessions(false)
      return
    }
    const timer = window.setTimeout(() => void refreshSessions(), 180)
    return () => {
      window.clearTimeout(timer)
      sessionsRequestRef.current?.abort()
      sessionsGenerationRef.current += 1
    }
  }, [enabled, refreshSessions])

  const loadMoreSessions = useCallback(async () => {
    if (!enabled || !sessionsHaveMore || !sessionsCursor || loadingSessions) return
    const generation = ++sessionsGenerationRef.current
    sessionsRequestRef.current?.abort()
    const controller = new AbortController()
    sessionsRequestRef.current = controller
    setLoadingSessions(true)
    try {
      const page = await api.listSessions(search, sessionsCursor, 50, controller.signal)
      if (controller.signal.aborted || generation !== sessionsGenerationRef.current) return
      setSessions((current) => {
        const byId = new Map(current.map((session) => [session.id, session]))
        page.data.forEach((session) => byId.set(session.id, session))
        return sortSessions([...byId.values()])
      })
      setSessionsCursor(page.page.nextCursor)
      setSessionsHaveMore(page.page.hasMore)
    } catch (cause) {
      if (controller.signal.aborted || generation !== sessionsGenerationRef.current) return
      setError(describeError(cause, 'More sessions could not be loaded.'))
    } finally {
      if (!controller.signal.aborted && generation === sessionsGenerationRef.current) {
        setLoadingSessions(false)
      }
      if (sessionsRequestRef.current === controller) sessionsRequestRef.current = null
    }
  }, [api, enabled, loadingSessions, search, sessionsCursor, sessionsHaveMore])

  useEffect(() => {
    if (!enabled || !activeSessionId || !shouldStreamActiveSession) {
      eventsPageRequestRef.current?.abort()
      eventsPageGenerationRef.current += 1
      setEvents([])
      setLiveModelSteps([])
      setEventsCursor(undefined)
      setEventsHaveMore(false)
      setLoadingEvents(false)
      setSubmittedTurnId(null)
      setStreamStatus('idle')
      return
    }

    const controller = new AbortController()
    eventsPageRequestRef.current?.abort()
    eventsPageGenerationRef.current += 1
    let retryDelay = 500
    let streamRetryDelay: number | undefined
    eventSequenceRef.current = 0
    setEvents([])
    setLiveModelSteps([])
    setSubmittedTurnId(null)
    setLoadingEvents(true)
    setStreamStatus('connecting')

    const loadLatestHistory = async (notice?: string) => {
      const page = await api.listLatestEvents(activeSessionId, 100, controller.signal)
      if (controller.signal.aborted) return false
      const latest = mergeAgentEvents([], page.data)
      eventSequenceRef.current = latestAgentSequence(latest)
      setEvents(latest)
      setLiveModelSteps([])
      setEventsCursor(page.page.nextCursor)
      setEventsHaveMore(page.page.hasMore)
      if (notice) showNotice(notice)
      return true
    }

    const refreshSessionMetadata = async () => {
      try {
        const detail = await api.getSession(activeSessionId, controller.signal)
        if (controller.signal.aborted) return
        const refreshed = detail.data
        setActiveSessionSnapshot(refreshed)
        setSessions((current) => {
          const exists = current.some((session) => session.id === refreshed.id)
          if (!exists && searchRef.current.trim()) return current
          return sortSessions([
            refreshed,
            ...current.filter((session) => session.id !== refreshed.id),
          ])
        })
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(describeError(cause, 'Conversation details could not be refreshed.'))
        }
      }
    }

    const run = async () => {
      try {
        if (!(await loadLatestHistory())) return
        setLoadingEvents(false)
      } catch (cause) {
        if (controller.signal.aborted) return
        setLoadingEvents(false)
        setStreamStatus('error')
        setError(describeError(cause, 'Conversation history is unavailable.'))
        return
      }

      while (!controller.signal.aborted) {
        try {
          setStreamStatus(eventSequenceRef.current > 0 ? 'reconnecting' : 'connecting')
          const response = await api.openEventStream(
            activeSessionId,
            eventSequenceRef.current,
            controller.signal,
          )
          setStreamStatus('live')
          retryDelay = 500
          const stream = response.body
          if (!stream) throw new Error('Router returned an empty Agent event stream.')
          for await (const event of parseAgentEventStream(stream, {
            onRetry: (milliseconds) => {
              streamRetryDelay = milliseconds
            },
          })) {
            if (controller.signal.aborted) return
            if (event.sessionId !== activeSessionId) {
              throw new Error('Router mixed events from another Agent session.')
            }
            if (isAgentLiveModelStepEvent(event)) {
              setLiveModelSteps((current) => reconcileAgentLiveModelSteps(current, event))
              continue
            }
            if (eventSequenceRef.current > 0 && event.sequence > eventSequenceRef.current + 1) {
              throw new AgentEventGapError(eventSequenceRef.current + 1, event.sequence)
            }
            eventSequenceRef.current = Math.max(eventSequenceRef.current, event.sequence)
            setEvents((current) => mergeAgentEvents(current, [event]))
            setLiveModelSteps((current) => reconcileAgentLiveModelSteps(current, event))
            if (event.type === 'terminal') {
              if (event.turnId) {
                setSubmittedTurnId((current) => (current === event.turnId ? null : current))
              }
              await refreshSessionMetadata()
            }
          }
          if (!controller.signal.aborted) throw new TypeError('Agent event stream disconnected.')
        } catch (cause) {
          if (controller.signal.aborted) return
          // Provisional frames are intentionally not resumable. A reconnect
          // starts from the durable sequence and waits for fresh previews.
          setLiveModelSteps([])
          if (
            cause instanceof AgentEventGapError ||
            (cause instanceof ManagementApiError && cause.status === 410)
          ) {
            try {
              const recovered = await loadLatestHistory(
                cause instanceof AgentEventGapError
                  ? 'Conversation updates were resynced.'
                  : 'Earlier events were archived. The latest conversation is loaded.',
              )
              if (!recovered) return
              setStreamStatus('reconnecting')
              continue
            } catch (recoveryCause) {
              setStreamStatus('error')
              setError(
                describeError(recoveryCause, 'The latest durable checkpoint is unavailable.'),
              )
              return
            }
          }
          if (!shouldReconnect(cause)) {
            setStreamStatus('error')
            setError(describeError(cause, 'Conversation updates are unavailable.'))
            return
          }
          setStreamStatus('reconnecting')
          try {
            const serverDelay =
              cause instanceof ManagementApiError ? cause.retryAfterMilliseconds : undefined
            const jitter = Math.round(retryDelay * (0.85 + Math.random() * 0.3))
            await waitForRetry(
              Math.max(serverDelay ?? 0, streamRetryDelay ?? 0, jitter),
              controller.signal,
            )
          } catch {
            return
          }
          retryDelay = Math.min(Math.round(retryDelay * 1.8), 8_000)
        }
      }
    }

    void run()
    return () => {
      controller.abort()
      eventsPageRequestRef.current?.abort()
      eventsPageGenerationRef.current += 1
    }
  }, [activeSessionId, api, enabled, shouldStreamActiveSession, showNotice, streamAttempt])

  const loadMoreEvents = useCallback(async () => {
    if (
      !activeSessionId ||
      !shouldStreamActiveSession ||
      !eventsHaveMore ||
      !eventsCursor ||
      loadingEvents
    )
      return
    const sessionId = activeSessionId
    const generation = ++eventsPageGenerationRef.current
    eventsPageRequestRef.current?.abort()
    const controller = new AbortController()
    eventsPageRequestRef.current = controller
    setLoadingEvents(true)
    try {
      const page = await api.listEarlierEvents(sessionId, eventsCursor, 100, controller.signal)
      if (
        controller.signal.aborted ||
        generation !== eventsPageGenerationRef.current ||
        activeSessionIdRef.current !== sessionId
      ) {
        return
      }
      setEvents((current) => mergeAgentEvents(current, page.data))
      setEventsCursor(page.page.nextCursor)
      setEventsHaveMore(page.page.hasMore)
    } catch (cause) {
      if (controller.signal.aborted || generation !== eventsPageGenerationRef.current) return
      if (cause instanceof ManagementApiError && cause.status === 410) {
        setEventsCursor(undefined)
        setEventsHaveMore(false)
        showNotice('Earlier conversation history has been archived.')
      } else {
        setError(describeError(cause, 'More conversation history could not be loaded.'))
      }
    } finally {
      if (!controller.signal.aborted && generation === eventsPageGenerationRef.current) {
        setLoadingEvents(false)
      }
      if (eventsPageRequestRef.current === controller) eventsPageRequestRef.current = null
    }
  }, [
    activeSessionId,
    api,
    eventsCursor,
    eventsHaveMore,
    loadingEvents,
    shouldStreamActiveSession,
    showNotice,
  ])

  const createSession = useCallback(
    async (input: AgentSessionInput): Promise<AgentSession> => {
      setMutating(true)
      try {
        const detail = await api.createSession(input)
        const session = detail.data
        setSessions((current) =>
          sortSessions([session, ...current.filter((item) => item.id !== session.id)]),
        )
        setActiveSessionSnapshot(session)
        setActiveSessionId(session.id)
        setError(null)
        return session
      } catch (cause) {
        const message = describeError(cause, 'The Agent session could not be created.')
        setError(message)
        throw cause
      } finally {
        setMutating(false)
      }
    },
    [api],
  )

  const deleteSession = useCallback(
    async (sessionId: string): Promise<void> => {
      setMutating(true)
      try {
        const detail = await api.getSession(sessionId)
        await api.deleteSession(sessionId, detail.etag)
        setSessions((current) => current.filter((session) => session.id !== sessionId))
        if (activeSessionIdRef.current === sessionId) {
          setActiveSessionId(null)
          setActiveSessionSnapshot(null)
        }
      } catch (cause) {
        setError(describeError(cause, 'The session could not be deleted.'))
        throw cause
      } finally {
        setMutating(false)
      }
    },
    [api],
  )

  const sendTurn = useCallback(
    async (content: AgentContentBlock[], sessionOverride?: string): Promise<string> => {
      const sessionId = sessionOverride ?? activeSessionIdRef.current
      if (!sessionId) throw new Error('Choose or create a session first.')
      setMutating(true)
      try {
        const receipt = await api.createTurn(sessionId, { input: { content } })
        setSubmittedTurnId(receipt.id)
        setError(null)
        return receipt.id
      } catch (cause) {
        setError(describeError(cause, 'The message could not be sent.'))
        throw cause
      } finally {
        setMutating(false)
      }
    },
    [api],
  )

  const eventActiveTurnId = useMemo(() => activeAgentTurnId(events), [events])
  const activeTurnId = eventActiveTurnId ?? submittedTurnId

  // A very short turn may publish its terminal event before the create-turn
  // response reaches the browser. Reconcile against durable events as well as
  // the live-stream callback so the composer cannot remain stuck in Stop mode.
  useEffect(() => {
    if (submittedTurnId && agentTurnIsTerminal(events, submittedTurnId)) {
      setSubmittedTurnId(null)
    }
  }, [events, submittedTurnId])

  const cancelTurn = useCallback(async () => {
    const sessionId = activeSessionIdRef.current
    if (!sessionId || !activeTurnId) return
    setMutating(true)
    try {
      await api.cancelTurn(sessionId, activeTurnId)
    } catch (cause) {
      setError(describeError(cause, 'The running turn could not be cancelled.'))
      throw cause
    } finally {
      setMutating(false)
    }
  }, [activeTurnId, api])

  const commitPublication = useCallback(
    async (approval: AgentApprovalRequestPayload): Promise<string> => {
      setMutating(true)
      try {
        const receipt = await api.commitPublication(
          approval.planId,
          approval.planDigest,
          approval.planEtag,
        )
        await api.waitForOperation(receipt.operationId)
        return receipt.operationId
      } catch (cause) {
        setError(describeError(cause, 'The reviewed Mixture-of-Models was not published.'))
        throw cause
      } finally {
        setMutating(false)
      }
    },
    [api],
  )

  const selectSession = useCallback(
    (sessionId: string | null) => {
      setActiveSessionId(sessionId)
      setActiveSessionSnapshot((current) => {
        if (!sessionId) return null
        return (
          sessions.find((session) => session.id === sessionId) ??
          (current?.id === sessionId ? current : null)
        )
      })
    },
    [sessions],
  )

  const recoverStream = useCallback(() => {
    setError(null)
    setStreamAttempt((current) => current + 1)
  }, [])

  return {
    activeSession,
    activeSessionId,
    activeTurnId,
    cancelTurn,
    commitPublication,
    createSession,
    deleteSession,
    error,
    events,
    eventsHaveMore,
    liveModelSteps,
    loadMoreEvents,
    loadMoreSessions,
    loadingEvents,
    loadingSessions,
    mutating,
    refreshSessions,
    recoverStream,
    selectSession,
    sendTurn,
    sessions,
    sessionsHaveMore,
    streamStatus,
  }
}
