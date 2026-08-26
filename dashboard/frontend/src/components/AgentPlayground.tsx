import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useAuth } from '../contexts/AuthContext'
import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import { useDelegatedInferenceSession } from '../hooks/useDelegatedInferenceSession'
import { useAgentSessionRuntime } from '../hooks/useAgentSessionRuntime'
import { useOpenAIPlaygroundRuntime } from '../hooks/useOpenAIPlaygroundRuntime'
import type { PlaygroundInvocation } from '../types/playgroundInvocation'
import {
  canAccessDashboardPath,
  canInvokeAgentTools,
  canPublishRouting,
  canReadAgent,
  canUseAgent,
  canUseBuilderAgent,
  canUseDelegatedInference,
} from '../utils/accessControl'
import { pendingApproval } from '../utils/agentEventProjection'
import type { AgentContentBlock } from '../generated/managementApiContract'
import { routingManagementApi } from '../utils/routingManagementApi'
import ConfirmDialog from './ConfirmDialog'
import InferenceKeySelector from './InferenceKeySelector'
import {
  buildPromptWithAttachments,
  buildPlaygroundUserContent,
  isPlaygroundImageAttachment,
  readPlaygroundAttachmentFile,
  validatePlaygroundAttachmentBudget,
  type PlaygroundAttachment,
} from './playgroundFileAttachments'
import { usePlaygroundRoutingModel } from './usePlaygroundRoutingModel'
import AgentComposer from './AgentComposer'
import type { PlaygroundQueuedTurn } from './AgentComposerQueue'
import AgentConversationSidebar, {
  type PlaygroundConversationListItem,
} from './AgentConversationSidebar'
import AgentPublicationReviewDialog from './AgentPublicationReviewDialog'
import AgentTimeline from './AgentTimeline'
import ProductIcon from './ProductIcon'
import { buildAgentSessionInput } from './agentPlaygroundSession'
import {
  agentSessionMode,
  playgroundModeForAgentSession,
  type PlaygroundMode,
} from './playgroundModes'
import styles from './AgentPlayground.module.css'

const MAX_QUEUED_TURNS = 8

function queuedTurnId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `queued-${crypto.randomUUID()}`
  }
  return `queued-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

interface AgentPlaygroundProps {
  endpoint: string
  fullscreen?: boolean
  invocation?: PlaygroundInvocation | null
  initialModel?: string | null
  onInvocationConsumed?: () => void
  onInitialModelConsumed?: () => void
}

function titleFromPrompt(prompt: string, mode: PlaygroundMode): string {
  const normalized = prompt.replace(/\s+/g, ' ').trim()
  if (!normalized) return mode === 'builder' ? 'New model path' : 'New conversation'
  return normalized.length > 62 ? `${normalized.slice(0, 59).trimEnd()}…` : normalized
}

function invocationPrompt(invocation: PlaygroundInvocation): string {
  const transcript = invocation.messages
    .map((message) => {
      const content =
        typeof message.content === 'string' ? message.content : JSON.stringify(message.content)
      return `${message.role}: ${content}`
    })
    .join('\n\n')
  const lead =
    invocation.intent === 'edit'
      ? 'Improve and validate this probe.'
      : 'Run and explain this probe.'
  return `${lead}\n\n${transcript}`.trim()
}

function StreamState({
  status,
}: {
  status: ReturnType<typeof useAgentSessionRuntime>['streamStatus']
}) {
  if (status === 'idle') return null
  const label = status === 'live' ? 'Live' : status === 'error' ? 'Offline' : 'Connecting'
  return (
    <span className={`${styles.streamState} ${status === 'live' ? styles.streamStateLive : ''}`}>
      <span aria-hidden="true" />
      {label}
    </span>
  )
}

export default function AgentPlayground({
  endpoint,
  fullscreen = false,
  invocation = null,
  initialModel = null,
  onInvocationConsumed,
  onInitialModelConsumed,
}: AgentPlaygroundProps) {
  const { user } = useAuth()
  const { selectedKey, setSelectedKeyId } = useInferenceRoutingAccess()
  const canUse = canUseDelegatedInference(user)
  const canReadAgentSessions = canReadAgent(user)
  const agentAvailable = canUseAgent(user) && canInvokeAgentTools(user)
  const builderAvailable = canUseBuilderAgent(user)
  const canPublish = canPublishRouting(user)
  const delegated = useDelegatedInferenceSession()
  const routing = usePlaygroundRoutingModel(
    endpoint,
    delegated.getAccessToken,
    canUse && delegated.status === 'ready',
  )
  const [search, setSearch] = useState('')
  const runtime = useAgentSessionRuntime({
    enabled: canReadAgentSessions,
    search,
  })
  const inference = useOpenAIPlaygroundRuntime({
    endpoint,
    getAccessToken: delegated.getAccessToken,
    storageScope: user?.id ?? '',
  })
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [draftMode, setDraftMode] = useState<PlaygroundMode>('chat')
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<PlaygroundAttachment[]>([])
  const [queuedTurns, setQueuedTurns] = useState<PlaygroundQueuedTurn[]>([])
  const [queuePaused, setQueuePaused] = useState(false)
  const [queueEpoch, setQueueEpoch] = useState(0)
  const [localError, setLocalError] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<PlaygroundConversationListItem | null>(null)
  const [reviewOpen, setReviewOpen] = useState(false)
  const [publishError, setPublishError] = useState<string | null>(null)
  const invocationHandled = useRef(false)
  const initialModelHandled = useRef(false)
  const queueDrainRef = useRef(false)

  const activeRouterSession = runtime.activeSession
  const activeBuilderSession = activeRouterSession?.mode === 'builder' ? activeRouterSession : null
  const activeAgentSession = activeRouterSession?.mode === 'chat' ? activeRouterSession : null
  const activeChatSession = inference.activeSession
  const activeMode: PlaygroundMode = activeBuilderSession
    ? 'builder'
    : activeAgentSession
      ? 'agent'
      : activeChatSession
        ? 'chat'
        : draftMode
  const effectiveTeamId = activeRouterSession?.effectiveTeamId ?? selectedKey?.contextTeamId ?? ''
  const activeSessionId = activeRouterSession?.id ?? activeChatSession?.id ?? null
  const displayName = user?.name.trim() || user?.email.split('@')[0] || 'there'
  const conversationSessions = useMemo<PlaygroundConversationListItem[]>(() => {
    const normalizedSearch = search.trim().toLocaleLowerCase()
    const browserSessions = inference.sessions
      .filter(
        (session) =>
          !normalizedSearch || session.title.toLocaleLowerCase().includes(normalizedSearch),
      )
      .map((session) => ({
        id: session.id,
        mode: 'chat' as const,
        source: 'browser' as const,
        title: session.title,
        updatedAt: session.updatedAt,
      }))
    const agentSessions = runtime.sessions.map((session) => ({
      id: session.id,
      mode: playgroundModeForAgentSession(session.mode),
      source: 'router' as const,
      title: session.title,
      updatedAt: session.updatedAt,
    }))
    return [...browserSessions, ...agentSessions].sort(
      (left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt),
    )
  }, [inference.sessions, runtime.sessions, search])
  const activeApproval = useMemo(
    () => (activeMode === 'builder' ? pendingApproval(runtime.events) : null),
    [activeMode, runtime.events],
  )
  const activeApprovalPlanId = activeApproval?.planId

  useEffect(() => {
    if (activeApprovalPlanId) {
      setPublishError(null)
      setReviewOpen(true)
    }
  }, [activeApprovalPlanId])

  useEffect(() => {
    if (!activeRouterSession) return
    inference.selectSession(null)
    setSelectedKeyId(activeRouterSession.keyId)
    routing.setModel(activeRouterSession.target.id)
    setDraftMode(playgroundModeForAgentSession(activeRouterSession.mode))
    setAttachments([])
    setInput('')
    setLocalError(null)
  }, [activeRouterSession?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!activeChatSession) return
    runtime.selectSession(null)
    setSelectedKeyId(activeChatSession.keyId)
    routing.setModel(activeChatSession.model)
    setDraftMode('chat')
    setAttachments([])
    setInput('')
    setLocalError(null)
  }, [activeChatSession?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!invocation || invocationHandled.current || routing.status !== 'ready') return
    invocationHandled.current = true
    runtime.selectSession(null)
    inference.selectSession(null)
    if (invocation.intent === 'edit' && builderAvailable) setDraftMode('builder')
    if (invocation.model && routing.models.some((model) => model.id === invocation.model)) {
      routing.setModel(invocation.model)
    }
    setInput(invocationPrompt(invocation))
    onInvocationConsumed?.()
  }, [builderAvailable, inference, invocation, onInvocationConsumed, routing, runtime])

  useEffect(() => {
    if (!initialModel || initialModelHandled.current || routing.status !== 'ready') return
    initialModelHandled.current = true
    runtime.selectSession(null)
    inference.selectSession(null)
    if (routing.models.some((model) => model.id === initialModel)) {
      routing.setModel(initialModel)
    } else {
      setLocalError('This model is not available to your account.')
    }
    onInitialModelConsumed?.()
  }, [inference, initialModel, onInitialModelConsumed, routing, runtime])

  const selectedTargetId =
    activeRouterSession?.target.id ?? activeChatSession?.model ?? routing.model
  const selectedOption = routing.models.find((model) => model.id === selectedTargetId)
  const running = activeMode !== 'chat' ? Boolean(runtime.activeTurnId) : inference.running
  const composerRunning = running || runtime.mutating
  const disabledReason = (() => {
    if (!canUse) return 'Playground access is not available.'
    if (delegated.status === 'loading' || routing.status === 'discovering') return 'Getting ready…'
    if (delegated.status === 'unavailable') return 'Create an API key to use Playground.'
    if (delegated.status === 'error') return 'API access is unavailable.'
    if (routing.status === 'error') return 'Models are unavailable.'
    if (activeMode === 'agent' && !agentAvailable) return 'Agent access is not available.'
    if (activeMode === 'builder' && !builderAvailable) return 'Builder access is not available.'
    if (activeChatSession && selectedKey?.keyId !== activeChatSession.keyId) {
      return 'The API key for this conversation is no longer available.'
    }
    if (activeRouterSession && selectedKey?.keyId !== activeRouterSession.keyId) {
      return 'The API key for this Agent session is no longer available.'
    }
    if (activeApproval && Date.parse(activeApproval.expiresAt) > Date.now()) {
      return 'Review this plan before continuing.'
    }
    if (!selectedOption) {
      return activeSessionId ? 'This model is no longer available.' : 'Choose a model.'
    }
    return undefined
  })()

  const startNew = useCallback(
    (mode: PlaygroundMode = 'chat') => {
      runtime.selectSession(null)
      inference.selectSession(null)
      setDraftMode(mode)
      setInput('')
      setAttachments([])
      setQueuedTurns([])
      setQueuePaused(false)
      setLocalError(null)
      inference.clearError()
      if (window.innerWidth < 960) setSidebarOpen(false)
    },
    [inference, runtime],
  )

  const handleModeChange = (mode: PlaygroundMode) => {
    if (
      activeSessionId ||
      (mode === 'agent' && !agentAvailable) ||
      (mode === 'builder' && !builderAvailable)
    )
      return
    setDraftMode(mode)
  }

  const handleAttach = async (files: FileList) => {
    setLocalError(null)
    const accepted: PlaygroundAttachment[] = []
    try {
      for (const file of Array.from(files)) {
        const attachment = await readPlaygroundAttachmentFile(file)
        const next = [...attachments, ...accepted, attachment]
        const budgetError = validatePlaygroundAttachmentBudget(next)
        if (budgetError) throw new Error(budgetError)
        accepted.push(attachment)
      }
      setAttachments((current) => [...current, ...accepted])
    } catch (cause) {
      setLocalError(cause instanceof Error ? cause.message : 'The attachment could not be read.')
    }
  }

  const submitTurn = useCallback(
    async (turn: Pick<PlaygroundQueuedTurn, 'input' | 'attachments'>) => {
      if (!selectedOption) throw new Error('Choose a model before sending a message.')
      const prompt = buildPromptWithAttachments(turn.input, turn.attachments)
      const content: AgentContentBlock[] = [
        ...(prompt ? [{ type: 'text' as const, text: prompt }] : []),
        ...turn.attachments.filter(isPlaygroundImageAttachment).map((attachment) => ({
          type: 'image_url' as const,
          url: attachment.content,
          detail: 'auto' as const,
        })),
      ]
      if (!content.length) return
      if (activeMode !== 'chat') {
        let sessionId = activeRouterSession?.id ?? null
        if (!sessionId) {
          if (!selectedKey) throw new Error('Choose an API key before starting a conversation.')
          const session = await runtime.createSession(
            buildAgentSessionInput({
              keyId: selectedKey.keyId,
              mode: agentSessionMode(activeMode),
              effectiveTeamId,
              model: selectedOption,
              title: titleFromPrompt(turn.input, activeMode),
            }),
          )
          sessionId = session.id
        }
        await runtime.sendTurn(content, sessionId)
      } else {
        let sessionId = activeChatSession?.id ?? null
        if (!sessionId) {
          if (!selectedKey) throw new Error('Choose an API key before starting a conversation.')
          sessionId = inference.createSession({
            keyId: selectedKey.keyId,
            model: selectedOption.id,
            title: titleFromPrompt(turn.input, 'chat'),
          }).id
        }
        await inference.send({
          content: buildPlaygroundUserContent(turn.input, turn.attachments),
          model: selectedOption.id,
          sessionId,
        })
      }
    },
    [
      activeChatSession?.id,
      activeMode,
      activeRouterSession?.id,
      effectiveTeamId,
      inference,
      runtime,
      selectedKey,
      selectedOption,
    ],
  )

  const handleSend = () => {
    if (disabledReason || composerRunning || !selectedOption) return
    const turn = { input, attachments }
    if (!turn.input.trim() && !turn.attachments.length) return
    setInput('')
    setAttachments([])
    setLocalError(null)
    void submitTurn(turn).catch((cause) => {
      setInput((current) => current || turn.input)
      setAttachments((current) => (current.length ? current : turn.attachments))
      setLocalError(cause instanceof Error ? cause.message : 'The message could not be sent.')
    })
  }

  const handleQueue = () => {
    if (!composerRunning || disabledReason) return
    if (!input.trim() && !attachments.length) return
    if (queuedTurns.length >= MAX_QUEUED_TURNS) {
      setLocalError(`Queue up to ${MAX_QUEUED_TURNS} messages at a time.`)
      return
    }
    setQueuedTurns((current) => [
      ...current,
      { id: queuedTurnId(), input, attachments: [...attachments] },
    ])
    setInput('')
    setAttachments([])
    setQueuePaused(false)
    setLocalError(null)
  }

  useEffect(() => {
    if (
      composerRunning ||
      queuePaused ||
      queueDrainRef.current ||
      !queuedTurns.length ||
      !activeSessionId ||
      disabledReason
    ) {
      return
    }
    const next = queuedTurns[0]
    queueDrainRef.current = true
    setQueuedTurns((current) => current.filter((turn) => turn.id !== next.id))
    setLocalError(null)
    void submitTurn(next)
      .catch((cause) => {
        setQueuedTurns((current) =>
          current.some((turn) => turn.id === next.id) ? current : [next, ...current],
        )
        setQueuePaused(true)
        setLocalError(
          cause instanceof Error ? cause.message : 'The queued message could not be sent.',
        )
      })
      .finally(() => {
        queueDrainRef.current = false
        setQueueEpoch((current) => current + 1)
      })
  }, [
    activeSessionId,
    composerRunning,
    disabledReason,
    queueEpoch,
    queuePaused,
    queuedTurns,
    submitTurn,
  ])

  const handlePublish = async () => {
    if (!activeApproval || !canPublish) return
    setPublishError(null)
    try {
      await runtime.commitPublication(activeApproval)
      const entrypointId = activeApproval.summary.entrypointId
      if (!entrypointId) {
        throw new Error('The publication did not identify its Mixture-of-Models.')
      }
      const entrypoint = await routingManagementApi.getEntrypoint(entrypointId)
      if (entrypoint.status !== 'active') {
        throw new Error('The Mixture-of-Models is not active yet.')
      }
      const expectedModelIds = [entrypoint.id, entrypoint.name, ...entrypoint.aliases]
      await routing.refresh({ expectedModelIds, timeoutMilliseconds: 12_000 })
      setReviewOpen(false)
    } catch (cause) {
      setPublishError(
        cause instanceof Error
          ? cause.message
          : 'The published model could not be confirmed from the Router.',
      )
    }
  }

  return (
    <section
      className={`${styles.root} ${fullscreen ? styles.fullscreen : ''}`}
      data-testid="agent-playground"
    >
      <AgentConversationSidebar
        activeSessionId={activeSessionId}
        busy={runtime.mutating || inference.running}
        loading={runtime.loadingSessions}
        open={sidebarOpen}
        search={search}
        sessions={conversationSessions}
        sessionsHaveMore={runtime.sessionsHaveMore}
        onDeleteRequest={setDeleteTarget}
        onLoadMore={() => void runtime.loadMoreSessions()}
        onNewChat={() => startNew('chat')}
        onSearchChange={setSearch}
        onSelect={(session) => {
          inference.clearError()
          setQueuedTurns([])
          setQueuePaused(false)
          if (session.source === 'browser') {
            runtime.selectSession(null)
            inference.selectSession(session.id)
          } else {
            inference.selectSession(null)
            runtime.selectSession(session.id)
          }
          if (window.matchMedia('(max-width: 959px)').matches) setSidebarOpen(false)
        }}
        onToggle={() => setSidebarOpen((current) => !current)}
      />

      <div className={styles.workspace}>
        <header className={styles.topbar}>
          <div className={styles.topbarActions}>
            <button
              type="button"
              className={`${styles.iconButton} ${styles.mobileSidebarButton}`}
              onClick={() => setSidebarOpen(true)}
              aria-label="Open conversations"
              aria-controls="agent-conversation-navigation"
              aria-expanded={sidebarOpen}
              data-testid="agent-mobile-conversation-trigger"
            >
              <ProductIcon name="chevron-right" />
            </button>
            <StreamState
              status={
                activeMode !== 'chat' ? runtime.streamStatus : inference.running ? 'live' : 'idle'
              }
            />
            <InferenceKeySelector
              className={styles.playgroundKeySelector}
              disabled={Boolean(activeSessionId) || runtime.mutating || inference.running}
              label="Use"
            />
            <button
              type="button"
              className={styles.iconButton}
              onClick={() => void runtime.refreshSessions()}
              aria-label="Refresh conversations"
            >
              <ProductIcon name="refresh" />
            </button>
          </div>
        </header>

        <AgentTimeline
          events={activeMode !== 'chat' ? runtime.events : []}
          liveModelSteps={activeMode !== 'chat' ? runtime.liveModelSteps : []}
          inferenceMessages={
            activeMode === 'chat' ? inference.messagesForSession(activeChatSession?.id ?? null) : []
          }
          hasEarlier={activeMode !== 'chat' && runtime.eventsHaveMore}
          loading={activeMode !== 'chat' && runtime.loadingEvents}
          mode={activeMode}
          canLoadArtifactContent={canReadAgent(user)}
          canReadRequestLogs={canAccessDashboardPath(user, '/logs')}
          userName={displayName}
          onLoadEarlier={() => void runtime.loadMoreEvents()}
          onReview={() => setReviewOpen(true)}
        />

        {localError ||
        inference.error ||
        (activeMode !== 'chat' ? runtime.error : null) ||
        routing.error ? (
          <div className={styles.errorBanner} role="alert">
            <ProductIcon name="alert" />
            <span>
              {localError ||
                inference.error ||
                (activeMode !== 'chat' ? runtime.error : null) ||
                routing.error}
            </span>
            {routing.status === 'error' ? (
              <button type="button" onClick={() => void routing.refresh().catch(() => undefined)}>
                Try again
              </button>
            ) : activeMode !== 'chat' && runtime.streamStatus === 'error' ? (
              <button type="button" onClick={runtime.recoverStream}>
                Reconnect
              </button>
            ) : null}
          </div>
        ) : null}

        <AgentComposer
          agentAvailable={agentAvailable && !activeSessionId}
          attachments={attachments}
          builderAvailable={builderAvailable && !activeSessionId}
          disabledReason={disabledReason}
          input={input}
          mode={activeMode}
          models={routing.models}
          queuePaused={queuePaused}
          queuedTurns={queuedTurns}
          running={composerRunning}
          selectedModel={selectedTargetId}
          targetLocked={Boolean(activeSessionId)}
          onAttach={(files) => void handleAttach(files)}
          onInputChange={setInput}
          onModeChange={handleModeChange}
          onModelChange={routing.setModel}
          onQueue={handleQueue}
          onQueueRemove={(turnId) =>
            setQueuedTurns((current) => current.filter((turn) => turn.id !== turnId))
          }
          onQueueResume={() => {
            inference.clearError()
            setLocalError(null)
            setQueuePaused(false)
          }}
          onRemoveAttachment={(id) =>
            setAttachments((current) => current.filter((item) => item.id !== id))
          }
          onSend={handleSend}
          onStop={() => {
            if (queuedTurns.length) setQueuePaused(true)
            if (activeMode !== 'chat') void runtime.cancelTurn()
            else inference.cancel()
          }}
        />
      </div>

      <ConfirmDialog
        isOpen={Boolean(deleteTarget)}
        title="Delete this conversation?"
        description="This conversation and its history will be removed."
        error={deleteTarget?.source === 'router' ? runtime.error : null}
        confirmLabel="Delete"
        pending={deleteTarget?.source === 'router' && runtime.mutating}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={async () => {
          if (!deleteTarget) return
          if (deleteTarget.source === 'browser') {
            inference.deleteSession(deleteTarget.id)
          } else {
            await runtime.deleteSession(deleteTarget.id)
          }
          setDeleteTarget(null)
        }}
      />
      {reviewOpen && activeApproval ? (
        <AgentPublicationReviewDialog
          approval={activeApproval}
          canPublish={canPublish}
          error={publishError}
          publishing={runtime.mutating}
          onClose={() => {
            setPublishError(null)
            setReviewOpen(false)
          }}
          onPublish={() => void handlePublish()}
        />
      ) : null}
    </section>
  )
}
