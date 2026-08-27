import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useAuth } from '../contexts/AuthContext'
import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import { useDelegatedInferenceSession } from '../hooks/useDelegatedInferenceSession'
import { useAgentSessionRuntime } from '../hooks/useAgentSessionRuntime'
import type { PlaygroundInvocation } from '../types/playgroundInvocation'
import {
  canAccessDashboardPath,
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
  const chatAvailable = canUseAgent(user)
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

  const activeSession = runtime.activeSession
  const activeMode: PlaygroundMode = activeSession?.mode ?? draftMode
  const effectiveTeamId = activeSession?.effectiveTeamId ?? selectedKey?.contextTeamId ?? ''
  const activeSessionId = activeSession?.id ?? null
  const displayName = user?.name.trim() || user?.email.split('@')[0] || 'there'
  const conversationSessions = useMemo<PlaygroundConversationListItem[]>(() => {
    const normalizedSearch = search.trim().toLocaleLowerCase()
    return runtime.sessions
      .filter(
        (session) =>
          !normalizedSearch || session.title.toLocaleLowerCase().includes(normalizedSearch),
      )
      .map((session) => ({
        id: session.id,
        mode: playgroundModeForAgentSession(session.mode),
        source: 'router' as const,
        title: session.title,
        updatedAt: session.updatedAt,
      }))
  }, [runtime.sessions, search])
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
    if (!activeSession) return
    setSelectedKeyId(activeSession.keyId)
    routing.setModel(activeSession.target.id)
    setDraftMode(playgroundModeForAgentSession(activeSession.mode))
    setAttachments([])
    setInput('')
    setLocalError(null)
  }, [activeSession?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!invocation || invocationHandled.current || routing.status !== 'ready') return
    invocationHandled.current = true
    runtime.selectSession(null)
    if (invocation.intent === 'edit' && builderAvailable) setDraftMode('builder')
    if (invocation.model && routing.models.some((model) => model.id === invocation.model)) {
      routing.setModel(invocation.model)
    }
    setInput(invocationPrompt(invocation))
    onInvocationConsumed?.()
  }, [builderAvailable, invocation, onInvocationConsumed, routing, runtime])

  useEffect(() => {
    if (!initialModel || initialModelHandled.current || routing.status !== 'ready') return
    initialModelHandled.current = true
    runtime.selectSession(null)
    if (routing.models.some((model) => model.id === initialModel)) {
      routing.setModel(initialModel)
    } else {
      setLocalError('This model is not available to your account.')
    }
    onInitialModelConsumed?.()
  }, [initialModel, onInitialModelConsumed, routing, runtime])

  const selectedTargetId = activeSession?.target.id ?? routing.model
  const selectedOption = routing.models.find((model) => model.id === selectedTargetId)
  const running = Boolean(runtime.activeTurnId)
  const composerRunning = running || runtime.mutating
  const disabledReason = (() => {
    if (!canUse) return 'Playground access is not available.'
    if (delegated.status === 'loading' || routing.status === 'discovering') return 'Getting ready…'
    if (delegated.status === 'unavailable') return 'Create an API key to use Playground.'
    if (delegated.status === 'error') return 'API access is unavailable.'
    if (routing.status === 'error') return 'Models are unavailable.'
    if (activeMode === 'chat' && !chatAvailable) return 'Playground access is not available.'
    if (activeMode === 'builder' && !builderAvailable) return 'Builder access is not available.'
    if (activeSession && selectedKey?.keyId !== activeSession.keyId) {
      return 'The API key for this conversation is no longer available.'
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
      setDraftMode(mode)
      setInput('')
      setAttachments([])
      setQueuedTurns([])
      setQueuePaused(false)
      setLocalError(null)
      if (window.innerWidth < 960) setSidebarOpen(false)
    },
    [runtime],
  )

  const handleModeChange = (mode: PlaygroundMode) => {
    if (
      activeSessionId ||
      (mode === 'chat' && !chatAvailable) ||
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
      let sessionId = activeSession?.id ?? null
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
    },
    [activeSession?.id, activeMode, effectiveTeamId, runtime, selectedKey, selectedOption],
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
        busy={runtime.mutating}
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
          setQueuedTurns([])
          setQueuePaused(false)
          runtime.selectSession(session.id)
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
            <StreamState status={runtime.streamStatus} />
            <InferenceKeySelector
              className={styles.playgroundKeySelector}
              disabled={Boolean(activeSessionId) || runtime.mutating}
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
          events={runtime.events}
          liveModelSteps={runtime.liveModelSteps}
          inferenceMessages={[]}
          hasEarlier={runtime.eventsHaveMore}
          loading={runtime.loadingEvents}
          mode={activeMode}
          canLoadArtifactContent={canReadAgent(user)}
          canReadRequestLogs={canAccessDashboardPath(user, '/logs')}
          userName={displayName}
          onLoadEarlier={() => void runtime.loadMoreEvents()}
          onReview={() => setReviewOpen(true)}
        />

        {localError || runtime.error || routing.error ? (
          <div className={styles.errorBanner} role="alert">
            <ProductIcon name="alert" />
            <span>{localError || runtime.error || routing.error}</span>
            {routing.status === 'error' ? (
              <button type="button" onClick={() => void routing.refresh().catch(() => undefined)}>
                Try again
              </button>
            ) : runtime.streamStatus === 'error' ? (
              <button type="button" onClick={runtime.recoverStream}>
                Reconnect
              </button>
            ) : null}
          </div>
        ) : null}

        <AgentComposer
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
            setLocalError(null)
            setQueuePaused(false)
          }}
          onRemoveAttachment={(id) =>
            setAttachments((current) => current.filter((item) => item.id !== id))
          }
          onSend={handleSend}
          onStop={() => {
            if (queuedTurns.length) setQueuePaused(true)
            void runtime.cancelTurn()
          }}
        />
      </div>

      <ConfirmDialog
        isOpen={Boolean(deleteTarget)}
        title="Delete this conversation?"
        description="This conversation and its history will be removed."
        error={runtime.error}
        confirmLabel="Delete"
        pending={runtime.mutating}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={async () => {
          if (!deleteTarget) return
          await runtime.deleteSession(deleteTarget.id)
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
