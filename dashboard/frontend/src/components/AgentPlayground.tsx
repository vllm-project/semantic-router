import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { useAuth } from '../contexts/AuthContext'
import { useInferenceRoutingAccess } from '../contexts/InferenceRoutingAccessContext'
import { useDelegatedInferenceSession } from '../hooks/useDelegatedInferenceSession'
import { useAgentSessionRuntime } from '../hooks/useAgentSessionRuntime'
import type { PlaygroundInvocation } from '../types/playgroundInvocation'
import {
  canManageRouting,
  canPublishRouting,
  canReadAgent,
  canUseAgent,
  canUseBuilderAgent,
} from '../utils/accessControl'
import { pendingApproval } from '../utils/agentEventProjection'
import type {
  AgentContentBlock,
  AgentSession,
  AgentSessionMode,
} from '../generated/managementApiContract'
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
import AgentConversationSidebar from './AgentConversationSidebar'
import AgentPublicationReviewDialog from './AgentPublicationReviewDialog'
import AgentTimeline from './AgentTimeline'
import ProductIcon from './ProductIcon'
import { buildAgentSessionInput } from './agentPlaygroundSession'
import styles from './AgentPlayground.module.css'

interface AgentPlaygroundProps {
  endpoint: string
  fullscreen?: boolean
  invocation?: PlaygroundInvocation | null
  onInvocationConsumed?: () => void
}

function titleFromPrompt(prompt: string, mode: AgentSessionMode): string {
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
  onInvocationConsumed,
}: AgentPlaygroundProps) {
  const { user } = useAuth()
  const { selectedKey, setSelectedKeyId } = useInferenceRoutingAccess()
  const canUse = canUseAgent(user)
  const builderAvailable = canUseBuilderAgent(user)
  const canPublish = canPublishRouting(user)
  const includeSingleModels = canManageRouting(user)
  const delegated = useDelegatedInferenceSession()
  const routing = usePlaygroundRoutingModel(
    endpoint,
    includeSingleModels,
    delegated.getAccessToken,
    canUse && delegated.status === 'ready',
  )
  const [search, setSearch] = useState('')
  const runtime = useAgentSessionRuntime({ enabled: canUse, search })
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth >= 960)
  const [draftMode, setDraftMode] = useState<AgentSessionMode>('chat')
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<PlaygroundAttachment[]>([])
  const [localError, setLocalError] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<AgentSession | null>(null)
  const [reviewOpen, setReviewOpen] = useState(false)
  const [publishError, setPublishError] = useState<string | null>(null)
  const invocationHandled = useRef(false)

  const activeMode = runtime.activeSession?.mode ?? draftMode
  const effectiveTeamId = runtime.activeSession?.effectiveTeamId ?? selectedKey?.contextTeamId ?? ''
  const activeApproval = useMemo(() => pendingApproval(runtime.events), [runtime.events])
  const activeApprovalPlanId = activeApproval?.planId

  useEffect(() => {
    if (activeApprovalPlanId) {
      setPublishError(null)
      setReviewOpen(true)
    }
  }, [activeApprovalPlanId])

  useEffect(() => {
    if (!runtime.activeSession) return
    setSelectedKeyId(runtime.activeSession.keyId)
    routing.setModel(runtime.activeSession.target.id)
    setDraftMode(runtime.activeSession.mode)
    setAttachments([])
    setInput('')
    setLocalError(null)
  }, [runtime.activeSession?.id]) // eslint-disable-line react-hooks/exhaustive-deps

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

  const selectedTargetId = runtime.activeSession?.target.id ?? routing.model
  const selectedOption = routing.models.find((model) => model.id === selectedTargetId)
  const running = Boolean(runtime.activeTurnId)
  const disabledReason = (() => {
    if (!canUse) return 'Playground access is not available.'
    if (delegated.status === 'loading' || routing.status === 'discovering') return 'Getting ready…'
    if (delegated.status === 'unavailable') return 'Create an API key to use Playground.'
    if (delegated.status === 'error') return 'API access is unavailable.'
    if (routing.status === 'error') return 'Models are unavailable.'
    if (activeApproval && Date.parse(activeApproval.expiresAt) > Date.now()) {
      return 'Review this plan before continuing.'
    }
    if (!selectedOption) {
      return runtime.activeSession ? 'This model is no longer available.' : 'Choose a model.'
    }
    return undefined
  })()

  const startNew = useCallback(
    (mode: AgentSessionMode = 'chat') => {
      runtime.selectSession(null)
      setDraftMode(mode)
      setInput('')
      setAttachments([])
      setLocalError(null)
      if (window.innerWidth < 960) setSidebarOpen(false)
    },
    [runtime],
  )

  const handleModeChange = (mode: AgentSessionMode) => {
    if (runtime.activeSession || (mode === 'builder' && !builderAvailable)) return
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

  const handleSend = async () => {
    if (disabledReason || running || !selectedOption) return
    const prompt = buildPromptWithAttachments(input, attachments)
    const content: AgentContentBlock[] = [
      ...(prompt ? [{ type: 'text' as const, text: prompt }] : []),
      ...attachments.filter(isPlaygroundImageAttachment).map((attachment) => ({
        type: 'image_url' as const,
        url: attachment.content,
        detail: 'auto' as const,
      })),
    ]
    if (!content.length) return
    setLocalError(null)
    try {
      let sessionId = runtime.activeSessionId
      if (!sessionId) {
        if (!selectedKey) throw new Error('Choose an API key before starting a conversation.')
        const session = await runtime.createSession(
          buildAgentSessionInput({
            keyId: selectedKey.keyId,
            mode: activeMode,
            effectiveTeamId,
            model: selectedOption,
            title: titleFromPrompt(input, activeMode),
          }),
        )
        sessionId = session.id
      }
      await runtime.sendTurn(content, sessionId)
      setInput('')
      setAttachments([])
    } catch (cause) {
      setLocalError(cause instanceof Error ? cause.message : 'The message could not be sent.')
    }
  }

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
        activeSessionId={runtime.activeSessionId}
        busy={runtime.mutating}
        loading={runtime.loadingSessions}
        open={sidebarOpen}
        search={search}
        sessions={runtime.sessions}
        sessionsHaveMore={runtime.sessionsHaveMore}
        onDeleteRequest={setDeleteTarget}
        onLoadMore={() => void runtime.loadMoreSessions()}
        onNewChat={() => startNew('chat')}
        onSearchChange={setSearch}
        onSelect={(session) => runtime.selectSession(session.id)}
        onToggle={() => setSidebarOpen((current) => !current)}
      />

      <div className={styles.workspace}>
        <header className={styles.topbar}>
          <div className={styles.topbarIdentity}>
            <div className={styles.topbarMark}>
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <strong>
                {runtime.activeSession?.title ||
                  (activeMode === 'builder' ? 'Builder' : 'Playground')}
              </strong>
              <span>
                {runtime.activeSession
                  ? routing.model || runtime.activeSession.target.id
                  : activeMode === 'builder'
                    ? 'Design a Mixture-of-Models'
                    : 'Router Agent'}
              </span>
            </div>
          </div>
          <div className={styles.topbarActions}>
            <StreamState status={runtime.streamStatus} />
            <InferenceKeySelector
              className={styles.playgroundKeySelector}
              disabled={Boolean(runtime.activeSession) || runtime.mutating}
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
          hasEarlier={runtime.eventsHaveMore}
          loading={runtime.loadingEvents}
          mode={activeMode}
          canLoadArtifactContent={canReadAgent(user)}
          onLoadEarlier={() => void runtime.loadMoreEvents()}
          onReview={() => setReviewOpen(true)}
        />

        {localError || runtime.error ? (
          <div className={styles.errorBanner} role="alert">
            <ProductIcon name="alert" />
            <span>{localError || runtime.error}</span>
            {runtime.streamStatus === 'error' ? (
              <button type="button" onClick={runtime.recoverStream}>
                Reconnect
              </button>
            ) : null}
          </div>
        ) : null}

        <AgentComposer
          attachments={attachments}
          builderAvailable={builderAvailable && !runtime.activeSession}
          disabledReason={disabledReason}
          input={input}
          mode={activeMode}
          models={routing.models}
          running={running || runtime.mutating}
          selectedModel={selectedTargetId}
          targetLocked={Boolean(runtime.activeSession)}
          onAttach={(files) => void handleAttach(files)}
          onInputChange={setInput}
          onModeChange={handleModeChange}
          onModelChange={routing.setModel}
          onRemoveAttachment={(id) =>
            setAttachments((current) => current.filter((item) => item.id !== id))
          }
          onSend={() => void handleSend()}
          onStop={() => void runtime.cancelTurn()}
        />
      </div>

      <ConfirmDialog
        isOpen={Boolean(deleteTarget)}
        title="Delete this conversation?"
        description="This conversation and its history will be removed."
        error={deleteTarget ? runtime.error : null}
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
