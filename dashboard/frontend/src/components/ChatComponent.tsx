import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import styles from './ChatComponent.module.css'
import ThinkingAnimation from './ThinkingAnimation'
import HeaderReveal from './HeaderReveal'
import ClawRoomChat from './ClawRoomChat'
import ChatComposerAddMenu from './ChatComposerAddMenu'
import ChatConversationSidebar from './ChatConversationSidebar'
import ChatComponentConversationViewport from './ChatComponentConversationViewport'
import ChatComponentErrors from './ChatComponentErrors'
import ChatComponentInputBar from './ChatComponentInputBar'
import ChatComponentSidebarShell from './ChatComponentSidebarShell'
import ChatTaskQueue from './ChatTaskQueue'
import { runPlaygroundTask } from './chatTaskExecution'
import {
  generateConversationId,
  generateMessageId,
  type PlaygroundTask,
  type Message,
} from './ChatComponentTypes'
import {
  buildConversationPreviews,
  type ChatComponentProps,
  type ClawPlaygroundView,
  findQueuedErrorConversationId,
  getLiveThinkingProcess,
  readClawModePreference,
  writeClawModePreference,
} from './chatComponentSupport'
import { useToolRegistry } from '../tools'
import { isOpenClawMCPToolName, useMCPToolSync } from '../tools/mcp'
import { ensureOpenClawServerConnected } from '../tools/mcp/api'
import { useConversationStorage, usePlaygroundQueue } from '../hooks'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canManageMCP } from '../utils/accessControl'
import { usePlaygroundAttachments } from './usePlaygroundAttachments'
import { useChatConversationState } from './useChatConversationState'
import { usePlaygroundConversationMessages } from './usePlaygroundConversationMessages'
import { usePlaygroundRoutingModel } from './usePlaygroundRoutingModel'
import {
  usePlaygroundInvocation,
  type ActivePlaygroundInvocationDraft,
} from './usePlaygroundInvocation'
import { usePlaygroundTaskSubmission } from './usePlaygroundTaskSubmission'
import { sanitizeMessagesForPersistence } from './chatPersistenceSupport'

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  invocation = null,
  isFullscreenMode = false,
  onInvocationConsumed,
}: ChatComponentProps) => {
  const [conversationMessages, setConversationMessages] = useState<Record<string, Message[]>>({})
  const [conversationId, setConversationId] = useState<string>(() => generateConversationId())
  const [inputValue, setInputValue] = useState('')
  const [activeTasks, setActiveTasks] = useState<Record<string, PlaygroundTask>>({})
  const [probeDraft, setProbeDraft] = useState<ActivePlaygroundInvocationDraft | null>(null)
  const {
    model,
    models: routingModels,
    retry: retryRoutingModelDiscovery,
    setModel,
    status: routingModelStatus,
  } = usePlaygroundRoutingModel(endpoint)
  const isRoutingModelReady = routingModelStatus === 'ready'
  const {
    conversationErrors,
    conversationThinking,
    headerRevealStates,
    setConversationError,
    setConversationHeaderReveal,
    setConversationThinkingState,
  } = useChatConversationState()
  const [isFullscreen] = useState(isFullscreenMode)
  const [enableWebSearch, setEnableWebSearch] = useState(false)
  const [enableClawMode, setEnableClawMode] = useState<boolean>(readClawModePreference)
  const [isTogglingClawMode, setIsTogglingClawMode] = useState(false)
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [clawView, setClawView] = useState<ClawPlaygroundView>(() => 'control')
  const [teamRoomCreateToken, setTeamRoomCreateToken] = useState(0)
  const { user, isLoading: authLoading } = useAuth()
  const { serverReadonly, isLoading: readonlyLoading } = useReadonly()

  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllersRef = useRef<Record<string, AbortController>>({})
  const hasHydratedConversation = useRef(false)
  const activeTasksRef = useRef<Record<string, PlaygroundTask>>({})
  const conversationIdRef = useRef(conversationId)

  const { conversations, saveConversation, getConversation, deleteConversation } =
    useConversationStorage<Message[]>({
      storageKey: 'sr:chat:conversations',
      maxConversations: 20,
      preparePayloadForPersistence: sanitizeMessagesForPersistence,
    })
  const {
    clearConversationQueue,
    enqueueTask,
    getQueue,
    queues,
    removeTask: removeQueuedTask,
    reorderTasks,
  } = usePlaygroundQueue()

  const {
    clearPendingAttachments,
    copyPendingAttachmentsForTask,
    handleAttachFiles,
    handleRemoveAttachment,
    pendingAttachments,
    restorePendingAttachments,
  } = usePlaygroundAttachments({
    conversationId,
    setConversationError,
  })

  const {
    getConversationMessagesSnapshot,
    getStoredMessagesForConversation,
    removeConversationMessages,
    restoreMessages,
    updateConversationMessages,
  } = usePlaygroundConversationMessages({
    conversationMessages,
    getConversation,
    setConversationMessages,
  })

  const setActiveTaskForConversation = useCallback((task: PlaygroundTask) => {
    if (activeTasksRef.current[task.conversationId]?.id === task.id) {
      return
    }
    const next = {
      ...activeTasksRef.current,
      [task.conversationId]: task,
    }
    activeTasksRef.current = next
    setActiveTasks(next)
  }, [])

  const clearActiveTaskForConversation = useCallback(
    (targetConversationId: string, taskId: string) => {
      const currentTask = activeTasksRef.current[targetConversationId]
      if (!currentTask || currentTask.id !== taskId) {
        return
      }
      const next = { ...activeTasksRef.current }
      delete next[targetConversationId]
      activeTasksRef.current = next
      setActiveTasks(next)
    },
    [],
  )

  const registerAbortController = useCallback(
    (targetConversationId: string, controller: AbortController | null) => {
      if (controller) {
        abortControllersRef.current[targetConversationId] = controller
        return
      }
      delete abortControllersRef.current[targetConversationId]
    },
    [],
  )

  useEffect(() => {
    conversationIdRef.current = conversationId
  }, [conversationId])

  const { refresh: refreshMCPTools } = useMCPToolSync({ enabled: true, pollInterval: 30000 })

  const { definitions: searchToolDefinitions } = useToolRegistry({
    enabledOnly: true,
    categories: ['search'],
  })
  // Other tools (always available, not controlled by web search toggle)
  const { definitions: otherToolDefinitions, executeAll: executeTools } = useToolRegistry({
    enabledOnly: true,
    categories: ['code', 'file', 'image', 'custom'],
  })

  const baseOtherToolDefinitions = useMemo(
    () => otherToolDefinitions.filter((def) => !isOpenClawMCPToolName(def.function.name)),
    [otherToolDefinitions],
  )
  const clawToolDefinitions = useMemo(
    () => otherToolDefinitions.filter((def) => isOpenClawMCPToolName(def.function.name)),
    [otherToolDefinitions],
  )
  const clawManagementDisabled =
    authLoading || readonlyLoading || serverReadonly || !canManageMCP(user)
  const currentHeaderRevealState = headerRevealStates[conversationId]

  useEffect(() => {
    if (!currentHeaderRevealState || currentHeaderRevealState.visible) {
      return
    }

    setConversationHeaderReveal(conversationId, currentHeaderRevealState.headers, true)
  }, [conversationId, currentHeaderRevealState, setConversationHeaderReveal])

  // Toggle fullscreen mode by adding/removing class to body
  useEffect(() => {
    if (isFullscreen) {
      document.body.classList.add('playground-fullscreen')
    } else {
      document.body.classList.remove('playground-fullscreen')
    }

    return () => {
      document.body.classList.remove('playground-fullscreen')
    }
  }, [isFullscreen])

  useEffect(() => {
    writeClawModePreference(enableClawMode)
  }, [enableClawMode])

  useEffect(() => {
    if (!enableClawMode) {
      setIsTogglingClawMode(false)
      setClawView('control')
      return
    }
    if (clawManagementDisabled) {
      setIsTogglingClawMode(false)
      return
    }

    let isCurrent = true
    const bootstrapClawTools = async () => {
      setIsTogglingClawMode(true)
      try {
        await ensureOpenClawServerConnected()
        await refreshMCPTools()
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to enable Claw Mode'
        console.warn(`[OpenClaw] UI mode enabled, but MCP bootstrap failed: ${message}`)
      } finally {
        if (isCurrent) {
          setIsTogglingClawMode(false)
        }
      }
    }

    void bootstrapClawTools()

    return () => {
      isCurrent = false
    }
  }, [clawManagementDisabled, enableClawMode, refreshMCPTools])

  useEffect(() => {
    if (enableClawMode && clawView === 'room') {
      setIsSidebarOpen(false)
    }
  }, [enableClawMode, clawView])

  // Hydrate the most recent conversation from localStorage once
  useEffect(() => {
    if (hasHydratedConversation.current) return

    if (conversations.length === 0) return

    const restoredConversationMessages = conversations.reduce<Record<string, Message[]>>(
      (acc, conv) => {
        if (Array.isArray(conv.payload)) {
          acc[conv.id] = restoreMessages(conv.payload)
        }
        return acc
      },
      {},
    )

    setConversationMessages(restoredConversationMessages)

    const latestConversation = getConversation()
    if (latestConversation?.payload && Array.isArray(latestConversation.payload)) {
      setConversationId(latestConversation.id)
    }

    hasHydratedConversation.current = true
  }, [conversations, getConversation, restoreMessages])

  // Persist changed conversations whenever in-memory messages change
  useEffect(() => {
    Object.entries(conversationMessages).forEach(([id, payload]) => {
      if (payload.length === 0) {
        return
      }
      saveConversation(id, payload)
    })
  }, [conversationMessages, saveConversation])

  const conversationPreviews = useMemo(
    () => buildConversationPreviews(conversations),
    [conversations],
  )

  const messages = useMemo(
    () => conversationMessages[conversationId] ?? getStoredMessagesForConversation(conversationId),
    [conversationId, conversationMessages, getStoredMessagesForConversation],
  )
  const queuedTasks = useMemo(() => getQueue(conversationId), [conversationId, getQueue])
  const generateId = generateMessageId
  const activeConversationTask = activeTasks[conversationId] ?? null
  const hasRunningTasks = Object.keys(activeTasks).length > 0
  const isCurrentConversationRunning = Boolean(activeConversationTask)

  const buildTaskRequestOptions = useCallback(
    () => ({
      enableClawMode: enableClawMode && !clawManagementDisabled,
      enableWebSearch,
      model,
    }),
    [clawManagementDisabled, enableClawMode, enableWebSearch, model],
  )

  const buildTaskTools = useCallback(
    (task: PlaygroundTask) => {
      const toolsDisabled =
        !task.requestOptions.enableWebSearch &&
        (!task.requestOptions.enableClawMode || clawManagementDisabled)
      if (toolsDisabled) return []
      const otherTools =
        task.requestOptions.enableClawMode && !clawManagementDisabled
          ? [...baseOtherToolDefinitions, ...clawToolDefinitions]
          : baseOtherToolDefinitions

      return [...otherTools, ...(task.requestOptions.enableWebSearch ? searchToolDefinitions : [])]
    },
    [baseOtherToolDefinitions, clawManagementDisabled, clawToolDefinitions, searchToolDefinitions],
  )

  const handleThinkingComplete = useCallback(() => {}, [])

  const handleHeaderRevealComplete = useCallback(() => {
    setConversationHeaderReveal(conversationId, null)
  }, [conversationId, setConversationHeaderReveal])

  const handleSelectConversation = useCallback(
    (id: string) => {
      const target = conversations.find((conv) => conv.id === id)
      if (!target) return

      setConversationId(target.id)
      setInputValue('')
      setProbeDraft(null)
      setExpandedToolCards(new Set())
    },
    [conversations],
  )

  const handleDeleteConversation = useCallback(
    (id: string) => {
      const remaining = conversations.filter((conv) => conv.id !== id)
      const deletingActiveConversation = Boolean(activeTasksRef.current[id])

      clearConversationQueue(id)
      deleteConversation(id)
      removeConversationMessages(id)

      if (deletingActiveConversation) {
        abortControllersRef.current[id]?.abort()
        clearActiveTaskForConversation(id, activeTasksRef.current[id].id)
      }

      registerAbortController(id, null)
      setConversationError(id, null)
      setConversationThinkingState(id, false)
      setConversationHeaderReveal(id, null)

      if (id === conversationId) {
        setExpandedToolCards(new Set())
        setInputValue('')
        setProbeDraft(null)

        const next = remaining[0]
        if (next) {
          setConversationId(next.id)
        } else {
          setConversationId(generateConversationId())
        }
      }
    },
    [
      clearActiveTaskForConversation,
      clearConversationQueue,
      conversationId,
      conversations,
      deleteConversation,
      removeConversationMessages,
      registerAbortController,
      setConversationError,
      setConversationHeaderReveal,
      setConversationThinkingState,
    ],
  )

  const executeTask = useCallback(
    (task: PlaygroundTask) =>
      runPlaygroundTask({
        buildTaskTools,
        clawManagementDisabled,
        clearConversationActiveTask: clearActiveTaskForConversation,
        endpoint,
        executeTools,
        expandedToolCardCount: expandedToolCards.size,
        generateId,
        getConversationMessagesSnapshot,
        getCurrentConversationId: () => conversationIdRef.current,
        registerAbortController,
        setConversationError,
        setConversationHeaderReveal,
        setConversationThinking: setConversationThinkingState,
        setExpandedToolCards,
        task,
        updateConversationMessages,
      }),
    [
      buildTaskTools,
      clawManagementDisabled,
      clearActiveTaskForConversation,
      endpoint,
      executeTools,
      expandedToolCards.size,
      generateId,
      getConversationMessagesSnapshot,
      registerAbortController,
      setConversationError,
      setConversationHeaderReveal,
      setConversationThinkingState,
      setExpandedToolCards,
      updateConversationMessages,
    ],
  )

  const startTask = useCallback(
    (task: PlaygroundTask) => {
      if (!isRoutingModelReady || activeTasksRef.current[task.conversationId]) {
        return
      }

      setActiveTaskForConversation(task)
      void executeTask(task)
    },
    [executeTask, isRoutingModelReady, setActiveTaskForConversation],
  )

  const activateProbeConversation = useCallback(
    (targetConversationId: string, initialMessages: Message[]) => {
      hasHydratedConversation.current = true
      conversationIdRef.current = targetConversationId
      clearPendingAttachments()
      setEnableClawMode(false)
      setEnableWebSearch(false)
      setExpandedToolCards(new Set())
      setConversationMessages((current) => ({
        ...current,
        [targetConversationId]: initialMessages,
      }))
      setConversationId(targetConversationId)
    },
    [clearPendingAttachments],
  )

  const focusComposer = useCallback(() => {
    if (typeof window === 'undefined') return
    window.requestAnimationFrame(() => {
      inputRef.current?.focus()
      const promptLength = inputRef.current?.value.length ?? 0
      inputRef.current?.setSelectionRange(promptLength, promptLength)
    })
  }, [])

  usePlaygroundInvocation({
    invocation,
    isRoutingModelReady,
    onInvocationConsumed,
    routingModels,
    activateConversation: activateProbeConversation,
    focusComposer,
    setConversationError,
    setDraft: setProbeDraft,
    setInputValue,
    setModel,
    startTask,
  })

  const handleSend = usePlaygroundTaskSubmission({
    activeTasksRef,
    buildTaskRequestOptions,
    clearPendingAttachments,
    conversationId,
    conversations,
    copyPendingAttachmentsForTask,
    enqueueTask,
    getConversationMessagesSnapshot,
    hasHydratedConversation,
    inputValue,
    isRoutingModelReady,
    model,
    pendingAttachments,
    probeDraft,
    saveConversation,
    setConversationError,
    setInputValue,
    setProbeDraft,
    startTask,
  })

  useEffect(() => {
    if (!isRoutingModelReady) {
      return
    }
    let activatedOrphanQueue = false
    Object.entries(queues).forEach(([targetConversationId, queue]) => {
      if (queue.length === 0 || activeTasksRef.current[targetConversationId]) {
        return
      }

      const nextTask = queue.reduce<PlaygroundTask>(
        (earliestTask, task) => (task.createdAt < earliestTask.createdAt ? task : earliestTask),
        queue[0],
      )
      if (!routingModels.some((modelOption) => modelOption.id === nextTask.requestOptions.model)) {
        setConversationError(
          targetConversationId,
          `Queued model "${nextTask.requestOptions.model}" is no longer available. Delete the queued task and resend it with an available model.`,
        )
        if (
          !activatedOrphanQueue &&
          conversationIdRef.current !== targetConversationId &&
          !conversations.some((conversation) => conversation.id === targetConversationId)
        ) {
          activatedOrphanQueue = true
          conversationIdRef.current = targetConversationId
          setConversationId(targetConversationId)
          setInputValue('')
          setProbeDraft(null)
          setExpandedToolCards(new Set())
        }
        return
      }

      removeQueuedTask(targetConversationId, nextTask.id)
      startTask(nextTask)
    })
  }, [
    activeTasks,
    conversations,
    isRoutingModelReady,
    queues,
    removeQueuedTask,
    routingModels,
    setConversationError,
    startTask,
  ])

  const handleDeleteQueuedTask = useCallback(
    (taskId: string) => {
      removeQueuedTask(conversationId, taskId)
    },
    [conversationId, removeQueuedTask],
  )

  const handleEditQueuedTask = useCallback(
    (taskId: string) => {
      const taskToEdit = queuedTasks.find((task) => task.id === taskId)
      if (!taskToEdit) {
        return
      }

      removeQueuedTask(conversationId, taskId)
      setInputValue(taskToEdit.prompt)
      restorePendingAttachments(taskToEdit.attachments)

      if (typeof window !== 'undefined') {
        window.requestAnimationFrame(() => {
          inputRef.current?.focus()
          const promptLength = taskToEdit.prompt.length
          inputRef.current?.setSelectionRange(promptLength, promptLength)
        })
      }
    },
    [conversationId, queuedTasks, removeQueuedTask, restorePendingAttachments],
  )

  const handleReorderQueuedTasks = useCallback(
    (sourceTaskId: string, targetTaskId: string) => {
      reorderTasks(conversationId, sourceTaskId, targetTaskId)
    },
    [conversationId, reorderTasks],
  )

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleStop = () => {
    abortControllersRef.current[conversationId]?.abort()
  }

  const handleNewConversation = useCallback(() => {
    setInputValue('')
    setProbeDraft(null)
    clearPendingAttachments()
    setExpandedToolCards(new Set())
    setConversationId(generateConversationId())
  }, [clearPendingAttachments])

  const handleToggleClawMode = useCallback(() => {
    if (hasRunningTasks || isTogglingClawMode) return
    if (enableClawMode) {
      setEnableClawMode(false)
      setConversationError(conversationId, null)
      return
    }
    setEnableClawMode(true)
    setConversationError(conversationId, null)
  }, [conversationId, enableClawMode, hasRunningTasks, isTogglingClawMode, setConversationError])

  const isTeamRoomView = enableClawMode && clawView === 'room',
    roomCreateDisabled = isTeamRoomView && clawManagementDisabled
  const hasActiveProbeDraft = probeDraft?.conversationId === conversationId
  const modeToggleDisabled =
    hasRunningTasks || isTogglingClawMode || clawManagementDisabled || hasActiveProbeDraft

  const handleToggleTeamView = useCallback(() => {
    if (!enableClawMode || modeToggleDisabled) return
    setClawView((prev) => (prev === 'room' ? 'control' : 'room'))
  }, [enableClawMode, modeToggleDisabled])

  const handleTopBarCreate = useCallback(() => {
    if (roomCreateDisabled) return
    if (isTeamRoomView) {
      setTeamRoomCreateToken((prev) => prev + 1)
      return
    }
    handleNewConversation()
  }, [handleNewConversation, isTeamRoomView, roomCreateDisabled])

  const handleToggleToolCard = useCallback((toolCallId: string) => {
    setExpandedToolCards((prev) => {
      const next = new Set(prev)
      if (next.has(toolCallId)) {
        next.delete(toolCallId)
      } else {
        next.add(toolCallId)
      }
      return next
    })
  }, [])

  const liveThinkingProcess = getLiveThinkingProcess(messages)
  const queuedErrorConversationId = findQueuedErrorConversationId(queues, conversationErrors)
  const visibleErrorConversationId = conversationErrors[conversationId]
    ? conversationId
    : queuedErrorConversationId
  const visibleError = visibleErrorConversationId
    ? conversationErrors[visibleErrorConversationId]
    : null
  const shouldShowThinking = !isTeamRoomView && Boolean(conversationThinking[conversationId])
  const shouldShowHeaderReveal =
    !isTeamRoomView &&
    Boolean(currentHeaderRevealState?.visible) &&
    Boolean(currentHeaderRevealState?.headers)

  return (
    <>
      {shouldShowThinking && (
        <ThinkingAnimation
          onComplete={handleThinkingComplete}
          thinkingProcess={liveThinkingProcess}
        />
      )}

      {shouldShowHeaderReveal && currentHeaderRevealState?.headers && (
        <HeaderReveal
          headers={currentHeaderRevealState.headers}
          onComplete={handleHeaderRevealComplete}
          displayDuration={2000}
        />
      )}

      <div className={`${styles.container} ${isFullscreen ? styles.fullscreen : ''}`}>
        <div className={styles.mainLayout}>
          <ChatComponentSidebarShell
            createDisabled={roomCreateDisabled}
            isOpen={isSidebarOpen}
            isTeamRoomView={isTeamRoomView}
            onCreate={handleTopBarCreate}
            onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
          >
            {!isTeamRoomView ? (
              <ChatConversationSidebar
                conversationId={conversationId}
                conversationPreviews={conversationPreviews}
                onDeleteConversation={handleDeleteConversation}
                onSelectConversation={handleSelectConversation}
              />
            ) : null}
          </ChatComponentSidebarShell>

          <div className={styles.chatArea}>
            {isTeamRoomView ? (
              <ClawRoomChat
                isSidebarOpen={isSidebarOpen}
                createRoomRequestToken={teamRoomCreateToken}
                inputModeControls={
                  <ChatComposerAddMenu
                    clawModeDisabled={modeToggleDisabled}
                    clawModeEnabled={enableClawMode}
                    clawRoom={{
                      active: true,
                      disabled: modeToggleDisabled,
                      onToggle: handleToggleTeamView,
                    }}
                    onToggleClawMode={handleToggleClawMode}
                    webSearchDisabled
                    webSearchEnabled
                    webSearchLocked
                  />
                }
              />
            ) : (
              <>
                <ChatComponentErrors
                  onDismissError={() => {
                    if (visibleErrorConversationId) {
                      setConversationError(visibleErrorConversationId, null)
                    }
                  }}
                  onRetryRoutingModelDiscovery={retryRoutingModelDiscovery}
                  routingModelStatus={routingModelStatus}
                  visibleError={visibleError}
                />
                <ChatComponentConversationViewport
                  conversationId={conversationId}
                  expandedToolCards={expandedToolCards}
                  messages={messages}
                  onToggleToolCard={handleToggleToolCard}
                />
                <ChatTaskQueue
                  queuedTasks={queuedTasks}
                  onEditTask={handleEditQueuedTask}
                  onDeleteTask={handleDeleteQueuedTask}
                  onReorderTasks={handleReorderQueuedTasks}
                />
                <ChatComponentInputBar
                  attachments={pendingAttachments}
                  attachFilesDisabled={readonlyLoading || serverReadonly || hasActiveProbeDraft}
                  enableClawMode={enableClawMode}
                  enableWebSearch={enableWebSearch}
                  inputRef={inputRef}
                  inputValue={inputValue}
                  isLoading={isCurrentConversationRunning}
                  isTogglingClawMode={isTogglingClawMode}
                  modeToggleDisabled={modeToggleDisabled}
                  modelOptions={routingModels}
                  modelSelectDisabled={!isRoutingModelReady || isCurrentConversationRunning}
                  selectedModel={model}
                  voiceInputDisabled={
                    isCurrentConversationRunning || readonlyLoading || serverReadonly
                  }
                  webSearchDisabled={hasActiveProbeDraft}
                  onAttachFiles={handleAttachFiles}
                  onChangeInput={setInputValue}
                  onKeyDown={handleKeyDown}
                  onModelChange={setModel}
                  onRemoveAttachment={handleRemoveAttachment}
                  onSend={handleSend}
                  onStop={handleStop}
                  onToggleClawMode={handleToggleClawMode}
                  onToggleClawRoom={handleToggleTeamView}
                  onToggleWebSearch={() => setEnableWebSearch((prev) => !prev)}
                  sendDisabled={!isRoutingModelReady}
                  sendDisabledReason={
                    routingModelStatus === 'error'
                      ? 'Retry model discovery before sending'
                      : 'Discovering an available router model'
                  }
                  showClawRoom={enableClawMode}
                />
              </>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

export default ChatComponent
