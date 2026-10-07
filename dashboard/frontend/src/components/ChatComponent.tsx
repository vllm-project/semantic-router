import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import styles from './ChatComponent.module.css'
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
  findQueuedErrorConversationId,
  getLiveThinkingProcess,
  readActiveConversationPreference,
  writeActiveConversationPreference,
} from './chatComponentSupport'
import { useToolRegistry } from '../tools'
import { useMCPToolSync } from '../tools/mcp'
import { useConversationStorage, usePlaygroundQueue } from '../hooks'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canSubmitFeedback } from '../utils/accessControl'
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
  feedbackInsightsBasePath,
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
    setConversationError,
    setConversationThinkingState,
  } = useChatConversationState()
  const [isFullscreen] = useState(isFullscreenMode)
  const [enableWebSearch, setEnableWebSearch] = useState(true)
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const { user } = useAuth()
  const { serverReadonly, isLoading: readonlyLoading } = useReadonly()

  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllersRef = useRef<Record<string, AbortController>>({})
  const hasHydratedConversation = useRef(false)
  const activeTasksRef = useRef<Record<string, PlaygroundTask>>({})
  const conversationIdRef = useRef(conversationId)
  const persistedConversationMessagesRef = useRef<Record<string, Message[]>>({})
  const hydratingConversationMessagesRef = useRef<Record<string, Message[]> | null>(null)

  const {
    conversations,
    isHydrated: areConversationsHydrated,
    saveConversation,
    getConversation,
    deleteConversation,
    renameConversation,
  } = useConversationStorage<Message[]>({
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

  // MCP 工具同步 - 自动将 MCP 服务器的工具同步到 toolRegistry
  useMCPToolSync({ enabled: true, pollInterval: 30000 })

  // Tool Registry integration
  // Search tools (controlled by web search toggle)
  const { definitions: searchToolDefinitions } = useToolRegistry({
    enabledOnly: true,
    categories: ['search'],
  })
  // Other tools (always available, not controlled by web search toggle)
  const { definitions: otherToolDefinitions, executeAll: executeTools } = useToolRegistry({
    enabledOnly: true,
    categories: ['code', 'file', 'image', 'custom'],
  })

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

  // Hydrate saved conversations once. Only restore a conversation the user
  // explicitly selected; otherwise keep the stable blank starting state.
  useEffect(() => {
    if (hasHydratedConversation.current || !areConversationsHydrated) return

    const restoredConversationMessages = conversations.reduce<Record<string, Message[]>>(
      (acc, conv) => {
        if (Array.isArray(conv.payload)) {
          acc[conv.id] = restoreMessages(conv.payload)
        }
        return acc
      },
      {},
    )

    hydratingConversationMessagesRef.current = restoredConversationMessages
    setConversationMessages(restoredConversationMessages)

    const selectedConversationId = readActiveConversationPreference(conversations)
    if (selectedConversationId) {
      conversationIdRef.current = selectedConversationId
      setConversationId(selectedConversationId)
    }

    hasHydratedConversation.current = true
  }, [areConversationsHydrated, conversations, restoreMessages])

  useEffect(() => {
    if (!hasHydratedConversation.current) return
    writeActiveConversationPreference(conversationId)
  }, [conversationId])

  // Persist changed conversations whenever in-memory messages change
  useEffect(() => {
    const hydratedMessages = hydratingConversationMessagesRef.current
    if (hydratedMessages) {
      if (conversationMessages === hydratedMessages) {
        persistedConversationMessagesRef.current = conversationMessages
        hydratingConversationMessagesRef.current = null
      }
      return
    }

    Object.entries(conversationMessages).forEach(([id, payload]) => {
      if (payload.length === 0 || persistedConversationMessagesRef.current[id] === payload) {
        return
      }
      saveConversation(id, payload)
    })
    persistedConversationMessagesRef.current = conversationMessages
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
  const isCurrentConversationRunning = Boolean(activeConversationTask)

  const buildTaskRequestOptions = useCallback(
    () => ({
      enableWebSearch,
      model,
    }),
    [enableWebSearch, model],
  )

  const buildTaskTools = useCallback(
    (task: PlaygroundTask) => [
      ...otherToolDefinitions,
      ...(task.requestOptions.enableWebSearch ? searchToolDefinitions : []),
    ],
    [otherToolDefinitions, searchToolDefinitions],
  )

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
      setConversationThinkingState,
    ],
  )

  const handleRenameConversation = useCallback(
    (id: string, title: string) => renameConversation(id, title),
    [renameConversation],
  )

  const executeTask = useCallback(
    (task: PlaygroundTask) =>
      runPlaygroundTask({
        buildTaskTools,
        clearConversationActiveTask: clearActiveTaskForConversation,
        endpoint,
        executeTools,
        expandedToolCardCount: expandedToolCards.size,
        generateId,
        getConversationMessagesSnapshot,
        registerAbortController,
        setConversationError,
        setConversationThinking: setConversationThinkingState,
        setExpandedToolCards,
        task,
        updateConversationMessages,
      }),
    [
      buildTaskTools,
      clearActiveTaskForConversation,
      endpoint,
      executeTools,
      expandedToolCards.size,
      generateId,
      getConversationMessagesSnapshot,
      registerAbortController,
      setConversationError,
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

  const hasActiveProbeDraft = probeDraft?.conversationId === conversationId

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
  const shouldShowThinking = Boolean(conversationThinking[conversationId])
  const isConversationEmpty = messages.length === 0 && !shouldShowThinking
  return (
    <>
      <div className={`${styles.container} ${isFullscreen ? styles.fullscreen : ''}`}>
        <div className={styles.mainLayout}>
          <ChatComponentSidebarShell
            isOpen={isSidebarOpen}
            onCreate={handleNewConversation}
            onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
          >
            <ChatConversationSidebar
              conversationId={conversationId}
              conversationPreviews={conversationPreviews}
              onDeleteConversation={handleDeleteConversation}
              onRenameConversation={handleRenameConversation}
              onSelectConversation={handleSelectConversation}
            />
          </ChatComponentSidebarShell>

          <div className={`${styles.chatArea} ${isConversationEmpty ? styles.chatAreaEmpty : ''}`}>
            <ChatComponentErrors
              overlay={isConversationEmpty}
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
              canSubmitFeedback={canSubmitFeedback(user)}
              conversationId={conversationId}
              expandedToolCards={expandedToolCards}
              messages={messages}
              feedbackInsightsBasePath={feedbackInsightsBasePath}
              onToggleToolCard={handleToggleToolCard}
              thinking={shouldShowThinking}
              thinkingProcess={liveThinkingProcess}
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
              enableWebSearch={enableWebSearch}
              inputRef={inputRef}
              inputValue={inputValue}
              isLoading={isCurrentConversationRunning}
              modelOptions={routingModels}
              modelSelectDisabled={!isRoutingModelReady || isCurrentConversationRunning}
              selectedModel={model}
              voiceInputDisabled={isCurrentConversationRunning || readonlyLoading || serverReadonly}
              webSearchDisabled={hasActiveProbeDraft}
              onAttachFiles={handleAttachFiles}
              onChangeInput={setInputValue}
              onKeyDown={handleKeyDown}
              onModelChange={setModel}
              onRemoveAttachment={handleRemoveAttachment}
              onSend={handleSend}
              onStop={handleStop}
              onToggleWebSearch={() => setEnableWebSearch((prev) => !prev)}
              sendDisabled={!isRoutingModelReady}
              sendDisabledReason={
                routingModelStatus === 'error'
                  ? 'Retry model discovery before sending'
                  : 'Discovering an available router model'
              }
            />
          </div>
        </div>
      </div>
    </>
  )
}

export default ChatComponent
