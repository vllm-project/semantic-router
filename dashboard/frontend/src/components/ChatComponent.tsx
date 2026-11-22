import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import ChatConversationSidebar from './ChatConversationSidebar'
import ChatComponentConversationViewport from './ChatComponentConversationViewport'
import ChatComponentErrors from './ChatComponentErrors'
import ChatComponentHeader from './ChatComponentHeader'
import ChatComponentInputBar from './ChatComponentInputBar'
import ChatComponentSidebarShell from './ChatComponentSidebarShell'
import ChatTaskQueue from './ChatTaskQueue'
import PlaygroundInspector from './PlaygroundInspector'
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
import { usePlaygroundPanes } from './usePlaygroundPanes'
import { sanitizeMessagesForPersistence } from './chatPersistenceSupport'

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  feedbackInsightsBasePath,
  invocation = null,
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
  const [enableWebSearch, setEnableWebSearch] = useState(true)
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())
  const [pinnedInspectId, setPinnedInspectId] = useState<string | null>(null)
  const panes = usePlaygroundPanes()
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
    if (!areConversationsHydrated || !hasHydratedConversation.current) return
    writeActiveConversationPreference(conversationId)
  }, [areConversationsHydrated, conversationId])

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
      setPinnedInspectId(null)
      if (panes.conversationsOverlay) panes.setConversationsOpen(false)
    },
    [conversations, panes],
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

  // A new turn returns the inspector to following the newest answer.
  const sendMessage = () => {
    setPinnedInspectId(null)
    handleSend()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
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
    setPinnedInspectId(null)
    setConversationId(generateConversationId())
  }, [clearPendingAttachments])

  const handleInspectMessage = useCallback(
    (messageId: string) => {
      setPinnedInspectId(messageId)
      panes.setInspectorOpen(true)
    },
    [panes],
  )

  const handleSuggestion = useCallback(
    (prompt: string) => {
      setInputValue(prompt)
      focusComposer()
    },
    [focusComposer],
  )

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
  const runningConversationIds = useMemo(() => new Set(Object.keys(activeTasks)), [activeTasks])
  const inspectedMessage = useMemo(() => {
    const answers = messages.filter((message) => message.role === 'assistant')
    return (
      (pinnedInspectId && answers.find((message) => message.id === pinnedInspectId)) ||
      [...answers].reverse().find((message) => message.headers) ||
      answers[answers.length - 1] ||
      null
    )
  }, [messages, pinnedInspectId])
  const inspectedPrompt = useMemo(() => {
    if (!inspectedMessage) return undefined
    const index = messages.findIndex((message) => message.id === inspectedMessage.id)
    for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
      if (messages[cursor].role === 'user') return messages[cursor].content
    }
    return undefined
  }, [inspectedMessage, messages])
  const selectedRecipe = routingModels.find((option) => option.id === model)?.recipe

  return (
    <div className="relative flex min-h-0 flex-1 overflow-hidden" data-testid="playground">
      <ChatComponentSidebarShell
        isOpen={panes.conversationsOpen}
        overlay={panes.conversationsOverlay}
        onCreate={handleNewConversation}
        onClose={() => panes.setConversationsOpen(false)}
      >
        <ChatConversationSidebar
          conversationId={conversationId}
          conversationPreviews={conversationPreviews}
          runningConversationIds={runningConversationIds}
          onDeleteConversation={handleDeleteConversation}
          onRenameConversation={handleRenameConversation}
          onSelectConversation={handleSelectConversation}
        />
      </ChatComponentSidebarShell>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <ChatComponentHeader
          conversationsOpen={panes.conversationsOpen}
          inspectorOpen={panes.inspectorOpen}
          models={routingModels}
          model={model}
          recipe={selectedRecipe}
          modelSelectDisabled={!isRoutingModelReady || isCurrentConversationRunning}
          onModelChange={setModel}
          onNewConversation={handleNewConversation}
          onToggleConversations={() => panes.setConversationsOpen(!panes.conversationsOpen)}
          onToggleInspector={() => panes.setInspectorOpen(!panes.inspectorOpen)}
        />
        <div className="relative flex min-h-0 flex-1 flex-col">
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
            inspectedMessageId={panes.inspectorOpen ? inspectedMessage?.id : undefined}
            messages={messages}
            feedbackInsightsBasePath={feedbackInsightsBasePath}
            onInspectMessage={handleInspectMessage}
            onSuggestion={handleSuggestion}
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
            voiceInputDisabled={isCurrentConversationRunning || readonlyLoading || serverReadonly}
            webSearchDisabled={hasActiveProbeDraft}
            onAttachFiles={handleAttachFiles}
            onChangeInput={setInputValue}
            onKeyDown={handleKeyDown}
            onRemoveAttachment={handleRemoveAttachment}
            onSend={sendMessage}
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

      <PlaygroundInspector
        open={panes.inspectorOpen}
        overlay={panes.inspectorOverlay}
        message={inspectedMessage}
        prompt={inspectedPrompt}
        insightsBasePath={feedbackInsightsBasePath}
        onClose={() => panes.setInspectorOpen(false)}
      />
    </div>
  )
}

export default ChatComponent
