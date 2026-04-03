import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import styles from './ChatComponent.module.css'
import ThinkingAnimation from './ThinkingAnimation'
import HeaderReveal from './HeaderReveal'
import ClawRoomChat from './ClawRoomChat'
import { ClawModeToggle } from './ChatComponentControls'
import ChatConversationSidebar from './ChatConversationSidebar'
import ChatComponentConversationViewport from './ChatComponentConversationViewport'
import ChatComponentInputBar from './ChatComponentInputBar'
import ChatComponentRoomToggle from './ChatComponentRoomToggle'
import ChatComponentSidebarShell from './ChatComponentSidebarShell'
import ChatTaskQueue from './ChatTaskQueue'
import { runPlaygroundTask } from './chatTaskExecution'
import {
  CLAW_MODE_STORAGE_KEY,
  type ConversationPreview,
  generateConversationId,
  generateMessageId,
  generatePlaygroundTaskId,
  type PlaygroundTask,
  type Message,
} from './ChatComponentTypes'
import { useToolRegistry } from '../tools'
import { isOpenClawMCPToolName, useMCPToolSync } from '../tools/mcp'
import { ensureOpenClawServerConnected } from '../tools/mcp/api'
import { useConversationStorage, usePlaygroundQueue } from '../hooks'
import { useReadonly } from '../contexts/ReadonlyContext'

interface ChatComponentProps {
  endpoint?: string
  isFullscreenMode?: boolean
}

type ClawPlaygroundView = 'control' | 'room'

interface ConversationHeaderRevealState {
  headers: Record<string, string>
  visible: boolean
}

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  isFullscreenMode = false,
}: ChatComponentProps) => {
  const [conversationMessages, setConversationMessages] = useState<Record<string, Message[]>>({})
  const [conversationId, setConversationId] = useState<string>(() => generateConversationId())
  const [inputValue, setInputValue] = useState('')
  const [activeTasks, setActiveTasks] = useState<Record<string, PlaygroundTask>>({})
  const model = 'MoM' // Fixed to MoM
  const [conversationErrors, setConversationErrors] = useState<Record<string, string>>({})
  const [conversationThinking, setConversationThinking] = useState<Record<string, boolean>>({})
  const [headerRevealStates, setHeaderRevealStates] = useState<Record<string, ConversationHeaderRevealState>>({})
  const [isFullscreen] = useState(isFullscreenMode)
  const [enableWebSearch, setEnableWebSearch] = useState(true)
  const [enableClawMode, setEnableClawMode] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    const saved = window.localStorage.getItem(CLAW_MODE_STORAGE_KEY)
    if (saved === null) return false
    return saved === 'true'
  })
  const [isTogglingClawMode, setIsTogglingClawMode] = useState(false)
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [clawView, setClawView] = useState<ClawPlaygroundView>(() => 'control')
  const [teamRoomCreateToken, setTeamRoomCreateToken] = useState(0)
  const { isReadonly, isLoading: readonlyLoading } = useReadonly()

  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllersRef = useRef<Record<string, AbortController>>({})
  const hasHydratedConversation = useRef(false)
  const activeTasksRef = useRef<Record<string, PlaygroundTask>>({})
  const conversationIdRef = useRef(conversationId)
  const conversationMessagesRef = useRef<Record<string, Message[]>>({})

  const { conversations, saveConversation, getConversation, deleteConversation } = useConversationStorage<Message[]>({
    storageKey: 'sr:chat:conversations',
    maxConversations: 20,
  })
  const {
    clearConversationQueue,
    enqueueTask,
    getQueue,
    queues,
    removeTask: removeQueuedTask,
    reorderTasks,
  } = usePlaygroundQueue()

  const setConversationError = useCallback((targetConversationId: string, error: string | null) => {
    setConversationErrors(prev => {
      if (!error) {
        if (!(targetConversationId in prev)) {
          return prev
        }
        const next = { ...prev }
        delete next[targetConversationId]
        return next
      }
      if (prev[targetConversationId] === error) {
        return prev
      }
      return {
        ...prev,
        [targetConversationId]: error,
      }
    })
  }, [])

  const setConversationThinkingState = useCallback((targetConversationId: string, visible: boolean) => {
    setConversationThinking(prev => {
      const current = prev[targetConversationId] ?? false
      if (current === visible) {
        return prev
      }
      if (!visible) {
        if (!(targetConversationId in prev)) {
          return prev
        }
        const next = { ...prev }
        delete next[targetConversationId]
        return next
      }
      return {
        ...prev,
        [targetConversationId]: true,
      }
    })
  }, [])

  const setConversationHeaderReveal = useCallback((
    targetConversationId: string,
    headers: Record<string, string> | null,
    visible = false
  ) => {
    setHeaderRevealStates(prev => {
      if (!headers || Object.keys(headers).length === 0) {
        if (!(targetConversationId in prev)) {
          return prev
        }
        const next = { ...prev }
        delete next[targetConversationId]
        return next
      }
      const current = prev[targetConversationId]
      if (current && current.visible === visible && current.headers === headers) {
        return prev
      }
      return {
        ...prev,
        [targetConversationId]: {
          headers,
          visible,
        },
      }
    })
  }, [])

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

  const clearActiveTaskForConversation = useCallback((targetConversationId: string, taskId: string) => {
    const currentTask = activeTasksRef.current[targetConversationId]
    if (!currentTask || currentTask.id !== taskId) {
      return
    }
    const next = { ...activeTasksRef.current }
    delete next[targetConversationId]
    activeTasksRef.current = next
    setActiveTasks(next)
  }, [])

  const registerAbortController = useCallback((targetConversationId: string, controller: AbortController | null) => {
    if (controller) {
      abortControllersRef.current[targetConversationId] = controller
      return
    }
    delete abortControllersRef.current[targetConversationId]
  }, [])

  const restoreMessages = useCallback((payload: Message[]) => {
    return payload.map(message => ({
      ...message,
      timestamp: new Date(message.timestamp),
    }))
  }, [])

  const getStoredMessagesForConversation = useCallback((id: string): Message[] => {
    const storedConversation = getConversation(id)
    if (!storedConversation?.payload || !Array.isArray(storedConversation.payload)) {
      return []
    }
    return restoreMessages(storedConversation.payload)
  }, [getConversation, restoreMessages])

  const updateConversationMessages = useCallback(
    (targetConversationId: string, updater: (prev: Message[]) => Message[]) => {
      setConversationMessages(prev => {
        const baseMessages = prev[targetConversationId] ?? getStoredMessagesForConversation(targetConversationId)
        const nextMessages = updater(baseMessages)
        if (nextMessages === baseMessages) {
          return prev
        }
        return {
          ...prev,
          [targetConversationId]: nextMessages,
        }
      })
    },
    [getStoredMessagesForConversation]
  )

  const removeConversationMessages = useCallback((targetConversationId: string) => {
    setConversationMessages(prev => {
      if (!(targetConversationId in prev)) {
        return prev
      }
      const next = { ...prev }
      delete next[targetConversationId]
      return next
    })
  }, [])

  const getConversationMessagesSnapshot = useCallback((targetConversationId: string) => (
    conversationMessagesRef.current[targetConversationId] ?? getStoredMessagesForConversation(targetConversationId)
  ), [getStoredMessagesForConversation])

  useEffect(() => {
    conversationIdRef.current = conversationId
  }, [conversationId])

  useEffect(() => {
    conversationMessagesRef.current = conversationMessages
  }, [conversationMessages])

  // MCP 工具同步 - 自动将 MCP 服务器的工具同步到 toolRegistry
  const { refresh: refreshMCPTools } = useMCPToolSync({ enabled: true, pollInterval: 30000 })

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

  const baseOtherToolDefinitions = useMemo(
    () => otherToolDefinitions.filter(def => !isOpenClawMCPToolName(def.function.name)),
    [otherToolDefinitions]
  )
  const clawToolDefinitions = useMemo(
    () => otherToolDefinitions.filter(def => isOpenClawMCPToolName(def.function.name)),
    [otherToolDefinitions]
  )
  const clawManagementDisabled = readonlyLoading || isReadonly
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
    if (typeof window === 'undefined') return
    window.localStorage.setItem(CLAW_MODE_STORAGE_KEY, String(enableClawMode))
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
        console.warn(`[ClawOS] UI mode enabled, but MCP bootstrap failed: ${message}`)
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

    const restoredConversationMessages = conversations.reduce<Record<string, Message[]>>((acc, conv) => {
      if (Array.isArray(conv.payload)) {
        acc[conv.id] = restoreMessages(conv.payload)
      }
      return acc
    }, {})

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

  const conversationPreviews = useMemo<ConversationPreview[]>(() => {
    return [...conversations]
      .sort((a, b) => a.createdAt - b.createdAt)
      .map(conv => {
        const firstUserMessage = Array.isArray(conv.payload)
          ? conv.payload.find(msg => msg.role === 'user')
          : undefined
        const title = (firstUserMessage?.content || 'New conversation').trim()
        const preview = title.length > 60 ? `${title.slice(0, 60)}…` : title || 'New conversation'

        return {
          id: conv.id,
          updatedAt: conv.updatedAt || conv.createdAt,
          preview,
        }
      })
  }, [conversations])

  const messages = useMemo(
    () => conversationMessages[conversationId] ?? getStoredMessagesForConversation(conversationId),
    [conversationId, conversationMessages, getStoredMessagesForConversation]
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
    [clawManagementDisabled, enableClawMode, enableWebSearch]
  )

  const buildTaskTools = useCallback(
    (task: PlaygroundTask) => {
      const otherTools = task.requestOptions.enableClawMode && !clawManagementDisabled
        ? [...baseOtherToolDefinitions, ...clawToolDefinitions]
        : baseOtherToolDefinitions

      return [
        ...otherTools,
        ...(task.requestOptions.enableWebSearch ? searchToolDefinitions : []),
      ]
    },
    [
      baseOtherToolDefinitions,
      clawManagementDisabled,
      clawToolDefinitions,
      searchToolDefinitions,
    ]
  )

  const handleThinkingComplete = useCallback(() => {}, [])

  const handleHeaderRevealComplete = useCallback(() => {
    setConversationHeaderReveal(conversationId, null)
  }, [conversationId, setConversationHeaderReveal])

  const handleSelectConversation = useCallback(
    (id: string) => {
      const target = conversations.find(conv => conv.id === id)
      if (!target) return

      setConversationId(target.id)
      setInputValue('')
      setExpandedToolCards(new Set())
    },
    [conversations]
  )

  const handleDeleteConversation = useCallback(
    (id: string) => {
      const remaining = conversations.filter(conv => conv.id !== id)
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
    ]
  )

  const executeTask = useCallback((task: PlaygroundTask) => runPlaygroundTask({
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
  }), [
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
  ])

  const startTask = useCallback((task: PlaygroundTask) => {
    if (activeTasksRef.current[task.conversationId]) {
      return
    }

    setActiveTaskForConversation(task)
    void executeTask(task)
  }, [executeTask, setActiveTaskForConversation])

  const handleSend = useCallback(() => {
    const trimmedInput = inputValue.trim()
    if (!trimmedInput) return

    const nextTask: PlaygroundTask = {
      id: generatePlaygroundTaskId(),
      conversationId,
      prompt: trimmedInput,
      createdAt: Date.now(),
      requestOptions: buildTaskRequestOptions(),
    }

    if (!conversations.some(conv => conv.id === conversationId)) {
      hasHydratedConversation.current = true
      saveConversation(conversationId, getConversationMessagesSnapshot(conversationId))
    }

    setConversationError(conversationId, null)
    setInputValue('')

    if (!activeTasksRef.current[conversationId]) {
      startTask(nextTask)
      return
    }
    enqueueTask(nextTask)
  }, [
    buildTaskRequestOptions,
    conversationId,
    conversations,
    enqueueTask,
    getConversationMessagesSnapshot,
    inputValue,
    saveConversation,
    setConversationError,
    startTask,
  ])

  useEffect(() => {
    Object.entries(queues).forEach(([targetConversationId, queue]) => {
      if (queue.length === 0 || activeTasksRef.current[targetConversationId]) {
        return
      }

      const nextTask = queue.reduce<PlaygroundTask>((earliestTask, task) => (
        task.createdAt < earliestTask.createdAt ? task : earliestTask
      ), queue[0])

      removeQueuedTask(targetConversationId, nextTask.id)
      startTask(nextTask)
    })
  }, [
    activeTasks,
    executeTask,
    queues,
    removeQueuedTask,
    startTask,
  ])

  const handleDeleteQueuedTask = useCallback((taskId: string) => {
    removeQueuedTask(conversationId, taskId)
  }, [conversationId, removeQueuedTask])

  const handleEditQueuedTask = useCallback((taskId: string) => {
    const taskToEdit = queuedTasks.find(task => task.id === taskId)
    if (!taskToEdit) {
      return
    }

    removeQueuedTask(conversationId, taskId)
    setInputValue(taskToEdit.prompt)

    if (typeof window !== 'undefined') {
      window.requestAnimationFrame(() => {
        inputRef.current?.focus()
        const promptLength = taskToEdit.prompt.length
        inputRef.current?.setSelectionRange(promptLength, promptLength)
      })
    }
  }, [conversationId, queuedTasks, removeQueuedTask])

  const handleReorderQueuedTasks = useCallback((sourceTaskId: string, targetTaskId: string) => {
    reorderTasks(conversationId, sourceTaskId, targetTaskId)
  }, [conversationId, reorderTasks])

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
    setExpandedToolCards(new Set())
    setConversationId(generateConversationId())
  }, [])

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

  const isTeamRoomView = enableClawMode && clawView === 'room', roomCreateDisabled = isTeamRoomView && clawManagementDisabled
  const modeToggleDisabled = hasRunningTasks || isTogglingClawMode || readonlyLoading

  const handleToggleTeamView = useCallback(() => { if (!enableClawMode || modeToggleDisabled) return; setClawView(prev => (prev === 'room' ? 'control' : 'room')) }, [enableClawMode, modeToggleDisabled])

  const handleTopBarCreate = useCallback(() => {
    if (roomCreateDisabled) return
    if (isTeamRoomView) {
      setTeamRoomCreateToken(prev => prev + 1)
      return
    }
    handleNewConversation()
  }, [handleNewConversation, isTeamRoomView, roomCreateDisabled])

  const handleToggleToolCard = useCallback((toolCallId: string) => {
    setExpandedToolCards(prev => {
      const next = new Set(prev)
      if (next.has(toolCallId)) {
        next.delete(toolCallId)
      } else {
        next.add(toolCallId)
      }
      return next
    })
  }, [])

  const roomChatToggleControl = enableClawMode
    ? <ChatComponentRoomToggle disabled={modeToggleDisabled} isTeamRoomView={isTeamRoomView} onToggle={handleToggleTeamView} />
    : null
  const liveThinkingProcess = messages.reduceRight((thinking, message) =>
    thinking || (message.role === 'assistant' && message.isStreaming ? message.thinkingProcess || '' : ''), '')
  const visibleError = conversationErrors[conversationId] ?? null
  const shouldShowThinking = !isTeamRoomView && Boolean(conversationThinking[conversationId])
  const shouldShowHeaderReveal = !isTeamRoomView
    && Boolean(currentHeaderRevealState?.visible)
    && Boolean(currentHeaderRevealState?.headers)

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
            onToggleSidebar={() => setIsSidebarOpen(prev => !prev)}
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
                inputModeControls={(
                  <>
                    <button
                      type="button"
                      className={`${styles.inputActionButton} ${styles.searchToggleActive}`}
                      onClick={event => event.preventDefault()}
                      data-tooltip="Web Search enabled in Room Chat"
                      aria-label="Web Search enabled in Room Chat"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M2 12h20" />
                        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                      </svg>
                    </button>
                    <ClawModeToggle
                      enabled={enableClawMode}
                      onToggle={handleToggleClawMode}
                      disabled={modeToggleDisabled}
                    />
                    {roomChatToggleControl}
                  </>
                )}
              />
            ) : (
              <>
                {visibleError && (
                  <div className={styles.error}>
                    <span className={styles.errorIcon}>⚠️</span>
                    <span>{visibleError}</span>
                    <button
                      className={styles.errorDismiss}
                      onClick={() => {
                        setConversationError(conversationId, null)
                      }}
                    >
                      ×
                    </button>
                  </div>
                )}
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
                  enableClawMode={enableClawMode}
                  enableWebSearch={enableWebSearch}
                  inputRef={inputRef}
                  inputValue={inputValue}
                  isLoading={isCurrentConversationRunning}
                  isTogglingClawMode={isTogglingClawMode}
                  modeToggleDisabled={modeToggleDisabled}
                  onChangeInput={setInputValue}
                  onKeyDown={handleKeyDown}
                  onSend={handleSend}
                  onStop={handleStop}
                  onToggleClawMode={handleToggleClawMode}
                  onToggleWebSearch={() => setEnableWebSearch(prev => !prev)}
                  roomChatToggleControl={roomChatToggleControl}
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
