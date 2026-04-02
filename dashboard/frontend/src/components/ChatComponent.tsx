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
  CLAW_TOOL_NAME_PREFIX,
  type ConversationPreview,
  generateConversationId,
  generateMessageId,
  generatePlaygroundTaskId,
  type PlaygroundTask,
  type Message,
} from './ChatComponentTypes'
import { useToolRegistry } from '../tools'
import { useMCPToolSync } from '../tools/mcp'
import { ensureOpenClawServerConnected } from '../tools/mcp/api'
import { useConversationStorage, usePlaygroundQueue } from '../hooks'
import { useReadonly } from '../contexts/ReadonlyContext'

interface ChatComponentProps {
  endpoint?: string
  isFullscreenMode?: boolean
}

type ClawPlaygroundView = 'control' | 'room'

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  isFullscreenMode = false,
}: ChatComponentProps) => {
  const [conversationMessages, setConversationMessages] = useState<Record<string, Message[]>>({})
  const [conversationId, setConversationId] = useState<string>(() => generateConversationId())
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [activeTask, setActiveTask] = useState<PlaygroundTask | null>(null)
  const model = 'MoM' // Fixed to MoM
  const [error, setError] = useState<string | null>(null)
  const [errorConversationId, setErrorConversationId] = useState<string | null>(null)
  const [showThinking, setShowThinking] = useState(false)
  const [showHeaderReveal, setShowHeaderReveal] = useState(false)
  const [pendingHeaders, setPendingHeaders] = useState<Record<string, string> | null>(null)
  const [headerRevealConversationId, setHeaderRevealConversationId] = useState<string | null>(null)
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
  const abortControllerRef = useRef<AbortController | null>(null)
  const hasHydratedConversation = useRef(false)
  const isLoadingRef = useRef(false)
  const activeTaskRef = useRef<PlaygroundTask | null>(null)
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
    () => otherToolDefinitions.filter(def => !def.function.name.startsWith(CLAW_TOOL_NAME_PREFIX)),
    [otherToolDefinitions]
  )
  const clawToolDefinitions = useMemo(
    () => otherToolDefinitions.filter(def => def.function.name.startsWith(CLAW_TOOL_NAME_PREFIX)),
    [otherToolDefinitions]
  )
  const clawManagementDisabled = readonlyLoading || isReadonly
  // When headers arrive, show HeaderReveal
  useEffect(() => {
    if (
      pendingHeaders
      && Object.keys(pendingHeaders).length > 0
      && headerRevealConversationId === conversationId
    ) {
      setShowHeaderReveal(true)
    }
  }, [conversationId, headerRevealConversationId, pendingHeaders])

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
  const nextQueuedTask = useMemo(() => {
    const allQueuedTasks = Object.values(queues).flat()
    if (allQueuedTasks.length === 0) {
      return null
    }

    return allQueuedTasks.reduce<PlaygroundTask>((earliestTask, task) => (
      task.createdAt < earliestTask.createdAt ? task : earliestTask
    ), allQueuedTasks[0])
  }, [queues])
  const generateId = generateMessageId
  const isCurrentConversationRunning = Boolean(
    isLoading && activeTask && activeTask.conversationId === conversationId
  )

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
    setShowHeaderReveal(false)
    setPendingHeaders(null)
    setHeaderRevealConversationId(null)
  }, [])

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
      const deletingActiveConversation = activeTaskRef.current?.conversationId === id

      clearConversationQueue(id)
      deleteConversation(id)
      removeConversationMessages(id)

      if (deletingActiveConversation) {
        abortControllerRef.current?.abort()
        activeTaskRef.current = null
        isLoadingRef.current = false
        setActiveTask(null)
        setIsLoading(false)
        setShowThinking(false)
        setShowHeaderReveal(false)
      }

      if (errorConversationId === id) {
        setError(null)
        setErrorConversationId(null)
      }

      if (headerRevealConversationId === id) {
        setPendingHeaders(null)
        setHeaderRevealConversationId(null)
        setShowHeaderReveal(false)
      }

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
      clearConversationQueue,
      conversationId,
      conversations,
      deleteConversation,
      errorConversationId,
      headerRevealConversationId,
      removeConversationMessages,
    ]
  )

  const executeTask = useCallback((task: PlaygroundTask) => runPlaygroundTask({
    abortControllerRef,
    activeTaskRef,
    buildTaskTools,
    clawManagementDisabled,
    endpoint,
    executeTools,
    expandedToolCardCount: expandedToolCards.size,
    generateId,
    getConversationMessagesSnapshot,
    getCurrentConversationId: () => conversationIdRef.current,
    isLoadingRef,
    setActiveTask,
    setError,
    setErrorConversationId,
    setExpandedToolCards,
    setHeaderRevealConversationId,
    setIsLoading,
    setPendingHeaders,
    setShowHeaderReveal,
    setShowThinking,
    task,
    updateConversationMessages,
  }), [
    activeTaskRef,
    buildTaskTools,
    clawManagementDisabled,
    endpoint,
    executeTools,
    expandedToolCards.size,
    generateId,
    getConversationMessagesSnapshot,
    isLoadingRef,
    setActiveTask,
    setError,
    setErrorConversationId,
    setExpandedToolCards,
    setHeaderRevealConversationId,
    setIsLoading,
    setPendingHeaders,
    setShowHeaderReveal,
    setShowThinking,
    updateConversationMessages,
  ])

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

    setError(null)
    setErrorConversationId(null)
    setInputValue('')

    if (!isLoadingRef.current && !activeTaskRef.current && !nextQueuedTask) {
      activeTaskRef.current = nextTask
      setActiveTask(nextTask)
      void executeTask(nextTask)
      return
    }
    enqueueTask(nextTask)
  }, [
    buildTaskRequestOptions,
    conversationId,
    conversations,
    enqueueTask,
    executeTask,
    getConversationMessagesSnapshot,
    inputValue,
    nextQueuedTask,
    saveConversation,
  ])

  useEffect(() => {
    if (isLoadingRef.current || activeTaskRef.current || !nextQueuedTask) {
      return
    }
    removeQueuedTask(nextQueuedTask.conversationId, nextQueuedTask.id)
    activeTaskRef.current = nextQueuedTask
    setActiveTask(nextQueuedTask)
    void executeTask(nextQueuedTask)
  }, [
    executeTask,
    nextQueuedTask,
    removeQueuedTask,
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
    if (activeTaskRef.current?.conversationId !== conversationId) {
      return
    }

    abortControllerRef.current?.abort()
    isLoadingRef.current = false
    setIsLoading(false)
  }

  const handleNewConversation = useCallback(() => {
    setInputValue('')
    setExpandedToolCards(new Set())
    setConversationId(generateConversationId())
  }, [])

  const handleToggleClawMode = useCallback(() => {
    if (isLoading || isTogglingClawMode) return
    if (enableClawMode) {
      setEnableClawMode(false)
      setError(null)
      return
    }
    setEnableClawMode(true)
    setError(null)
  }, [enableClawMode, isLoading, isTogglingClawMode])

  const isTeamRoomView = enableClawMode && clawView === 'room', roomCreateDisabled = isTeamRoomView && clawManagementDisabled
  const modeToggleDisabled = isLoading || isTogglingClawMode || readonlyLoading

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
  const visibleError = errorConversationId === conversationId ? error : null
  const shouldShowThinking = !isTeamRoomView && showThinking && activeTask?.conversationId === conversationId
  const shouldShowHeaderReveal = !isTeamRoomView
    && showHeaderReveal
    && pendingHeaders
    && headerRevealConversationId === conversationId

  return (
    <>
      {shouldShowThinking && (
        <ThinkingAnimation
          onComplete={handleThinkingComplete}
          thinkingProcess={liveThinkingProcess}
        />
      )}

      {shouldShowHeaderReveal && pendingHeaders && (
        <HeaderReveal
          headers={pendingHeaders}
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
                        setError(null)
                        setErrorConversationId(null)
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
