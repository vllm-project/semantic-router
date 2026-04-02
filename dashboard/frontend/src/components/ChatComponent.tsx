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
import {
  buildChoicesArray,
  consumeEventStream,
  getFirstChoice,
  isEventStreamContentType,
  mergeParsedChoices,
  parseChatCompletionPayload,
  type ChoiceAccumulator,
  type ParsedChatCompletion,
  type ParsedToolCallChunk,
} from './chatResponseParsing'
import {
  buildChatMessages,
  buildChatRequestBody,
  collectResponseHeaders,
  type OutboundChatMessage,
} from './chatRequestSupport'
import {
  CLAW_MODE_STORAGE_KEY,
  CLAW_TOOL_NAME_PREFIX,
  type Choice,
  type ConversationPreview,
  generateConversationId,
  generateMessageId,
  generatePlaygroundTaskId,
  type Message,
  type PlaygroundTask,
  type ReMoMRoundResponse,
} from './ChatComponentTypes'
import { useToolRegistry } from '../tools'
import { useMCPToolSync } from '../tools/mcp'
import { ensureOpenClawServerConnected } from '../tools/mcp/api'
import { useConversationStorage, usePlaygroundQueue } from '../hooks'
import { useReadonly } from '../contexts/ReadonlyContext'
import type { ToolCall, ToolResult } from '../tools'
import { serializeToolResultForModel } from '../tools/toolResultSupport'

interface ChatComponentProps {
  endpoint?: string
  isFullscreenMode?: boolean
}

type ClawPlaygroundView = 'control' | 'room'

const ChatComponent = ({
  endpoint = '/api/router/v1/chat/completions',
  isFullscreenMode = false,
}: ChatComponentProps) => {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversationId, setConversationId] = useState<string>(() => generateConversationId())
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [activeTask, setActiveTask] = useState<PlaygroundTask | null>(null)
  const model = 'MoM' // Fixed to MoM
  const [error, setError] = useState<string | null>(null)
  const [showThinking, setShowThinking] = useState(false)
  const [showHeaderReveal, setShowHeaderReveal] = useState(false)
  const [pendingHeaders, setPendingHeaders] = useState<Record<string, string> | null>(null)
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
  const messagesRef = useRef<Message[]>([])

  const { conversations, saveConversation, getConversation, deleteConversation } = useConversationStorage<Message[]>({
    storageKey: 'sr:chat:conversations',
    maxConversations: 20,
  })
  const {
    clearConversationQueue,
    enqueueTask,
    getQueue,
    removeTask: removeQueuedTask,
    reorderTasks,
  } = usePlaygroundQueue()

  const restoreMessages = useCallback((payload: Message[]) => {
    return payload.map(message => ({
      ...message,
      timestamp: new Date(message.timestamp),
    }))
  }, [])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

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
    if (pendingHeaders && Object.keys(pendingHeaders).length > 0) {
      setShowHeaderReveal(true)
    }
  }, [pendingHeaders])

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

    const latestConversation = getConversation()
    if (latestConversation?.payload && Array.isArray(latestConversation.payload)) {
      setConversationId(latestConversation.id)
      setMessages(restoreMessages(latestConversation.payload))
    }

    hasHydratedConversation.current = true
  }, [conversations, getConversation, restoreMessages])

  // Persist conversation whenever messages change
  useEffect(() => {
    if (messages.length === 0) return
    saveConversation(conversationId, messages)
  }, [conversationId, messages, saveConversation])

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
  const queuedTasks = useMemo(() => getQueue(conversationId), [conversationId, getQueue])
  const generateId = generateMessageId

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
  }, [])

  const handleSelectConversation = useCallback(
    (id: string) => {
      const target = conversations.find(conv => conv.id === id)
      if (!target) return

      abortControllerRef.current?.abort()
      activeTaskRef.current = null
      isLoadingRef.current = false
      setActiveTask(null)
      setIsLoading(false)
      setConversationId(target.id)
      setMessages(restoreMessages(Array.isArray(target.payload) ? target.payload : []))
      setInputValue('')
      setError(null)
      setPendingHeaders(null)
      setShowHeaderReveal(false)
      setShowThinking(false)
      setExpandedToolCards(new Set())
    },
    [conversations, restoreMessages]
  )

  const handleDeleteConversation = useCallback(
    (id: string) => {
      const remaining = conversations.filter(conv => conv.id !== id)

      clearConversationQueue(id)
      deleteConversation(id)

      if (id === conversationId) {
        abortControllerRef.current?.abort()
        activeTaskRef.current = null
        isLoadingRef.current = false
        setActiveTask(null)
        setIsLoading(false)
        setError(null)
        setPendingHeaders(null)
        setShowHeaderReveal(false)
        setShowThinking(false)
        setExpandedToolCards(new Set())
        setInputValue('')

        const next = remaining[0]
        if (next && Array.isArray(next.payload)) {
          setConversationId(next.id)
          setMessages(restoreMessages(next.payload))
        } else {
          setConversationId(generateConversationId())
          setMessages([])
        }
      }
    },
    [clearConversationQueue, conversationId, conversations, deleteConversation, restoreMessages]
  )

  const executeTask = useCallback(async (task: PlaygroundTask) => {
    const trimmedInput = task.prompt.trim()
    if (!trimmedInput) return

    setError(null)

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    isLoadingRef.current = true
    setIsLoading(true)

    setPendingHeaders(null)
    setShowHeaderReveal(false)
    setShowThinking(true)

    const assistantMessageId = generateId()
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      abortControllerRef.current = new AbortController()

      const activeTools = buildTaskTools(task)
      const chatMessages = buildChatMessages(
        messagesRef.current,
        trimmedInput,
        task.requestOptions.enableClawMode && !clawManagementDisabled
      )
      const requestBody = buildChatRequestBody(task.requestOptions.model, chatMessages, activeTools)

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`API error: ${response.status} - ${errorText}`)
      }

      const responseHeaders = collectResponseHeaders(response)

      if (Object.keys(responseHeaders).length > 0) {
        console.log('Headers received, showing HeaderReveal')
        setPendingHeaders(responseHeaders)
        setShowThinking(false)
        setShowHeaderReveal(true)
      }

      const choiceContents: Map<number, ChoiceAccumulator> = new Map()
      let isRatingsMode = false
      const toolCallsMap: Map<number, ToolCall> = new Map()
      let hasToolCalls = false
      let reasoningMomResponses: ReMoMRoundResponse[] | undefined
      let latestThinkingProcess = ''

      const syncAssistantToolCalls = () => {
        const currentToolCalls = Array.from(toolCallsMap.values())
        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? { ...m, toolCalls: currentToolCalls }
              : m
          )
        )
      }

      const mergeToolCallsIntoState = (
        parsedToolCalls: ParsedToolCallChunk[],
        idPrefix: string,
        status: ToolCall['status']
      ) => {
        if (parsedToolCalls.length === 0) {
          return false
        }

        hasToolCalls = true

        for (const parsedToolCall of parsedToolCalls) {
          const toolCallIndex = parsedToolCall.index
          if (!toolCallsMap.has(toolCallIndex)) {
            toolCallsMap.set(toolCallIndex, {
              id: parsedToolCall.id || `${idPrefix}-${toolCallIndex}`,
              type: 'function',
              function: {
                name: parsedToolCall.functionName || '',
                arguments: ''
              },
              status,
            })
          }

          const existingToolCall = toolCallsMap.get(toolCallIndex)!
          existingToolCall.status = status

          if (parsedToolCall.functionName) {
            existingToolCall.function.name = parsedToolCall.functionName
          }

          if (parsedToolCall.functionArguments) {
            existingToolCall.function.arguments += parsedToolCall.functionArguments
          }

          if (parsedToolCall.id) {
            existingToolCall.id = parsedToolCall.id
          }
        }

        return true
      }

      const syncAssistantChoices = (streaming: boolean) => {
        if (hasToolCalls && !getFirstChoice(choiceContents)?.content) {
          return
        }

        if (isRatingsMode) {
          const choicesArray = buildChoicesArray(choiceContents)
          const thinkingProcess = getFirstChoice(choiceContents)?.reasoningContent || latestThinkingProcess

          if (thinkingProcess) {
            latestThinkingProcess = thinkingProcess
          }

          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? {
                  ...m,
                  content: choicesArray[0]?.content || '',
                  choices: choicesArray,
                  thinkingProcess: thinkingProcess || m.thinkingProcess,
                  isStreaming: streaming,
                }
                : m
            )
          )
          return
        }

        const firstChoice = getFirstChoice(choiceContents)
        if (!firstChoice) {
          return
        }

        if (firstChoice.reasoningContent) {
          latestThinkingProcess = firstChoice.reasoningContent
        }

        setMessages(prev =>
          prev.map(m =>
            m.id === assistantMessageId
              ? {
                ...m,
                content: firstChoice.content,
                thinkingProcess: firstChoice.reasoningContent || m.thinkingProcess,
                isStreaming: streaming,
              }
              : m
          )
        )
      }

      const applyParsedCompletion = (parsedCompletion: ParsedChatCompletion, streaming: boolean) => {
        if (parsedCompletion.reasoningMomResponses) {
          reasoningMomResponses = parsedCompletion.reasoningMomResponses
          console.log('[ReMoM] Extracted reasoning_mom_responses:', reasoningMomResponses)
        }

        if (parsedCompletion.choices.length > 1) {
          isRatingsMode = true
        }

        mergeParsedChoices(choiceContents, parsedCompletion.choices)

        let shouldSyncToolCalls = false
        for (const parsedChoice of parsedCompletion.choices) {
          if (mergeToolCallsIntoState(parsedChoice.toolCalls, 'tool', streaming ? 'running' : 'pending')) {
            shouldSyncToolCalls = true
          }
        }

        if (shouldSyncToolCalls) {
          syncAssistantToolCalls()
        }

        syncAssistantChoices(streaming)
      }

      if (!isEventStreamContentType(response.headers.get('content-type'))) {
        const responseText = await response.text()
        const parsedResponse = parseChatCompletionPayload(responseText)

        if (!parsedResponse) {
          throw new Error('Invalid JSON response')
        }

        if (parsedResponse.errorMessage) {
          throw new Error(parsedResponse.errorMessage)
        }

        if (parsedResponse.choices.length === 0) {
          throw new Error('No choices in response')
        }

        applyParsedCompletion(parsedResponse, false)
      } else {
        if (!response.body) {
          throw new Error('No response body')
        }

        await consumeEventStream(response.body, data => {
          const parsedChunk = parseChatCompletionPayload(data)
          if (!parsedChunk) {
            return
          }

          if (parsedChunk.errorMessage) {
            throw new Error(parsedChunk.errorMessage)
          }

          applyParsedCompletion(parsedChunk, true)
        })
      }

      if (hasToolCalls) {
        const MAX_TOOL_ITERATIONS = 30
        let iteration = 0
        let allToolCalls = Array.from(toolCallsMap.values())
        let allToolResults: ToolResult[] = []
        let finalContent = ''
        let currentMessages: OutboundChatMessage[] = [...chatMessages]

        while (iteration < MAX_TOOL_ITERATIONS) {
          iteration++
          console.log(`Tool iteration ${iteration}/${MAX_TOOL_ITERATIONS}`)

          const currentToolCalls = iteration === 1
            ? allToolCalls
            : Array.from(toolCallsMap.values())

          if (currentToolCalls.length === 0) break

          currentToolCalls.forEach(tc => { tc.status = 'running' })

          const uiToolCalls = [...allToolCalls]
          if (iteration > 1) {
            currentToolCalls.forEach(tc => {
              if (!uiToolCalls.find(t => t.id === tc.id)) {
                uiToolCalls.push(tc)
              }
            })
            allToolCalls = uiToolCalls
          }

          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, toolCalls: [...uiToolCalls] }
                : m
            )
          )

          const toolResults = await executeTools(currentToolCalls, {
            signal: abortControllerRef.current?.signal,
          })

          toolResults.forEach(result => {
            const tc = currentToolCalls.find(t => t.id === result.callId)
            if (tc) {
              tc.status = result.error ? 'failed' : 'completed'
            }
          })

          allToolResults = [...allToolResults, ...toolResults]

          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, toolCalls: [...uiToolCalls], toolResults: allToolResults }
                : m
            )
          )

          if (uiToolCalls.length > 0 && expandedToolCards.size === 0) {
            setExpandedToolCards(new Set([uiToolCalls[0].id]))
          }

          currentMessages = [
            ...currentMessages,
            {
              role: 'assistant',
              content: null,
              tool_calls: currentToolCalls.map(tc => ({
                id: tc.id,
                type: 'function',
                function: {
                  name: tc.function.name,
                  arguments: tc.function.arguments
                }
              }))
            },
            ...toolResults.map(tr => ({
              role: 'tool',
              tool_call_id: tr.callId,
              content: serializeToolResultForModel(tr),
            }))
          ]

          const followUpResponse = await fetch(endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model: task.requestOptions.model,
              messages: currentMessages,
              stream: true,
              tools: activeTools,
              tool_choice: 'auto',
            }),
            signal: abortControllerRef.current?.signal,
          })

          if (!followUpResponse.ok) {
            console.error('Follow-up API call failed:', followUpResponse.status, followUpResponse.statusText)
            break
          }

          let followUpContent = ''
          let followUpThinking = ''
          let hasMoreToolCalls = false
          let streamFinishReason = ''

          toolCallsMap.clear()

          const syncFollowUpMessage = (streaming: boolean) => {
            if (followUpThinking) {
              latestThinkingProcess = followUpThinking
            }

            if (!followUpContent && !followUpThinking) {
              return
            }

            setMessages(prev =>
              prev.map(m =>
                m.id === assistantMessageId
                  ? {
                    ...m,
                    content: followUpContent || m.content,
                    thinkingProcess: followUpThinking || m.thinkingProcess,
                    isStreaming: streaming,
                  }
                  : m
              )
            )
          }

          const applyFollowUpCompletion = (parsedCompletion: ParsedChatCompletion, streaming: boolean) => {
            const firstChoice = parsedCompletion.choices[0]
            const resolvedFinishReason = firstChoice?.finishReason

            if (resolvedFinishReason) {
              streamFinishReason = resolvedFinishReason
              console.log(`Iteration ${iteration} finish_reason: ${resolvedFinishReason}, hasContent: ${followUpContent.length > 0}`)
            }

            let shouldSyncToolCalls = false
            for (const parsedChoice of parsedCompletion.choices) {
              if (parsedChoice.toolCalls.length > 0) {
                hasMoreToolCalls = true
                if (mergeToolCallsIntoState(parsedChoice.toolCalls, `tool-${iteration}`, 'pending')) {
                  shouldSyncToolCalls = true
                }
              }

              if (parsedChoice.content) {
                followUpContent += parsedChoice.content
              }

              if (parsedChoice.reasoningContent) {
                followUpThinking += parsedChoice.reasoningContent
              }
            }

            if (!streamFinishReason) {
              streamFinishReason = hasMoreToolCalls ? 'tool_calls' : 'stop'
            }

            if (shouldSyncToolCalls) {
              syncAssistantToolCalls()
            }

            syncFollowUpMessage(streaming)
          }

          if (!isEventStreamContentType(followUpResponse.headers.get('content-type'))) {
            const followUpText = await followUpResponse.text()
            const parsedFollowUp = parseChatCompletionPayload(followUpText)

            if (parsedFollowUp?.errorMessage) {
              console.error('Follow-up API call returned an error:', parsedFollowUp.errorMessage)
              break
            }

            if (parsedFollowUp && parsedFollowUp.choices.length > 0) {
              applyFollowUpCompletion(parsedFollowUp, false)
            }
          } else {
            if (!followUpResponse.body) break

            await consumeEventStream(followUpResponse.body, data => {
              const parsedFollowUpChunk = parseChatCompletionPayload(data)
              if (!parsedFollowUpChunk) {
                return
              }

              if (parsedFollowUpChunk.errorMessage) {
                console.error('Follow-up streaming chunk returned an error:', parsedFollowUpChunk.errorMessage)
                return
              }

              applyFollowUpCompletion(parsedFollowUpChunk, true)
            })
          }

          if (followUpContent) {
            finalContent = followUpContent
            console.log(`Iteration ${iteration} content: ${followUpContent.substring(0, 100)}`)
          }

          if (streamFinishReason === 'tool_calls' && toolCallsMap.size > 0) {
            console.log(`Model requested ${toolCallsMap.size} more tool call(s) (finish_reason: tool_calls), will continue loop`)
            continue
          } else if (streamFinishReason === 'stop' || streamFinishReason === 'length') {
            console.log(`Model finished (finish_reason: ${streamFinishReason}), exiting tool loop`)
            break
          } else if (!hasMoreToolCalls) {
            console.log('No more tool calls detected, exiting tool loop')
            break
          }

          console.log(`Default case: hasMoreToolCalls=${hasMoreToolCalls}, finish_reason=${streamFinishReason}, continuing`)
        }

        if (iteration >= MAX_TOOL_ITERATIONS) {
          console.warn('Reached maximum tool iterations, stopping')
        }

        console.log('Tool loop finished, final content length:', finalContent.length)
        if (finalContent) {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, content: finalContent }
                : m
            )
          )
        } else {
          console.warn('Tool loop finished but no content received from model, generating fallback summary')

          let fallbackContent = ''
          if (allToolResults.length > 0) {
            const successResults = allToolResults.filter(tr => !tr.error)
            const failedResults = allToolResults.filter(tr => tr.error)

            if (successResults.length > 0) {
              fallbackContent = 'Based on the tool results, here is the relevant information:\n\n'
              for (const tr of successResults) {
                if (typeof tr.content === 'string' && tr.content.length > 0) {
                  const summary = tr.content.length > 500
                    ? `${tr.content.substring(0, 500)}...`
                    : tr.content
                  fallbackContent += `${summary}\n\n`
                }
              }
            }

            if (failedResults.length > 0 && !fallbackContent) {
              fallbackContent = 'Some tool calls failed. Please try again or refine the request.'
            }
          }

          if (!fallbackContent) {
            fallbackContent = 'The model did not generate a response. Please try again.'
          }

          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, content: fallbackContent }
                : m
            )
          )
        }
      }

      const finalChoices: Choice[] | undefined = isRatingsMode
        ? buildChoicesArray(choiceContents)
        : undefined
      const finalThinkingProcess = latestThinkingProcess || getFirstChoice(choiceContents)?.reasoningContent || ''

      console.log('[ReMoM] Setting reasoning_mom_responses:', reasoningMomResponses)
      setShowThinking(false)
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMessageId
            ? {
              ...m,
              isStreaming: false,
              headers: Object.keys(responseHeaders).length > 0 ? responseHeaders : undefined,
              choices: finalChoices,
              thinkingProcess: finalThinkingProcess || m.thinkingProcess,
              reasoning_mom_responses: reasoningMomResponses
            }
            : m
        )
      )
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return
      }
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      setMessages(prev => prev.filter(m => m.id !== assistantMessageId))
    } finally {
      isLoadingRef.current = false
      setIsLoading(false)
      setShowThinking(false)
      if (activeTaskRef.current?.id === task.id) {
        activeTaskRef.current = null
      }
      setActiveTask(current => (current?.id === task.id ? null : current))
      abortControllerRef.current = null
    }
  }, [
    buildTaskTools,
    clawManagementDisabled,
    endpoint,
    executeTools,
    expandedToolCards.size,
    generateId,
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
      saveConversation(conversationId, messagesRef.current)
    }

    setError(null)
    setInputValue('')

    if (!isLoadingRef.current && !activeTaskRef.current && queuedTasks.length === 0) {
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
    inputValue,
    queuedTasks.length,
    saveConversation,
  ])

  useEffect(() => {
    if (enableClawMode && clawView === 'room') {
      return
    }
    if (isLoadingRef.current || activeTaskRef.current || queuedTasks.length === 0) {
      return
    }
    const nextTask = queuedTasks[0]
    removeQueuedTask(conversationId, nextTask.id)
    activeTaskRef.current = nextTask
    setActiveTask(nextTask)
    void executeTask(nextTask)
  }, [
    clawView,
    conversationId,
    enableClawMode,
    executeTask,
    queuedTasks,
    removeQueuedTask,
  ])

  const handleDeleteQueuedTask = useCallback((taskId: string) => {
    removeQueuedTask(conversationId, taskId)
  }, [conversationId, removeQueuedTask])

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
    abortControllerRef.current?.abort()
    isLoadingRef.current = false
    setIsLoading(false)
  }

  const handleNewConversation = useCallback(() => {
    abortControllerRef.current?.abort()
    activeTaskRef.current = null
    isLoadingRef.current = false
    setActiveTask(null)
    setIsLoading(false)
    setMessages([])
    setError(null)
    setPendingHeaders(null)
    setShowHeaderReveal(false)
    setShowThinking(false)
    setExpandedToolCards(new Set())
    setInputValue('')
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

  return (
    <>
      {showThinking && (
        <ThinkingAnimation
          onComplete={handleThinkingComplete}
          thinkingProcess={liveThinkingProcess}
        />
      )}

      {showHeaderReveal && pendingHeaders && (
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
                {error && (
                  <div className={styles.error}>
                    <span className={styles.errorIcon}>⚠️</span>
                    <span>{error}</span>
                    <button
                      className={styles.errorDismiss}
                      onClick={() => setError(null)}
                    >
                      ×
                    </button>
                  </div>
                )}
                <ChatComponentConversationViewport
                  expandedToolCards={expandedToolCards}
                  messages={messages}
                  onToggleToolCard={handleToggleToolCard}
                />
                <ChatTaskQueue
                  activeTask={activeTask}
                  queuedTasks={queuedTasks}
                  onDeleteTask={handleDeleteQueuedTask}
                  onReorderTasks={handleReorderQueuedTasks}
                />
                <ChatComponentInputBar
                  enableClawMode={enableClawMode}
                  enableWebSearch={enableWebSearch}
                  inputRef={inputRef}
                  inputValue={inputValue}
                  isLoading={isLoading}
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
