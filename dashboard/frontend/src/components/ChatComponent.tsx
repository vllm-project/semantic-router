import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import styles from './ChatComponent.module.css'
import ThinkingAnimation from './ThinkingAnimation'
import HeaderReveal from './HeaderReveal'
import ClawRoomChat from './ClawRoomChat'
import { ClawModeToggle } from './ChatComponentControls'
import ChatConversationSidebar from './ChatConversationSidebar'
import ChatComponentInputBar from './ChatComponentInputBar'
import ChatComponentMessages from './ChatComponentMessages'
import ChatComponentRoomToggle from './ChatComponentRoomToggle'
import ChatComponentTopBar from './ChatComponentTopBar'
import {
  CLAW_MODE_STORAGE_KEY,
  CLAW_MODE_SYSTEM_PROMPT,
  CLAW_TOOL_NAME_PREFIX,
  type Choice,
  type ConversationPreview,
  generateConversationId,
  generateMessageId,
  type Message,
  type ReMoMRoundResponse,
} from './ChatComponentTypes'
import { useToolRegistry } from '../tools'
import { useMCPToolSync } from '../tools/mcp'
import { ensureOpenClawServerConnected } from '../tools/mcp/api'
import { useConversationStorage } from '../hooks'
import { useReadonly } from '../contexts/ReadonlyContext'
import type { ToolCall, ToolResult } from '../tools'

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
  const model = 'MoM' // Fixed to MoM
  const [error, setError] = useState<string | null>(null)
  const [showThinking, setShowThinking] = useState(false)
  const [showHeaderReveal, setShowHeaderReveal] = useState(false)
  const [pendingHeaders, setPendingHeaders] = useState<Record<string, string> | null>(null)
  const [isFullscreen] = useState(isFullscreenMode)
  const [enableWebSearch, setEnableWebSearch] = useState(true)
  const [enableClawMode, setEnableClawMode] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true
    const saved = window.localStorage.getItem(CLAW_MODE_STORAGE_KEY)
    if (saved === null) return true
    return saved === 'true'
  })
  const [isTogglingClawMode, setIsTogglingClawMode] = useState(false)
  const [expandedToolCards, setExpandedToolCards] = useState<Set<string>>(new Set())
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [clawView, setClawView] = useState<ClawPlaygroundView>(() => 'control')
  const [teamRoomCreateToken, setTeamRoomCreateToken] = useState(0)
  const { isReadonly, isLoading: readonlyLoading } = useReadonly()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const hasHydratedConversation = useRef(false)

  const { conversations, saveConversation, getConversation, deleteConversation } = useConversationStorage<Message[]>({
    storageKey: 'sr:chat:conversations',
    maxConversations: 20,
  })

  const restoreMessages = useCallback((payload: Message[]) => {
    return payload.map(message => ({
      ...message,
      timestamp: new Date(message.timestamp),
    }))
  }, [])

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
  const activeOtherToolDefinitions = useMemo(
    () => (
      enableClawMode && !clawManagementDisabled
        ? [...baseOtherToolDefinitions, ...clawToolDefinitions]
        : baseOtherToolDefinitions
    ),
    [baseOtherToolDefinitions, clawManagementDisabled, clawToolDefinitions, enableClawMode]
  )

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

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
  const generateId = generateMessageId

  const handleThinkingComplete = useCallback(() => {
    // Thinking animation will be hidden when headers arrive
    // This callback is kept for ThinkingAnimation component compatibility
  }, [])

  const handleHeaderRevealComplete = useCallback(() => {
    setShowHeaderReveal(false)
    setPendingHeaders(null)
  }, [])

  const handleSelectConversation = useCallback(
    (id: string) => {
      const target = conversations.find(conv => conv.id === id)
      if (!target) return

      abortControllerRef.current?.abort()
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

      deleteConversation(id)

      if (id === conversationId) {
        abortControllerRef.current?.abort()
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
    [conversationId, conversations, deleteConversation, restoreMessages]
  )

  const handleSend = async () => {
    const trimmedInput = inputValue.trim()
    if (!trimmedInput || isLoading) return

    setError(null)
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: trimmedInput,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    // Reset animation states and show initial thinking animation (no content)
    setPendingHeaders(null)
    setShowHeaderReveal(false)
    setShowThinking(true)  // Show immediately when user sends message

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

      // Build chat messages with proper tool call history
      // This ensures the model knows which tool calls have been completed
      type ChatMessage = {
        role: string
        content: string | null
        tool_calls?: Array<{
          id: string
          type: 'function'
          function: { name: string; arguments: string }
        }>
        tool_call_id?: string
      }

      const chatMessages: ChatMessage[] = []

      // Process each message for context
      // IMPORTANT: For history messages, we only include the final text content,
      // NOT the tool calls and results. This prevents context pollution where
      // the model might be confused by previous tool usage when answering new questions.
      for (const m of messages) {
        if (m.role === 'user') {
          chatMessages.push({ role: 'user', content: m.content })
        } else if (m.role === 'assistant') {
          // For assistant messages, only include the final text content
          // Don't include tool_calls or tool results in history
          // This keeps the context clean for new questions
          if (m.content) {
            chatMessages.push({ role: 'assistant', content: m.content })
          }
        }
      }

      if (enableClawMode && !clawManagementDisabled) {
        chatMessages.unshift({ role: 'system', content: CLAW_MODE_SYSTEM_PROMPT })
      }

      // Add the new user message
      chatMessages.push({ role: 'user', content: trimmedInput })

      // Build request body
      const requestBody: Record<string, unknown> = {
        model,
        messages: chatMessages,
        stream: true,
      }

      // Add tools to request:
      // - Search tools: only when web search is enabled
      // - Other tools: always available
      const activeTools = [
        ...activeOtherToolDefinitions,
        ...(enableWebSearch ? searchToolDefinitions : []),
      ]
      if (activeTools.length > 0) {
        requestBody.tools = activeTools
        requestBody.tool_choice = 'auto'
      }

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

      // Extract key headers from response
      const responseHeaders: Record<string, string> = {}
      const headerKeys = [
        'x-vsr-selected-model',
        'x-vsr-selected-decision',
        'x-vsr-cache-hit',
        'x-vsr-selected-reasoning',
        'x-vsr-jailbreak-blocked',
        'x-vsr-pii-violation',
        'x-vsr-hallucination-detected',
        'x-vsr-fact-check-needed',
        'x-vsr-matched-keywords',
        'x-vsr-matched-embeddings',
        'x-vsr-matched-domains',
        'x-vsr-matched-fact-check',
        'x-vsr-matched-user-feedback',
        'x-vsr-matched-preference',
        'x-vsr-matched-language',
        'x-vsr-matched-context',
        'x-vsr-context-token-count',
        'x-vsr-matched-complexity',
        // Looper headers
        'x-vsr-looper-model',
        'x-vsr-looper-models-used',
        'x-vsr-looper-iterations',
        'x-vsr-looper-algorithm',
      ]

      headerKeys.forEach(key => {
        const value = response.headers.get(key)
        if (value) {
          responseHeaders[key] = value
        }
      })

      // Store headers and hide thinking animation, show HeaderReveal
      if (Object.keys(responseHeaders).length > 0) {
        console.log('Headers received, showing HeaderReveal')
        setPendingHeaders(responseHeaders)
        setShowThinking(false)  // Hide full-screen thinking animation
        setShowHeaderReveal(true)  // Show HeaderReveal
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      // Track content and reasoning for each choice (for ratings mode)
      const choiceContents: Map<number, { content: string; reasoningContent: string; model?: string }> = new Map()
      // Check if this is ratings mode (multiple choices)
      let isRatingsMode = false
      // Track tool calls
      const toolCallsMap: Map<number, ToolCall> = new Map()
      let hasToolCalls = false
      // Track ReMoM intermediate responses
      let reasoningMomResponses: ReMoMRoundResponse[] | undefined
      // Buffer for incomplete lines
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk
        const lines = buffer.split('\n')

        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') continue

            try {
              const parsed = JSON.parse(data)
              const choices = parsed.choices || []

              // Extract reasoning_mom_responses if present (ReMoM algorithm)
              if (parsed.reasoning_mom_responses) {
                reasoningMomResponses = parsed.reasoning_mom_responses
                console.log('[ReMoM] Extracted reasoning_mom_responses:', reasoningMomResponses)
              }

              // Detect ratings mode (multiple choices)
              if (choices.length > 1) {
                isRatingsMode = true
              }

              // Process each choice
              for (const choice of choices) {
                const index = choice.index ?? 0
                const content = choice.delta?.content || ''
                const reasoningContent = choice.delta?.reasoning_content || ''
                const model = choice.model
                const deltaToolCalls = choice.delta?.tool_calls

                // Handle tool calls
                if (deltaToolCalls && Array.isArray(deltaToolCalls)) {
                  hasToolCalls = true
                  for (const tc of deltaToolCalls) {
                    const tcIndex = tc.index ?? 0
                    if (!toolCallsMap.has(tcIndex)) {
                      toolCallsMap.set(tcIndex, {
                        id: tc.id || `tool-${tcIndex}`,
                        type: 'function',
                        function: {
                          name: tc.function?.name || '',
                          arguments: ''
                        },
                        status: 'running'
                      })
                    }
                    const existingTc = toolCallsMap.get(tcIndex)!
                    if (tc.function?.name) {
                      existingTc.function.name = tc.function.name
                    }
                    if (tc.function?.arguments) {
                      existingTc.function.arguments += tc.function.arguments
                    }
                    if (tc.id) {
                      existingTc.id = tc.id
                    }
                  }

                  // Update message with tool calls
                  const currentToolCalls = Array.from(toolCallsMap.values())
                  setMessages(prev =>
                    prev.map(m =>
                      m.id === assistantMessageId
                        ? { ...m, toolCalls: currentToolCalls }
                        : m
                    )
                  )
                }

                if (!choiceContents.has(index)) {
                  choiceContents.set(index, { content: '', reasoningContent: '', model })
                }

                const current = choiceContents.get(index)!
                if (content) {
                  current.content += content
                }
                if (reasoningContent) {
                  current.reasoningContent += reasoningContent
                }
                if (model && !current.model) {
                  current.model = model
                }
              }

              // Update message state (only if we have content, not just tool calls)
              if (!hasToolCalls || choiceContents.get(0)?.content) {
                if (isRatingsMode) {
                  // Ratings mode: update choices array
                  const choicesArray: Choice[] = []

                  choiceContents.forEach((value, index) => {
                    choicesArray[index] = { content: value.content, model: value.model }
                  })

                  // Get thinking process (reasoning_content) from first choice
                  const firstChoice = choiceContents.get(0)
                  const thinkingProcess = firstChoice?.reasoningContent || ''

                  setMessages(prev =>
                    prev.map(m =>
                      m.id === assistantMessageId
                        ? {
                          ...m,
                          content: choicesArray[0]?.content || '',
                          choices: choicesArray,
                          thinkingProcess: thinkingProcess
                        }
                        : m
                    )
                  )
                } else {
                  // Single choice mode
                  const firstChoice = choiceContents.get(0)
                  if (firstChoice) {
                    setMessages(prev =>
                      prev.map(m =>
                        m.id === assistantMessageId
                          ? {
                            ...m,
                            content: firstChoice.content,
                            thinkingProcess: firstChoice.reasoningContent,
                            isStreaming: true
                          }
                          : m
                      )
                    )
                  }
                }
              }
            } catch {
              // Skip malformed JSON chunks
            }
          }
        }
      }

      // If we had tool calls, execute tools in a loop until model gives final answer
      if (hasToolCalls) {
        // Maximum tool call iterations to prevent infinite loops
        const MAX_TOOL_ITERATIONS = 30
        let iteration = 0

        // Accumulated tool calls and results across all iterations
        let allToolCalls = Array.from(toolCallsMap.values())
        let allToolResults: ToolResult[] = []

        // Track final content from tool loop
        let finalContent = ''

        // Current conversation state for tool loop - use chatMessages as base (already built correctly)
        type ChatMessage = { role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string }
        let currentMessages: ChatMessage[] = [...chatMessages]

        while (iteration < MAX_TOOL_ITERATIONS) {
          iteration++
          console.log(`Tool iteration ${iteration}/${MAX_TOOL_ITERATIONS}`)

          // Get current tool calls to execute
          const currentToolCalls = iteration === 1
            ? allToolCalls
            : Array.from(toolCallsMap.values())

          if (currentToolCalls.length === 0) break

          // Mark all tools as running
          currentToolCalls.forEach(tc => { tc.status = 'running' })

          // Update UI with current tool calls
          const uiToolCalls = [...allToolCalls]
          if (iteration > 1) {
            // Add new tool calls from subsequent iterations
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

          // Execute all current tools in parallel
          const toolResults = await executeTools(currentToolCalls, {
            signal: abortControllerRef.current?.signal,
          })

          // Update tool statuses based on results
          toolResults.forEach(result => {
            const tc = currentToolCalls.find(t => t.id === result.callId)
            if (tc) {
              tc.status = result.error ? 'failed' : 'completed'
            }
          })

          // Accumulate results
          allToolResults = [...allToolResults, ...toolResults]

          // Update message with completed tool calls and all results
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMessageId
                ? { ...m, toolCalls: [...uiToolCalls], toolResults: allToolResults }
                : m
            )
          )

          // Auto-expand the first tool card
          if (uiToolCalls.length > 0 && expandedToolCards.size === 0) {
            setExpandedToolCards(new Set([uiToolCalls[0].id]))
          }

          // Build messages for next API call
          currentMessages = [
            ...currentMessages,
            // Assistant message with tool_calls
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
            // Tool results (truncate long content to avoid exceeding model context)
            ...toolResults.map(tr => {
              const MAX_TOOL_RESULT_LENGTH = 15000 // ~4k tokens
              let content = typeof tr.content === 'string'
                ? tr.content
                : JSON.stringify(tr.content)

              // Truncate if too long
              if (content.length > MAX_TOOL_RESULT_LENGTH) {
                content = content.substring(0, MAX_TOOL_RESULT_LENGTH) + '\n\n...[Content truncated due to length]'
                console.log(`Tool result for ${tr.name} truncated from ${typeof tr.content === 'string' ? tr.content.length : JSON.stringify(tr.content).length} to ${MAX_TOOL_RESULT_LENGTH} chars`)
              }

              return {
                role: 'tool',
                tool_call_id: tr.callId,
                content
              }
            })
          ]

          // Make follow-up API call with tools enabled
          const followUpResponse = await fetch(endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              model,
              messages: currentMessages,
              stream: true,
              // Keep tools enabled for multi-step tool usage (same logic as initial request)
              tools: activeTools,
              tool_choice: 'auto',
            }),
            signal: abortControllerRef.current?.signal,
          })

          if (!followUpResponse.ok) {
            console.error('Follow-up API call failed:', followUpResponse.status, followUpResponse.statusText)
            break
          }

          if (!followUpResponse.body) break

          const followUpReader = followUpResponse.body.getReader()
          const followUpDecoder = new TextDecoder()
          let followUpContent = ''
          let hasMoreToolCalls = false
          let streamFinishReason = ''

          // Reset tool calls map for this iteration
          toolCallsMap.clear()

          while (true) {
            const { done, value } = await followUpReader.read()
            if (done) break

            const chunk = followUpDecoder.decode(value, { stream: true })
            const lines = chunk.split('\n').filter(line => line.trim() !== '')

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data === '[DONE]') continue

                try {
                  const parsed = JSON.parse(data)
                  const delta = parsed.choices?.[0]?.delta
                  const deltaToolCalls = delta?.tool_calls
                  const finishReason = parsed.choices?.[0]?.finish_reason

                  // Capture finish reason when present
                  if (finishReason) {
                    streamFinishReason = finishReason
                    console.log(`Iteration ${iteration} finish_reason: ${finishReason}, hasContent: ${followUpContent.length > 0}`)
                  }

                  // Check for new tool calls
                  if (deltaToolCalls && Array.isArray(deltaToolCalls)) {
                    hasMoreToolCalls = true
                    for (const tc of deltaToolCalls) {
                      const tcIndex = tc.index ?? 0
                      if (!toolCallsMap.has(tcIndex)) {
                        toolCallsMap.set(tcIndex, {
                          id: tc.id || `tool-${iteration}-${tcIndex}`,
                          type: 'function',
                          function: {
                            name: tc.function?.name || '',
                            arguments: ''
                          },
                          status: 'pending'
                        })
                      }
                      const existingTc = toolCallsMap.get(tcIndex)!
                      if (tc.function?.name) {
                        existingTc.function.name = tc.function.name
                      }
                      if (tc.function?.arguments) {
                        existingTc.function.arguments += tc.function.arguments
                      }
                      if (tc.id) {
                        existingTc.id = tc.id
                      }
                    }
                  }

                  // Accumulate content
                  if (delta?.content) {
                    followUpContent += delta.content
                    setMessages(prev =>
                      prev.map(m =>
                        m.id === assistantMessageId
                          ? { ...m, content: followUpContent }
                          : m
                      )
                    )
                  }
                } catch {
                  // Ignore parse errors
                }
              }
            }
          }

          // Save content from this iteration
          if (followUpContent) {
            finalContent = followUpContent
            console.log(`Iteration ${iteration} content: ${followUpContent.substring(0, 100)}`)
          }

          // Check if we should continue the loop
          if (streamFinishReason === 'tool_calls' && toolCallsMap.size > 0) {
            // Model wants to call more tools, continue loop
            console.log(`Model requested ${toolCallsMap.size} more tool call(s) (finish_reason: tool_calls), will continue loop`)
            continue
          } else if (streamFinishReason === 'stop' || streamFinishReason === 'length') {
            // Model finished, exit loop
            console.log(`Model finished (finish_reason: ${streamFinishReason}), exiting tool loop`)
            break
          } else if (!hasMoreToolCalls) {
            // No more tool calls, exit loop
            console.log('No more tool calls detected, exiting tool loop')
            break
          }

          console.log(`Default case: hasMoreToolCalls=${hasMoreToolCalls}, finish_reason=${streamFinishReason}, continuing`)
        }

        if (iteration >= MAX_TOOL_ITERATIONS) {
          console.warn('Reached maximum tool iterations, stopping')
        }

        // Ensure final content is set after all tool iterations
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
          // If no content after tool loop, generate a fallback summary from tool results
          console.warn('Tool loop finished but no content received from model, generating fallback summary')

          // Generate fallback content based on tool results
          let fallbackContent = ''
          if (allToolResults.length > 0) {
            const successResults = allToolResults.filter(tr => !tr.error)
            const failedResults = allToolResults.filter(tr => tr.error)

            if (successResults.length > 0) {
              fallbackContent = '基于搜索结果，以下是相关信息：\n\n'
              for (const tr of successResults) {
                if (typeof tr.content === 'string' && tr.content.length > 0) {
                  // 截取前 500 字符作为摘要
                  const summary = tr.content.length > 500
                    ? tr.content.substring(0, 500) + '...'
                    : tr.content
                  fallbackContent += summary + '\n\n'
                }
              }
            }

            if (failedResults.length > 0 && !fallbackContent) {
              fallbackContent = '抱歉，部分工具执行失败。请尝试重新查询或使用其他关键词。'
            }
          }

          if (!fallbackContent) {
            fallbackContent = '模型没有生成响应内容，请尝试重新提问。'
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

      // Finalize message
      const finalChoices: Choice[] | undefined = isRatingsMode
        ? Array.from(choiceContents.entries())
          .sort(([a], [b]) => a - b)
          .map(([, v]) => ({ content: v.content, model: v.model }))
        : undefined

      // Get final thinking process from reasoning_content
      const finalThinkingProcess = choiceContents.get(0)?.reasoningContent || ''

      // Streaming finished - no need to control ThinkingAnimation here
      // It was already hidden when headers arrived

      console.log('[ReMoM] Setting reasoning_mom_responses:', reasoningMomResponses)
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
      setIsLoading(false)
      abortControllerRef.current = null
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleStop = () => {
    abortControllerRef.current?.abort()
    setIsLoading(false)
  }

  const handleNewConversation = useCallback(() => {
    abortControllerRef.current?.abort()
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
  const modeToggleDisabled = isLoading || isTogglingClawMode

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

  return (
    <>
      {showThinking && (
        <ThinkingAnimation
          onComplete={handleThinkingComplete}
          thinkingProcess=""
        />
      )}

      {showHeaderReveal && pendingHeaders && (
        <HeaderReveal
          headers={pendingHeaders}
          onComplete={handleHeaderRevealComplete}
          displayDuration={2000}
        />
      )}

      <div
        className={`${styles.container} ${isFullscreen ? styles.fullscreen : ''}`}
      >
        <div className={styles.mainLayout}>
          {!isTeamRoomView && isSidebarOpen && (
            <ChatConversationSidebar
              conversationId={conversationId}
              conversationPreviews={conversationPreviews}
              onDeleteConversation={handleDeleteConversation}
              onSelectConversation={handleSelectConversation}
            />
          )}

          <div className={styles.chatArea}>
            <ChatComponentTopBar
              isSidebarOpen={isSidebarOpen}
              isTeamRoomView={isTeamRoomView}
              createDisabled={roomCreateDisabled}
              onCreate={handleTopBarCreate}
              onToggleSidebar={() => setIsSidebarOpen(prev => !prev)}
            />
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
                <ChatComponentMessages
                  expandedToolCards={expandedToolCards}
                  messages={messages}
                  messagesEndRef={messagesEndRef}
                  onToggleToolCard={handleToggleToolCard}
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
