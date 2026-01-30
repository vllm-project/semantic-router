/**
 * AgentPlayground - Main component for agentic workflow visualization
 * Provides enhanced chat agent with visualization toggle for step-by-step view
 */

import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import styles from './AgentPlayground.module.css'
import StepTimeline from './StepTimeline'
import MarkdownRenderer from '../MarkdownRenderer'
import { useToolRegistry } from '../../tools'
import { getTranslateAttr } from '../../hooks/useNoTranslate'
import type { ToolCall, ToolResult } from '../../tools'
import type {
  AgentSession,
  AgentStep,
  VisualizationMode,
  AgentPlaygroundSettings,
} from '../../types/agent'
import {
  createSession,
  createStep,
} from '../../types/agent'

interface AgentPlaygroundProps {
  endpoint?: string
  defaultModel?: string
  defaultSystemPrompt?: string
}

// Visualization toggle component
const VisualizationToggle = ({
  mode,
  onToggle,
  disabled = false,
}: {
  mode: VisualizationMode
  onToggle: () => void
  disabled?: boolean
}) => (
  <div className={styles.visualizationToggle}>
    <span className={styles.toggleLabel}>Step View</span>
    <button
      className={`${styles.toggleButton} ${mode === 'steps' ? styles.toggleActive : ''}`}
      onClick={onToggle}
      disabled={disabled}
      aria-pressed={mode === 'steps'}
      title={mode === 'steps' ? 'Show step-by-step visualization' : 'Show direct response'}
    >
      <div className={styles.toggleTrack}>
        <div className={styles.toggleThumb} />
      </div>
    </button>
  </div>
)

// Settings panel component
const SettingsPanel = ({
  settings,
  onUpdate,
  models,
}: {
  settings: AgentPlaygroundSettings
  onUpdate: <K extends keyof AgentPlaygroundSettings>(key: K, value: AgentPlaygroundSettings[K]) => void
  models: string[]
}) => (
  <div className={styles.settingsPanel}>
    <div className={styles.settingItem}>
      <label className={styles.settingLabel}>Model</label>
      <select
        className={styles.settingSelect}
        value={settings.model}
        onChange={e => onUpdate('model', e.target.value)}
      >
        <option value="">Default (Router)</option>
        {models.map(model => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
    </div>
    <div className={styles.settingItem}>
      <label className={styles.settingLabel}>Max Steps</label>
      <input
        type="number"
        className={styles.settingInput}
        value={settings.maxSteps}
        onChange={e => onUpdate('maxSteps', parseInt(e.target.value) || 30)}
        min={1}
        max={100}
      />
    </div>
    <div className={styles.settingItem}>
      <label className={styles.settingCheckbox}>
        <input
          type="checkbox"
          checked={settings.enableWebSearch}
          onChange={e => onUpdate('enableWebSearch', e.target.checked)}
        />
        <span>Web Search</span>
      </label>
    </div>
  </div>
)

const AgentPlayground = ({
  endpoint = '/api/router/v1/chat/completions',
  defaultModel = 'MoM',
  defaultSystemPrompt = '',
}: AgentPlaygroundProps) => {
  // State
  const [session, setSession] = useState<AgentSession | null>(null)
  const [selectedStep, setSelectedStep] = useState<AgentStep | null>(null)
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [directModeContent, setDirectModeContent] = useState('')
  const [settings, setSettings] = useState<AgentPlaygroundSettings>({
    visualizationMode: 'steps',
    agentMode: 'chat',
    model: defaultModel,
    maxSteps: 30,
    enableWebSearch: true,
    systemPrompt: defaultSystemPrompt,
    autoScroll: true,
  })

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)

  // Tool registry
  const { definitions: searchToolDefinitions, executeAll: executeSearchTools } = useToolRegistry({
    enabledOnly: true,
    categories: ['search'],
  })

  const { definitions: otherToolDefinitions, executeAll: executeOtherTools } = useToolRegistry({
    enabledOnly: true,
    categories: ['code', 'file', 'image', 'custom'],
  })

  // Available models (can be fetched from config)
  const availableModels = useMemo(() => ['MoM', 'gpt-4', 'gpt-3.5-turbo', 'claude-3'], [])

  // Update settings helper
  const updateSetting = useCallback(<K extends keyof AgentPlaygroundSettings>(
    key: K,
    value: AgentPlaygroundSettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }, [])

  // Toggle visualization mode
  const toggleVisualization = useCallback(() => {
    setSettings(prev => ({
      ...prev,
      visualizationMode: prev.visualizationMode === 'steps' ? 'direct' : 'steps',
    }))
  }, [])

  // Get active tools
  const activeTools = useMemo(() => {
    const tools = [...otherToolDefinitions]
    if (settings.enableWebSearch) {
      tools.push(...searchToolDefinitions)
    }
    return tools
  }, [searchToolDefinitions, otherToolDefinitions, settings.enableWebSearch])

  // Execute tools and get results
  const executeTools = useCallback(async (toolCalls: ToolCall[]) => {
    const searchCalls = toolCalls.filter(tc =>
      searchToolDefinitions.some(t => t.function.name === tc.function.name)
    )
    const otherCalls = toolCalls.filter(tc =>
      otherToolDefinitions.some(t => t.function.name === tc.function.name)
    )

    const results: ToolResult[] = []

    if (searchCalls.length > 0) {
      const searchResults = await executeSearchTools(searchCalls, {
        signal: abortControllerRef.current?.signal,
      })
      results.push(...searchResults)
    }

    if (otherCalls.length > 0) {
      const otherResults = await executeOtherTools(otherCalls, {
        signal: abortControllerRef.current?.signal,
      })
      results.push(...otherResults)
    }

    return results
  }, [searchToolDefinitions, otherToolDefinitions, executeSearchTools, executeOtherTools])

  // Add step to session
  const addStep = useCallback((step: AgentStep) => {
    setSession(prev => {
      if (!prev) return null
      return {
        ...prev,
        steps: [...prev.steps, step],
        step_count: prev.steps.length + 1,
      }
    })
  }, [])

  // Update step in session
  const updateStep = useCallback((stepId: string, updates: Partial<AgentStep>) => {
    setSession(prev => {
      if (!prev) return null
      return {
        ...prev,
        steps: prev.steps.map(s => (s.id === stepId ? { ...s, ...updates } : s)),
      }
    })
  }, [])

  // Handle submit
  const handleSubmit = useCallback(async () => {
    if (!inputValue.trim() || isLoading) return

    const task = inputValue.trim()
    setInputValue('')
    setError(null)
    setIsLoading(true)
    setDirectModeContent('')

    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()

    // Create new session
    const newSession = createSession(task, 'chat', settings.model)
    newSession.status = 'running'
    newSession.max_steps = settings.maxSteps
    setSession(newSession)
    setSelectedStep(null)

    // Build messages
    type ChatMessage = { role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string }
    const messages: ChatMessage[] = []

    if (settings.systemPrompt) {
      messages.push({ role: 'system', content: settings.systemPrompt })
    }
    messages.push({ role: 'user', content: task })

    // Tool execution loop
    const MAX_ITERATIONS = settings.maxSteps
    let iteration = 0
    let allToolCalls: ToolCall[] = []
    let allToolResults: ToolResult[] = []
    let currentMessages = [...messages]
    let hasToolCalls = false
    let stepIndex = 0

    try {
      // Initial API call
      const requestBody: Record<string, unknown> = {
        model: settings.model || 'MoM',
        messages: currentMessages,
        stream: true,
      }

      if (activeTools.length > 0) {
        requestBody.tools = activeTools
        requestBody.tool_choice = 'auto'
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: abortControllerRef.current?.signal,
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`)
      }

      if (!response.body) {
        throw new Error('No response body')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let content = ''
      let thinkingContent = ''
      const toolCallsMap = new Map<number, ToolCall>()

      // Stream initial response
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter(line => line.trim())

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data)
            const delta = parsed.choices?.[0]?.delta

            // Handle thinking/reasoning content
            if (delta?.reasoning_content) {
              thinkingContent += delta.reasoning_content

              // In steps mode, create/update thought step
              if (settings.visualizationMode === 'steps') {
                const existingThought = session?.steps.find(
                  s => s.type === 'thought' && s.status === 'running'
                )
                if (existingThought) {
                  updateStep(existingThought.id, { thought: thinkingContent })
                } else {
                  const thoughtStep = createStep('thought', stepIndex++, {
                    thought: thinkingContent,
                    status: 'running',
                  })
                  addStep(thoughtStep)
                }
              }
            }

            // Handle content
            if (delta?.content) {
              content += delta.content
              setDirectModeContent(content)

              // Complete thought step if we have content now
              if (settings.visualizationMode === 'steps' && thinkingContent) {
                const thoughtStep = session?.steps.find(
                  s => s.type === 'thought' && s.status === 'running'
                )
                if (thoughtStep) {
                  updateStep(thoughtStep.id, { status: 'completed' })
                }
              }
            }

            // Handle tool calls
            if (delta?.tool_calls && Array.isArray(delta.tool_calls)) {
              hasToolCalls = true
              for (const tc of delta.tool_calls) {
                const tcIndex = tc.index ?? 0
                if (!toolCallsMap.has(tcIndex)) {
                  toolCallsMap.set(tcIndex, {
                    id: tc.id || `tool-${tcIndex}`,
                    type: 'function',
                    function: { name: tc.function?.name || '', arguments: '' },
                    status: 'pending',
                  })
                }
                const existing = toolCallsMap.get(tcIndex)!
                if (tc.function?.name) existing.function.name = tc.function.name
                if (tc.function?.arguments) existing.function.arguments += tc.function.arguments
                if (tc.id) existing.id = tc.id
              }
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }

      // Process tool calls in a loop
      if (hasToolCalls) {
        allToolCalls = Array.from(toolCallsMap.values())

        while (iteration < MAX_ITERATIONS) {
          iteration++
          console.log(`[AgentPlayground] Tool iteration ${iteration}/${MAX_ITERATIONS}`)

          const currentToolCalls = iteration === 1 ? allToolCalls : Array.from(toolCallsMap.values())
          if (currentToolCalls.length === 0) break

          // Mark tools as running
          currentToolCalls.forEach(tc => { tc.status = 'running' })

          // Create action steps for each tool call
          if (settings.visualizationMode === 'steps') {
            for (const tc of currentToolCalls) {
              let parsedArgs: Record<string, unknown> = {}
              try {
                parsedArgs = JSON.parse(tc.function.arguments || '{}')
              } catch {
                parsedArgs = { raw: tc.function.arguments }
              }

              const actionStep = createStep('action', stepIndex++, {
                action: {
                  type: 'function',
                  name: tc.function.name,
                  arguments: parsedArgs,
                },
                toolCall: tc,
                status: 'running',
              })
              addStep(actionStep)
            }
          }

          // Execute tools
          const toolResults = await executeTools(currentToolCalls)

          // Update tool statuses
          toolResults.forEach(result => {
            const tc = currentToolCalls.find(t => t.id === result.callId)
            if (tc) {
              tc.status = result.error ? 'failed' : 'completed'
            }
          })

          allToolResults = [...allToolResults, ...toolResults]

          // Create observation steps
          if (settings.visualizationMode === 'steps') {
            for (const result of toolResults) {
              const actionStep = session?.steps.find(
                s => s.type === 'action' && s.toolCall?.id === result.callId
              )
              if (actionStep) {
                updateStep(actionStep.id, { status: result.error ? 'failed' : 'completed' })
              }

              const observationStep = createStep('observation', stepIndex++, {
                observation: {
                  content: result.content,
                  error: result.error,
                  duration_ms: 0, // Could track actual duration
                },
                toolResult: result,
                status: result.error ? 'failed' : 'completed',
              })
              addStep(observationStep)
            }
          }

          // Build messages for follow-up
          currentMessages = [
            ...currentMessages,
            {
              role: 'assistant',
              content: null,
              tool_calls: currentToolCalls.map(tc => ({
                id: tc.id,
                type: 'function',
                function: { name: tc.function.name, arguments: tc.function.arguments },
              })),
            },
            ...toolResults.map(tr => ({
              role: 'tool',
              tool_call_id: tr.callId,
              content: typeof tr.content === 'string' ? tr.content : JSON.stringify(tr.content),
            })),
          ]

          // Make follow-up API call
          const followUpResponse = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: settings.model || 'MoM',
              messages: currentMessages,
              stream: true,
              tools: activeTools,
              tool_choice: 'auto',
            }),
            signal: abortControllerRef.current?.signal,
          })

          if (!followUpResponse.ok || !followUpResponse.body) {
            throw new Error('Follow-up API call failed')
          }

          const followUpReader = followUpResponse.body.getReader()
          const followUpDecoder = new TextDecoder()
          let followUpContent = ''
          let hasMoreToolCalls = false
          toolCallsMap.clear()

          while (true) {
            const { done, value } = await followUpReader.read()
            if (done) break

            const chunk = followUpDecoder.decode(value, { stream: true })
            const lines = chunk.split('\n').filter(line => line.trim())

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue
              const data = line.slice(6)
              if (data === '[DONE]') continue

              try {
                const parsed = JSON.parse(data)
                const delta = parsed.choices?.[0]?.delta

                if (delta?.tool_calls && Array.isArray(delta.tool_calls)) {
                  hasMoreToolCalls = true
                  for (const tc of delta.tool_calls) {
                    const tcIndex = tc.index ?? 0
                    if (!toolCallsMap.has(tcIndex)) {
                      toolCallsMap.set(tcIndex, {
                        id: tc.id || `tool-${iteration}-${tcIndex}`,
                        type: 'function',
                        function: { name: tc.function?.name || '', arguments: '' },
                        status: 'pending',
                      })
                    }
                    const existing = toolCallsMap.get(tcIndex)!
                    if (tc.function?.name) existing.function.name = tc.function.name
                    if (tc.function?.arguments) existing.function.arguments += tc.function.arguments
                    if (tc.id) existing.id = tc.id
                  }
                }

                if (delta?.content) {
                  followUpContent += delta.content
                  setDirectModeContent(followUpContent)
                }
              } catch {
                // Ignore parse errors
              }
            }
          }

          // Check if we should continue
          if (!hasMoreToolCalls) {
            content = followUpContent
            break
          }

          // Add new tool calls
          const newToolCalls = Array.from(toolCallsMap.values())
          allToolCalls = [...allToolCalls, ...newToolCalls]
        }
      }

      // Create final answer step
      if (settings.visualizationMode === 'steps' && content) {
        const finalStep = createStep('final_answer', stepIndex++, {
          answer: content,
          status: 'completed',
        })
        addStep(finalStep)
      }

      // Update session status
      setSession(prev =>
        prev
          ? {
              ...prev,
              status: 'completed',
              completed_at: Date.now(),
              final_response: content,
            }
          : null
      )
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setSession(prev => (prev ? { ...prev, status: 'cancelled' } : null))
      } else {
        const errorMessage = err instanceof Error ? err.message : 'An error occurred'
        setError(errorMessage)
        setSession(prev =>
          prev ? { ...prev, status: 'failed', error: errorMessage } : null
        )
      }
    } finally {
      setIsLoading(false)
    }
  }, [
    inputValue,
    isLoading,
    settings,
    endpoint,
    activeTools,
    executeTools,
    addStep,
    updateStep,
    session?.steps,
  ])

  // Handle cancel
  const handleCancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }, [])

  // Handle clear
  const handleClear = useCallback(() => {
    setSession(null)
    setSelectedStep(null)
    setDirectModeContent('')
    setError(null)
  }, [])

  // Handle key press in input
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit]
  )

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>Agent Playground</h2>
          <span className={styles.subtitle}>Enhanced Chat Agent with Step Visualization</span>
        </div>
        <div className={styles.headerRight}>
          <VisualizationToggle
            mode={settings.visualizationMode}
            onToggle={toggleVisualization}
            disabled={isLoading}
          />
          <button
            className={styles.settingsButton}
            onClick={() => setShowSettings(!showSettings)}
            aria-label="Settings"
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>
          <button
            className={styles.clearButton}
            onClick={handleClear}
            disabled={isLoading || !session}
            aria-label="Clear"
          >
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M3 6h18" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
              <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            </svg>
          </button>
        </div>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <SettingsPanel
          settings={settings}
          onUpdate={updateSetting}
          models={availableModels}
        />
      )}

      {/* Main content area */}
      <div className={styles.content} ref={contentRef}>
        {/* Error message */}
        {error && (
          <div className={styles.errorBanner}>
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span>{error}</span>
            <button onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}

        {/* Visualization based on mode */}
        {settings.visualizationMode === 'steps' ? (
          <div className={styles.stepsView}>
            {/* Step timeline */}
            <div className={styles.timelinePanel}>
              <StepTimeline
                session={session}
                selectedStep={selectedStep}
                onStepSelect={setSelectedStep}
                autoScroll={settings.autoScroll}
              />
            </div>

            {/* Detail panel (optional - shows selected step details) */}
            {selectedStep && (
              <div className={styles.detailPanel}>
                <div className={styles.detailHeader}>
                  <h3>Step {selectedStep.index + 1} Details</h3>
                  <button onClick={() => setSelectedStep(null)}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
                <div className={styles.detailContent}>
                  {selectedStep.type === 'thought' && selectedStep.thought && (
                    <MarkdownRenderer content={selectedStep.thought} />
                  )}
                  {selectedStep.type === 'action' && selectedStep.action && (
                    <div>
                      <h4>Tool: {selectedStep.action.name}</h4>
                      <pre>{JSON.stringify(selectedStep.action.arguments, null, 2)}</pre>
                    </div>
                  )}
                  {selectedStep.type === 'observation' && selectedStep.observation && (
                    <div>
                      {selectedStep.observation.error ? (
                        <div className={styles.errorText}>{selectedStep.observation.error}</div>
                      ) : (
                        <MarkdownRenderer
                          content={
                            typeof selectedStep.observation.content === 'string'
                              ? selectedStep.observation.content
                              : JSON.stringify(selectedStep.observation.content, null, 2)
                          }
                        />
                      )}
                    </div>
                  )}
                  {selectedStep.type === 'final_answer' && selectedStep.answer && (
                    <MarkdownRenderer content={selectedStep.answer} />
                  )}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Direct mode - simple response view */
          <div className={styles.directView}>
            {session && (
              <div className={styles.directContent}>
                <div className={styles.taskDisplay}>
                  <span className={styles.taskLabel}>Task:</span>
                  <span className={styles.taskText}>{session.task}</span>
                </div>
                {directModeContent || session.final_response ? (
                  <div className={styles.responseContent} translate={getTranslateAttr(isLoading)}>
                    <MarkdownRenderer content={directModeContent || session.final_response || ''} />
                  </div>
                ) : session.status === 'running' ? (
                  <div className={styles.loadingIndicator}>
                    <div className={styles.loadingDot} />
                    <div className={styles.loadingDot} />
                    <div className={styles.loadingDot} />
                  </div>
                ) : null}
              </div>
            )}
            {!session && (
              <div className={styles.emptyState}>
                <svg
                  width="48"
                  height="48"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                >
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <p>Enter a task to start the agent</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input area */}
      <div className={styles.inputArea}>
        <div className={styles.inputWrapper}>
          <textarea
            ref={inputRef}
            className={styles.input}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe a task for the agent..."
            disabled={isLoading}
            rows={1}
          />
          <div className={styles.inputActions}>
            {isLoading ? (
              <button
                className={styles.cancelButton}
                onClick={handleCancel}
                aria-label="Cancel"
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                </svg>
              </button>
            ) : (
              <button
                className={styles.submitButton}
                onClick={handleSubmit}
                disabled={!inputValue.trim()}
                aria-label="Execute"
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default AgentPlayground
