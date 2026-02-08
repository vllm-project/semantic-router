/**
 * ComputerUsePlayground - Computer-use agent that controls a browser
 * Inspired by HuggingFace's smolagents/computer-use-agent
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import styles from './ComputerUsePlayground.module.css'
import StepTimeline from './StepTimeline'
import ScreenshotViewer from './ScreenshotViewer'
import * as browserApi from '../../utils/browserApi'
import type { BrowserActionResult } from '../../utils/browserApi'
import type { AgentSession, AgentStep } from '../../types/agent'
import { createSession, createStep } from '../../types/agent'

interface ComputerUsePlaygroundProps {
  endpoint?: string
  model?: string
}

// LLM tool call arguments
interface ComputerUseToolArgs {
  action: string
  url?: string
  selector?: string
  text?: string
  x?: number
  y?: number
  delta_x?: number
  delta_y?: number
  duration?: number
  key?: string
}

// System prompt inspired by HuggingFace's smolagents approach
const SYSTEM_PROMPT = `You are a computer-use agent that controls a web browser to accomplish tasks.

## CRITICAL: YOU MUST COMPLETE ALL STEPS
DO NOT provide a final answer until you have ACTUALLY SEEN the search results in a screenshot.
After typing a search query, you MUST press Enter to submit the search.
After pressing Enter, you MUST wait for results and READ them from the screenshot.

## YOUR CAPABILITIES
You control a REAL browser. You MUST use the computer_use tool for EVERY action. Never describe actions - DO them.

## AVAILABLE ACTIONS
- navigate: Go to a URL. Example: {"action": "navigate", "url": "https://www.google.com"}
- click: Click on an element. Example: {"action": "click", "selector": "textarea[name='q']"} or {"action": "click", "x": 640, "y": 300}
- type: Type text into an input. Example: {"action": "type", "text": "weather tel aviv"}
- key: Press a keyboard key. Example: {"action": "key", "key": "Enter"}
- scroll: Scroll the page. Example: {"action": "scroll", "delta_y": 300}
- wait: Wait for page to load. Example: {"action": "wait", "duration": 2000}

## REQUIRED WORKFLOW FOR SEARCH TASKS
You MUST follow ALL of these steps in order:

1. NAVIGATE to Google: computer_use({"action": "navigate", "url": "https://www.google.com"})
2. CLICK the search box: computer_use({"action": "click", "selector": "textarea[name='q']"})
3. TYPE the query: computer_use({"action": "type", "text": "your search query"})
4. PRESS ENTER to submit: computer_use({"action": "key", "key": "Enter"})
5. WAIT for results: computer_use({"action": "wait", "duration": 2000})
6. READ the results from the screenshot, THEN provide your final answer

## CRITICAL RULES
1. After typing, you MUST press Enter - typing alone does NOT submit the search
2. After pressing Enter, you MUST wait for results to load
3. You can ONLY provide a final answer after you SEE search results in the screenshot
4. If you see "Google" homepage with a search box, you have NOT searched yet - keep going
5. Each step requires a separate tool call - never skip steps

## WHEN TO PROVIDE FINAL ANSWER
ONLY provide a text answer (without tool call) when:
- You have pressed Enter to submit the search
- You have waited for results to load
- You can SEE the actual search results/answer in the current screenshot

The browser viewport is 1280x800 pixels.`

// Tool definition with all actions
const computerUseToolDefinition = {
  type: 'function' as const,
  function: {
    name: 'computer_use',
    description: 'Control the web browser. MUST be used for every action.',
    parameters: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['navigate', 'click', 'type', 'key', 'scroll', 'screenshot', 'wait', 'back', 'forward', 'refresh'],
          description: 'The action to perform',
        },
        url: {
          type: 'string',
          description: 'URL to navigate to (for navigate action)',
        },
        selector: {
          type: 'string',
          description: 'CSS selector to click (for click action)',
        },
        text: {
          type: 'string',
          description: 'Text to type (for type action)',
        },
        key: {
          type: 'string',
          description: 'Key to press (for key action). Examples: Enter, Tab, Escape, Backspace',
        },
        x: {
          type: 'number',
          description: 'X coordinate for click',
        },
        y: {
          type: 'number',
          description: 'Y coordinate for click',
        },
        delta_y: {
          type: 'number',
          description: 'Scroll amount in pixels (positive = down)',
        },
        duration: {
          type: 'number',
          description: 'Wait duration in milliseconds',
        },
      },
      required: ['action'],
    },
  },
}

const ComputerUsePlayground = ({
  endpoint = '/api/router/v1/chat/completions',
  model = 'qwen2.5:3b',
}: ComputerUsePlaygroundProps) => {
  // State
  const [session, setSession] = useState<AgentSession | null>(null)
  const [browserSessionId, setBrowserSessionId] = useState<string | null>(null)
  const [currentScreenshot, setCurrentScreenshot] = useState<BrowserActionResult | null>(null)
  const [selectedStep, setSelectedStep] = useState<AgentStep | null>(null)
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isBrowserReady, setIsBrowserReady] = useState(false)

  // Refs
  const abortControllerRef = useRef<AbortController | null>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  // Start browser session
  const startBrowser = useCallback(async () => {
    try {
      setError(null)
      const result = await browserApi.startBrowserSession(true)
      if (result.success) {
        setBrowserSessionId(result.session_id)
        setIsBrowserReady(true)
        // Navigate to blank page and take initial screenshot
        const screenshot = await browserApi.navigateTo(result.session_id, 'about:blank')
        setCurrentScreenshot(screenshot)
      } else {
        throw new Error(result.error || 'Failed to start browser')
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start browser'
      setError(message)
      setIsBrowserReady(false)
    }
  }, [])

  // Stop browser session
  const stopBrowser = useCallback(async () => {
    if (browserSessionId) {
      try {
        await browserApi.stopBrowserSession(browserSessionId)
      } catch (err) {
        console.error('Failed to stop browser:', err)
      }
      setBrowserSessionId(null)
      setIsBrowserReady(false)
      setCurrentScreenshot(null)
    }
  }, [browserSessionId])

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

  // Execute a browser action
  const executeBrowserAction = useCallback(async (args: ComputerUseToolArgs): Promise<BrowserActionResult> => {
    if (!browserSessionId) {
      throw new Error('Browser session not started')
    }

    let result: BrowserActionResult

    switch (args.action) {
      case 'navigate':
        result = await browserApi.executeBrowserAction(browserSessionId, {
          type: 'navigate',
          url: args.url,
        })
        break

      case 'click':
        result = await browserApi.executeBrowserAction(browserSessionId, {
          type: 'click',
          selector: args.selector,
          x: args.x,
          y: args.y,
        })
        break

      case 'type':
        // For type, we need to handle it specially - type into the currently focused element
        if (args.selector) {
          // Click first to focus, then type
          await browserApi.executeBrowserAction(browserSessionId, {
            type: 'click',
            selector: args.selector,
          })
        }
        result = await browserApi.executeBrowserAction(browserSessionId, {
          type: 'type',
          text: args.text,
          selector: args.selector,
        })
        break

      case 'key':
        // Handle keyboard key press using backend key action
        result = await browserApi.executeBrowserAction(browserSessionId, {
          type: 'key',
          key: args.key || 'Enter',
        })
        break

      case 'scroll':
        result = await browserApi.executeBrowserAction(browserSessionId, {
          type: 'scroll',
          delta_y: args.delta_y || 300,
        })
        break

      case 'wait':
        result = await browserApi.executeBrowserAction(browserSessionId, {
          type: 'wait',
          duration: args.duration || 1000,
        })
        break

      case 'screenshot':
        result = await browserApi.takeScreenshot(browserSessionId)
        break

      case 'back':
        result = await browserApi.executeBrowserAction(browserSessionId, { type: 'back' })
        break

      case 'forward':
        result = await browserApi.executeBrowserAction(browserSessionId, { type: 'forward' })
        break

      case 'refresh':
        result = await browserApi.executeBrowserAction(browserSessionId, { type: 'refresh' })
        break

      default:
        throw new Error(`Unknown action: ${args.action}`)
    }

    if (result.screenshot) {
      setCurrentScreenshot(result)
    }

    return result
  }, [browserSessionId])

  // Main agent loop
  const runAgentLoop = useCallback(async (task: string) => {
    let sessionId = browserSessionId

    if (!sessionId) {
      // Start browser first
      try {
        const result = await browserApi.startBrowserSession(true)
        if (result.success) {
          sessionId = result.session_id
          setBrowserSessionId(sessionId)
          setIsBrowserReady(true)
        } else {
          throw new Error(result.error || 'Failed to start browser')
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to start browser')
        return
      }
    }

    setIsLoading(true)
    setError(null)

    // Create new session
    const newSession = createSession(task, 'computer_use', model)
    newSession.status = 'running'
    newSession.max_steps = 20
    setSession(newSession)
    setSelectedStep(null)

    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()

    // Take initial screenshot
    let initialScreenshot: BrowserActionResult | null = null
    try {
      initialScreenshot = await browserApi.takeScreenshot(sessionId)
      setCurrentScreenshot(initialScreenshot)
    } catch (err) {
      console.error('Failed to take initial screenshot:', err)
    }

    // Build initial messages WITH the screenshot
    const messages: Array<{ role: string; content: string | Array<{ type: string; text?: string; image_url?: { url: string } }> }> = [
      { role: 'system', content: SYSTEM_PROMPT },
    ]

    // Add user task with initial screenshot context
    if (initialScreenshot?.screenshot) {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Task: ${task}\n\nHere is the current browser state. The browser is ready for your first action. Start by navigating to an appropriate website.`,
          },
          {
            type: 'image_url',
            image_url: {
              url: `data:image/png;base64,${initialScreenshot.screenshot}`,
            },
          },
        ],
      })
    } else {
      messages.push({ role: 'user', content: task })
    }

    let stepIndex = 0
    const MAX_ITERATIONS = 20
    let iteration = 0

    // Track actions to prevent premature completion
    const actionsPerformed: string[] = []
    let hasNavigated = false
    let hasTyped = false
    let hasSubmitted = false  // True only after pressing Enter
    let hasWaitedAfterSubmit = false

    try {
      while (iteration < MAX_ITERATIONS) {
        iteration++
        console.log(`[ComputerUse] Iteration ${iteration}/${MAX_ITERATIONS}`)
        console.log(`[ComputerUse] State: navigated=${hasNavigated}, typed=${hasTyped}, submitted=${hasSubmitted}, waited=${hasWaitedAfterSubmit}`)

        // Create thought step
        const thoughtStep = createStep('thought', stepIndex++, {
          thought: `Step ${iteration}: Analyzing screenshot and deciding next action...`,
          status: 'running',
        })
        addStep(thoughtStep)

        // Force tool use until we've completed the search workflow
        // Only allow 'auto' after we've submitted the search and waited
        const shouldForceToolUse = !hasSubmitted || !hasWaitedAfterSubmit

        // Call LLM
        const requestBody = {
          model,
          messages,
          stream: false,
          tools: [computerUseToolDefinition],
          tool_choice: shouldForceToolUse ? 'required' : 'auto',
        }

        console.log(`[ComputerUse] Sending request with ${messages.length} messages`)

        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          signal: abortControllerRef.current?.signal,
        })

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`API error: ${response.status} - ${errorText}`)
        }

        const data = await response.json()
        const choice = data.choices?.[0]
        const message = choice?.message
        const finishReason = choice?.finish_reason

        console.log(`[ComputerUse] Response:`, { finishReason, hasToolCalls: !!message?.tool_calls, content: message?.content?.substring(0, 100) })

        // Update thought with LLM's reasoning
        if (message?.content) {
          updateStep(thoughtStep.id, {
            thought: message.content,
            status: 'completed',
          })
        } else {
          updateStep(thoughtStep.id, { status: 'completed' })
        }

        // Check if LLM wants to use a tool
        if (message?.tool_calls && message.tool_calls.length > 0) {
          for (const toolCall of message.tool_calls) {
            if (toolCall.function.name === 'computer_use') {
              let args: ComputerUseToolArgs
              try {
                const parsedArgs = JSON.parse(toolCall.function.arguments)

                // Infer action if not provided
                let action = parsedArgs.action
                if (!action) {
                  if (parsedArgs.url) action = 'navigate'
                  else if (parsedArgs.key) action = 'key'
                  else if (parsedArgs.text) action = 'type'
                  else if (parsedArgs.selector || parsedArgs.x !== undefined) action = 'click'
                  else if (parsedArgs.delta_y !== undefined) action = 'scroll'
                  else if (parsedArgs.duration !== undefined) action = 'wait'
                  else action = 'screenshot'
                  console.log(`[ComputerUse] Inferred action: ${action}`)
                }

                args = { ...parsedArgs, action }
              } catch {
                console.error('Failed to parse tool arguments:', toolCall.function.arguments)
                continue
              }

              console.log(`[ComputerUse] Executing action:`, args)

              // Create action step
              const actionStep = createStep('action', stepIndex++, {
                action: {
                  type: 'function',
                  name: 'computer_use',
                  arguments: args as unknown as Record<string, unknown>,
                },
                status: 'running',
              })
              addStep(actionStep)

              // Execute the action
              try {
                const result = await executeBrowserAction(args)

                // Track which actions have been performed
                actionsPerformed.push(args.action)
                if (result.success) {
                  if (args.action === 'navigate') {
                    hasNavigated = true
                  } else if (args.action === 'type') {
                    hasTyped = true
                  } else if (args.action === 'key' && args.key?.toLowerCase() === 'enter') {
                    hasSubmitted = true
                  } else if (args.action === 'wait' && hasSubmitted) {
                    hasWaitedAfterSubmit = true
                  }
                }

                console.log(`[ComputerUse] After action: navigated=${hasNavigated}, typed=${hasTyped}, submitted=${hasSubmitted}, waited=${hasWaitedAfterSubmit}`)

                // Update action step
                updateStep(actionStep.id, { status: result.success ? 'completed' : 'failed' })

                // Create observation step
                const observationStep = createStep('observation', stepIndex++, {
                  observation: {
                    content: result.success
                      ? `Action "${args.action}" completed successfully. ${result.url ? `Current URL: ${result.url}` : ''} ${result.title ? `Page title: ${result.title}` : ''}`
                      : `Action "${args.action}" failed: ${result.error}`,
                    error: result.error,
                    screenshot: result.screenshot ? `data:image/png;base64,${result.screenshot}` : undefined,
                    duration_ms: 0,
                  },
                  status: result.success ? 'completed' : 'failed',
                })
                addStep(observationStep)

                // Add assistant message (the tool call)
                messages.push({
                  role: 'assistant',
                  content: message.content || '',
                })

                // Build the next prompt based on what step we're at
                let nextPrompt: string
                if (!result.success) {
                  nextPrompt = `Action "${args.action}" failed: ${result.error}. Try a different approach.`
                } else if (args.action === 'type' && !hasSubmitted) {
                  // Just typed - MUST press Enter next
                  nextPrompt = `You typed the search query. Now you MUST press Enter to submit the search. Use: computer_use({"action": "key", "key": "Enter"})`
                } else if (args.action === 'key' && args.key?.toLowerCase() === 'enter') {
                  // Just pressed Enter - MUST wait for results
                  nextPrompt = `Search submitted. Now WAIT for results to load: computer_use({"action": "wait", "duration": 2000})`
                } else if (args.action === 'wait' && hasSubmitted) {
                  // Waited after submit - NOW can provide answer
                  nextPrompt = `Results should now be loaded. Look at the screenshot and provide your final answer based on what you see.`
                } else if (hasSubmitted && hasWaitedAfterSubmit) {
                  // Ready for final answer
                  nextPrompt = `Analyze the search results in the screenshot and provide your final answer.`
                } else {
                  // Normal continuation
                  nextPrompt = `Action "${args.action}" completed. ${result.url ? `URL: ${result.url}` : ''} ${result.title ? `Title: ${result.title}` : ''}\n\nAnalyze the screenshot and continue with the next action.`
                }

                // Add the result with screenshot
                if (result.screenshot) {
                  messages.push({
                    role: 'user',
                    content: [
                      {
                        type: 'text',
                        text: nextPrompt,
                      },
                      {
                        type: 'image_url',
                        image_url: {
                          url: `data:image/png;base64,${result.screenshot}`,
                        },
                      },
                    ],
                  })
                } else {
                  messages.push({
                    role: 'user',
                    content: nextPrompt,
                  })
                }

                // Small delay between actions
                await new Promise(resolve => setTimeout(resolve, 500))

              } catch (err) {
                const errorMsg = err instanceof Error ? err.message : 'Unknown error'
                console.error(`[ComputerUse] Action error:`, errorMsg)
                updateStep(actionStep.id, { status: 'failed' })

                const observationStep = createStep('observation', stepIndex++, {
                  observation: {
                    content: `Error: ${errorMsg}`,
                    error: errorMsg,
                    duration_ms: 0,
                  },
                  status: 'failed',
                })
                addStep(observationStep)

                messages.push({
                  role: 'user',
                  content: `The action failed with error: ${errorMsg}. Please try a different approach.`,
                })
              }
            }
          }
        } else if (finishReason === 'stop' && message?.content) {
          // LLM provided final answer
          console.log(`[ComputerUse] Final answer received`)
          const finalStep = createStep('final_answer', stepIndex++, {
            answer: message.content,
            status: 'completed',
          })
          addStep(finalStep)

          setSession(prev =>
            prev
              ? {
                  ...prev,
                  status: 'completed',
                  completed_at: Date.now(),
                  final_response: message.content,
                }
              : null
          )
          break
        } else {
          // No tool call and no stop
          console.log('[ComputerUse] No tool call, prompting to continue')
          messages.push({
            role: 'user',
            content: 'Please use the computer_use tool to perform an action, or provide your final answer if the task is complete.',
          })
        }
      }

      if (iteration >= MAX_ITERATIONS) {
        setSession(prev =>
          prev
            ? {
                ...prev,
                status: 'failed',
                error: 'Maximum iterations reached',
              }
            : null
        )
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setSession(prev => (prev ? { ...prev, status: 'cancelled' } : null))
      } else {
        const errorMessage = err instanceof Error ? err.message : 'An error occurred'
        console.error('[ComputerUse] Error:', errorMessage)
        setError(errorMessage)
        setSession(prev =>
          prev ? { ...prev, status: 'failed', error: errorMessage } : null
        )
      }
    } finally {
      setIsLoading(false)
    }
  }, [browserSessionId, model, endpoint, addStep, updateStep, executeBrowserAction])

  // Handle submit
  const handleSubmit = useCallback(async () => {
    if (!inputValue.trim() || isLoading) return
    const task = inputValue.trim()
    setInputValue('')
    await runAgentLoop(task)
  }, [inputValue, isLoading, runAgentLoop])

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
    setError(null)
  }, [])

  // Handle key press
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit]
  )

  // Start browser on mount
  useEffect(() => {
    startBrowser()
    return () => {
      stopBrowser()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>Computer-Use Agent</h2>
          <div className={styles.browserStatus}>
            <div
              className={`${styles.statusDot} ${isBrowserReady ? styles.statusReady : styles.statusNotReady}`}
            />
            <span>{isBrowserReady ? 'Browser Ready' : 'Browser Not Ready'}</span>
          </div>
        </div>
        <div className={styles.headerRight}>
          <button
            className={styles.headerButton}
            onClick={isBrowserReady ? stopBrowser : startBrowser}
            disabled={isLoading}
          >
            {isBrowserReady ? 'Stop Browser' : 'Start Browser'}
          </button>
          <button
            className={styles.headerButton}
            onClick={handleClear}
            disabled={isLoading || !session}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className={styles.errorBanner}>
          <span>{error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {/* Main content */}
      <div className={styles.content}>
        {/* Screenshot panel */}
        <div className={styles.screenshotPanel}>
          <ScreenshotViewer
            screenshot={currentScreenshot?.screenshot || null}
            url={currentScreenshot?.url}
            title={currentScreenshot?.title}
            width={currentScreenshot?.width || 1280}
            height={currentScreenshot?.height || 800}
            isLoading={!isBrowserReady && !error}
            error={!isBrowserReady && error ? error : undefined}
          />
        </div>

        {/* Steps panel */}
        <div className={styles.stepsPanel}>
          <StepTimeline
            session={session}
            selectedStep={selectedStep}
            onStepSelect={setSelectedStep}
            autoScroll={true}
          />
        </div>
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
            placeholder="Describe what you want the agent to do... (e.g., 'Search for weather in Tel Aviv on Google')"
            disabled={isLoading || !isBrowserReady}
            rows={1}
          />
          <div className={styles.inputActions}>
            {isLoading ? (
              <button className={styles.cancelButton} onClick={handleCancel}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                </svg>
              </button>
            ) : (
              <button
                className={styles.submitButton}
                onClick={handleSubmit}
                disabled={!inputValue.trim() || !isBrowserReady}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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

export default ComputerUsePlayground
