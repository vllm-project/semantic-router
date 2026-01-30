/**
 * E2BComputerUsePlayground - Computer-use agent that controls E2B cloud desktop
 * Uses E2B Desktop Sandbox for isolated cloud-based desktop automation
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import styles from './E2BComputerUsePlayground.module.css'
import VncViewer from './VncViewer'
import { useAgentWebSocket, AgentStepEvent, AgentTraceMetadata } from '../../hooks/useAgentWebSocket'

interface E2BComputerUsePlaygroundProps {
  /** Default model to use */
  defaultModel?: string
}

// Step card component for displaying agent steps
const StepCard = ({
  step,
  isExpanded,
  onToggle,
}: {
  step: AgentStepEvent
  isExpanded: boolean
  onToggle: () => void
}) => {
  const hasThought = step.thought && step.thought.trim()
  const hasActions = step.actions && step.actions.length > 0
  const hasError = step.error

  return (
    <div className={`${styles.stepCard} ${hasError ? styles.stepError : ''}`}>
      <div className={styles.stepHeader} onClick={onToggle}>
        <div className={styles.stepNumber}>Step {step.stepId}</div>
        <div className={styles.stepMeta}>
          <span className={styles.stepTokens}>
            {step.inputTokensUsed + step.outputTokensUsed} tokens
          </span>
          <span className={styles.stepDuration}>
            {step.duration.toFixed(1)}s
          </span>
        </div>
        <svg
          className={`${styles.expandIcon} ${isExpanded ? styles.expanded : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {isExpanded && (
        <div className={styles.stepContent}>
          {/* Screenshot */}
          {step.image && (
            <div className={styles.stepScreenshot}>
              <img src={step.image} alt={`Step ${step.stepId} screenshot`} />
            </div>
          )}

          {/* Thought */}
          {hasThought && (
            <div className={styles.stepSection}>
              <div className={styles.sectionLabel}>Thought</div>
              <div className={styles.sectionContent}>{step.thought}</div>
            </div>
          )}

          {/* Actions */}
          {hasActions && (
            <div className={styles.stepSection}>
              <div className={styles.sectionLabel}>Actions</div>
              <div className={styles.actionsGrid}>
                {step.actions.map((action, idx) => (
                  <div key={idx} className={styles.actionItem}>
                    <code>{action.description || action.original_string}</code>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {hasError && (
            <div className={styles.stepSection}>
              <div className={styles.sectionLabel}>Error</div>
              <div className={styles.errorContent}>{step.error}</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Metadata display component
const MetadataBar = ({ metadata }: { metadata: AgentTraceMetadata | null }) => {
  if (!metadata) return null

  return (
    <div className={styles.metadataBar}>
      <div className={styles.metaItem}>
        <span className={styles.metaLabel}>Steps</span>
        <span className={styles.metaValue}>
          {metadata.numberOfSteps} / {metadata.maxSteps}
        </span>
      </div>
      <div className={styles.metaItem}>
        <span className={styles.metaLabel}>Tokens</span>
        <span className={styles.metaValue}>
          {metadata.inputTokensUsed + metadata.outputTokensUsed}
        </span>
      </div>
      <div className={styles.metaItem}>
        <span className={styles.metaLabel}>Duration</span>
        <span className={styles.metaValue}>{metadata.duration.toFixed(1)}s</span>
      </div>
      {metadata.final_state && (
        <div className={`${styles.metaItem} ${styles.stateItem}`}>
          <span
            className={`${styles.stateBadge} ${
              metadata.final_state === 'success'
                ? styles.stateSuccess
                : metadata.final_state === 'stopped'
                ? styles.stateStopped
                : styles.stateError
            }`}
          >
            {metadata.final_state.replace('_', ' ')}
          </span>
        </div>
      )}
    </div>
  )
}

// Available models
const AVAILABLE_MODELS = [
  { id: 'envoy/auto', name: 'Auto', provider: 'Semantic Router' },
  { id: 'ollama/qwen3-vl:8b', name: 'Qwen3-VL 8B', provider: 'Ollama (Recommended)' },
  { id: 'Qwen/Qwen3-VL-8B-Instruct', name: 'Qwen3-VL 8B', provider: 'HuggingFace' },
  { id: 'Qwen/Qwen3-VL-30B-A3B-Instruct', name: 'Qwen3-VL 30B', provider: 'HuggingFace' },
  { id: 'ollama/qwen2.5vl:7b', name: 'Qwen2.5-VL 7B', provider: 'Ollama' },
  { id: 'ollama/llava:7b', name: 'LLaVA 7B', provider: 'Ollama' },
  { id: 'openai/gpt-4o', name: 'GPT-4o', provider: 'OpenAI' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI' },
]

const E2BComputerUsePlayground = ({
  defaultModel = 'envoy/auto',
}: E2BComputerUsePlaygroundProps) => {
  // State
  const [inputValue, setInputValue] = useState('')
  const [selectedModel, setSelectedModel] = useState(defaultModel)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const stepsContainerRef = useRef<HTMLDivElement>(null)

  // WebSocket hook
  const {
    connectionState,
    vncUrl,
    isRunning,
    steps,
    metadata,
    error,
    connect,
    disconnect,
    submitTask,
    stopTask,
  } = useAgentWebSocket({
    autoConnect: false,
    onAgentProgress: (event) => {
      // Auto-expand new steps
      setExpandedSteps((prev) => new Set([...prev, event.agentStep.stepId]))

      // Auto-scroll to latest step
      setTimeout(() => {
        stepsContainerRef.current?.scrollTo({
          top: stepsContainerRef.current.scrollHeight,
          behavior: 'smooth',
        })
      }, 100)
    },
  })

  const isConnected = connectionState === 'connected'

  // Handle submit
  const handleSubmit = useCallback(() => {
    if (!inputValue.trim() || isRunning || !isConnected) return
    submitTask(inputValue.trim(), selectedModel)
    setInputValue('')
    setExpandedSteps(new Set())
  }, [inputValue, isRunning, isConnected, selectedModel, submitTask])

  // Handle stop
  const handleStop = useCallback(() => {
    stopTask()
  }, [stopTask])

  // Handle connect/disconnect
  const handleConnectionToggle = useCallback(() => {
    if (isConnected) {
      disconnect()
    } else {
      connect()
    }
  }, [isConnected, connect, disconnect])

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

  // Toggle step expansion
  const toggleStep = useCallback((stepId: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(stepId)) {
        next.delete(stepId)
      } else {
        next.add(stepId)
      }
      return next
    })
  }, [])

  // Clear steps
  const handleClear = useCallback(() => {
    setExpandedSteps(new Set())
    // Steps will be cleared when a new task is submitted
  }, [])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>E2B Computer-Use Agent</h2>
          <div
            className={`${styles.connectionBadge} ${
              isConnected ? styles.connected : styles.disconnected
            }`}
          >
            <div className={styles.connectionDot} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
        <div className={styles.headerRight}>
          <select
            className={styles.modelSelect}
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isRunning}
          >
            {AVAILABLE_MODELS.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.provider})
              </option>
            ))}
          </select>
          <button
            className={`${styles.headerButton} ${isConnected ? styles.disconnectButton : ''}`}
            onClick={handleConnectionToggle}
            disabled={isRunning}
          >
            {isConnected ? 'Disconnect' : 'Connect'}
          </button>
          {isRunning && (
            <button className={styles.stopButton} onClick={handleStop}>
              Stop
            </button>
          )}
          <button
            className={styles.headerButton}
            onClick={handleClear}
            disabled={isRunning || steps.length === 0}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className={styles.errorBanner}>
          <span>{error}</span>
        </div>
      )}

      {/* Metadata bar */}
      <MetadataBar metadata={metadata} />

      {/* Main content */}
      <div className={styles.content}>
        {/* VNC viewer panel */}
        <div className={styles.vncPanel}>
          <VncViewer
            vncUrl={vncUrl}
            isConnected={isConnected}
            isRunning={isRunning}
            error={!isConnected && !isRunning ? 'Connect to start' : null}
          />
        </div>

        {/* Steps panel */}
        <div className={styles.stepsPanel}>
          <div className={styles.stepsPanelHeader}>
            <h3>Agent Steps</h3>
            <span className={styles.stepCount}>{steps.length} steps</span>
          </div>
          <div className={styles.stepsContainer} ref={stepsContainerRef}>
            {steps.length === 0 ? (
              <div className={styles.stepsPlaceholder}>
                <p>Submit a task to see agent steps</p>
              </div>
            ) : (
              steps.map((step) => (
                <StepCard
                  key={step.stepId}
                  step={step}
                  isExpanded={expandedSteps.has(step.stepId)}
                  onToggle={() => toggleStep(step.stepId)}
                />
              ))
            )}
          </div>
        </div>
      </div>

      {/* Input area */}
      <div className={styles.inputArea}>
        <div className={styles.inputWrapper}>
          <textarea
            ref={inputRef}
            className={styles.input}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isConnected
                ? 'Describe what you want the agent to do... (e.g., "Search for weather in Tel Aviv on Google")'
                : 'Connect to the agent service to start...'
            }
            disabled={isRunning || !isConnected}
            rows={1}
          />
          <div className={styles.inputActions}>
            <button
              className={styles.submitButton}
              onClick={handleSubmit}
              disabled={!inputValue.trim() || isRunning || !isConnected}
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
          </div>
        </div>
      </div>
    </div>
  )
}

export default E2BComputerUsePlayground
