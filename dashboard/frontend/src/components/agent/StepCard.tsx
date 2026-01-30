/**
 * StepCard - Individual step visualization component
 * Displays thought/action/observation/final_answer steps with appropriate styling
 */

import { useState, useCallback, useMemo } from 'react'
import styles from './StepCard.module.css'
import MarkdownRenderer from '../MarkdownRenderer'
import type { AgentStep, StepStatus, AgentStepType } from '../../types/agent'
import { STEP_THEMES, STATUS_COLORS, getStepTypeName, formatDuration } from '../../types/agent'

interface StepCardProps {
  step: AgentStep
  isSelected?: boolean
  onSelect?: (step: AgentStep) => void
  showDetails?: boolean
}

// Step type icons as SVG components
const StepIcon = ({ type }: { type: AgentStepType }) => {
  switch (type) {
    case 'thought':
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 18h6" />
          <path d="M10 22h4" />
          <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8c0-3.31-2.69-6-6-6S6 4.69 6 8c0 1.33.47 2.55 1.5 3.5.76.76 1.23 1.52 1.41 2.5" />
        </svg>
      )
    case 'action':
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
        </svg>
      )
    case 'observation':
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      )
    case 'final_answer':
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      )
  }
}

// Status indicator component
const StatusIndicator = ({ status }: { status: StepStatus }) => {
  const color = STATUS_COLORS[status]

  return (
    <div
      className={`${styles.statusIndicator} ${status === 'running' ? styles.statusRunning : ''}`}
      style={{ backgroundColor: color }}
      title={status.charAt(0).toUpperCase() + status.slice(1)}
    >
      {status === 'completed' && (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      )}
      {status === 'failed' && (
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      )}
    </div>
  )
}

const StepCard = ({ step, isSelected = false, onSelect, showDetails = true }: StepCardProps) => {
  const [isExpanded, setIsExpanded] = useState(true)
  const theme = STEP_THEMES[step.type]

  const handleClick = useCallback(() => {
    if (onSelect) {
      onSelect(step)
    }
  }, [onSelect, step])

  const handleToggle = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    setIsExpanded(!isExpanded)
  }, [isExpanded])

  // Get the main content based on step type
  const mainContent = useMemo(() => {
    switch (step.type) {
      case 'thought':
        return step.thought || ''
      case 'action':
        if (step.action) {
          return `**${step.action.name}**(${JSON.stringify(step.action.arguments, null, 2)})`
        }
        return ''
      case 'observation':
        if (step.observation) {
          if (step.observation.error) {
            return `**Error:** ${step.observation.error}`
          }
          const content = step.observation.content
          if (typeof content === 'string') {
            return content
          }
          return '```json\n' + JSON.stringify(content, null, 2) + '\n```'
        }
        return ''
      case 'final_answer':
        return step.answer || ''
      default:
        return ''
    }
  }, [step])

  // Get truncated preview
  const preview = useMemo(() => {
    const text = mainContent.replace(/[#*`]/g, '').trim()
    if (text.length > 100) {
      return text.slice(0, 100) + '...'
    }
    return text
  }, [mainContent])

  // Calculate duration display
  const durationDisplay = useMemo(() => {
    if (step.type === 'observation' && step.observation?.duration_ms) {
      return formatDuration(step.observation.duration_ms)
    }
    return null
  }, [step])

  return (
    <div
      className={`${styles.container} ${isSelected ? styles.selected : ''} ${step.status === 'failed' ? styles.failed : ''}`}
      style={{
        '--step-primary': theme.primary,
        '--step-background': theme.background,
        '--step-border': theme.border,
      } as React.CSSProperties}
      onClick={handleClick}
    >
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <StatusIndicator status={step.status} />
          <div className={styles.iconWrapper} style={{ color: theme.primary }}>
            <StepIcon type={step.type} />
          </div>
          <div className={styles.headerInfo}>
            <span className={styles.stepIndex}>Step {step.index + 1}</span>
            <span className={styles.stepType} style={{ color: theme.primary }}>
              {getStepTypeName(step.type)}
              {step.type === 'action' && step.action && (
                <span className={styles.actionName}>: {step.action.name}</span>
              )}
            </span>
          </div>
        </div>
        <div className={styles.headerRight}>
          {durationDisplay && (
            <span className={styles.duration}>{durationDisplay}</span>
          )}
          {showDetails && mainContent && (
            <button
              className={styles.expandButton}
              onClick={handleToggle}
              aria-expanded={isExpanded}
              aria-label={isExpanded ? 'Collapse' : 'Expand'}
            >
              <svg
                className={`${styles.expandIcon} ${isExpanded ? styles.expanded : ''}`}
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Progress bar for running state */}
      {step.status === 'running' && (
        <div className={styles.progressBar}>
          <div className={styles.progressFill} style={{ backgroundColor: theme.primary }} />
        </div>
      )}

      {/* Content area */}
      {showDetails && mainContent && (
        <>
          {/* Preview when collapsed */}
          {!isExpanded && (
            <div className={styles.preview}>
              {preview}
            </div>
          )}

          {/* Full content when expanded */}
          {isExpanded && (
            <div className={styles.content}>
              {step.type === 'action' && step.action ? (
                <div className={styles.actionContent}>
                  <div className={styles.actionHeader}>
                    <span className={styles.actionLabel}>Tool:</span>
                    <code className={styles.toolName}>{step.action.name}</code>
                  </div>
                  <div className={styles.actionArgs}>
                    <span className={styles.actionLabel}>Arguments:</span>
                    <pre className={styles.codeBlock}>
                      {JSON.stringify(step.action.arguments, null, 2)}
                    </pre>
                  </div>
                  {step.action.code && (
                    <div className={styles.actionCode}>
                      <span className={styles.actionLabel}>Code:</span>
                      <pre className={styles.codeBlock}>{step.action.code}</pre>
                    </div>
                  )}
                </div>
              ) : step.type === 'observation' && step.observation?.screenshot ? (
                <div className={styles.observationContent}>
                  <img
                    src={step.observation.screenshot}
                    alt="Screenshot"
                    className={styles.screenshot}
                  />
                  {step.observation.error && (
                    <div className={styles.errorMessage}>
                      {step.observation.error}
                    </div>
                  )}
                </div>
              ) : (
                <MarkdownRenderer content={mainContent} />
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default StepCard
