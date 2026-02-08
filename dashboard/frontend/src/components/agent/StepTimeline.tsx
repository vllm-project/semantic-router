/**
 * StepTimeline - Vertical timeline of agent steps
 * Displays a scrollable list of steps with connecting timeline
 */

import { useRef, useEffect, useCallback } from 'react'
import styles from './StepTimeline.module.css'
import StepCard from './StepCard'
import type { AgentStep, AgentSession } from '../../types/agent'
import { STEP_THEMES, STATUS_COLORS } from '../../types/agent'

interface StepTimelineProps {
  session: AgentSession | null
  selectedStep: AgentStep | null
  onStepSelect: (step: AgentStep) => void
  autoScroll?: boolean
}

// Empty state component
const EmptyState = () => (
  <div className={styles.emptyState}>
    <svg
      width="48"
      height="48"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={styles.emptyIcon}
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
    <p className={styles.emptyText}>No steps yet</p>
    <p className={styles.emptySubtext}>Enter a task to start the agent</p>
  </div>
)

// Running indicator for the timeline
const RunningIndicator = () => (
  <div className={styles.runningIndicator}>
    <div className={styles.runningDot} />
    <span className={styles.runningText}>Processing...</span>
  </div>
)

const StepTimeline = ({
  session,
  selectedStep,
  onStepSelect,
  autoScroll = true
}: StepTimelineProps) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const lastStepRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to the latest step
  useEffect(() => {
    if (autoScroll && lastStepRef.current && session?.steps.length) {
      lastStepRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      })
    }
  }, [autoScroll, session?.steps.length])

  const handleStepSelect = useCallback((step: AgentStep) => {
    onStepSelect(step)
  }, [onStepSelect])

  // No session or empty steps
  if (!session || session.steps.length === 0) {
    return (
      <div className={styles.container}>
        <EmptyState />
      </div>
    )
  }

  const { steps, status } = session
  const isRunning = status === 'running'

  return (
    <div className={styles.container} ref={containerRef}>
      {/* Session header */}
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <div
            className={`${styles.statusBadge} ${styles[status]}`}
            style={{ backgroundColor: STATUS_COLORS[status === 'idle' ? 'pending' : status === 'cancelled' ? 'failed' : status === 'running' ? 'running' : 'completed'] }}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </div>
          <span className={styles.stepCount}>
            {steps.length} step{steps.length !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Timeline */}
      <div className={styles.timeline}>
        {/* Timeline line */}
        <div className={styles.timelineLine} />

        {/* Steps */}
        {steps.map((step, index) => {
          const isLast = index === steps.length - 1
          const isSelected = selectedStep?.id === step.id
          const theme = STEP_THEMES[step.type]

          return (
            <div
              key={step.id}
              className={styles.stepWrapper}
              ref={isLast ? lastStepRef : undefined}
            >
              {/* Timeline node */}
              <div className={styles.timelineNode}>
                <div
                  className={`${styles.timelineDot} ${step.status === 'running' ? styles.timelineDotRunning : ''}`}
                  style={{
                    backgroundColor: step.status === 'running'
                      ? STATUS_COLORS.running
                      : step.status === 'failed'
                        ? STATUS_COLORS.failed
                        : theme.primary
                  }}
                />
              </div>

              {/* Step card */}
              <div className={styles.stepContent}>
                <StepCard
                  step={step}
                  isSelected={isSelected}
                  onSelect={handleStepSelect}
                  showDetails={true}
                />
              </div>
            </div>
          )
        })}

        {/* Running indicator at the end */}
        {isRunning && (
          <div className={styles.stepWrapper}>
            <div className={styles.timelineNode}>
              <div className={`${styles.timelineDot} ${styles.timelineDotPending}`} />
            </div>
            <div className={styles.stepContent}>
              <RunningIndicator />
            </div>
          </div>
        )}
      </div>

      {/* Session completion message */}
      {session.status === 'completed' && session.final_response && (
        <div className={styles.completionMessage}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          <span>Task completed</span>
        </div>
      )}

      {/* Session error message */}
      {session.status === 'failed' && session.error && (
        <div className={styles.errorMessage}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span>{session.error}</span>
        </div>
      )}
    </div>
  )
}

export default StepTimeline
