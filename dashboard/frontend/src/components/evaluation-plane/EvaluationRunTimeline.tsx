import { useId } from 'react'

import type { EvaluationRun, EvaluationRunEvent } from '../../types/evaluationPlane'
import { formatDateTime } from '../../utils/dateTime'
import EvaluationIssueDetails, { type EvaluationIssueDetail } from './EvaluationIssueDetails'
import { EvaluationActionButton } from './EvaluationPrimitives'
import { evaluationRunEventLabel } from './evaluationRunTimelinePresentation'
import planeStyles from './EvaluationPlane.module.css'
import styles from './EvaluationRunTimeline.module.css'

interface EvaluationRunTimelineProps {
  run: EvaluationRun
  events: EvaluationRunEvent[]
  connected: boolean
  error: string | null
  onReconnect: () => void
}

function eventProductMessage(event: EvaluationRunEvent): string {
  switch (event.type) {
    case 'snapshot':
      return 'The run state was saved.'
    case 'progress':
      return event.progress && event.progress.total > 0
        ? `${Math.round(event.progress.percent)}% complete · ${event.progress.completed} of ${event.progress.total} steps finished.`
        : 'Run progress was updated.'
    case 'track': {
      const resultLabel = event.payload.record_count === 1 ? 'result' : 'results'
      return `${event.payload.record_count.toLocaleString()} ${resultLabel} recorded for this evaluation area.`
    }
    case 'gate':
      return 'A readiness check was evaluated.'
    case 'artifact':
      return 'A report output was saved.'
    case 'completed':
      return 'The run completed and its report is ready.'
    case 'failed':
      return 'The run stopped before completing. Review the available evidence before retrying.'
    case 'cancelled':
      return 'The run was cancelled before completion.'
  }
}

export default function EvaluationRunTimeline({
  run,
  events,
  connected,
  error,
  onReconnect,
}: EvaluationRunTimelineProps) {
  const timelineTitleID = useId()
  const active = run.status === 'running' || run.status === 'sealing'
  const eventDetails: EvaluationIssueDetail[] = events
    .filter((event) => event.message.trim())
    .map((event) => ({
      label: `${evaluationRunEventLabel(event)} · ${formatDateTime(event.timestamp)}`,
      message: event.message,
    }))

  return (
    <>
      <div className={styles.eventHeader}>
        <h4 id={timelineTitleID}>Execution timeline</h4>
        <span className={connected ? styles.live : styles.offline}>
          {connected
            ? 'Live updates'
            : error
              ? 'Updates unavailable'
              : active
                ? 'Connecting'
                : 'Run history'}
        </span>
      </div>
      {error ? (
        <div className={planeStyles.inlineError} role="alert">
          <div>
            <strong>Live updates unavailable</strong>
            <span>Reconnect to resume new timeline updates. Saved run evidence is unaffected.</span>
            <EvaluationIssueDetails issues={[{ label: 'Live updates', message: error }]} />
          </div>
          <EvaluationActionButton type="button" compact onClick={onReconnect}>
            Reconnect
          </EvaluationActionButton>
        </div>
      ) : null}
      {events.length === 0 ? (
        <p className={planeStyles.emptyCopy}>
          {active
            ? run.status === 'sealing'
              ? 'Finalizing results…'
              : 'Waiting for the first event…'
            : 'No run updates are available.'}
        </p>
      ) : (
        <div
          className={styles.eventList}
          role="region"
          aria-labelledby={timelineTitleID}
          tabIndex={0}
        >
          <ol className={styles.eventItems}>
            {events.map((event, index) => (
              <li key={event.id || `${event.timestamp}-${index}`}>
                <time>{formatDateTime(event.timestamp)}</time>
                <div>
                  <strong>{evaluationRunEventLabel(event)}</strong>
                  <span>{eventProductMessage(event)}</span>
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
      <EvaluationIssueDetails className={styles.timelineTechnicalDetails} issues={eventDetails} />
    </>
  )
}
