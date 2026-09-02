import { useDeferredValue, useMemo, useState } from 'react'

import ProductIcon from '../ProductIcon'
import type {
  EvaluationRun,
  EvaluationRunEvent,
  EvaluationRunStatus,
} from '../../types/evaluationPlane'
import { formatDateTime, formatDurationBetween } from '../../utils/dateTime'
import { RunStatusBadge, TrackChips } from './EvaluationPrimitives'
import styles from './EvaluationPlane.module.css'

interface EvaluationRunsProps {
  runs: EvaluationRun[]
  selectedRunID: string | null
  events: EvaluationRunEvent[]
  eventsConnected: boolean
  eventsError: string | null
  canRun: boolean
  canDelete: boolean
  pending: boolean
  onSelect: (run: EvaluationRun) => void
  onStart: (run: EvaluationRun) => void
  onCancel: (run: EvaluationRun) => void
  onDelete: (run: EvaluationRun) => void
  onOpenReport: (run: EvaluationRun) => void
  onRefresh: () => void
}

export default function EvaluationRuns({
  runs,
  selectedRunID,
  events,
  eventsConnected,
  eventsError,
  canRun,
  canDelete,
  pending,
  onSelect,
  onStart,
  onCancel,
  onDelete,
  onOpenReport,
  onRefresh,
}: EvaluationRunsProps) {
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState<EvaluationRunStatus | 'all'>('all')
  const deferredSearch = useDeferredValue(search.trim().toLowerCase())
  const selectedRun = runs.find((run) => run.id === selectedRunID) || null
  const filteredRuns = useMemo(
    () =>
      runs.filter((run) => {
        if (status !== 'all' && run.status !== status) return false
        if (!deferredSearch) return true
        return [
          run.id,
          run.name,
          run.description,
          run.target_id,
          run.change_profile,
          ...run.track_ids,
        ]
          .join(' ')
          .toLowerCase()
          .includes(deferredSearch)
      }),
    [deferredSearch, runs, status],
  )

  return (
    <div className={styles.sectionStack}>
      <section className={styles.panel}>
        <div className={styles.panelHeader}>
          <div>
            <span className={styles.eyebrow}>Execution ledger</span>
            <h2>Evaluation runs</h2>
            <p>Every run is an immutable experiment snapshot with its own event stream.</p>
          </div>
          <button
            type="button"
            className={styles.iconButton}
            onClick={onRefresh}
            aria-label="Refresh evaluation runs"
          >
            <ProductIcon name="refresh" /> Refresh
          </button>
        </div>
        <div className={styles.filters}>
          <label>
            <span>Search</span>
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Run, target, or track"
            />
          </label>
          <label>
            <span>Status</span>
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value as EvaluationRunStatus | 'all')}
            >
              <option value="all">All statuses</option>
              <option value="pending">Pending</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </label>
          <span className={styles.resultCount}>
            {filteredRuns.length} of {runs.length} runs
          </span>
        </div>

        {filteredRuns.length === 0 ? (
          <div className={styles.emptyState}>
            <p>No evaluation runs match this view.</p>
          </div>
        ) : (
          <div className={styles.runList}>
            {filteredRuns.map((run) => (
              <article
                key={run.id}
                className={`${styles.runRow} ${selectedRunID === run.id ? styles.runSelected : ''}`}
              >
                <button type="button" className={styles.runSummary} onClick={() => onSelect(run)}>
                  <div className={styles.runIdentity}>
                    <div>
                      <strong>{run.name}</strong>
                      <code>{run.id}</code>
                    </div>
                    <p>{run.description || 'No experiment description.'}</p>
                    <TrackChips trackIDs={run.track_ids} />
                  </div>
                  <div className={styles.runMetadata}>
                    <RunStatusBadge status={run.status} />
                    <span>
                      {run.mode} · {run.evidence_level} · {run.change_profile}
                    </span>
                    <span>{formatDateTime(run.created_at)}</span>
                  </div>
                </button>
                <div className={styles.runProgress}>
                  <div
                    className={styles.progressTrack}
                    role="progressbar"
                    aria-label={`${run.name} progress`}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(run.progress.percent)}
                  >
                    <span
                      style={{ width: `${Math.max(0, Math.min(100, run.progress.percent))}%` }}
                    />
                  </div>
                  <small>
                    {run.progress.completed}/{run.progress.total} ·{' '}
                    {run.progress.message || 'Awaiting execution'}
                  </small>
                </div>
                <div className={styles.rowActions}>
                  {run.status === 'pending' && canRun ? (
                    <button type="button" disabled={pending} onClick={() => onStart(run)}>
                      <ProductIcon name="play" /> Start
                    </button>
                  ) : null}
                  {run.status === 'running' && canRun ? (
                    <button
                      type="button"
                      disabled={pending}
                      className={styles.warningButton}
                      onClick={() => onCancel(run)}
                    >
                      <ProductIcon name="close" /> Cancel
                    </button>
                  ) : null}
                  {['completed', 'failed', 'cancelled'].includes(run.status) ? (
                    <button type="button" onClick={() => onOpenReport(run)}>
                      <ProductIcon name="chart" /> Report
                    </button>
                  ) : null}
                  {run.status !== 'running' && canDelete ? (
                    <button
                      type="button"
                      disabled={pending}
                      className={styles.dangerButton}
                      onClick={() => onDelete(run)}
                    >
                      <ProductIcon name="trash" /> Delete
                    </button>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      {selectedRun ? (
        <section className={styles.panel} aria-label="Selected evaluation run details">
          <div className={styles.panelHeader}>
            <div>
              <span className={styles.eyebrow}>Run inspector</span>
              <h2>{selectedRun.name}</h2>
              <p>
                {selectedRun.target_id} · seed {selectedRun.seed} · sample limit{' '}
                {selectedRun.sample_limit} · concurrency {selectedRun.concurrency}
              </p>
            </div>
            <RunStatusBadge status={selectedRun.status} />
          </div>
          <dl className={styles.definitionGrid}>
            <div>
              <dt>Mode</dt>
              <dd>{selectedRun.mode}</dd>
            </div>
            <div>
              <dt>Evidence</dt>
              <dd>{selectedRun.evidence_level}</dd>
            </div>
            <div>
              <dt>Change profile</dt>
              <dd>{selectedRun.change_profile}</dd>
            </div>
            <div>
              <dt>Started</dt>
              <dd>{formatDateTime(selectedRun.started_at)}</dd>
            </div>
            <div>
              <dt>Duration</dt>
              <dd>{formatDurationBetween(selectedRun.started_at, selectedRun.completed_at)}</dd>
            </div>
            <div>
              <dt>Suites</dt>
              <dd>{selectedRun.suite_ids.join(', ') || '-'}</dd>
            </div>
            <div>
              <dt>Baseline</dt>
              <dd>{selectedRun.baseline_run_id || '-'}</dd>
            </div>
          </dl>
          {selectedRun.error ? (
            <div className={styles.errorBanner} role="alert">
              {selectedRun.error}
            </div>
          ) : null}

          <div className={styles.eventHeader}>
            <h3>Live events</h3>
            {selectedRun.status === 'running' ? (
              <span className={eventsConnected ? styles.live : styles.offline}>
                {eventsConnected ? 'Connected' : 'Reconnecting'}
              </span>
            ) : (
              <span className={styles.offline}>Terminal</span>
            )}
          </div>
          {eventsError ? <p className={styles.inlineError}>{eventsError}</p> : null}
          {events.length === 0 ? (
            <p className={styles.emptyCopy}>
              {selectedRun.status === 'running'
                ? 'Waiting for the first event…'
                : 'No live events retained in this browser session.'}
            </p>
          ) : (
            <ol className={styles.eventList}>
              {events.map((event, index) => (
                <li key={event.id || `${event.timestamp}-${index}`}>
                  <time>{formatDateTime(event.timestamp)}</time>
                  <div>
                    <strong>{event.track_id || event.type}</strong>
                    <span>{event.message}</span>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </section>
      ) : null}
    </div>
  )
}
