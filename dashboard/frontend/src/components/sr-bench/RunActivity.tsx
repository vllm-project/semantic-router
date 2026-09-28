import { useState } from 'react'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import BenchPagination from './BenchPagination'
import { active, number } from './model'
import { duration, elapsedSeconds, responseBytes } from './activityPresentation'
import { useActiveCalls } from './useActiveCalls'
import { useActivityClock } from './useActivityClock'
import type { CallActivity, CallRecord, Run } from './types'
import styles from './RunActivity.module.css'

const phases: Record<CallActivity['phase'], string> = {
  preparing: 'Preparing request',
  waiting: 'Waiting for response',
  streaming: 'Receiving response',
}

function callLabel(run: Run, call: CallRecord) {
  const role =
    call.role === 'subject'
      ? 'Model response'
      : call.role === 'judge'
        ? 'Judge'
        : call.role === 'simulator'
          ? 'Simulator'
          : 'Auxiliary call'
  const question =
    run.manifest.cases?.findIndex(
      (value) =>
        value && typeof value === 'object' && (value as { id?: string }).id === call.case_id,
    ) ?? -1
  return question >= 0 ? `${role} · Question ${question + 1}` : role
}

export default function RunActivity({ run, revision }: { run: Run; revision: number }) {
  const [page, setPage] = useState(0)
  const enabled = active(run.status)
  const { calls, readAt, error } = useActiveCalls(run.id, enabled, revision)
  const now = useActivityClock(enabled)
  const currentPage = Math.min(page, Math.max(0, Math.ceil((calls?.length ?? 0) / 5) - 1))
  if (!enabled) return null
  return (
    <section className={styles.section} aria-label="Current activity">
      <div className={styles.heading}>
        <h3>
          <ProductIcon name="trace" width={16} height={16} /> Current activity{' '}
          {calls !== null && <small>· {number(calls.length)} in progress</small>}
        </h3>
        {readAt && (
          <small>
            Last checked <time dateTime={readAt}>{new Date(readAt).toLocaleTimeString()}</time>
          </small>
        )}
      </div>
      <p className={styles.description}>
        Response activity only; tokens and cost remain unknown until usage is recorded.
      </p>
      {error && (
        <p role="alert" className={styles.error}>
          Activity could not be refreshed. {error}{' '}
          {calls !== null && 'Showing the last observed calls below.'}
        </p>
      )}
      {calls === null ? (
        error ? (
          <p>Current activity is unavailable.</p>
        ) : (
          <ProductLoadingState compact label="Reading current activity…" />
        )
      ) : calls.length === 0 ? (
        <p>No model call was in progress at the last check. The run is still {run.status}.</p>
      ) : (
        <ul className={styles.calls}>
          {calls.slice(currentPage * 5, currentPage * 5 + 5).map((call) => (
            <li key={call.id}>
              <div className={styles.identity}>
                <strong>{call.selected_model ?? call.model ?? 'Model not recorded'}</strong>
                <span>{callLabel(run, call)}</span>
              </div>
              <dl>
                <div>
                  <dt>Phase</dt>
                  <dd>{call.activity ? phases[call.activity.phase] : 'Activity not recorded'}</dd>
                </div>
                <div>
                  <dt>Call elapsed</dt>
                  <dd>{duration(elapsedSeconds(call.started_at, now))}</dd>
                </div>
                <div>
                  <dt>Last response activity</dt>
                  <dd>
                    {call.activity?.last_activity_at
                      ? `${duration(elapsedSeconds(call.activity.last_activity_at, now))} ago`
                      : call.activity
                        ? 'No response bytes observed'
                        : 'Not recorded'}
                  </dd>
                </div>
                <div>
                  <dt>Response received</dt>
                  <dd>
                    {call.activity ? responseBytes(call.activity.received_bytes) : 'Not recorded'}
                  </dd>
                </div>
              </dl>
            </li>
          ))}
        </ul>
      )}
      {calls && (
        <BenchPagination
          label="Active calls"
          total={calls.length}
          page={currentPage}
          pageSize={5}
          onChange={setPage}
        />
      )}
    </section>
  )
}
