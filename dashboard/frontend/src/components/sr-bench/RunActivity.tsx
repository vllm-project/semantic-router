import { useState } from 'react'
import { Activity } from 'lucide-react'
import { StatusDot } from '../../ui'
import BenchPagination from './BenchPagination'
import { active, number } from './model'
import { duration, elapsedSeconds, responseBytes } from './activityPresentation'
import { useActiveCalls } from './useActiveCalls'
import { useActivityClock } from './useActivityClock'
import { BenchBlock, BenchLoading, Notice } from './benchUi'
import { benchText } from './benchStyles'
import type { CallActivity, CallRecord, Run } from './types'

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
    <BenchBlock className="mt-4" aria-label="Current activity">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className={`${benchText.h3} flex flex-wrap items-center gap-2`}>
          <Activity aria-hidden="true" className="size-4 text-info" />
          Current activity{' '}
          {calls !== null && (
            <small className="text-xs font-normal text-muted-foreground">
              · {number(calls.length)} in progress
            </small>
          )}
        </h3>
        {readAt && (
          <small className="text-xs text-subtle-foreground">
            Last checked <time dateTime={readAt}>{new Date(readAt).toLocaleTimeString()}</time>
          </small>
        )}
      </div>
      <p className={`${benchText.muted} mt-1`}>
        Response activity only; tokens and cost remain unknown until usage is recorded.
      </p>
      {error && (
        <Notice tone="danger" as="p" role="alert">
          Activity could not be refreshed. {error}{' '}
          {calls !== null && 'Showing the last observed calls below.'}
        </Notice>
      )}
      {calls === null ? (
        error ? (
          <p className={`${benchText.body} mt-3`}>Current activity is unavailable.</p>
        ) : (
          <BenchLoading label="Reading current activity…" />
        )
      ) : calls.length === 0 ? (
        <p className={`${benchText.body} mt-3`}>
          No model call was in progress at the last check. The run is still {run.status}.
        </p>
      ) : (
        <ul className="mt-3 divide-y divide-border border-t border-border">
          {calls.slice(currentPage * 5, currentPage * 5 + 5).map((call) => (
            <li key={call.id} className="py-3">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 break-words">
                <StatusDot tone="info" pulse />
                <strong className="text-[13px] font-medium text-foreground">
                  {call.selected_model ?? call.model ?? 'Model not recorded'}
                </strong>
                <span className="text-xs text-muted-foreground">{callLabel(run, call)}</span>
              </div>
              <dl className="mt-2.5 grid grid-cols-2 gap-3 md:grid-cols-4 [&_dd]:mt-0.5 [&_dd]:break-words [&_dd]:text-[13px] [&_dd]:text-foreground [&_dd]:tabular-nums [&_dt]:text-xs [&_dt]:text-muted-foreground">
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
    </BenchBlock>
  )
}
