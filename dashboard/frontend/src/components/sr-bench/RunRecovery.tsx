import { useEffect, useRef, useState } from 'react'
import { benchApi, SrBenchRequestError } from './api'
import { money, number } from './model'
import type { RecoveryCell, RecoveryPlan, RecoveryRequest, Run } from './types'
import styles from './SrBench.module.css'
import {
  clearRecovery,
  readRecovery,
  saveRecovery,
  type PendingRecovery,
} from './recoverySubmission'

const cellKey = (cell: RecoveryCell) => `${cell.target_id}\0${cell.case_id}`

export default function RunRecovery({
  run,
  actorID,
  canRun,
  onRecovered,
}: {
  run: Run
  actorID: string
  canRun: boolean
  onRecovered: (run: Run) => void
}) {
  const [saved] = useState(() => readRecovery(actorID, run.id))
  const [request, setRequest] = useState<RecoveryRequest | null>(saved.saved?.request ?? null)
  const [mode, setMode] = useState<RecoveryPlan['mode']>('undispatched')
  const [plan, setPlan] = useState<RecoveryPlan | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [acknowledged, setAcknowledged] = useState(false)
  const [page, setPage] = useState(0)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const mounted = useRef(false)
  const requestSequence = useRef(0)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      requestSequence.current += 1
    }
  }, [])
  async function review() {
    if (!canRun || pending || request || saved.error) return
    const sequence = ++requestSequence.current
    const current = () => mounted.current && requestSequence.current === sequence
    setPending(true)
    setError('')
    setPlan(null)
    setSelected(new Set())
    setAcknowledged(false)
    setPage(0)
    try {
      const reviewed = await benchApi.recoveryPlan(run.id, mode)
      if (current()) setPlan(reviewed)
    } catch (cause) {
      if (current())
        setError(cause instanceof Error ? cause.message : 'Could not inspect recovery eligibility.')
    } finally {
      if (current()) setPending(false)
    }
  }
  async function recover() {
    if (!canRun || pending || saved.error || (!request && !plan)) return
    const sequence = ++requestSequence.current
    const current = () => mounted.current && requestSequence.current === sequence
    const body = request ?? {
      mode,
      plan_sha256: plan!.plan_sha256,
      cells: plan!.eligible_cells.filter((cell) => selected.has(cellKey(cell))),
      idempotency_key: crypto.randomUUID(),
      ...(mode === 'failed' ? { acknowledge_new_attempt: acknowledged } : {}),
    }
    const submission: PendingRecovery = { version: 1, actorID, parentID: run.id, request: body }
    setPending(true)
    setError('')
    try {
      // Persist the exact attempt before posting. Reloading or a lost response must
      // reuse its idempotency key, never create a second recovery implicitly.
      saveRecovery(submission)
      setRequest(body)
      const child = await benchApi.recover(run.id, body)
      if (!current()) return
      if (!clearRecovery(submission))
        throw new Error(
          'The saved recovery changed while this response was pending. Reload to reconcile the current request.',
        )
      setRequest(null)
      onRecovered(child)
    } catch (cause) {
      if (!current()) return
      if (
        cause instanceof SrBenchRequestError &&
        cause.code === 'recovery_plan_required' &&
        cause.dispatchStarted === false
      ) {
        try {
          if (!clearRecovery(submission)) throw new Error('Saved recovery identity changed')
        } catch {
          setError(
            'The service confirmed no dispatch, but the saved recovery could not be cleared safely. Reload and reconcile its identity before creating another plan.',
          )
          return
        }
        setRequest(null)
        setPlan(null)
        setSelected(new Set())
      }
      setError(
        cause instanceof Error
          ? cause.message
          : 'Recovery response unavailable. Reconcile this same request before creating another attempt.',
      )
    } finally {
      if (current()) setPending(false)
    }
  }
  return (
    <section className={styles.recovery} aria-labelledby="recovery-title">
      <h3 id="recovery-title">Recover unfinished work</h3>
      <p>
        Inspect eligibility, select cases and create a separate child run. The original run and its
        costs remain unchanged; the child reports only its own scope.
      </p>
      {saved.error ? (
        <p className={styles.error} role="alert">
          {saved.error}
        </p>
      ) : request ? (
        <div className={styles.notice}>
          <strong>Recovery submission needs reconciliation</strong>
          <p>
            The exact request is saved for this account and parent in this tab. Checking it again
            uses the same submission ID and selected cases.
          </p>
          <p>
            {number(request.cells.length)} cases · {request.mode}
          </p>
          <button disabled={!canRun || pending} onClick={() => void recover()}>
            {pending ? 'Reconciling…' : 'Check or submit same recovery'}
          </button>
        </div>
      ) : (
        <>
          <div className={styles.formGrid}>
            <label>
              Recovery scope
              <select
                disabled={!canRun || pending}
                value={mode}
                onChange={(event) => {
                  setMode(event.target.value as RecoveryPlan['mode'])
                  setPlan(null)
                  setSelected(new Set())
                  setAcknowledged(false)
                }}
              >
                <option value="undispatched">Continue undispatched cases</option>
                <option value="failed">Retry known failed cases as new attempts</option>
              </select>
            </label>
          </div>
          <p className={styles.muted}>
            {mode === 'undispatched'
              ? 'Only cases with no dispatched model call are eligible. Completed, previously attempted and claimed cases are excluded.'
              : 'Only failed cases with fully reconciled calls and known spend are eligible. Each selected case incurs a new attempt and new cost; ambiguous calls cannot be retried here.'}
          </p>
          <button disabled={!canRun || pending} onClick={() => void review()}>
            {pending ? 'Inspecting…' : 'Review recovery plan'}
          </button>
          {plan && (
            <div className={styles.plan}>
              <h4>
                {number(plan.counts.eligible)} eligible · {number(plan.counts.excluded)} excluded
              </h4>
              <p>
                Parent completed {number(plan.parent.progress.completed)} /{' '}
                {number(plan.parent.progress.total)} · known spend{' '}
                {money(plan.parent.known_spend_usd)}
                {plan.parent.spend_complete ? '' : ' (incomplete usage accounting)'}
              </p>
              <p>
                New attempt budget: {money(plan.new_attempt_budget_usd)} ·{' '}
                {plan.scope ?? 'selected cells only'}
              </p>
              {!!plan.eligible_cells.length && (
                <>
                  <label className={styles.checkbox}>
                    <input
                      type="checkbox"
                      checked={selected.size === plan.eligible_cells.length}
                      onChange={(event) =>
                        setSelected(
                          event.target.checked
                            ? new Set(plan.eligible_cells.map(cellKey))
                            : new Set(),
                        )
                      }
                    />
                    Select all {number(plan.eligible_cells.length)} eligible cases
                  </label>
                  <div className={styles.tableScroll}>
                    <table>
                      <thead>
                        <tr>
                          <th>Select</th>
                          <th>Target</th>
                          <th>Case</th>
                        </tr>
                      </thead>
                      <tbody>
                        {plan.eligible_cells.slice(page * 25, page * 25 + 25).map((cell) => (
                          <tr key={cellKey(cell)}>
                            <td>
                              <input
                                type="checkbox"
                                aria-label={`Recover ${cell.target_id} ${cell.case_id}`}
                                checked={selected.has(cellKey(cell))}
                                onChange={(event) =>
                                  setSelected((previous) => {
                                    const next = new Set(previous)
                                    if (event.target.checked) next.add(cellKey(cell))
                                    else next.delete(cellKey(cell))
                                    return next
                                  })
                                }
                              />
                            </td>
                            <td>{cell.target_id}</td>
                            <td>{cell.case_id}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {plan.eligible_cells.length > 25 && (
                    <div className={styles.actions}>
                      <button disabled={page === 0} onClick={() => setPage(page - 1)}>
                        Previous eligible cases
                      </button>
                      <span>Page {page + 1}</span>
                      <button
                        disabled={(page + 1) * 25 >= plan.eligible_cells.length}
                        onClick={() => setPage(page + 1)}
                      >
                        Next eligible cases
                      </button>
                    </div>
                  )}
                  {mode === 'failed' && (
                    <label className={styles.checkbox}>
                      <input
                        type="checkbox"
                        checked={acknowledged}
                        onChange={(event) => setAcknowledged(event.target.checked)}
                      />
                      I acknowledge these are new model attempts with additional cost, recorded
                      separately from the original run.
                    </label>
                  )}
                  <button
                    className={styles.primary}
                    disabled={
                      !canRun || pending || !selected.size || (mode === 'failed' && !acknowledged)
                    }
                    onClick={() => void recover()}
                  >
                    Create recovery run ({number(selected.size)} cases)
                  </button>
                </>
              )}
              {!!plan.excluded.length && (
                <details>
                  <summary>Excluded cases and reasons</summary>
                  <pre>{JSON.stringify(plan.excluded, null, 2)}</pre>
                </details>
              )}
            </div>
          )}
        </>
      )}
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
