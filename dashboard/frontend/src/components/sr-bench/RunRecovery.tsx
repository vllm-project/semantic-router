import { useState } from 'react'
import { benchApi, SrBenchRequestError } from './api'
import { money, number } from './model'
import type { RecoveryCell, RecoveryPlan, RecoveryRequest, Run } from './types'
import styles from './SrBench.module.css'

const cellKey = (cell: RecoveryCell) => `${cell.target_id}\0${cell.case_id}`
function savedRequest(key: string): RecoveryRequest | null {
  try {
    return JSON.parse(sessionStorage.getItem(key) ?? 'null') as RecoveryRequest | null
  } catch {
    return null
  }
}

export default function RunRecovery({
  run,
  canRun,
  onRecovered,
}: {
  run: Run
  canRun: boolean
  onRecovered: (run: Run) => void
}) {
  const storageKey = `sr-bench-recovery:${run.id}`
  const [request, setRequest] = useState<RecoveryRequest | null>(() => savedRequest(storageKey))
  const [mode, setMode] = useState<RecoveryPlan['mode']>('undispatched')
  const [plan, setPlan] = useState<RecoveryPlan | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [acknowledged, setAcknowledged] = useState(false)
  const [page, setPage] = useState(0)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  async function review() {
    setPending(true)
    setError('')
    setPlan(null)
    setSelected(new Set())
    setAcknowledged(false)
    setPage(0)
    try {
      setPlan(await benchApi.recoveryPlan(run.id, mode))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not inspect recovery eligibility.')
    } finally {
      setPending(false)
    }
  }
  async function recover() {
    if (!request && !plan) return
    const body = request ?? {
      mode,
      plan_sha256: plan!.plan_sha256,
      cells: plan!.eligible_cells.filter((cell) => selected.has(cellKey(cell))),
      idempotency_key: crypto.randomUUID(),
      ...(mode === 'failed' ? { acknowledge_new_attempt: acknowledged } : {}),
    }
    setPending(true)
    setError('')
    try {
      // Persist the exact attempt before posting. Reloading or a lost response must
      // reuse its idempotency key, never create a second recovery implicitly.
      sessionStorage.setItem(storageKey, JSON.stringify(body))
      setRequest(body)
      const child = await benchApi.recover(run.id, body)
      sessionStorage.removeItem(storageKey)
      setRequest(null)
      onRecovered(child)
    } catch (cause) {
      if (
        cause instanceof SrBenchRequestError &&
        cause.code === 'recovery_plan_required' &&
        cause.dispatchStarted === false
      ) {
        sessionStorage.removeItem(storageKey)
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
      setPending(false)
    }
  }
  return (
    <section className={styles.recovery} aria-labelledby="recovery-title">
      <h3 id="recovery-title">Recover unfinished work</h3>
      <p>
        Inspect eligibility, select cases and create a separate child run. The original run and its
        costs remain unchanged; the child reports only its own scope.
      </p>
      {request ? (
        <div className={styles.notice}>
          <strong>Recovery submission needs reconciliation</strong>
          <p>
            The exact request is saved in this tab. Checking it again uses the same submission ID
            and selected cases.
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
