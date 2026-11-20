import { useEffect, useRef, useState } from 'react'
import { History } from 'lucide-react'
import { Button, Checkbox, Label } from '../../ui'
import { benchApi, SrBenchRequestError } from './api'
import { money, number } from './model'
import { targetName } from './targetPresentation'
import type { RecoveryCell, RecoveryPlan, RecoveryRequest, Run } from './types'
import BenchSelect from './BenchSelect'
import BenchPagination from './BenchPagination'
import { ActionRow, BenchBlock, Disclosure, Notice, TableFrame } from './benchUi'
import { benchTable, benchText } from './benchStyles'
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
  const [excludedPage, setExcludedPage] = useState(0)
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
    setExcludedPage(0)
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
  const allSelected = !!plan && selected.size === plan.eligible_cells.length
  return (
    <BenchBlock className="mt-6" aria-labelledby="recovery-title">
      <h3 id="recovery-title" className={`${benchText.h3} flex items-center gap-2`}>
        <History aria-hidden="true" className="size-4 text-muted-foreground" />
        Recover unfinished work
      </h3>
      <p className={`${benchText.muted} mt-1 max-w-3xl`}>
        Inspect eligibility, select cases and create a separate child run. The original run and its
        costs remain unchanged; the child reports only its own scope.
      </p>
      {saved.error ? (
        <Notice tone="danger" as="p" role="alert">
          {saved.error}
        </Notice>
      ) : request ? (
        <Notice tone="warning">
          <strong>Recovery submission needs reconciliation</strong>
          <p>
            The exact request is saved for this account and parent in this tab. Checking it again
            uses the same submission ID and selected cases.
          </p>
          <p>
            {number(request.cells.length)} cases · {request.mode}
          </p>
          <div>
            <Button size="sm" disabled={!canRun || pending} onClick={() => void recover()}>
              {pending ? 'Reconciling…' : 'Check or submit same recovery'}
            </Button>
          </div>
        </Notice>
      ) : (
        <>
          <div className="mt-3 grid max-w-md gap-3">
            <BenchSelect
              label="Recovery scope"
              disabled={!canRun || pending}
              value={mode}
              onChange={(value) => {
                setMode(value as RecoveryPlan['mode'])
                setPlan(null)
                setSelected(new Set())
                setAcknowledged(false)
              }}
              options={[
                { value: 'undispatched', label: 'Continue undispatched cases' },
                { value: 'failed', label: 'Retry known failed cases as new attempts' },
              ]}
            />
          </div>
          <p className={`${benchText.muted} mt-2 max-w-3xl`}>
            {mode === 'undispatched'
              ? 'Only cases with no dispatched model call are eligible. Completed, previously attempted and claimed cases are excluded.'
              : 'Only failed cases with fully reconciled calls and known spend are eligible. Each selected case incurs a new attempt and new cost; ambiguous calls cannot be retried here.'}
          </p>
          <Button
            variant="outline"
            size="sm"
            className="mt-3"
            disabled={!canRun || pending}
            onClick={() => void review()}
          >
            {pending ? 'Inspecting…' : 'Review recovery plan'}
          </Button>
          {plan && (
            <Notice tone="success" icon={false} className="mt-4 bg-surface-subtle">
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
                  <Label className="mt-1 font-normal text-foreground">
                    <Checkbox
                      checked={allSelected}
                      onCheckedChange={(checked) =>
                        setSelected(
                          checked === true ? new Set(plan.eligible_cells.map(cellKey)) : new Set(),
                        )
                      }
                    />
                    Select all {number(plan.eligible_cells.length)} eligible cases
                  </Label>
                  <TableFrame className="my-1">
                    <table className={benchTable}>
                      <thead>
                        <tr>
                          <th className="w-14">Select</th>
                          <th>Target</th>
                          <th>Case</th>
                        </tr>
                      </thead>
                      <tbody>
                        {plan.eligible_cells.slice(page * 25, page * 25 + 25).map((cell) => (
                          <tr key={cellKey(cell)}>
                            <td>
                              <Checkbox
                                aria-label={`Recover ${targetName(run.manifest, cell.target_id)} ${cell.case_id}`}
                                checked={selected.has(cellKey(cell))}
                                onCheckedChange={(checked) =>
                                  setSelected((previous) => {
                                    const next = new Set(previous)
                                    if (checked === true) next.add(cellKey(cell))
                                    else next.delete(cellKey(cell))
                                    return next
                                  })
                                }
                              />
                            </td>
                            <td>{targetName(run.manifest, cell.target_id)}</td>
                            <td className="font-mono">{cell.case_id}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </TableFrame>
                  <BenchPagination
                    label="Eligible cases"
                    total={plan.eligible_cells.length}
                    page={page}
                    pageSize={25}
                    onChange={setPage}
                  />
                  {mode === 'failed' && (
                    <Label className="items-start font-normal leading-relaxed text-foreground">
                      <Checkbox
                        className="mt-0.5"
                        checked={acknowledged}
                        onCheckedChange={(checked) => setAcknowledged(checked === true)}
                      />
                      I acknowledge these are new model attempts with additional cost, recorded
                      separately from the original run.
                    </Label>
                  )}
                  <ActionRow>
                    <Button
                      size="sm"
                      disabled={
                        !canRun || pending || !selected.size || (mode === 'failed' && !acknowledged)
                      }
                      onClick={() => void recover()}
                    >
                      Create recovery run ({number(selected.size)} cases)
                    </Button>
                  </ActionRow>
                </>
              )}
              {!!plan.excluded.length && (
                <Disclosure summary="Excluded cases and reasons" bare className="my-0">
                  <TableFrame className="my-1">
                    <table className={benchTable} aria-label="Excluded recovery cases">
                      <thead>
                        <tr>
                          <th>Target</th>
                          <th>Case</th>
                          <th>Reason</th>
                        </tr>
                      </thead>
                      <tbody>
                        {plan.excluded
                          .slice(excludedPage * 25, excludedPage * 25 + 25)
                          .map((cell) => (
                            <tr key={cellKey(cell)}>
                              <td>{targetName(run.manifest, cell.target_id)}</td>
                              <td className="font-mono">{cell.case_id}</td>
                              <td>{cell.reason}</td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </TableFrame>
                  <BenchPagination
                    label="Excluded cases"
                    total={plan.excluded.length}
                    page={excludedPage}
                    pageSize={25}
                    onChange={setExcludedPage}
                  />
                </Disclosure>
              )}
            </Notice>
          )}
        </>
      )}
      {error && (
        <Notice tone="danger" as="p" role="alert">
          {error}
        </Notice>
      )}
    </BenchBlock>
  )
}
