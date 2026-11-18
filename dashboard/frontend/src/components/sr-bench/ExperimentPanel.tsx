import { useEffect, useRef, useState } from 'react'
import {
  Activity,
  ArrowRight,
  Box,
  ChartLine,
  Check,
  Layers,
  Link2,
  Play,
  Split,
  TestTubes,
  type LucideIcon,
} from 'lucide-react'
import {
  Button,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  IconTile,
  Label,
  Skeleton,
  Textarea,
  cn,
} from '../../ui'
import BenchSelect from './BenchSelect'
import ExperimentDelete from './ExperimentDelete'
import { Notice } from './benchUi'
import { canReuseBaseline } from './baselineReuse'
import { experimentApi, type Experiment, type ExperimentMember } from './experimentApi'
import { experimentRoleLabels } from './experimentEvidencePresentation'
import { RunStatusCell } from './evalShared'
import { active } from './runPresentation'
import type { ExperimentRunContext, Run } from './types'

export interface Membership {
  members: ExperimentMember[]
  complete: boolean
  loading: boolean
  error: string
  reload: () => void
}

/** The one emphasized next step; the data attribute lets checks find it without styling hooks. */
const primary = (on: boolean) =>
  on
    ? ({ variant: 'default', 'data-primary-action': '' } as const)
    : ({ variant: 'outline' } as const)

export default function ExperimentPanel({
  id,
  actorID,
  runs,
  membership,
  canWrite,
  canRun,
  onLoaded,
  onOpenRun,
  onCandidate,
  onCompare,
  onDeleted,
  onChanged,
}: {
  id: string
  actorID: string
  runs: Run[]
  membership: Membership
  canWrite: boolean
  canRun: boolean
  onLoaded: (experiment: Experiment) => void
  onOpenRun: (id: string) => void
  onCandidate: (
    baseline: string,
    mode: 'live' | 'preview',
    role: ExperimentRunContext['role'],
  ) => void
  onCompare: (baseline: string) => void
  onDeleted: () => void
  onChanged: () => void
}) {
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [readError, setReadError] = useState('')
  const [revision, setRevision] = useState(0)
  const [attaching, setAttaching] = useState(false)
  const [notice, setNotice] = useState('')
  const [referenceID, setReferenceID] = useState('')
  const [activeRunID, setActiveRunID] = useState('')
  const loadedRef = useRef(onLoaded)
  loadedRef.current = onLoaded

  useEffect(() => {
    const controller = new AbortController()
    setReadError('')
    experimentApi
      .runs(id, 0, controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return
        setExperiment(value.experiment)
        loadedRef.current(value.experiment)
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setReadError(cause instanceof Error ? cause.message : 'Could not read the experiment.')
      })
    return () => controller.abort()
  }, [id, revision])

  const knownRoles = Object.fromEntries(
    membership.members.map((member) => [member.run_id, member.role]),
  )
  const references = runs.filter(
    (run) => knownRoles[run.id] === 'baseline' && canReuseBaseline(run),
  )
  const reference = references.find((run) => run.id === referenceID) ?? references[0]
  const activeRuns = runs.filter((run) => knownRoles[run.id] && active(run.status))
  const activeRun =
    activeRuns.find((run) => run.id === activeRunID) ??
    activeRuns.find((run) => knownRoles[run.id] === 'baseline') ??
    activeRuns[0]
  const hasStartingRecipe = Object.values(knownRoles).includes('initial')
  const hasRecipeResults = runs.some(
    (run) =>
      ['initial', 'candidate', 'validation'].includes(knownRoles[run.id]) &&
      ['completed', 'failed'].includes(run.status) &&
      run.progress.total > 0 &&
      run.progress.completed + run.progress.failed === run.progress.total,
  )
  const empty = membership.complete && membership.members.length === 0
  const versions = membership.members.filter((member) => member.role === 'candidate').length
  const journey: Array<[label: string, icon: LucideIcon, done: boolean]> = [
    ['Single-model reference', Box, Object.values(knownRoles).includes('baseline')],
    ['Starting recipe', Layers, hasStartingRecipe],
    [
      versions
        ? `${versions} recipe ${versions === 1 ? 'version' : 'versions'}`
        : 'Recipe versions',
      ChartLine,
      versions > 0,
    ],
  ]
  const title = activeRun
    ? knownRoles[activeRun.id] === 'baseline'
      ? 'Reference evaluation in progress'
      : 'Evaluation in progress'
    : empty
      ? 'Start with your single models'
      : reference
        ? hasRecipeResults
          ? 'Review your recipe versions'
          : 'Test a recipe on the same questions'
        : 'Choose your reference results'
  const copy = activeRun
    ? 'Open the run to follow response activity and results as attempts finish.'
    : empty
      ? 'Run the connected models on shared questions, or use results you already have.'
      : reference
        ? hasRecipeResults
          ? 'Compare saved results to see how quality and cost changed. Test another version when you have a change to evaluate.'
          : 'A reference fixes the questions and settings. Measure your starting recipe, then compare changes against your best single model.'
        : 'Use a finished single-model run as the reference for recipe versions. Running evaluations stay listed below.'
  const compareAction = reference && (
    <Button
      size="sm"
      {...primary(hasRecipeResults && !activeRun)}
      onClick={() => onCompare(reference.id)}
    >
      <ChartLine aria-hidden="true" /> Compare results
    </Button>
  )

  return (
    <section
      aria-label="Experiment"
      className="mb-4 overflow-hidden rounded-xl border border-border bg-card shadow-xs"
    >
      <div className="flex flex-wrap items-start gap-x-3 gap-y-2 px-4 pt-3.5 sm:px-5">
        <IconTile icon={TestTubes} tone="orange" size="md" />
        <div className="flex min-w-[12rem] flex-1 flex-col justify-center gap-0.5 self-center">
          <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-subtle-foreground">
            Experiment
          </span>
          <h2 className="truncate text-[15px] font-semibold tracking-[-0.01em]">
            {experiment?.name ?? (readError ? 'Experiment' : <Skeleton className="h-4 w-40" />)}
          </h2>
        </div>
        <div className="flex max-w-full flex-wrap items-center justify-end gap-1.5">
          {canWrite && membership.complete && (
            <Button variant="outline" size="sm" onClick={() => setAttaching(true)}>
              <Link2 aria-hidden="true" /> Add saved results
            </Button>
          )}
          {experiment && canWrite && !readError && (
            <ExperimentDelete
              key={`${actorID}:${id}`}
              experiment={experiment}
              disabled={false}
              onDeleted={onDeleted}
              onRefresh={() => {
                setRevision((value) => value + 1)
                membership.reload()
              }}
            />
          )}
        </div>
      </div>
      <ol
        className="flex flex-wrap items-center gap-1.5 px-4 pb-3.5 pt-2.5 text-xs text-muted-foreground sm:px-5"
        aria-label="Experiment sequence"
      >
        {journey.map(([label, Icon, done], index) => (
          <li key={label} className="inline-flex items-center gap-1.5 whitespace-nowrap">
            {index > 0 && (
              <ArrowRight aria-hidden="true" className="size-3 text-subtle-foreground" />
            )}
            <span
              className={cn(
                'inline-flex items-center gap-1 rounded-full border px-2 py-0.5',
                done
                  ? 'border-success/30 bg-success-soft text-success-foreground'
                  : 'border-border bg-surface text-foreground/80',
              )}
            >
              {done ? (
                <Check aria-hidden="true" className="size-3" strokeWidth={2.5} />
              ) : (
                <Icon aria-hidden="true" className="size-3" />
              )}
              {label}
            </span>
          </li>
        ))}
      </ol>
      {readError && (
        <Notice tone="danger" as="p" role="alert" className="mx-4 mt-0 sm:mx-5">
          {readError}
        </Notice>
      )}
      {notice && (
        <Notice tone="success" as="p" role="status" className="mx-4 mt-0 sm:mx-5">
          {notice}
        </Notice>
      )}
      <div className="border-t border-border bg-surface-subtle/50 px-4 py-3.5 sm:px-5">
        {membership.loading ? (
          <div role="status" className="flex flex-col gap-2">
            <span className="text-xs text-muted-foreground">Finding the next step…</span>
            <Skeleton className="h-3 w-2/3" />
          </div>
        ) : membership.error ? (
          <Notice tone="danger" role="alert" className="my-0">
            <p>Could not load all runs in this experiment. {membership.error}</p>
            <div>
              <Button variant="outline" size="xs" onClick={membership.reload}>
                Reload experiment runs
              </Button>
            </div>
          </Notice>
        ) : (
          membership.complete && (
            <div className="flex flex-col gap-3">
              <div className="flex items-start gap-3">
                <span className="grid size-7 shrink-0 place-items-center rounded-md bg-card text-muted-foreground ring-1 ring-inset ring-border">
                  {activeRun ? (
                    <Activity aria-hidden="true" className="size-3.5 text-info" />
                  ) : (
                    <ArrowRight aria-hidden="true" className="size-3.5" />
                  )}
                </span>
                <div className="min-w-0">
                  <h3 className="text-[13.5px] font-semibold">{title}</h3>
                  <p className="mt-0.5 max-w-[80ch] text-[13px] text-muted-foreground">{copy}</p>
                </div>
              </div>
              {activeRun && (
                <section
                  aria-label="Active evaluation"
                  className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-border bg-card px-3.5 py-3"
                >
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    {activeRuns.length > 1 ? (
                      <div className="max-w-md">
                        <BenchSelect
                          label="Active run"
                          value={activeRun.id}
                          options={activeRuns.map((run) => ({
                            value: run.id,
                            label: run.manifest.name,
                            description: `${experimentRoleLabels[knownRoles[run.id]]} · ${run.status}`,
                          }))}
                          onChange={setActiveRunID}
                        />
                      </div>
                    ) : (
                      <strong className="truncate text-[13px] font-semibold">
                        {activeRun.manifest.name}
                      </strong>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {experimentRoleLabels[knownRoles[activeRun.id]]}
                      {activeRuns.length > 1 && ` · ${activeRuns.length} active runs`}
                    </span>
                    <span className="text-xs text-subtle-foreground tabular-nums">
                      {activeRun.progress.completed.toLocaleString()} /{' '}
                      {activeRun.progress.total.toLocaleString()} attempts completed ·{' '}
                      {activeRun.progress.failed.toLocaleString()} failed
                    </span>
                  </div>
                  <RunStatusCell run={activeRun} className="w-36" />
                  <Button size="sm" {...primary(true)} onClick={() => onOpenRun(activeRun.id)}>
                    Open run <ArrowRight aria-hidden="true" />
                  </Button>
                </section>
              )}
              {reference ? (
                <div className="flex flex-wrap items-end gap-3">
                  {references.length > 1 ? (
                    <div className="w-full max-w-sm">
                      <BenchSelect
                        label="Single-model reference"
                        value={reference.id}
                        options={references.map((run) => ({
                          value: run.id,
                          label: run.manifest.name,
                          description: `${run.manifest.profile} · ${run.status}`,
                        }))}
                        onChange={setReferenceID}
                      />
                    </div>
                  ) : (
                    <div className="flex min-w-0 flex-col items-start gap-0.5">
                      <span className="text-xs text-muted-foreground">Single-model reference</span>
                      <Button
                        variant="link"
                        size="sm"
                        className="h-auto max-w-full whitespace-normal p-0 text-left text-foreground"
                        onClick={() => onOpenRun(reference.id)}
                      >
                        {reference.manifest.name} <ArrowRight aria-hidden="true" />
                      </Button>
                    </div>
                  )}
                  <div className="flex flex-wrap items-center gap-2 sm:ml-auto">
                    {hasRecipeResults && compareAction}
                    {canRun && (
                      <>
                        <Button
                          size="sm"
                          {...primary(!(hasRecipeResults || activeRun))}
                          onClick={() =>
                            onCandidate(
                              reference.id,
                              'live',
                              reference.manifest.profile === 'standard'
                                ? 'validation'
                                : hasStartingRecipe
                                  ? 'candidate'
                                  : 'initial',
                            )
                          }
                        >
                          <Play aria-hidden="true" />
                          {reference.manifest.profile === 'standard'
                            ? 'Validate final recipe'
                            : hasStartingRecipe
                              ? 'Test another recipe'
                              : 'Evaluate starting recipe'}
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onCandidate(reference.id, 'preview', 'preview')}
                        >
                          <Split aria-hidden="true" /> Check routing first
                        </Button>
                      </>
                    )}
                    {!hasRecipeResults && compareAction}
                  </div>
                </div>
              ) : (
                <div className="flex flex-wrap items-center gap-2">
                  {canRun && !activeRun && (
                    <Button
                      size="sm"
                      {...primary(true)}
                      onClick={() => onCandidate('', 'live', 'baseline')}
                    >
                      <Play aria-hidden="true" /> Run single models
                    </Button>
                  )}
                  {canWrite && empty && (
                    <Button variant="outline" size="sm" onClick={() => setAttaching(true)}>
                      <Link2 aria-hidden="true" /> Use saved results
                    </Button>
                  )}
                </div>
              )}
            </div>
          )
        )}
      </div>
      {attaching && (
        <AttachDialog
          experimentID={id}
          runs={runs}
          knownRoles={knownRoles}
          onClose={() => setAttaching(false)}
          onAttached={() => {
            setAttaching(false)
            setNotice('Saved run added to this experiment.')
            setRevision((value) => value + 1)
            membership.reload()
            onChanged()
          }}
        />
      )}
    </section>
  )
}

function AttachDialog({
  experimentID,
  runs,
  knownRoles,
  onClose,
  onAttached,
}: {
  experimentID: string
  runs: Run[]
  knownRoles: Record<string, ExperimentRunContext['role']>
  onClose: () => void
  onAttached: () => void
}) {
  const [runID, setRunID] = useState('')
  const [role, setRole] = useState<ExperimentRunContext['role']>('candidate')
  const [hypothesis, setHypothesis] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const busy = useRef(false)
  const mounted = useRef(false)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  const selected = runs.find((run) => run.id === runID)
  const roles = selected?.experiment_roles ?? []
  async function attach() {
    if (busy.current) return
    busy.current = true
    setPending(true)
    setError('')
    try {
      if (!roles.includes(role)) throw new Error('Choose an eligible role for this saved run.')
      await experimentApi.attach(experimentID, runID, role, hypothesis)
      if (mounted.current) onAttached()
    } catch (cause) {
      if (mounted.current)
        setError(cause instanceof Error ? cause.message : 'Could not save the experiment.')
    } finally {
      busy.current = false
      if (mounted.current) setPending(false)
    }
  }
  return (
    <Dialog open onOpenChange={(open) => !open && !pending && onClose()}>
      <DialogContent size="md">
        <DialogHeader>
          <DialogTitle>Add saved results</DialogTitle>
          <DialogDescription>
            Reuse a run you already have. Adding it here does not rerun the evaluation.
          </DialogDescription>
        </DialogHeader>
        <DialogBody className="flex flex-col gap-4">
          <BenchSelect
            label="Saved run"
            value={runID}
            disabled={pending}
            searchable
            options={runs
              .filter((run) => !knownRoles[run.id] && run.experiment_roles?.length)
              .map((run) => ({
                value: run.id,
                label: run.manifest.name,
                description: `${run.manifest.profile} · ${run.status}`,
              }))}
            onChange={(value) => {
              setRunID(value)
              setRole(runs.find((run) => run.id === value)!.experiment_roles![0])
            }}
          />
          {roles.length > 1 && (
            <BenchSelect
              label="Use this result as"
              value={role}
              disabled={pending}
              options={roles.map((value) => ({ value, label: experimentRoleLabels[value] }))}
              onChange={(value) => setRole(value as ExperimentRunContext['role'])}
            />
          )}
          {roles.length === 1 && (
            <p className="text-[13px] text-muted-foreground">{experimentRoleLabels[roles[0]]}</p>
          )}
          <div className="grid gap-1.5">
            <Label htmlFor="experiment-hypothesis">What changed? (optional)</Label>
            <Textarea
              id="experiment-hypothesis"
              disabled={pending}
              value={hypothesis}
              maxLength={2000}
              onChange={(event) => setHypothesis(event.target.value)}
              placeholder="For example: prefer the less expensive model for simple questions."
            />
          </div>
          {error && (
            <Notice tone="danger" as="p" role="alert" className="my-0">
              {error}
            </Notice>
          )}
        </DialogBody>
        <DialogFooter>
          <Button variant="ghost" size="sm" disabled={pending} onClick={onClose}>
            Cancel
          </Button>
          <Button
            size="sm"
            disabled={pending || !runID || !roles.includes(role)}
            loading={pending}
            onClick={() => void attach()}
          >
            {pending ? null : <Link2 aria-hidden="true" />}
            {pending ? 'Saving…' : 'Add run'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
