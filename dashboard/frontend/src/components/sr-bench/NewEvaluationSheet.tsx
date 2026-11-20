import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Database,
  FlaskConical,
  Play,
  RefreshCw,
} from 'lucide-react'
import {
  Button,
  DetailSheet,
  DetailSheetBody,
  DetailSheetFooter,
  DetailSheetHeader,
  IconTile,
  Input,
  KeyValueGrid,
  Label,
  MiniStat,
  SectionLabel,
  Segmented,
  Skeleton,
  StatusDot,
  Textarea,
  cn,
  type StatusTone,
} from '../../ui'
import { benchApi, SrBenchRequestError } from './api'
import { DEFAULT_LIMITS, makeManifest, number, validateManifest } from './model'
import type { RerunSource } from './rerunSource'
import type { Catalog, Dataset, Manifest, Plan, Run, Target } from './types'
import FrozenBaselineProtocol from './FrozenBaselineProtocol'
import { canReuseBaseline } from './baselineReuse'
import RunSettings, { type SamplingSettings } from './RunSettings'
import TargetRequestProfile from './TargetRequestProfile'
import TargetPicker from './TargetPicker'
import ReplayComposer from './ReplayComposer'
import { targetLabel } from './targetPresentation'
import { nativeOutputIssue } from './nativeOutput'
import { experimentRoleLabels } from './experimentEvidencePresentation'
import RunDatasetScope, { type ResolvedDatasetScope } from './RunDatasetScope'
import {
  prepareEvaluationDataset,
  type EvaluationPreparationProgress,
} from './evaluationPreparation'
import { preparationPhase, type PreparationProfile } from './datasetPreparationApi'
import { benchmarkTitle, profileTitle } from './datasetPresentation'
import { BenchmarkChips, TargetChip } from './evalShared'
import {
  clearSubmission,
  readSubmission,
  saveSubmission,
  type PendingSubmission,
} from './submission'
import { Disclosure, Notice } from './benchUi'
import { codeBlockClass, identityClass } from './benchStyles'

export type EvaluationMode = 'live' | 'preview' | 'estimate'

const STEPS = ['What to measure', 'What to compare', 'Limits', 'Review'] as const

const modeCopy: Record<EvaluationMode, string> = {
  live: 'sr-bench sends each question to every target and scores the answers.',
  preview:
    'A dry run routes every question through your recipe without calling models: decisions and selected models only.',
  estimate:
    'Estimates a routing change from saved answers. No new model calls are made and nothing is scored.',
}

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  mode: EvaluationMode
  onModeChange?: (mode: EvaluationMode) => void
  allowEstimate?: boolean
  catalog: Catalog | null
  datasets: Dataset[]
  targets: Target[]
  ready: boolean
  canRun: boolean
  actorID: string
  baselineID?: string
  experiment?: Manifest['experiment']
  initialModel?: string
  initialDataset?: string
  rerun?: RerunSource
  onStarted: (run: Run) => void
}

export default function NewEvaluationSheet(props: Props) {
  const { open, onOpenChange, mode, onModeChange, ready, catalog, actorID, canRun } = props
  const title = props.rerun
    ? 'Re-run evaluation'
    : props.baselineID
      ? mode === 'preview'
        ? 'Dry-run a candidate recipe'
        : 'Evaluate a candidate recipe'
      : 'New evaluation'
  return (
    <DetailSheet open={open} onOpenChange={onOpenChange} size="lg">
      <DetailSheetHeader
        icon={<IconTile icon={FlaskConical} tone="orange" size="lg" />}
        title={title}
        description={modeCopy[mode]}
      />
      {onModeChange ? (
        <div className="border-b border-border px-5 py-2.5">
          <Segmented<EvaluationMode>
            aria-label="Evaluation type"
            value={mode}
            onValueChange={onModeChange}
            options={[
              { value: 'live', label: 'Evaluate' },
              { value: 'preview', label: 'Dry run' },
              ...(props.baselineID || props.allowEstimate === false
                ? []
                : [{ value: 'estimate' as const, label: 'Offline estimate' }]),
            ]}
          />
        </div>
      ) : null}
      {open && mode === 'estimate' ? (
        <>
          <DetailSheetBody className="px-5 py-5">
            <ReplayComposer
              key={actorID}
              actorID={actorID}
              canRun={canRun}
              onCreated={props.onStarted}
            />
          </DetailSheetBody>
          <DetailSheetFooter>
            <span className="text-xs text-muted-foreground">Uses saved answers only</span>
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto"
              onClick={() => onOpenChange(false)}
            >
              Close
            </Button>
          </DetailSheetFooter>
        </>
      ) : open && ready && catalog ? (
        <Composer
          key={`${actorID}:${mode}:${props.initialDataset ?? ''}:${props.baselineID ?? ''}:${props.experiment?.id ?? ''}:${props.experiment?.role ?? ''}:${props.rerun?.name ?? ''}`}
          {...props}
          catalog={catalog}
          mode={mode === 'preview' ? 'preview' : 'live'}
        />
      ) : open ? (
        <DetailSheetBody className="px-5 py-5">
          <div role="status" className="flex flex-col gap-2.5">
            <span className="text-xs text-muted-foreground">
              Waiting for the benchmark catalog, prepared datasets and configured targets.
            </span>
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        </DetailSheetBody>
      ) : null}
    </DetailSheet>
  )
}

function Stepper({
  step,
  onStep,
  incomplete,
}: {
  step: number
  onStep: (step: number) => void
  incomplete: (step: number) => boolean
}) {
  return (
    <nav
      aria-label="Steps"
      className="flex items-center gap-2 overflow-x-auto border-b border-border px-5 py-3"
    >
      {STEPS.map((label, index) => (
        <span key={label} className="flex shrink-0 items-center gap-2">
          {index > 0 ? (
            <span
              aria-hidden="true"
              className={cn('h-px w-5 sm:w-8', index <= step ? 'bg-primary/50' : 'bg-border')}
            />
          ) : null}
          <button
            type="button"
            aria-current={index === step ? 'step' : undefined}
            onClick={() => onStep(index)}
            className={cn(
              'flex items-center gap-2 rounded-md text-[13px] outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/40',
              index === step
                ? 'font-medium text-foreground'
                : index < step
                  ? 'text-muted-foreground hover:text-foreground'
                  : 'text-subtle-foreground hover:text-foreground',
            )}
          >
            <span
              className={cn(
                'grid size-5 place-items-center rounded-full border text-[11px] font-semibold tabular',
                index < step && incomplete(index)
                  ? 'border-warning bg-warning-soft text-warning-foreground'
                  : index < step
                    ? 'border-transparent bg-success text-primary-foreground'
                    : index === step
                      ? 'border-primary bg-primary text-primary-foreground'
                      : 'border-border-strong',
              )}
            >
              {index < step && !incomplete(index) ? (
                <Check aria-hidden="true" className="size-3" strokeWidth={3} />
              ) : (
                index + 1
              )}
            </span>
            {label}
          </button>
        </span>
      ))}
    </nav>
  )
}

function StepSection({
  title,
  aside,
  children,
}: {
  title: string
  aside?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="flex min-w-0 flex-col gap-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionLabel>{title}</SectionLabel>
        {aside}
      </div>
      {children}
    </section>
  )
}

const itemTone = (status: string): StatusTone =>
  status === 'completed'
    ? 'success'
    : status === 'failed'
      ? 'danger'
      : status === 'running'
        ? 'info'
        : 'neutral'

function duration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—'
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`
  const hours = seconds / 3600
  return `${Number.isInteger(hours) ? hours : hours.toFixed(1)} h`
}

function Composer({
  onOpenChange,
  catalog,
  datasets,
  targets: registeredTargets,
  canRun,
  actorID,
  initialModel,
  initialDataset,
  mode,
  baselineID,
  experiment,
  rerun,
  onStarted,
}: Omit<Props, 'mode' | 'catalog' | 'allowEstimate'> & {
  mode: 'live' | 'preview'
  catalog: Catalog
}) {
  const targetKind =
    experiment?.role === 'baseline'
      ? 'single'
      : baselineID ||
          mode === 'preview' ||
          experiment?.role === 'initial' ||
          experiment?.role === 'candidate' ||
          experiment?.role === 'validation'
        ? 'mom'
        : undefined
  const availableTargets = registeredTargets.filter(
    (target) => !targetKind || target.kind === targetKind,
  )
  const [step, setStep] = useState(0)
  const goTo = setStep
  const [baseline, setBaseline] = useState<Run | null>(null)
  const [baselineError, setBaselineError] = useState('')
  const [baselineRevision, setBaselineRevision] = useState(0)
  useEffect(() => {
    if (!baselineID) return
    const controller = new AbortController()
    setBaseline(null)
    setBaselineError('')
    void benchApi
      .run(baselineID, controller.signal)
      .then((run) => {
        if (!controller.signal.aborted) {
          if (run.id !== baselineID)
            throw new Error('The baseline identity changed. Reload the selected run.')
          if (!canReuseBaseline(run))
            throw new Error('Choose a finished live single-model baseline with its full plan.')
          setBaseline(run)
        }
      })
      .catch((cause) => {
        if (!controller.signal.aborted)
          setBaselineError(cause instanceof Error ? cause.message : 'Could not read the baseline.')
      })
    return () => controller.abort()
  }, [baselineID, baselineRevision])
  const [saved] = useState(() => readSubmission(actorID))
  const [submission, setSubmission] = useState<PendingSubmission | null>(saved.request)
  const [name, setName] = useState(
    rerun?.name ??
      (mode === 'preview'
        ? 'Routing dry run'
        : experiment
          ? experimentRoleLabels[experiment.role]
          : 'Model comparison'),
  )
  const [hypothesis, setHypothesis] = useState(experiment?.hypothesis ?? '')
  const [costPolicy, setCostPolicy] = useState<'require_priced' | 'capability_only'>(
    rerun?.costPolicy ?? 'require_priced',
  )
  const [outputPolicy, setOutputPolicy] = useState<'bounded' | 'native'>(
    rerun?.outputPolicy ?? 'bounded',
  )
  const [profile, setProfile] = useState(
    rerun?.profile ?? datasets.find((item) => item.id === initialDataset)?.profile ?? 'quick',
  )
  const [frozenDataset, setFrozenDataset] = useState(rerun?.dataset)
  const experimentContext = useMemo(
    () =>
      experiment
        ? {
            id: experiment.id,
            role:
              (baseline?.manifest.profile ?? profile) === 'standard' &&
              (experiment.role === 'initial' || experiment.role === 'candidate')
                ? ('validation' as const)
                : experiment.role,
            ...(hypothesis.trim() ? { hypothesis: hypothesis.trim() } : {}),
          }
        : undefined,
    [experiment, hypothesis, baseline, profile],
  )
  const [benchmarks, setBenchmarks] = useState<string[]>(
    rerun?.benchmarks ?? datasets.find((item) => item.id === initialDataset)?.benchmarks ?? [],
  )
  const [scope, setScope] = useState<ResolvedDatasetScope>({
    profile,
    sourceIDs: [],
    missingBenchmarks: [],
    seed: null,
    ready: false,
    error: 'Select at least one benchmark.',
  })
  const [targets, setTargets] = useState<Target[]>(() =>
    availableTargets.filter((target) =>
      rerun ? rerun.targetIDs.includes(target.id) : target.model === initialModel,
    ),
  )
  const [limits, setLimits] = useState(rerun ? { ...rerun.limits } : { ...DEFAULT_LIMITS })
  function changeCostPolicy(value: 'require_priced' | 'capability_only') {
    setCostPolicy(value)
    if (value === 'capability_only')
      setLimits((previous) =>
        !Number.isFinite(previous.max_cost_usd) || previous.max_cost_usd <= 0
          ? { ...previous, max_cost_usd: DEFAULT_LIMITS.max_cost_usd }
          : previous,
      )
  }
  const missingPrices = targets.some(
    (target) =>
      !target.prices ||
      Object.keys(target.prices).length === 0 ||
      (target.kind === 'single' &&
        !Object.prototype.hasOwnProperty.call(target.prices, target.model)),
  )
  const [sampling, setSampling] = useState<SamplingSettings>(
    rerun?.sampling ?? { temperature: 0, top_p: 1 },
  )
  const [previewContext, setPreviewContext] = useState<NonNullable<Manifest['preview_context']>>({})
  const [plan, setPlan] = useState<{
    manifest: Manifest
    evidence: Plan
    fingerprint: string
    idempotencyKey: string
  } | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [preparation, setPreparation] = useState<EvaluationPreparationProgress | null>(null)
  const preparationController = useRef<AbortController | null>(null)
  const mounted = useRef(false)
  const requestSequence = useRef(0)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      requestSequence.current += 1
      preparationController.current?.abort()
    }
  }, [])
  const formManifest = useMemo(() => {
    if (baseline && baseline.id === baselineID)
      return {
        ...baseline.manifest,
        name,
        mode,
        targets,
        ...(experimentContext ? { experiment: experimentContext } : {}),
      }
    const manifest = makeManifest(name, mode, profile, undefined, targets, limits)
    manifest.seed = (frozenDataset ? rerun?.sampling.seed : undefined) ?? scope.seed ?? 20260918
    manifest.sampling.seed = manifest.seed
    const defaults = { ...manifest.sampling, ...sampling }
    if (outputPolicy === 'native') {
      delete defaults.max_tokens
      delete manifest.limits.max_output_tokens
    }
    for (const key of ['temperature', 'top_p', 'seed'] as const) {
      // An operator-fixed field has no editable default; discard stale form input
      // so a disabled field cannot leave the reviewed manifest invalid.
      if (
        targets.length &&
        targets.every((target) => typeof target.request_params?.[key] === 'number')
      ) {
        Object.assign(defaults, { [key]: manifest.sampling[key] })
      }
    }
    const context = {
      ...(previewContext.session_id?.trim()
        ? { session_id: previewContext.session_id.trim() }
        : {}),
      ...(previewContext.conversation_id?.trim()
        ? { conversation_id: previewContext.conversation_id.trim() }
        : {}),
      ...(previewContext.sampling_seed !== undefined
        ? { sampling_seed: previewContext.sampling_seed }
        : {}),
    }
    return {
      ...manifest,
      ...(experimentContext ? { experiment: experimentContext } : {}),
      cost_policy: costPolicy,
      output_policy: outputPolicy,
      sampling: defaults,
      ...(mode === 'preview' && Object.keys(context).length ? { preview_context: context } : {}),
    }
  }, [
    name,
    mode,
    profile,
    scope.seed,
    targets,
    limits,
    costPolicy,
    outputPolicy,
    sampling,
    previewContext,
    baseline,
    baselineID,
    experimentContext,
    frozenDataset,
    rerun,
  ])
  const nativeOutput = formManifest.output_policy === 'native'
  const nativeIssues = nativeOutput
    ? targets.flatMap((target) => {
        const issue = nativeOutputIssue(target)
        return issue ? [`${targetLabel(target)}: ${issue}`] : []
      })
    : []
  const fingerprint = JSON.stringify({ manifest: formManifest, benchmarks, scope, frozenDataset })
  const planCurrent = plan?.fingerprint === fingerprint
  const scopeReady = baselineID
    ? baseline?.id === baselineID
    : frozenDataset
      ? true
      : scope.ready && scope.profile === profile
  const blocker = !scopeReady
    ? baselineID
      ? 'Waiting for the reference protocol'
      : scope.error || 'Choose what to measure'
    : !targets.length
      ? targetKind === 'mom'
        ? 'Choose a recipe to compare'
        : 'Choose at least one model or recipe'
      : nativeIssues.length
        ? 'Remove targets without native capacity'
        : ''
  const incomplete = (index: number) =>
    index === 0 ? !scopeReady : index === 1 ? !targets.length || nativeIssues.length > 0 : false

  async function reviewPlan() {
    if (pending || submission || saved.error || !canRun) return
    const sequence = ++requestSequence.current
    const current = () => mounted.current && requestSequence.current === sequence
    setError('')
    setPending(true)
    setPlan(null)
    setPreparation(null)
    const controller = new AbortController()
    preparationController.current = controller
    try {
      let evidence: Plan
      if (baselineID) {
        if (baseline?.id !== baselineID) throw new Error('Wait for the frozen baseline protocol.')
        if (!targets.length) throw new Error('Choose a configured MoM target.')
        evidence = await benchApi.candidatePlan(baselineID, {
          target_ids: targets.map((target) => target.id),
          mode,
          name,
          ...(experimentContext ? { experiment: experimentContext } : {}),
        })
      } else {
        let manifest = formManifest
        if (!targets.length) throw new Error('Choose at least one model or recipe.')
        let dataset: { path: string; sha256: string }
        if (frozenDataset) dataset = frozenDataset
        else {
          if (!scope.ready || scope.profile !== profile)
            throw new Error(scope.error || 'Check the selected benchmarks before reviewing a plan.')
          dataset = scope.missingBenchmarks.length
            ? await prepareEvaluationDataset(
                {
                  benchmarks,
                  profile: profile as PreparationProfile,
                  seed: scope.seed ?? 20260918,
                },
                controller.signal,
                (progress) => {
                  if (current()) setPreparation(progress)
                },
              )
            : (await benchApi.composeDatasets(scope.sourceIDs, benchmarks)).dataset
        }
        if (!current()) return
        manifest = {
          ...manifest,
          dataset: { path: dataset.path, sha256: dataset.sha256 },
        }
        const issue = validateManifest(manifest)
        if (issue) throw new Error(issue)
        evidence = await benchApi.plan(manifest)
      }
      if (!current()) return
      setPreparation(null)
      setPlan({
        manifest: evidence.manifest,
        evidence,
        fingerprint,
        idempotencyKey: crypto.randomUUID(),
      })
    } catch (cause) {
      if (current())
        setError(cause instanceof Error ? cause.message : 'Could not prepare this run.')
    } finally {
      if (current()) setPending(false)
    }
  }

  async function startRun() {
    if (pending || !canRun || saved.error || (!submission && (!plan || !planCurrent))) return
    const sequence = ++requestSequence.current
    const current = () => mounted.current && requestSequence.current === sequence
    const body = submission ?? {
      version: 1 as const,
      actorID,
      manifest: plan!.manifest,
      idempotencyKey: plan!.idempotencyKey,
    }
    setError('')
    setPending(true)
    try {
      saveSubmission(body)
      setSubmission(body)
      const run = await benchApi.start(body.manifest, body.idempotencyKey)
      if (!current()) return
      if (!clearSubmission(body))
        throw new Error(
          'The saved submission changed while this response was pending. Reload to reconcile the current request.',
        )
      setSubmission(null)
      onStarted(run)
    } catch (cause) {
      if (!current()) return
      if (cause instanceof SrBenchRequestError && cause.dispatchStarted === false) {
        try {
          if (!clearSubmission(body)) throw new Error('The saved submission identity changed')
          setSubmission(null)
          setPlan(null)
        } catch {
          setError(
            'The service confirmed no dispatch, but this tab could not clear the saved submission. Reconcile browser storage before creating another plan.',
          )
          return
        }
      }
      setError(
        cause instanceof Error
          ? cause.message
          : 'Could not start this run. Check the run list before submitting again.',
      )
    } finally {
      if (current()) setPending(false)
    }
  }

  const errorNotice = error ? (
    <Notice tone="danger" as="p" role="alert" className="my-0">
      {error}
    </Notice>
  ) : null

  if (submission || saved.error)
    return (
      <>
        <DetailSheetBody className="flex flex-col gap-3 px-5 py-5">
          {saved.error ? (
            <Notice tone="danger" as="p" role="alert" className="my-0">
              {saved.error}
            </Notice>
          ) : (
            submission && (
              <Notice tone="warning" className="my-0">
                <h3>Evaluation submission needs reconciliation</h3>
                <p>
                  The frozen request is saved for this account in this tab. No new plan can replace
                  it until its outcome is known. Checking again uses the same submission ID and
                  manifest; it may finish the original submission if the service never received it.
                </p>
                <p className="font-medium text-foreground">
                  {submission.manifest.name} ·{' '}
                  {submission.manifest.targets.map(targetLabel).join(', ')}
                </p>
                <Disclosure summary="Frozen submission" bare className="my-0">
                  <pre className={codeBlockClass}>
                    {JSON.stringify(submission.manifest, null, 2)}
                  </pre>
                </Disclosure>
              </Notice>
            )
          )}
          {errorNotice}
        </DetailSheetBody>
        <DetailSheetFooter>
          <Button variant="ghost" size="sm" className="ml-auto" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          {submission ? (
            <Button size="sm" disabled={!canRun || pending} onClick={() => void startRun()}>
              {pending ? 'Reconciling…' : 'Check or submit same evaluation'}
            </Button>
          ) : null}
        </DetailSheetFooter>
      </>
    )

  if (baselineID && baseline?.id !== baselineID)
    return (
      <DetailSheetBody className="px-5 py-5">
        {baselineError ? (
          <Notice tone="danger" role="alert" className="my-0">
            <p>{baselineError}</p>
            <div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setBaselineRevision((value) => value + 1)}
              >
                Retry baseline read
              </Button>
            </div>
          </Notice>
        ) : (
          <div role="status" className="flex flex-col gap-2.5">
            <span className="text-xs text-muted-foreground">
              Loading the frozen baseline protocol…
            </span>
            <Skeleton className="h-24 w-full" />
          </div>
        )}
      </DetailSheetBody>
    )

  const missingPriceNotice = !baselineID &&
    mode === 'live' &&
    costPolicy === 'require_priced' &&
    missingPrices && (
      <Notice tone="warning" role="note" aria-label="Missing target prices" className="my-0">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="min-w-0 flex-1">
            Prices are missing for selected targets. Quality only removes the USD budget; time,
            token and call limits remain.
          </span>
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() => changeCostPolicy('capability_only')}
          >
            Use Quality only
          </Button>
        </div>
      </Notice>
    )
  const nativeNotice = nativeIssues.length > 0 && (
    <Notice tone="warning" as="p" role="status" className="my-0">
      Native capacity is unavailable for the selected targets. {nativeIssues.join(' ')} Remove
      unsupported targets or choose bounded output.
    </Notice>
  )
  const evaluations = plan?.evidence.total
  const questions =
    evaluations !== undefined && plan!.manifest.targets.length
      ? Math.round(evaluations / plan!.manifest.targets.length)
      : undefined
  const reviewSettings = baseline?.manifest ?? formManifest

  return (
    <>
      <Stepper step={step} onStep={setStep} incomplete={incomplete} />
      <DetailSheetBody>
        <fieldset
          className="m-0 flex min-w-0 flex-col gap-6 border-0 px-5 py-5"
          disabled={pending}
          aria-label={STEPS[step]}
        >
          {!baselineID && !frozenDataset && (
            <div hidden={step !== 0}>
              <RunDatasetScope
                catalog={catalog}
                datasets={datasets}
                profile={profile}
                onProfile={setProfile}
                benchmarks={benchmarks}
                onBenchmarks={setBenchmarks}
                initialDataset={initialDataset}
                onResolved={setScope}
              />
            </div>
          )}
          {step === 0 &&
            (baselineID && baseline ? (
              <FrozenBaselineProtocol run={baseline} />
            ) : frozenDataset ? (
              <StepSection title="Questions">
                <div className="flex items-start gap-3 rounded-xl border border-border bg-card p-3.5 shadow-xs">
                  <IconTile icon={Database} tone="orange" size="md" />
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <span className="text-[13px] font-semibold">Same questions as before</span>
                    <span className="text-xs text-muted-foreground">
                      {profileTitle(profile)} · the frozen dataset of the original run is reused, so
                      results stay comparable.
                    </span>
                    <BenchmarkChips ids={benchmarks} catalog={catalog} max={4} />
                    <code className="truncate font-mono text-[11px] text-subtle-foreground">
                      {frozenDataset.sha256}
                    </code>
                  </div>
                </div>
                <div>
                  <Button
                    variant="link"
                    size="sm"
                    className="p-0"
                    onClick={() => setFrozenDataset(undefined)}
                  >
                    Choose questions again
                  </Button>
                </div>
              </StepSection>
            ) : null)}
          {step === 0 && mode === 'preview' && (
            <Notice tone="info" as="p" className="my-0">
              A dry run checks decisions and selected models. It does not generate answers or
              measure capability.
            </Notice>
          )}
          {step === 1 && (
            <>
              <StepSection
                title={
                  targetKind === 'single'
                    ? 'Single models'
                    : targetKind === 'mom'
                      ? 'Recipe'
                      : 'Models and recipes'
                }
                aside={
                  <span className="text-xs text-muted-foreground">
                    {targetKind === 'mom'
                      ? 'Routed by the Router on the same questions'
                      : 'Compare a recipe against the models it routes to'}
                  </span>
                }
              >
                {registeredTargets.length ? (
                  <TargetPicker
                    targets={availableTargets}
                    selected={targets}
                    onChange={setTargets}
                    unavailable={(target) => {
                      if (!nativeOutput) return null
                      const issue = nativeOutputIssue(target)
                      return issue ? `Native output unavailable: ${issue}` : null
                    }}
                  />
                ) : (
                  <Notice tone="warning" as="p" className="my-0">
                    No targets are registered. Run vllm-sr benchmark target register --file
                    targets.json against this service’s store; this list updates automatically.
                  </Notice>
                )}
              </StepSection>
              {missingPriceNotice}
              {nativeNotice}
              {targets.length > 0 && (
                <StepSection title="Request profiles">
                  <div className="flex flex-col gap-3">
                    {targets.map((target) => (
                      <div
                        key={target.id}
                        className="min-w-0 rounded-lg border border-border bg-surface px-3.5 py-3"
                      >
                        <div className="mb-1 flex flex-wrap items-center gap-2">
                          <TargetChip target={target} />
                          <span className="text-xs text-muted-foreground">
                            Credentials stay on the server.
                            {target.prices
                              ? ' Configured prices included.'
                              : ' Unpriced usage will remain unknown.'}
                          </span>
                        </div>
                        <TargetRequestProfile
                          target={target}
                          sampling={formManifest.sampling}
                          outputPolicy={formManifest.output_policy}
                        />
                        <Disclosure summary="Connection and configuration" bare className="my-0">
                          <dl className={`${identityClass} my-1`}>
                            <dt>Endpoint</dt>
                            <dd className="font-mono">{target.base_url}</dd>
                            {target.config_hash && (
                              <>
                                <dt>Frozen configuration</dt>
                                <dd>
                                  <code>{target.config_hash}</code>
                                </dd>
                              </>
                            )}
                          </dl>
                        </Disclosure>
                      </div>
                    ))}
                  </div>
                </StepSection>
              )}
            </>
          )}
          {step === 2 && (
            <>
              {nativeNotice}
              {baselineID ? (
                <Notice tone="info" as="p" className="my-0">
                  Questions, limits and sampling come from the frozen reference run, so the
                  candidate is measured under the same protocol.
                </Notice>
              ) : (
                <RunSettings
                  limits={limits}
                  onLimitsChange={setLimits}
                  sampling={sampling}
                  onSamplingChange={setSampling}
                  seed={formManifest.seed}
                  targets={targets}
                  costPolicy={costPolicy}
                  outputPolicy={outputPolicy}
                  onOutputPolicyChange={setOutputPolicy}
                  nativeAvailable={availableTargets.some((target) => !nativeOutputIssue(target))}
                  onCostPolicyChange={changeCostPolicy}
                  mode={mode}
                  previewContext={previewContext}
                  onPreviewContextChange={setPreviewContext}
                />
              )}
              <StepSection title="Run details">
                <div className="grid min-w-0 gap-1.5">
                  <Label htmlFor="evaluation-run-name">Run name</Label>
                  <Input
                    id="evaluation-run-name"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                  />
                </div>
                {experiment && (
                  <div className="grid min-w-0 gap-1.5">
                    <Label htmlFor="evaluation-hypothesis">What changed? (optional)</Label>
                    <Textarea
                      id="evaluation-hypothesis"
                      value={hypothesis}
                      maxLength={2000}
                      placeholder="For example: prefer the less expensive model for simple questions."
                      onChange={(event) => setHypothesis(event.target.value)}
                    />
                  </div>
                )}
              </StepSection>
            </>
          )}
          {step === 3 && (
            <>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <MiniStat
                  label={mode === 'preview' ? 'Routed questions' : 'Evaluations'}
                  value={evaluations !== undefined ? number(evaluations) : pending ? '…' : '—'}
                  hint={
                    questions !== undefined
                      ? `${number(questions)} questions × ${plan!.manifest.targets.length} ${plan!.manifest.targets.length === 1 ? 'target' : 'targets'}`
                      : 'Counted by the service when the plan is ready'
                  }
                />
                <MiniStat
                  label="Cost"
                  value={
                    mode === 'preview'
                      ? 'No model calls'
                      : reviewSettings.cost_policy === 'capability_only'
                        ? 'No USD cap'
                        : `Up to $${number(reviewSettings.limits.max_cost_usd, 2)}`
                  }
                  hint={
                    mode === 'preview'
                      ? 'Routing only'
                      : reviewSettings.cost_policy === 'capability_only'
                        ? 'Quality only: token and call limits remain'
                        : 'The run stops at this budget'
                  }
                />
                <MiniStat
                  label="Time"
                  value={`Up to ${duration(reviewSettings.limits.max_run_seconds)}`}
                  hint="The run stops at this deadline"
                />
              </div>
              <StepSection title="Plan">
                <KeyValueGrid
                  items={[
                    { label: 'Name', value: name },
                    { label: 'Size', value: profileTitle(reviewSettings.profile) },
                    {
                      label: 'Benchmarks',
                      value: (
                        <BenchmarkChips
                          ids={baseline?.manifest.dataset?.benchmarks ?? benchmarks}
                          catalog={catalog}
                          max={4}
                        />
                      ),
                    },
                    {
                      label: 'Targets',
                      value: (
                        <span className="flex flex-wrap gap-1">
                          {targets.map((target) => (
                            <TargetChip key={target.id} target={target} />
                          ))}
                        </span>
                      ),
                    },
                  ]}
                />
              </StepSection>
              {!baselineID &&
                !frozenDataset &&
                scope.missingBenchmarks.length > 0 &&
                !preparation &&
                !plan && (
                  <Notice tone="info" icon={Database} as="p" className="my-0">
                    {scope.missingBenchmarks.map(benchmarkTitle).join(', ')}{' '}
                    {scope.missingBenchmarks.length === 1 ? 'is' : 'are'} prepared on the service
                    before planning: downloaded, normalized and pinned.
                  </Notice>
                )}
              {preparation && (
                <section
                  className="min-w-0 rounded-lg border border-info/25 bg-info-soft px-4 py-3"
                  aria-label="Preparing evaluation"
                  aria-live="polite"
                >
                  <div className="flex items-center gap-2 text-[13px]">
                    {pending && (
                      <RefreshCw aria-hidden="true" className="size-4 shrink-0 text-info" />
                    )}
                    <strong className="font-medium text-foreground">
                      {preparation.phase === 'waiting_for_slot'
                        ? 'Waiting for another preparation to finish'
                        : preparation.job
                          ? preparationPhase(preparation.job)
                          : 'Checking evaluation data'}
                    </strong>
                  </div>
                  <p className="mt-1 text-[13px] text-muted-foreground">
                    {preparation.phase === 'waiting_for_slot'
                      ? 'Keep this page open until preparation starts.'
                      : 'Preparation runs on the service. You can leave this page and return later.'}
                  </p>
                  {preparation.job?.items && (
                    <ul className="mt-2.5 grid gap-1.5">
                      {preparation.job.items.map((item) => (
                        <li
                          key={item.benchmark}
                          className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-[13px]"
                        >
                          <span className="text-foreground">{benchmarkTitle(item.benchmark)}</span>
                          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                            <StatusDot
                              tone={itemTone(item.status)}
                              pulse={item.status === 'running'}
                            />
                            {item.status === 'completed'
                              ? item.reused
                                ? 'Already available'
                                : 'Ready'
                              : item.status === 'failed'
                                ? 'Needs attention'
                                : item.status === 'running'
                                  ? 'Preparing…'
                                  : 'Waiting'}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              )}
              {pending && !preparation && !plan && (
                <div role="status" className="flex flex-col gap-2">
                  <span className="text-xs text-muted-foreground">
                    Checking the plan with sr-bench…
                  </span>
                  <Skeleton className="h-3 w-2/3" />
                </div>
              )}
              {errorNotice}
              {missingPriceNotice}
              {plan && (
                <Notice
                  tone={planCurrent ? 'success' : 'warning'}
                  icon={planCurrent ? CheckCircle2 : RefreshCw}
                  className="my-0"
                >
                  <h3>{planCurrent ? 'Plan ready for review' : 'Plan changed — review again'}</h3>
                  <p>
                    {number(plan.evidence.total)} planned evaluations ·{' '}
                    {plan.manifest.targets.length} targets · {plan.manifest.mode}
                  </p>
                  <Disclosure summary="Preflight and frozen manifest" bare className="my-0">
                    <pre className={codeBlockClass}>{JSON.stringify(plan.evidence, null, 2)}</pre>
                    <pre className={codeBlockClass}>{JSON.stringify(plan.manifest, null, 2)}</pre>
                  </Disclosure>
                </Notice>
              )}
            </>
          )}
          {!canRun && (
            <Notice tone="neutral" as="p" className="my-0">
              Run controls require evaluation permissions and a writable Dashboard session.
            </Notice>
          )}
        </fieldset>
      </DetailSheetBody>
      <DetailSheetFooter highlight={step === 3 && planCurrent}>
        <span
          className={cn(
            'hidden min-w-0 truncate text-xs tabular sm:inline',
            step >= 2 && blocker ? 'text-warning-foreground' : 'text-muted-foreground',
          )}
        >
          {step >= 2 && blocker
            ? blocker
            : `Step ${step + 1} of ${STEPS.length}${step < 3 && targets.length ? ` · ${targets.length} ${targets.length === 1 ? 'target' : 'targets'}` : ''}`}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {step > 0 ? (
            <Button variant="ghost" size="sm" disabled={pending} onClick={() => setStep(step - 1)}>
              <ArrowLeft aria-hidden="true" />
              Back
            </Button>
          ) : (
            <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
          )}
          {step < 2 ? (
            <Button size="sm" onClick={() => goTo(step + 1)}>
              Continue
              <ArrowRight aria-hidden="true" />
            </Button>
          ) : step === 2 || !planCurrent ? (
            <Button
              size="sm"
              disabled={
                pending || !canRun || nativeIssues.length > 0 || !scopeReady || !targets.length
              }
              onClick={() => {
                goTo(3)
                void reviewPlan()
              }}
            >
              {pending ? 'Working…' : 'Review plan'}
            </Button>
          ) : (
            <Button
              size="sm"
              disabled={pending || !canRun || !planCurrent || nativeIssues.length > 0}
              onClick={() => void startRun()}
            >
              <Play aria-hidden="true" />
              {mode === 'preview' ? 'Start dry run' : 'Start evaluation'}
            </Button>
          )}
        </div>
      </DetailSheetFooter>
    </>
  )
}
