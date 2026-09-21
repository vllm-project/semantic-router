import { useEffect, useMemo, useRef, useState } from 'react'
import { benchApi, SrBenchRequestError } from './api'
import { DEFAULT_LIMITS, makeManifest, number, validateManifest } from './model'
import type { Catalog, Dataset, Manifest, Plan, Run, Target } from './types'
import styles from './SrBench.module.css'
import ProductLoadingState from '../ProductLoadingState'
import FrozenBaselineProtocol from './FrozenBaselineProtocol'
import { canReuseBaseline } from './baselineReuse'
import ProductIcon from '../ProductIcon'
import BenchSelect from './BenchSelect'
import RunSettings, { type SamplingSettings } from './RunSettings'
import TargetRequestProfile from './TargetRequestProfile'
import { targetLabel } from './targetPresentation'
import { nativeOutputIssue } from './nativeOutput'
import { experimentRoleLabels } from './experimentEvidencePresentation'
import controls from './BenchControls.module.css'
import RunDatasetScope, { type ResolvedDatasetScope } from './RunDatasetScope'
import {
  prepareEvaluationDataset,
  type EvaluationPreparationProgress,
} from './evaluationPreparation'
import { preparationPhase, type PreparationProfile } from './datasetPreparationApi'
import { benchmarkTitle } from './datasetPresentation'
import composer from './RunComposer.module.css'
import {
  clearSubmission,
  readSubmission,
  saveSubmission,
  type PendingSubmission,
} from './submission'

interface Props {
  catalog: Catalog
  datasets: Dataset[]
  targets: Target[]
  canRun: boolean
  actorID: string
  initialModel?: string
  initialDataset?: string
  mode?: 'live' | 'preview'
  baselineID?: string
  experiment?: Manifest['experiment']
  onStarted: (run: Run) => void
}

export default function RunComposer({
  catalog,
  datasets,
  targets: registeredTargets,
  canRun,
  actorID,
  initialModel,
  initialDataset,
  mode = 'live',
  baselineID,
  experiment,
  onStarted,
}: Props) {
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
    mode === 'preview'
      ? 'Routing preview'
      : experiment
        ? experimentRoleLabels[experiment.role]
        : 'Model comparison',
  )
  const [hypothesis, setHypothesis] = useState(experiment?.hypothesis ?? '')
  const [costPolicy, setCostPolicy] = useState<'require_priced' | 'capability_only'>(
    'require_priced',
  )
  const [outputPolicy, setOutputPolicy] = useState<'bounded' | 'native'>('bounded')
  const [profile, setProfile] = useState(
    datasets.find((item) => item.id === initialDataset)?.profile ?? 'quick',
  )
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
    datasets.find((item) => item.id === initialDataset)?.benchmarks ?? [],
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
    availableTargets.filter((target) => target.model === initialModel),
  )
  const [limits, setLimits] = useState({ ...DEFAULT_LIMITS })
  const [sampling, setSampling] = useState<SamplingSettings>({ temperature: 0, top_p: 1 })
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
    manifest.seed = scope.seed ?? 20260918
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
  ])
  const nativeOutput = formManifest.output_policy === 'native'
  const nativeIssues = nativeOutput
    ? targets.flatMap((target) => {
        const issue = nativeOutputIssue(target)
        return issue ? [`${targetLabel(target)}: ${issue}`] : []
      })
    : []
  const fingerprint = JSON.stringify({ manifest: formManifest, benchmarks, scope })
  const planCurrent = plan?.fingerprint === fingerprint

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
        if (!scope.ready || scope.profile !== profile)
          throw new Error(scope.error || 'Check the selected benchmarks before reviewing a plan.')
        const dataset = scope.missingBenchmarks.length
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

  if (submission || saved.error)
    return (
      <section className={styles.panel} aria-labelledby="new-run-title">
        <h2 id="new-run-title">{mode === 'preview' ? 'Preview routing' : 'Create evaluation'}</h2>
        {saved.error ? (
          <p className={styles.error} role="alert">
            {saved.error}
          </p>
        ) : (
          submission && (
            <div className={styles.notice}>
              <h3>Evaluation submission needs reconciliation</h3>
              <p>
                The frozen request is saved for this account in this tab. No new plan can replace it
                until its outcome is known. Checking again uses the same submission ID and manifest;
                it may finish the original submission if the service never received it.
              </p>
              <p>
                {submission.manifest.name} ·{' '}
                {submission.manifest.targets.map(targetLabel).join(', ')}
              </p>
              <button disabled={!canRun || pending} onClick={() => void startRun()}>
                {pending ? 'Reconciling…' : 'Check or submit same evaluation'}
              </button>
              <details className={styles.details}>
                <summary>Frozen submission</summary>
                <pre>{JSON.stringify(submission.manifest, null, 2)}</pre>
              </details>
            </div>
          )
        )}
        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}
      </section>
    )

  if (baselineID && baseline?.id !== baselineID)
    return (
      <section className={styles.panel}>
        {baselineError ? (
          <div className={styles.error} role="alert">
            <p>{baselineError}</p>
            <button onClick={() => setBaselineRevision((value) => value + 1)}>
              Retry baseline read
            </button>
          </div>
        ) : (
          <ProductLoadingState compact label="Loading the frozen baseline protocol…" />
        )}
      </section>
    )

  return (
    <section className={styles.panel} aria-labelledby="new-run-title">
      <div className={styles.sectionHeading}>
        <div>
          <h2 id="new-run-title">{mode === 'preview' ? 'Preview routing' : 'Create evaluation'}</h2>
          <p>Use the same frozen questions for single models and your MoM.</p>
        </div>
        <span className={styles.badge}>sr-bench 1.0</span>
      </div>
      <fieldset className={composer.fields} disabled={pending}>
        <h3>1. Choose scope</h3>
        {baselineID && baseline ? (
          <FrozenBaselineProtocol run={baseline} />
        ) : (
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
        )}
        <div className={styles.formGrid}>
          <label>
            Run name
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
        </div>
        {experiment && (
          <label>
            What changed? (optional)
            <textarea
              value={hypothesis}
              maxLength={2000}
              placeholder="For example: prefer the less expensive model for simple questions."
              onChange={(event) => setHypothesis(event.target.value)}
            />
          </label>
        )}
        {mode === 'preview' && (
          <p className={styles.notice}>
            Preview checks decisions and selected models. It does not generate answers or measure
            capability.
          </p>
        )}
        <div className={styles.sectionHeading}>
          <h3>
            2.{' '}
            {targetKind === 'single'
              ? 'Choose single models'
              : targetKind === 'mom'
                ? 'Choose a recipe'
                : 'Choose models or recipes'}
          </h3>
          {registeredTargets.length > 0 && (
            <BenchSelect
              label="Add configured target"
              className={controls.targetPicker}
              value=""
              searchable
              placeholder="Choose target"
              options={availableTargets
                .filter((item) => !targets.some((target) => target.id === item.id))
                .filter((item) => !nativeOutput || !nativeOutputIssue(item))
                .map((target) => ({
                  value: target.id,
                  label: targetLabel(target),
                  description: target.kind === 'mom' ? 'Mixture of models' : 'Single model',
                }))}
              onChange={(value) => {
                const target = availableTargets.find((item) => item.id === value)
                if (target) setTargets((previous) => [...previous, { ...target }])
              }}
            />
          )}
        </div>
        {!registeredTargets.length && (
          <p className={styles.notice}>
            No targets are registered. Run vllm-sr benchmark target register --file targets.json
            against this service’s store, then refresh.
          </p>
        )}
        <div className={styles.targetList}>
          {targets.map((target, index) => (
            <fieldset key={index} className={styles.target}>
              <legend>
                <ProductIcon name={target.kind === 'mom' ? 'mixture' : 'model'} />
                {targetLabel(target)}
              </legend>
              <p>
                <strong>{target.kind === 'mom' ? 'Mixture of models' : 'Single model'}</strong>
              </p>
              <details className={styles.details}>
                <summary>Connection and configuration</summary>
                <dl className={styles.identity}>
                  <dt>Endpoint</dt>
                  <dd>{target.base_url}</dd>
                  {target.config_hash && (
                    <>
                      <dt>Frozen configuration</dt>
                      <dd>
                        <code>{target.config_hash}</code>
                      </dd>
                    </>
                  )}
                </dl>
              </details>
              <TargetRequestProfile
                target={target}
                sampling={formManifest.sampling}
                outputPolicy={formManifest.output_policy}
              />
              <div className={styles.targetFooter}>
                <span>
                  Credentials stay on the server.
                  {target.prices
                    ? ' Configured prices included.'
                    : ' Unpriced usage will remain unknown.'}
                  {target.kind === 'mom' &&
                    (target.capture_recipe
                      ? ' A matching recipe snapshot will be captured.'
                      : ' Recipe capture is not enabled for this target.')}
                </span>
                <button
                  type="button"
                  className={controls.compactButton}
                  onClick={() => setTargets((previous) => previous.filter((_, i) => i !== index))}
                >
                  <ProductIcon name="close" />
                  Remove target {index + 1}
                </button>
              </div>
            </fieldset>
          ))}
        </div>
        {nativeIssues.length > 0 && (
          <p className={styles.notice} role="status">
            Native capacity is unavailable for the selected targets. {nativeIssues.join(' ')} Remove
            unsupported targets or choose bounded output.
          </p>
        )}
        {!baselineID && (
          <>
            <h3>3. Set budget and limits</h3>
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
              onCostPolicyChange={(value) => {
                setCostPolicy(value)
                if (value === 'capability_only')
                  setLimits((previous) =>
                    !Number.isFinite(previous.max_cost_usd) || previous.max_cost_usd <= 0
                      ? { ...previous, max_cost_usd: DEFAULT_LIMITS.max_cost_usd }
                      : previous,
                  )
              }}
              mode={mode}
              previewContext={previewContext}
              onPreviewContextChange={setPreviewContext}
            />
          </>
        )}
        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}
        {preparation && (
          <section
            className={composer.preparation}
            aria-label="Preparing evaluation"
            aria-live="polite"
          >
            <div className={composer.preparationHeading}>
              {pending && <ProductIcon name="refresh" />}
              <strong>
                {preparation.phase === 'waiting_for_slot'
                  ? 'Waiting for another preparation to finish'
                  : preparation.job
                    ? preparationPhase(preparation.job)
                    : 'Checking evaluation data'}
              </strong>
            </div>
            <p>
              {preparation.phase === 'waiting_for_slot'
                ? 'Keep this page open until preparation starts.'
                : 'Preparation runs on the service. You can leave this page and return later.'}
            </p>
            {preparation.job?.items && (
              <ul>
                {preparation.job.items.map((item) => (
                  <li key={item.benchmark}>
                    <span>{benchmarkTitle(item.benchmark)}</span>
                    <span>
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
        {plan && (
          <div className={styles.plan}>
            <h3>{planCurrent ? 'Plan ready for review' : 'Plan changed — review again'}</h3>
            <p>
              {number(plan.evidence.total)} planned evaluations · {plan.manifest.targets.length}{' '}
              targets · {plan.manifest.mode}
            </p>
            <details>
              <summary>Preflight and frozen manifest</summary>
              <pre>{JSON.stringify(plan.evidence, null, 2)}</pre>
              <pre>{JSON.stringify(plan.manifest, null, 2)}</pre>
            </details>
          </div>
        )}
        {!canRun && (
          <p className={styles.notice}>
            Run controls require evaluation permissions and a writable Dashboard session.
          </p>
        )}
        <div className={styles.actions}>
          <button
            type="button"
            className={!planCurrent ? styles.primary : undefined}
            disabled={
              pending ||
              !canRun ||
              nativeIssues.length > 0 ||
              (baselineID ? baseline?.id !== baselineID : !scope.ready || scope.profile !== profile)
            }
            onClick={() => void reviewPlan()}
          >
            {pending ? 'Working…' : 'Review plan'}
          </button>
          <button
            type="button"
            className={planCurrent ? styles.primary : undefined}
            disabled={pending || !canRun || !planCurrent || nativeIssues.length > 0}
            onClick={() => void startRun()}
          >
            {mode === 'preview' ? 'Start route preview' : 'Start evaluation'}
          </button>
        </div>
      </fieldset>
    </section>
  )
}
