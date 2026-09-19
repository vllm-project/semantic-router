import { useEffect, useMemo, useRef, useState } from 'react'
import { benchApi, SrBenchRequestError } from './api'
import { DEFAULT_LIMITS, makeManifest, number, validateManifest } from './model'
import type { Catalog, Dataset, Manifest, Plan, Run, Target } from './types'
import styles from './SrBench.module.css'
import ProductIcon from '../ProductIcon'
import BenchSelect from './BenchSelect'
import RunSettings, { type SamplingSettings } from './RunSettings'
import TargetRequestProfile from './TargetRequestProfile'
import controls from './BenchControls.module.css'
import {
  benchmarkTitle,
  friendlyDatasetName,
  profileTitle,
  profileDescription,
} from './datasetPresentation'
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
  onStarted,
}: Props) {
  const [saved] = useState(() => readSubmission(actorID))
  const [submission, setSubmission] = useState<PendingSubmission | null>(saved.request)
  const [name, setName] = useState('Balance comparison')
  const [mode, setMode] = useState<Manifest['mode']>('live')
  const [costPolicy, setCostPolicy] = useState<'require_priced' | 'capability_only'>(
    'require_priced',
  )
  const [profile, setProfile] = useState(
    datasets.find((item) => item.id === initialDataset)?.profile ?? 'quick',
  )
  const initialSource =
    datasets.find((item) => item.id === initialDataset) ??
    datasets.find((item) => !item.profile || item.profile === 'quick')
  const [datasetID, setDatasetID] = useState(initialSource?.id ?? '')
  const [benchmarks, setBenchmarks] = useState<string[]>(initialSource?.benchmarks ?? [])
  const [targets, setTargets] = useState<Target[]>(() =>
    registeredTargets.filter((target) => target.model === initialModel),
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
  const mounted = useRef(false)
  const requestSequence = useRef(0)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      requestSequence.current += 1
    }
  }, [])
  const profiles = ['smoke', 'quick', 'standard'].filter((value) =>
    catalog.profiles.some((item) => item.id === value),
  )
  const dataset = datasets.find((item) => item.id === datasetID)
  const formManifest = useMemo(() => {
    const manifest = makeManifest(name, mode, profile, dataset, targets, limits)
    const defaults = { ...manifest.sampling, ...sampling }
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
      cost_policy: costPolicy,
      sampling: defaults,
      ...(mode === 'preview' && Object.keys(context).length ? { preview_context: context } : {}),
    }
  }, [name, mode, profile, dataset, targets, limits, costPolicy, sampling, previewContext])
  const fingerprint = JSON.stringify({ manifest: formManifest, benchmarks })
  const sources = datasets.filter((item) => !item.profile || item.profile === profile)
  const compatibleSources = dataset
    ? sources.filter((item) => item.seed === dataset.seed && item.split === dataset.split)
    : []
  const availableBenchmarks = new Set(compatibleSources.flatMap((item) => item.benchmarks ?? []))
  function chooseProfile(value: string) {
    const source = datasets.find((item) => !item.profile || item.profile === value)
    setProfile(value)
    setDatasetID(source?.id ?? '')
    setBenchmarks(source?.benchmarks ?? [])
  }
  function selectedSourceIDs(): string[] {
    if (!dataset) return []
    const compatible = compatibleSources
    const selected = [dataset]
    for (const benchmark of benchmarks) {
      if (selected.some((item) => item.benchmarks?.includes(benchmark))) continue
      const covered = new Set(
        selected.flatMap((item) => item.benchmarks ?? []).filter((id) => benchmarks.includes(id)),
      )
      const source = compatible
        .filter((item) => item.benchmarks?.includes(benchmark))
        .sort((a, b) => {
          const overlap = (item: Dataset) =>
            item.benchmarks?.filter((id) => covered.has(id)).length ?? 0
          return (
            overlap(a) - overlap(b) ||
            (a.benchmarks?.length ?? 0) - (b.benchmarks?.length ?? 0) ||
            a.id.localeCompare(b.id)
          )
        })[0]
      if (!source)
        throw new Error(
          `No compatible prepared source for ${benchmark}. Choose another source collection; development and holdout data cannot be mixed.`,
        )
      selected.push(source)
    }
    return selected
      .filter((item) => item.benchmarks?.some((benchmark) => benchmarks.includes(benchmark)))
      .map((item) => item.id)
  }
  const planCurrent = plan?.fingerprint === fingerprint

  async function reviewPlan() {
    if (pending || submission || saved.error) return
    const sequence = ++requestSequence.current
    const current = () => mounted.current && requestSequence.current === sequence
    setError('')
    setPending(true)
    setPlan(null)
    try {
      let manifest = formManifest
      const issue = validateManifest(manifest)
      if (issue) throw new Error(issue)
      if (!benchmarks.length) throw new Error('Select at least one prepared benchmark.')
      const composed = await benchApi.composeDatasets(selectedSourceIDs(), benchmarks)
      if (!current()) return
      manifest = {
        ...manifest,
        dataset: { path: composed.dataset.path, sha256: composed.dataset.sha256 },
      }
      const evidence = await benchApi.plan(manifest)
      if (!current()) return
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
        <h2 id="new-run-title">Create evaluation</h2>
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
                {submission.manifest.targets.map((target) => target.id).join(', ')}
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

  return (
    <section className={styles.panel} aria-labelledby="new-run-title">
      <div className={styles.sectionHeading}>
        <div>
          <h2 id="new-run-title">Create evaluation</h2>
          <p>Use the same frozen questions for single models and your MoM.</p>
        </div>
        <span className={styles.badge}>sr-bench 1.0</span>
      </div>
      <h3>1. Choose scope</h3>
      <div className={styles.profileCards} role="radiogroup" aria-label="Evaluation size">
        {profiles.map((value) => (
          <label
            key={value}
            className={`${styles.profileCard} ${profile === value ? styles.profileSelected : ''}`}
          >
            <input
              type="radio"
              name="evaluation-profile"
              checked={profile === value}
              onChange={() => chooseProfile(value)}
            />
            <strong>{profileTitle(value)}</strong>
            <span>{profileDescription(value)}</span>
          </label>
        ))}
      </div>
      {!sources.length && (
        <p className={styles.notice}>
          No prepared {profileTitle(profile).toLowerCase()} datasets are registered. Prepare a
          dataset with the CLI to use this size.
        </p>
      )}
      <div className={controls.selectionHeading}>
        <h4>Benchmarks</h4>
        <button
          className={controls.compactButton}
          onClick={() =>
            setBenchmarks(
              benchmarks.length === availableBenchmarks.size ? [] : [...availableBenchmarks],
            )
          }
          disabled={!availableBenchmarks.size}
        >
          {benchmarks.length === availableBenchmarks.size && benchmarks.length
            ? 'Clear benchmarks'
            : 'Select all benchmarks'}
        </button>
      </div>
      <div className={styles.benchmarkChoices} role="group" aria-label="Included benchmarks">
        {catalog.benchmarks.map((benchmark) => (
          <label key={benchmark.id} className={styles.benchmarkChoice}>
            <input
              type="checkbox"
              checked={benchmarks.includes(benchmark.id)}
              disabled={!availableBenchmarks.has(benchmark.id)}
              onChange={(event) =>
                setBenchmarks((previous) =>
                  event.target.checked
                    ? [...previous, benchmark.id]
                    : previous.filter((id) => id !== benchmark.id),
                )
              }
            />
            <span>
              {benchmarkTitle(benchmark.id)}
              <small>
                {availableBenchmarks.has(benchmark.id)
                  ? 'Prepared'
                  : 'No compatible prepared source'}
              </small>
            </span>
          </label>
        ))}
      </div>
      <details className={styles.details}>
        <summary>Prepared source collection</summary>
        <BenchSelect
          label="Prepared dataset"
          className={controls.sourcePicker}
          value={datasetID}
          searchable
          placeholder="Select a prepared source"
          options={sources.map((item) => ({
            value: item.id,
            label: friendlyDatasetName(item),
            description: `${number(item.case_count)} cases · ${item.split ?? 'split unspecified'}`,
          }))}
          onChange={(value) => {
            setDatasetID(value)
            setBenchmarks(datasets.find((item) => item.id === value)?.benchmarks ?? [])
          }}
        />
        <p className={styles.muted}>
          Selected benchmarks are composed without resampling. Sources must share the same size,
          seed and split.
        </p>
      </details>
      <div className={styles.formGrid}>
        <label>
          Run name
          <input value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <BenchSelect
          label="Mode"
          value={mode}
          options={[
            {
              value: 'live',
              label: 'Live evaluation',
              description: 'Generate answers and measure quality, cost and latency.',
            },
            {
              value: 'preview',
              label: 'Route preview',
              description: 'Inspect routing decisions without generating answers.',
            },
          ]}
          onChange={(value) => setMode(value as Manifest['mode'])}
        />
      </div>
      {mode === 'preview' && (
        <p className={styles.notice}>
          Preview checks decisions and selected models. It does not generate answers or measure
          capability.
        </p>
      )}
      {datasets.length === 0 && (
        <p className={styles.notice}>
          No prepared datasets are registered. Prepare a frozen dataset with the sr-bench CLI
          connected to this service, then refresh this page.
        </p>
      )}
      {dataset && (
        <p className={styles.muted}>
          {benchmarks.length} benchmarks selected. The reviewed plan shows the exact frozen case
          count.
        </p>
      )}
      <div className={styles.sectionHeading}>
        <h3>2. Choose targets</h3>
        {registeredTargets.length > 0 && (
          <BenchSelect
            label="Add configured target"
            className={controls.targetPicker}
            value=""
            searchable
            placeholder="Choose target"
            options={registeredTargets
              .filter((item) => !targets.some((target) => target.id === item.id))
              .map((target) => ({
                value: target.id,
                label: target.id,
                description: `${target.kind === 'mom' ? 'Mixture of models' : 'Single model'} · ${target.model}`,
              }))}
            onChange={(value) => {
              const target = registeredTargets.find((item) => item.id === value)
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
              {target.id}
            </legend>
            <p>
              <strong>{target.kind === 'mom' ? 'Mixture of models' : 'Single model'}</strong> ·{' '}
              {target.model}
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
            <TargetRequestProfile target={target} sampling={formManifest.sampling} />
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
      <BenchSelect
        label="Cost accounting"
        className={controls.costControl}
        value={costPolicy}
        options={[
          {
            value: 'require_priced',
            label: 'Quality and cost',
            description: 'Require complete prices and a model-cost budget.',
          },
          {
            value: 'capability_only',
            label: 'Quality only',
            description: 'Run without cost-saving claims when prices are unavailable.',
          },
        ]}
        onChange={(value) => {
          setCostPolicy(value as 'require_priced' | 'capability_only')
          if (value === 'capability_only')
            setLimits((previous) =>
              !Number.isFinite(previous.max_cost_usd) || previous.max_cost_usd <= 0
                ? { ...previous, max_cost_usd: DEFAULT_LIMITS.max_cost_usd }
                : previous,
            )
        }}
      />
      {costPolicy === 'capability_only' && (
        <p className={styles.notice}>
          Unknown or unpriced usage cannot be bounded by a USD budget. Time, output and call limits
          still apply; this run cannot prove cost savings.
        </p>
      )}
      <h3>3. Set budget and limits</h3>
      <RunSettings
        limits={limits}
        onLimitsChange={setLimits}
        sampling={sampling}
        onSamplingChange={setSampling}
        seed={formManifest.seed}
        targets={targets}
        costPolicy={costPolicy}
        mode={mode}
        previewContext={previewContext}
        onPreviewContextChange={setPreviewContext}
      />
      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
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
        <button type="button" disabled={pending || !canRun} onClick={() => void reviewPlan()}>
          {pending ? 'Working…' : 'Review plan'}
        </button>
        <button
          type="button"
          className={styles.primary}
          disabled={pending || !canRun || !planCurrent}
          onClick={() => void startRun()}
        >
          Start evaluation
        </button>
      </div>
    </section>
  )
}
