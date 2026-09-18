import { useEffect, useMemo, useRef, useState } from 'react'
import { benchApi, SrBenchRequestError } from './api'
import {
  DEFAULT_LIMITS,
  effectiveRequestProfile,
  makeManifest,
  number,
  validateManifest,
} from './model'
import type { Catalog, Dataset, Manifest, Plan, Run, Target } from './types'
import styles from './SrBench.module.css'
import ProductIcon from '../ProductIcon'
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
  const [advanced, setAdvanced] = useState(false)
  const [json, setJson] = useState('')
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
  const formManifest = useMemo(
    () => ({
      ...makeManifest(name, mode, profile, dataset, targets, limits),
      cost_policy: costPolicy,
    }),
    [name, mode, profile, dataset, targets, limits, costPolicy],
  )
  const fingerprint = advanced ? json : JSON.stringify({ manifest: formManifest, benchmarks })
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
      let manifest = advanced ? (JSON.parse(json) as Manifest) : formManifest
      const issue = validateManifest(manifest)
      if (issue) throw new Error(issue)
      if (!advanced) {
        if (!benchmarks.length) throw new Error('Select at least one prepared benchmark.')
        const composed = await benchApi.composeDatasets(selectedSourceIDs(), benchmarks)
        if (!current()) return
        manifest = {
          ...manifest,
          dataset: { path: composed.dataset.path, sha256: composed.dataset.sha256 },
        }
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
      <div className={styles.sectionHeading}>
        <h4>Benchmarks</h4>
        <button
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
        <label>
          Prepared dataset
          <select
            value={datasetID}
            onChange={(event) => {
              setDatasetID(event.target.value)
              setBenchmarks(
                datasets.find((item) => item.id === event.target.value)?.benchmarks ?? [],
              )
            }}
          >
            <option value="">Select a prepared source</option>
            {sources.map((item) => (
              <option key={item.id} value={item.id}>
                {friendlyDatasetName(item)} · {number(item.case_count)} cases ·{' '}
                {item.split ?? 'split unspecified'}
              </option>
            ))}
          </select>
        </label>
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
        <label>
          Mode
          <select
            value={mode}
            onChange={(event) => setMode(event.target.value as Manifest['mode'])}
          >
            <option value="live">Live evaluation</option>
            <option value="preview">Route preview</option>
          </select>
        </label>
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
        <div className={styles.actions}>
          {registeredTargets.length > 0 && (
            <label className={styles.inlineLabel}>
              Add configured target
              <select
                value=""
                onChange={(event) => {
                  const target = registeredTargets.find((item) => item.id === event.target.value)
                  if (target) setTargets((previous) => [...previous, { ...target }])
                }}
              >
                <option value="">Choose target</option>
                {registeredTargets
                  .filter((item) => !targets.some((target) => target.id === item.id))
                  .map((target) => (
                    <option key={target.id} value={target.id}>
                      {target.id} · {target.kind === 'mom' ? 'MoM' : 'Single model'}
                    </option>
                  ))}
              </select>
            </label>
          )}
        </div>
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
            <section aria-label={`${target.id} request profile`}>
              {!advanced && (
                <>
                  <h4>Effective request profile</h4>
                  <p>
                    Output tokens:{' '}
                    {number(effectiveRequestProfile(target, formManifest.sampling).max_tokens)}
                    {target.request_params?.max_tokens === undefined
                      ? ' (run default)'
                      : ' (registered override)'}
                  </p>
                </>
              )}
              {advanced && (
                <p className={styles.muted}>
                  Request parameters come from the edited manifest and reviewed plan.
                </p>
              )}
              {target.request_params && Object.keys(target.request_params).length > 0 && (
                <details>
                  <summary>Registered sampling overrides</summary>
                  <pre>{JSON.stringify(target.request_params, null, 2)}</pre>
                </details>
              )}
              <p className={styles.muted}>
                Fixed server-registered values override run sampling defaults. Select another
                registered target to use a different fixed profile. The run output cap still
                applies; this form does not change the registered profile or raise the cap
                automatically.
              </p>
            </section>
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
                onClick={() => setTargets((previous) => previous.filter((_, i) => i !== index))}
              >
                Remove target {index + 1}
              </button>
            </div>
          </fieldset>
        ))}
      </div>
      <label>
        Cost accounting
        <select
          value={costPolicy}
          onChange={(event) =>
            setCostPolicy(event.target.value as 'require_priced' | 'capability_only')
          }
        >
          <option value="require_priced">Require complete prices and cost bounds</option>
          <option value="capability_only">Capability only — no cost-saving claim</option>
        </select>
      </label>
      {costPolicy === 'capability_only' && (
        <p className={styles.notice}>
          Unknown or unpriced usage cannot be bounded by a USD budget. Time, output and call limits
          still apply; this run cannot prove cost savings.
        </p>
      )}
      <h3>3. Set budget and limits</h3>
      <div className={styles.formGrid}>
        <label>
          Budget (USD)
          <input
            type="number"
            min="0.01"
            step="0.01"
            value={limits.max_cost_usd}
            onChange={(event) => setLimits({ ...limits, max_cost_usd: Number(event.target.value) })}
          />
        </label>
        <label>
          Run deadline (seconds)
          <input
            type="number"
            min="1"
            value={limits.max_run_seconds}
            onChange={(event) =>
              setLimits({ ...limits, max_run_seconds: Number(event.target.value) })
            }
          />
        </label>
        <label>
          Request deadline (seconds)
          <input
            type="number"
            min="1"
            value={limits.total_timeout_s}
            onChange={(event) =>
              setLimits({ ...limits, total_timeout_s: Number(event.target.value) })
            }
          />
        </label>
        <label>
          Idle timeout (seconds)
          <input
            type="number"
            min="1"
            value={limits.idle_timeout_s}
            onChange={(event) =>
              setLimits({ ...limits, idle_timeout_s: Number(event.target.value) })
            }
          />
        </label>
        <label>
          Max output tokens
          <input
            type="number"
            min="1"
            value={limits.max_output_tokens}
            onChange={(event) =>
              setLimits({ ...limits, max_output_tokens: Number(event.target.value) })
            }
          />
        </label>
        <label>
          Concurrency
          <input
            type="number"
            min="1"
            max="32"
            value={limits.concurrency}
            onChange={(event) => setLimits({ ...limits, concurrency: Number(event.target.value) })}
          />
        </label>
      </div>
      <details className={styles.details}>
        <summary>Advanced manifest</summary>
        <p>
          Set sampling and run limits. Target endpoints, prices, credentials and harness settings
          are managed by the server; target edits are rejected.
        </p>
        <label className={styles.checkbox}>
          <input
            type="checkbox"
            checked={advanced}
            onChange={(event) => {
              setAdvanced(event.target.checked)
              if (event.target.checked) setJson(JSON.stringify(formManifest, null, 2))
            }}
          />
          Use edited manifest
        </label>
        <textarea
          aria-label="Manifest JSON"
          rows={14}
          value={advanced ? json : JSON.stringify(formManifest, null, 2)}
          readOnly={!advanced}
          onChange={(event) => setJson(event.target.value)}
        />
      </details>
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
