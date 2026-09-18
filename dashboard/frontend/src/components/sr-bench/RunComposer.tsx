import { useMemo, useState } from 'react'
import { benchApi } from './api'
import { DEFAULT_LIMITS, makeManifest, number, validateManifest } from './model'
import type { Catalog, Dataset, Manifest, Plan, Run, Target } from './types'
import styles from './SrBench.module.css'

interface Props {
  catalog: Catalog
  datasets: Dataset[]
  targets: Target[]
  canRun: boolean
  initialModel?: string
  onStarted: (run: Run) => void
}

export default function RunComposer({
  catalog,
  datasets,
  targets: registeredTargets,
  canRun,
  initialModel,
  onStarted,
}: Props) {
  const [name, setName] = useState('Balance comparison')
  const [mode, setMode] = useState<Manifest['mode']>('live')
  const [costPolicy, setCostPolicy] = useState<'require_priced' | 'capability_only'>('require_priced')
  const [profile, setProfile] = useState('quick')
  const [datasetID, setDatasetID] = useState('')
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
  const profiles = catalog.profiles.map((item) => item.id)
  const dataset = datasets.find((item) => item.id === datasetID)
  const formManifest = useMemo(
    () => ({ ...makeManifest(name, mode, profile, dataset, targets, limits), cost_policy: costPolicy }),
    [name, mode, profile, dataset, targets, limits, costPolicy],
  )
  const fingerprint = advanced ? json : JSON.stringify(formManifest)
  const planCurrent = plan?.fingerprint === fingerprint

  async function reviewPlan() {
    setError('')
    setPending(true)
    setPlan(null)
    try {
      const manifest = advanced ? (JSON.parse(json) as Manifest) : formManifest
      const issue = validateManifest(manifest)
      if (issue) throw new Error(issue)
      const evidence = await benchApi.plan(manifest)
      setPlan({
        manifest: evidence.manifest,
        evidence,
        fingerprint,
        idempotencyKey: crypto.randomUUID(),
      })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not prepare this run.')
    } finally {
      setPending(false)
    }
  }

  async function startRun() {
    if (!plan || !planCurrent || !canRun) return
    setError('')
    setPending(true)
    try {
      onStarted(await benchApi.start(plan.manifest, plan.idempotencyKey))
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : 'Could not start this run. Check the run list before submitting again.',
      )
    } finally {
      setPending(false)
    }
  }

  return (
    <section className={styles.panel} aria-labelledby="new-run-title">
      <div className={styles.sectionHeading}>
        <div>
          <h2 id="new-run-title">Create evaluation</h2>
          <p>Use the same frozen questions for single models and your MoM.</p>
        </div>
        <span className={styles.badge}>sr-bench 1.0</span>
      </div>
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
        <label>
          Profile
          <select value={profile} onChange={(event) => { setProfile(event.target.value); setDatasetID('') }}>
            {(profiles.length ? profiles : ['smoke', 'quick', 'standard']).map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          Prepared dataset
          <select value={datasetID} onChange={(event) => setDatasetID(event.target.value)}>
            <option value="">Select a dataset</option>
            {datasets.filter(item => !item.profile || item.profile === profile).map((item) => (
              <option key={item.id} value={item.id}>
                {item.name ?? item.id} · {number(item.case_count)} cases
              </option>
            ))}
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
          {number(dataset.case_count)} frozen cases ·{' '}
          {dataset.benchmarks?.join(', ') || 'Benchmark scope recorded in the dataset'}
          <br />
          SHA-256 <code className={styles.hash}>{dataset.sha256}</code>
        </p>
      )}
      <div className={styles.sectionHeading}>
        <h3>Targets</h3>
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
          No targets are registered. Run vllm-sr benchmark target register --file targets.json against this service’s store, then refresh.
        </p>
      )}
      <div className={styles.targetList}>
        {targets.map((target, index) => (
          <fieldset key={index} className={styles.target}>
            <legend>Target {index + 1}</legend>
            <div className={styles.formGrid}>
              <label>
                Name
                <input value={target.id} readOnly />
              </label>
              <label>
                Type
                <input
                  value={target.kind === 'mom' ? 'Mixture of models' : 'Single model'}
                  readOnly
                />
              </label>
              <label>
                Model / entrypoint
                <input value={target.model} readOnly />
              </label>
              <label>
                Endpoint
                <input value={target.base_url} readOnly />
              </label>
              {target.config_hash && (
                <label>
                  Frozen configuration
                  <input value={target.config_hash} readOnly />
                </label>
              )}
            </div>
            <div className={styles.targetFooter}>
              <span>
                Credentials stay on the server.
                {target.prices
                  ? ' Configured prices included.'
                  : ' Unpriced usage will remain unknown.'}
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
      <label>Cost accounting<select value={costPolicy} onChange={event => setCostPolicy(event.target.value as 'require_priced' | 'capability_only')}><option value="require_priced">Require complete prices and cost bounds</option><option value="capability_only">Capability only — no cost-saving claim</option></select></label>
      {costPolicy === 'capability_only' && <p className={styles.notice}>Unknown or unpriced usage cannot be bounded by a USD budget. Time, output and call limits still apply; this run cannot prove cost savings.</p>}
      <h3>Run budget and limits</h3>
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
          Set sampling and run limits. Target endpoints, prices, credentials and harness settings are managed by the server; target edits are rejected.
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
