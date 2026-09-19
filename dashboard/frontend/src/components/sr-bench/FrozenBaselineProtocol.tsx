import { number } from './model'
import { profileTitle } from './datasetPresentation'
import type { Run } from './types'
import styles from './SrBench.module.css'

export default function FrozenBaselineProtocol({ run }: { run: Run }) {
  const { manifest } = run
  return (
    <section className={styles.notice} aria-label="Frozen baseline protocol">
      <h3>Reuse {manifest.name}</h3>
      <p>
        The service preserves the exact questions, grading protocol, sampling defaults and limits
        from this baseline. Only the run name and configured MoM targets change.
      </p>
      <dl className={styles.identity}>
        <dt>Profile</dt>
        <dd>{profileTitle(manifest.profile)}</dd>
        <dt>Frozen questions</dt>
        <dd>{number(manifest.cases?.length)}</dd>
        <dt>Sampling defaults</dt>
        <dd>
          Temperature {number(manifest.sampling.temperature, 2)} · Top P{' '}
          {manifest.sampling.top_p == null ? 'Not set' : number(manifest.sampling.top_p, 2)} · seed{' '}
          {manifest.sampling.seed == null ? 'Not set' : number(manifest.sampling.seed)}
        </dd>
        <dt>Cost accounting</dt>
        <dd>
          {manifest.cost_policy === 'capability_only'
            ? 'Quality only · no USD budget'
            : `Quality and cost · $${number(manifest.limits.max_cost_usd, 2)} budget`}
        </dd>
        <dt>Execution limits</dt>
        <dd>
          {number(manifest.limits.max_output_tokens)} output tokens ·{' '}
          {number(manifest.limits.concurrency)} concurrent cases ·{' '}
          {number(manifest.limits.max_run_seconds)} seconds
        </dd>
      </dl>
    </section>
  )
}
