import { useState } from 'react'
import { number } from './model'
import type { Dataset, Run } from './types'
import styles from './SrBench.module.css'

export default function DatasetInventory({
  datasets,
  runs,
  canRun,
  onUse,
}: {
  datasets: Dataset[]
  runs: Run[]
  canRun: boolean
  onUse: (dataset: Dataset) => void
}) {
  const [query, setQuery] = useState('')
  const [profile, setProfile] = useState('all')
  const visible = datasets.filter(
    (dataset) =>
      `${dataset.name ?? ''} ${dataset.id} ${dataset.benchmarks?.join(' ') ?? ''}`
        .toLowerCase()
        .includes(query.toLowerCase()) &&
      (profile === 'all' || dataset.profile === profile),
  )
  return (
    <section className={styles.panel}>
      <div className={styles.sectionHeading}>
        <div>
          <h2>Prepared datasets</h2>
          <p>Frozen case sets shared by the CLI, Dashboard, single models and MoM.</p>
        </div>
        <span className={styles.badge}>{number(datasets.length)} datasets</span>
      </div>
      <div className={styles.formGrid}>
        <label>
          Search datasets
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Name, benchmark or dataset ID"
          />
        </label>
        <label>
          Dataset profile
          <select value={profile} onChange={(event) => setProfile(event.target.value)}>
            <option value="all">All profiles</option>
            {[...new Set(datasets.flatMap((dataset) => (dataset.profile ? [dataset.profile] : [])))]
              .sort()
              .map((item) => (
                <option key={item}>{item}</option>
              ))}
          </select>
        </label>
      </div>
      <p className={styles.muted}>
        Datasets are prepared and versioned with the CLI. A different case hash is a different
        evaluation scope; compare runs on the same frozen cases.
      </p>
      {!visible.length && (
        <p className={styles.emptyState}>
          {datasets.length
            ? 'No datasets match these filters.'
            : 'No prepared datasets are registered. Prepare one with the CLI connected to this service, then refresh.'}
        </p>
      )}
      <div className={styles.datasetGrid}>
        {visible.map((dataset) => {
          const usedBy = runs.filter((run) => run.manifest.dataset?.sha256 === dataset.sha256)
          return (
            <article className={styles.datasetCard} key={dataset.id}>
              <div className={styles.sectionHeading}>
                <span className={styles.badge}>{dataset.profile ?? 'Custom profile'}</span>
                <strong>{number(dataset.case_count)} cases</strong>
              </div>
              <h3>{dataset.name ?? dataset.benchmarks?.join(' + ') ?? dataset.id}</h3>
              <p>
                {dataset.benchmarks?.join(' · ') ||
                  'Benchmark scope is recorded in the dataset manifest.'}
              </p>
              <p className={styles.muted}>
                {number(usedBy.length)} evaluation runs use this case hash
              </p>
              <details>
                <summary>Dataset identity and provenance</summary>
                <dl className={styles.identity}>
                  <dt>Dataset ID</dt>
                  <dd>
                    <code>{dataset.id}</code>
                  </dd>
                  <dt>Cases SHA-256</dt>
                  <dd>
                    <code>{dataset.sha256}</code>
                  </dd>
                  <dt>Case location</dt>
                  <dd>
                    <code>{dataset.path}</code>
                  </dd>
                </dl>
                <pre>{JSON.stringify(dataset, null, 2)}</pre>
              </details>
              <div className={styles.actions}>
                <button disabled={!canRun} onClick={() => onUse(dataset)}>
                  Evaluate this dataset
                </button>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}
