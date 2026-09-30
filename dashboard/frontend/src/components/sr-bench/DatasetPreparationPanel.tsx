import { useState } from 'react'
import BenchSelect from './BenchSelect'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import { number } from './model'
import { benchmarkTitle, profileTitle } from './datasetPresentation'
import {
  preparationBenchmarkIDs,
  preparationIsActive,
  preparationPhase,
  type DatasetPreparationJob,
  type PreparationProfile,
} from './datasetPreparationApi'
import { useDatasetPreparations } from './useDatasetPreparations'
import type { Dataset } from './types'
import controls from './BenchControls.module.css'
import shared from './SrBench.module.css'
import styles from './DatasetPreparationPanel.module.css'

export default function DatasetPreparationPanel({
  disabledReason,
  accessRefreshing,
  onRefreshAccess,
  onCompleted,
  onOpenDataset,
  onClose,
}: {
  disabledReason: string | null
  accessRefreshing: boolean
  onRefreshAccess: () => void
  onCompleted: () => void
  onOpenDataset: (dataset: Dataset) => void
  onClose: () => void
}) {
  const state = useDatasetPreparations(onCompleted)
  const [benchmarkID, setBenchmarkID] = useState('')
  const [profile, setProfile] = useState<PreparationProfile>('smoke')
  const benchmark = state.benchmarks.find((item) => item.id === benchmarkID) ?? state.benchmarks[0]
  const active = state.preparations.filter(preparationIsActive)
  const pending = state.preparations.filter((job) => job.status !== 'completed')
  const completed = state.preparations.filter((job) => job.status === 'completed')
  const duplicate = active.some(
    (job) => preparationBenchmarkIDs(job).includes(benchmark?.id ?? '') && job.profile === profile,
  )
  const profiles = ['smoke', 'quick', 'standard'] as const
  const selectionDisabled = state.submitting || state.loading
  const mutationDisabled = !!disabledReason || selectionDisabled

  const jobs = (items: DatasetPreparationJob[], label: string) => (
    <ul className={styles.jobs} aria-label={label}>
      {items.map((job) => (
        <li key={job.id} className={styles.job} data-status={job.status}>
          <div className={styles.jobRow}>
            <div className={styles.jobIdentity}>
              <strong>
                {preparationBenchmarkIDs(job)
                  .map(
                    (id) =>
                      state.benchmarks.find((item) => item.id === id)?.name ?? benchmarkTitle(id),
                  )
                  .join(', ')}
              </strong>
              <span>{profileTitle(job.profile)}</span>
              {job.status === 'completed' && job.dataset && (
                <span>{number(job.dataset.case_count)} questions ready</span>
              )}
            </div>
            <div className={styles.jobMeta}>
              <span className={styles.status} role="status">
                {preparationPhase(job)}
              </span>
              <time dateTime={job.updated_at} title={new Date(job.updated_at).toLocaleString()}>
                {new Date(job.updated_at).toLocaleDateString()}
              </time>
              {job.status === 'completed' && job.dataset && (
                <button
                  type="button"
                  className={controls.compactButton}
                  onClick={() => onOpenDataset(job.dataset!)}
                >
                  View prepared dataset
                </button>
              )}
              {job.status === 'failed' && (
                <button
                  type="button"
                  className={controls.compactButton}
                  disabled={mutationDisabled || active.length > 0}
                  onClick={() => {
                    const ids = preparationBenchmarkIDs(job)
                    if (!mutationDisabled && !active.length && ids.length)
                      void state.prepare(
                        job.benchmarks
                          ? { benchmarks: ids, profile: job.profile, seed: job.seed ?? undefined }
                          : {
                              benchmark: ids[0],
                              profile: job.profile,
                              seed: job.seed ?? undefined,
                              limit: job.limit ?? undefined,
                            },
                      )
                  }}
                >
                  Retry preparation
                </button>
              )}
            </div>
          </div>
          {job.error && <p className={styles.jobError}>{job.error}</p>}
        </li>
      ))}
    </ul>
  )

  return (
    <section className={shared.panel} aria-label="Prepare datasets">
      <div className={`${shared.sectionHeading} ${styles.heading}`}>
        <div>
          <h2>Manage datasets</h2>
          <p className={styles.intro}>
            Create evaluation prepares datasets automatically. Manage downloads here.
          </p>
        </div>
        <button
          type="button"
          className={controls.compactButton}
          onClick={onClose}
          aria-label="Close dataset preparation"
        >
          <ProductIcon name="close" />
        </button>
      </div>
      {state.readError && (
        <div className={shared.error} role="alert">
          <p>{state.readError}</p>
          <button type="button" className={controls.compactButton} onClick={state.refresh}>
            Refresh preparation status
          </button>
        </div>
      )}
      {state.loading && <ProductLoadingState compact label="Loading available datasets…" />}
      {benchmark && (
        <form
          onSubmit={(event) => {
            event.preventDefault()
            if (!mutationDisabled && !active.length)
              void state.prepare({ benchmark: benchmark.id, profile })
          }}
        >
          <div className={`${shared.formGrid} ${styles.formFields}`}>
            <BenchSelect
              label="Benchmark to download"
              value={benchmark.id}
              options={state.benchmarks.map((item) => ({ value: item.id, label: item.name }))}
              onChange={setBenchmarkID}
              disabled={selectionDisabled}
              searchable
            />
            <BenchSelect
              label="Dataset size"
              value={profile}
              options={profiles.map((value) => ({
                value,
                label: profileTitle(value),
                description: `${number(benchmark.profiles[value])} questions`,
              }))}
              onChange={(value) => setProfile(value as PreparationProfile)}
              disabled={selectionDisabled}
            />
          </div>
          <div className={styles.selection}>
            <strong>{number(benchmark.profiles[profile])} questions</strong>
            <button
              type="submit"
              className={`${shared.primary} ${controls.compactButton}`}
              disabled={mutationDisabled || active.length > 0}
            >
              {state.submitting
                ? 'Starting preparation…'
                : duplicate
                  ? 'Preparation in progress'
                  : active.length
                    ? 'Another preparation is running'
                    : 'Download and prepare'}
            </button>
          </div>
          <details className={styles.supporting}>
            <summary>Source and requirements</summary>
            <div className={styles.supportingBody}>
              <a href={benchmark.source_url} target="_blank" rel="noreferrer">
                View source dataset
              </a>
              {benchmark.access_note && <p>{benchmark.access_note}</p>}
              <p>Downloads continue in the background and do not generate model answers.</p>
              {benchmark.dependencies.length > 0 && (
                <p>Dependencies: {benchmark.dependencies.join(', ')} · installed automatically</p>
              )}
            </div>
          </details>
        </form>
      )}
      {!state.loading && !state.readError && !benchmark && (
        <p className={shared.notice}>No downloadable benchmarks are available from this service.</p>
      )}
      {disabledReason && (
        <div className={styles.accessNotice}>
          <p>{disabledReason}</p>
          <button
            type="button"
            className={controls.compactButton}
            onClick={onRefreshAccess}
            disabled={accessRefreshing}
          >
            <ProductIcon name="refresh" /> Refresh access
          </button>
        </div>
      )}
      {state.submitError && (
        <p className={shared.error} role="alert">
          {state.submitError}
        </p>
      )}
      {state.preparations.length > 0 && (
        <div className={styles.history}>
          <div className={styles.historyHeading}>
            <h3>Downloads</h3>
            <button type="button" className={controls.compactButton} onClick={state.refresh}>
              <ProductIcon name="refresh" /> Refresh status
            </button>
          </div>
          {pending.length > 0 && jobs(pending, 'Dataset preparations')}
          {completed.length > 0 && (
            <details className={styles.completed}>
              <summary>Completed ({number(completed.length)})</summary>
              {jobs(completed, 'Completed preparations')}
            </details>
          )}
        </div>
      )}
    </section>
  )
}
