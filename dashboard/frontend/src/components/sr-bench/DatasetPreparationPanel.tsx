import { useState } from 'react'
import BenchSelect from './BenchSelect'
import ProductIcon from '../ProductIcon'
import ProductLoadingState from '../ProductLoadingState'
import { number } from './model'
import { benchmarkTitle, profileTitle } from './datasetPresentation'
import {
  preparationIsActive,
  preparationPhase,
  type PreparationProfile,
} from './datasetPreparationApi'
import { useDatasetPreparations } from './useDatasetPreparations'
import type { Dataset } from './types'
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
  const duplicate = active.some((job) => job.benchmark === benchmark?.id && job.profile === profile)
  const profiles = ['smoke', 'quick', 'standard'] as const
  const selectionDisabled = state.submitting || state.loading
  const mutationDisabled = !!disabledReason || selectionDisabled
  return (
    <section className={shared.panel} aria-label="Prepare datasets">
      <div className={shared.sectionHeading}>
        <div>
          <h2>Prepare dataset</h2>
          <p>Download a benchmark and freeze a reusable set of questions.</p>
        </div>
        <button type="button" onClick={onClose} aria-label="Close dataset preparation">
          <ProductIcon name="close" />
        </button>
      </div>
      <p className={shared.muted}>
        Required dependencies are installed automatically on the service. Preparation continues when
        you leave this page and does not generate model answers.
      </p>
      {state.readError && (
        <div className={shared.error} role="alert">
          <p>{state.readError}</p>
          <button type="button" onClick={state.refresh}>
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
          <p className={shared.muted}>
            {number(state.benchmarks.length)} benchmarks · {profiles.length} dataset sizes:{' '}
            {profiles.map(profileTitle).join(', ')}. Choose a benchmark to compare question counts.
          </p>
          <div className={shared.formGrid}>
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
            <span>
              {profileTitle(profile)} · {benchmark.name}
            </span>
            <a href={benchmark.source_url} target="_blank" rel="noreferrer">
              View source dataset
            </a>
          </div>
          <p className={shared.muted}>
            {profiles
              .map((value) => `${profileTitle(value)}: ${number(benchmark.profiles[value])}`)
              .join(' · ')}{' '}
            questions
          </p>
          {benchmark.access_note && <p className={shared.muted}>{benchmark.access_note}</p>}
          <div className={shared.actions}>
            <button
              type="submit"
              className={shared.primary}
              disabled={mutationDisabled || active.length > 0}
            >
              <ProductIcon name="database" />
              {state.submitting
                ? 'Starting preparation…'
                : duplicate
                  ? 'Preparation in progress'
                  : active.length
                    ? 'Another preparation is running'
                    : 'Download and prepare'}
            </button>
            {benchmark.dependencies.length > 0 && (
              <span className={shared.muted}>
                Dependencies: {benchmark.dependencies.join(', ')} · installed if needed
              </span>
            )}
          </div>
        </form>
      )}
      {!state.loading && !state.readError && !benchmark && (
        <p className={shared.notice}>No downloadable benchmarks are available from this service.</p>
      )}
      {disabledReason && (
        <div className={shared.notice}>
          <p>{disabledReason}</p>
          <button type="button" onClick={onRefreshAccess} disabled={accessRefreshing}>
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
          <div className={shared.sectionHeading}>
            <h3>Preparation history</h3>
            <button type="button" onClick={state.refresh}>
              <ProductIcon name="refresh" /> Refresh status
            </button>
          </div>
          <ul className={styles.jobs} aria-label="Dataset preparations">
            {state.preparations.map((job) => (
              <li key={job.id} className={styles.job}>
                <div className={shared.sectionHeading}>
                  <div>
                    <strong>
                      {state.benchmarks.find((item) => item.id === job.benchmark)?.name ??
                        benchmarkTitle(job.benchmark)}
                    </strong>
                    <span className={styles.profile}>{profileTitle(job.profile)}</span>
                  </div>
                  <span className={shared.badge} role="status">
                    {preparationPhase(job)}
                  </span>
                </div>
                <p className={shared.muted}>Updated {new Date(job.updated_at).toLocaleString()}</p>
                {job.error && <p className={shared.error}>{job.error}</p>}
                {job.status === 'completed' && job.dataset && (
                  <div className={shared.actions}>
                    <span>{number(job.dataset.case_count)} questions ready</span>
                    <button type="button" onClick={() => onOpenDataset(job.dataset!)}>
                      View prepared dataset
                    </button>
                  </div>
                )}
                {job.status === 'failed' && (
                  <button
                    type="button"
                    disabled={mutationDisabled || active.length > 0}
                    onClick={() => {
                      if (!mutationDisabled && !active.length)
                        void state.prepare({
                          benchmark: job.benchmark,
                          profile: job.profile,
                          seed: job.seed ?? undefined,
                          limit: job.limit ?? undefined,
                        })
                    }}
                  >
                    Retry preparation
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
