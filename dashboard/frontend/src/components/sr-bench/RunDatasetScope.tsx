import { useEffect, useMemo, useRef, useState } from 'react'
import ProductLoadingState from '../ProductLoadingState'
import { benchApi } from './api'
import BenchSelect from './BenchSelect'
import {
  benchmarkTitle,
  friendlyDatasetName,
  profileDescription,
  profileTitle,
} from './datasetPresentation'
import type { Catalog, Dataset, DatasetSelection } from './types'
import styles from './SrBench.module.css'
import controls from './BenchControls.module.css'

export interface ResolvedDatasetScope {
  profile: string
  sourceIDs: string[]
  missingBenchmarks: string[]
  seed: number | null
  ready: boolean
  error: string
}

export default function RunDatasetScope({
  catalog,
  datasets,
  profile,
  onProfile,
  benchmarks,
  onBenchmarks,
  initialDataset,
  onResolved,
}: {
  catalog: Catalog
  datasets: Dataset[]
  profile: string
  onProfile: (value: string) => void
  benchmarks: string[]
  onBenchmarks: (value: string[]) => void
  initialDataset?: string
  onResolved: (value: ResolvedDatasetScope) => void
}) {
  const [custom, setCustom] = useState(!!initialDataset)
  const [datasetID, setDatasetID] = useState(initialDataset ?? '')
  const [selection, setSelection] = useState<DatasetSelection | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const initialized = useRef(!!initialDataset)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    if (custom) {
      setLoading(false)
      setError('')
      return
    }
    const controller = new AbortController()
    setLoading(true)
    setError('')
    setSelection(null)
    void benchApi
      .datasetSelection(profile, controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return
        if (value.profile !== profile)
          throw new Error('The prepared scope did not match the selected profile.')
        setSelection(value)
        if (!initialized.current) {
          initialized.current = true
          onBenchmarks(value.benchmarks.filter((entry) => entry.eligible).map((entry) => entry.id))
        }
      })
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(
            cause instanceof Error ? cause.message : 'Prepared benchmarks could not be checked.',
          )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [custom, profile, revision, onBenchmarks])
  const dataset = datasets.find((item) => item.id === datasetID)
  const entries = catalog.benchmarks.map((benchmark) => {
    if (custom) {
      const eligible = dataset?.profile === profile && !!dataset.benchmarks?.includes(benchmark.id)
      return {
        id: benchmark.id,
        eligible,
        reason: eligible ? null : 'Not included in this dataset',
        reason_code: eligible ? null : 'not_in_dataset',
        case_count: 0,
        source_ids: eligible ? [dataset!.id] : [],
      }
    }
    return selection?.profile === profile
      ? selection.benchmarks.find((entry) => entry.id === benchmark.id)
      : undefined
  })
  const available = entries
    .filter((entry) => entry?.eligible || (!custom && entry?.reason_code === 'not_prepared'))
    .map((entry) => entry!.id)
  const resolved = useMemo((): ResolvedDatasetScope => {
    if (custom) {
      const ready =
        !!dataset &&
        dataset.profile === profile &&
        benchmarks.length > 0 &&
        benchmarks.every((id) => dataset.benchmarks?.includes(id))
      return {
        profile,
        sourceIDs: ready ? [dataset.id] : [],
        missingBenchmarks: [],
        seed: dataset?.seed ?? null,
        ready,
        error: ready
          ? ''
          : 'Choose a dataset and benchmarks included in its profile. Existing selections are preserved until you change them.',
      }
    }
    if (loading || selection?.profile !== profile || error)
      return {
        profile,
        sourceIDs: [],
        missingBenchmarks: [],
        seed: null,
        ready: false,
        error: error || 'Wait for the service to check this profile.',
      }
    const chosen = benchmarks.map((id) => selection.benchmarks.find((entry) => entry.id === id))
    const ready =
      chosen.length > 0 &&
      chosen.every(
        (entry) =>
          (entry?.eligible && entry.source_ids.length > 0 && selection.seed !== null) ||
          entry?.reason_code === 'not_prepared',
      )
    return {
      profile,
      sourceIDs: ready ? [...new Set(chosen.flatMap((entry) => entry!.source_ids))] : [],
      missingBenchmarks: ready
        ? chosen.filter((entry) => entry?.reason_code === 'not_prepared').map((entry) => entry!.id)
        : [],
      seed: selection.seed ?? 20260918,
      ready,
      error: ready
        ? ''
        : benchmarks.length
          ? 'Some selected benchmarks are unavailable for this profile. Deselect them or choose a specific dataset.'
          : 'Select at least one benchmark.',
    }
  }, [custom, dataset, profile, benchmarks, loading, selection, error])
  useEffect(() => onResolved(resolved), [resolved, onResolved])
  const allSelected = available.length > 0 && available.every((id) => benchmarks.includes(id))
  return (
    <>
      <div className={styles.profileCards} role="radiogroup" aria-label="Evaluation size">
        {['smoke', 'quick', 'standard']
          .filter((id) => catalog.profiles.some((item) => item.id === id))
          .map((id) => (
            <label
              key={id}
              className={`${styles.profileCard} ${profile === id ? styles.profileSelected : ''}`}
            >
              <input
                type="radio"
                name="evaluation-profile"
                checked={profile === id}
                onChange={() => onProfile(id)}
              />
              <strong>{profileTitle(id)}</strong>
              <span>{profileDescription(id)}</span>
            </label>
          ))}
      </div>
      <div className={controls.selectionHeading}>
        <h4>Benchmarks</h4>
        <button
          className={controls.compactButton}
          disabled={!available.length || loading}
          onClick={() =>
            onBenchmarks(
              allSelected
                ? benchmarks.filter((id) => !available.includes(id))
                : [...new Set([...benchmarks, ...available])],
            )
          }
        >
          {allSelected ? 'Clear benchmarks' : 'Select all benchmarks'}
        </button>
      </div>
      {!custom && loading && <ProductLoadingState compact label="Checking prepared benchmarks…" />}
      {!custom && error && (
        <div className={styles.error} role="alert">
          <p>{error}</p>
          <button onClick={() => setRevision((value) => value + 1)}>Retry benchmark check</button>
        </div>
      )}
      <div className={styles.benchmarkChoices} role="group" aria-label="Included benchmarks">
        {catalog.benchmarks.map((benchmark, index) => {
          const entry = entries[index]
          const selected = benchmarks.includes(benchmark.id)
          return (
            <label key={benchmark.id} className={styles.benchmarkChoice}>
              <input
                type="checkbox"
                checked={selected}
                disabled={!selected && (!available.includes(benchmark.id) || loading)}
                onChange={(event) =>
                  onBenchmarks(
                    event.target.checked
                      ? [...benchmarks, benchmark.id]
                      : benchmarks.filter((id) => id !== benchmark.id),
                  )
                }
              />
              <span>
                {benchmarkTitle(benchmark.id)}
                <small>
                  {entry?.eligible
                    ? custom
                      ? 'Included in selected dataset'
                      : `${entry.case_count} questions`
                    : entry?.reason_code === 'not_prepared'
                      ? 'Prepared automatically'
                      : (entry?.reason ?? 'Not available')}
                </small>
              </span>
            </label>
          )
        })}
      </div>
      {!custom && (
        <p className={styles.muted}>
          Missing datasets and required dependencies are prepared automatically when you review.
        </p>
      )}
      {benchmarks.length > 0 && !resolved.ready && !loading && !error && (
        <p className={styles.notice}>{resolved.error}</p>
      )}
      {custom ? (
        <section className={styles.customDataset} aria-label="Specific dataset">
          <BenchSelect
            label="Specific dataset"
            value={datasetID}
            searchable
            placeholder="Choose a frozen dataset"
            options={datasets
              .filter((item) => item.profile === profile)
              .map((item) => ({
                value: item.id,
                label: friendlyDatasetName(item),
                description: `${item.case_count} questions · ${item.split ?? 'split unspecified'}`,
              }))}
            onChange={setDatasetID}
          />
          <p className={styles.muted}>
            This explicit dataset selection replaces automatic source resolution. Benchmark
            selections are preserved; no questions are resampled.
          </p>
          <button className={styles.linkButton} onClick={() => setCustom(false)}>
            Choose benchmarks automatically
          </button>
        </section>
      ) : (
        <button className={styles.linkButton} onClick={() => setCustom(true)}>
          Use a specific dataset
        </button>
      )}
    </>
  )
}
