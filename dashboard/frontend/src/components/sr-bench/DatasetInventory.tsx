import { useState } from 'react'
import ProductIcon from '../ProductIcon'
import { number } from './model'
import {
  benchmarkTitle,
  datasetEvaluationTitle,
  friendlyDatasetName,
  profileDescription,
  profileTitle,
} from './datasetPresentation'
import DatasetDetails from './DatasetDetails'
import DatasetExposure from './DatasetExposure'
import BenchSelect from './BenchSelect'
import type { Dataset, Run } from './types'
import shared from './SrBench.module.css'
import styles from './DatasetInventory.module.css'

const PAGE_SIZE = 12

export default function DatasetInventory({
  datasets,
  runs,
  runsLoaded = true,
  canRun,
  onUse,
  selectedDatasetId = '',
  onSelectDataset,
  onBackToDatasets,
  onPrepare,
}: {
  datasets: Dataset[]
  runs: Run[]
  runsLoaded?: boolean
  canRun: boolean
  onUse: (dataset: Dataset) => void
  selectedDatasetId?: string
  onSelectDataset?: (id: string) => void
  onBackToDatasets?: () => void
  onPrepare?: () => void
}) {
  const [query, setQuery] = useState('')
  const [profile, setProfile] = useState('all')
  const [benchmark, setBenchmark] = useState('all')
  const [page, setPage] = useState(0)
  const [localSelection, setLocalSelection] = useState('')
  const selected = selectedDatasetId || localSelection
  const open = (id: string) => (onSelectDataset ? onSelectDataset(id) : setLocalSelection(id))
  const back = () => {
    setLocalSelection('')
    onBackToDatasets?.()
  }
  const allBenchmarks = [...new Set(datasets.flatMap((dataset) => dataset.benchmarks ?? []))].sort()
  const questionTotal = (items: Dataset[]) =>
    items.reduce((total, dataset) => total + dataset.case_count, 0)
  const preparedSetCount = (count: number) =>
    `${number(count)} prepared ${count === 1 ? 'case set' : 'case sets'}`
  const profiles = [...new Set(datasets.map((dataset) => dataset.profile ?? 'custom'))].sort(
    (a, b) =>
      ['smoke', 'quick', 'standard', 'custom'].indexOf(a) -
      ['smoke', 'quick', 'standard', 'custom'].indexOf(b),
  )
  const visible = datasets.filter(
    (dataset) =>
      `${friendlyDatasetName(dataset)} ${(dataset.benchmarks ?? []).map(benchmarkTitle).join(' ')}`
        .toLowerCase()
        .includes(query.trim().toLowerCase()) &&
      (profile === 'all' || (dataset.profile ?? 'custom') === profile) &&
      (benchmark === 'all' || dataset.benchmarks?.includes(benchmark)),
  )
  const currentPage = Math.min(page, Math.max(0, Math.ceil(visible.length / PAGE_SIZE) - 1))
  const current = visible.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE)
  const filter = (set: (value: string) => void, value: string) => {
    set(value)
    setPage(0)
  }
  if (selected)
    return (
      <DatasetDetails
        key={selected}
        id={selected}
        dataset={datasets.find((dataset) => dataset.id === selected)}
        canRun={canRun}
        onUse={onUse}
        onBack={back}
      />
    )
  return (
    <section className={styles.inventory} aria-label="Dataset library">
      <div className={shared.sectionHeading}>
        <div>
          <h2>Dataset library</h2>
          <p>Choose a case set. Explore its coverage and questions before evaluating.</p>
        </div>
        <div className={shared.actions}>
          <span className={shared.badge}>
            {number(datasets.length)} {datasets.length === 1 ? 'dataset' : 'datasets'} ·{' '}
            {number(allBenchmarks.length)} {allBenchmarks.length === 1 ? 'benchmark' : 'benchmarks'}
          </span>
          {onPrepare && (
            <button type="button" onClick={onPrepare}>
              <ProductIcon name="database" /> Prepare dataset
            </button>
          )}
        </div>
      </div>
      <div className={styles.modes} role="group" aria-label="Filter by evaluation mode">
        <button
          className={styles.mode}
          aria-pressed={profile === 'all'}
          onClick={() => filter(setProfile, 'all')}
        >
          <ProductIcon name="database" width={22} height={22} />
          <strong>All datasets</strong>
          <span className={styles.modeQuestions}>
            <strong>{number(questionTotal(datasets))}</strong> total questions
          </span>
          <span>{preparedSetCount(datasets.length)}</span>
        </button>
        {profiles.map((item) => (
          <button
            className={styles.mode}
            key={item}
            aria-pressed={profile === item}
            onClick={() => filter(setProfile, item)}
          >
            <ProductIcon
              name={item === 'smoke' ? 'status' : item === 'quick' ? 'play' : 'chart'}
              width={22}
              height={22}
            />
            <strong>{profileTitle(item)}</strong>
            <span className={styles.modeQuestions}>
              <strong>
                {number(
                  questionTotal(
                    datasets.filter((dataset) => (dataset.profile ?? 'custom') === item),
                  ),
                )}
              </strong>{' '}
              total questions
            </span>
            <span title={profileDescription(item)}>
              {preparedSetCount(
                datasets.filter((dataset) => (dataset.profile ?? 'custom') === item).length,
              )}
            </span>
          </button>
        ))}
      </div>
      <p className={styles.countNote}>
        Question totals are across prepared sets. Sets may overlap.
      </p>
      <div className={styles.filters}>
        <label className={styles.search}>
          Search datasets
          <div>
            <ProductIcon name="search" width={18} height={18} />
            <input
              type="search"
              value={query}
              onChange={(event) => filter(setQuery, event.target.value)}
              placeholder="Search by name or benchmark"
            />
          </div>
        </label>
        <BenchSelect
          label="Benchmark"
          value={benchmark}
          onChange={(value) => filter(setBenchmark, value)}
          options={[
            { value: 'all', label: 'All benchmarks' },
            ...allBenchmarks.map((item) => ({ value: item, label: benchmarkTitle(item) })),
          ]}
        />
      </div>
      {!visible.length && (
        <div className={styles.empty}>
          <ProductIcon name="database" width={32} height={32} />
          <h3>{datasets.length ? 'No matching datasets' : 'Your dataset library is empty'}</h3>
          <p>
            {datasets.length
              ? 'Try another benchmark or mode.'
              : 'Choose Prepare dataset to download a benchmark and explore its questions here.'}
          </p>
        </div>
      )}
      {profiles
        .filter((item) => current.some((dataset) => (dataset.profile ?? 'custom') === item))
        .map((group) => (
          <section
            key={group}
            className={styles.group}
            aria-label={`${profileTitle(group)} datasets`}
          >
            <div className={styles.groupHeading}>
              <h3>{profileTitle(group)}</h3>
              <span>
                {visible.filter((dataset) => (dataset.profile ?? 'custom') === group).length}{' '}
                {visible.filter((dataset) => (dataset.profile ?? 'custom') === group).length === 1
                  ? 'dataset'
                  : 'datasets'}
              </span>
            </div>
            <div className={styles.cards}>
              {current
                .filter((dataset) => (dataset.profile ?? 'custom') === group)
                .map((dataset) => {
                  const usedBy = runs.filter(
                    (run) => run.manifest.dataset?.sha256 === dataset.sha256,
                  ).length
                  return (
                    <article key={dataset.id} className={styles.card}>
                      <button
                        className={styles.cardLink}
                        onClick={() => open(dataset.id)}
                        aria-label={`Explore ${friendlyDatasetName(dataset)}, ${number(dataset.case_count)} questions`}
                      >
                        <div className={styles.cardTop}>
                          <ProductIcon name="database" width={21} height={21} />
                          <span>
                            {datasetEvaluationTitle(dataset.preparation, dataset.benchmarks) ??
                              profileTitle(dataset.profile)}
                          </span>
                          <ProductIcon name="arrow-right" width={18} height={18} />
                        </div>
                        <h4>{friendlyDatasetName(dataset)}</h4>
                        <p className={styles.cardDescription}>
                          {(dataset.benchmarks ?? []).slice(0, 3).map(benchmarkTitle).join(' · ') ||
                            'Custom evaluation questions'}
                          {(dataset.benchmarks?.length ?? 0) > 3 &&
                            ` +${dataset.benchmarks!.length - 3} more`}
                        </p>
                        <DatasetExposure
                          preparation={dataset.preparation}
                          benchmarks={dataset.benchmarks}
                          compact
                        />
                        <div className={styles.cardMetrics}>
                          <strong>
                            {number(dataset.case_count)} <span>questions</span>
                          </strong>
                          <span>
                            {number(dataset.benchmarks?.length ?? 0)}{' '}
                            {dataset.benchmarks?.length === 1 ? 'benchmark' : 'benchmarks'}
                          </span>
                        </div>
                        <span className={styles.explore}>
                          Explore questions{' '}
                          <ProductIcon name="chevron-right" width={14} height={14} />
                        </span>
                      </button>
                      <div className={styles.cardFooter}>
                        <span>
                          {runsLoaded
                            ? `${number(usedBy)} ${usedBy === 1 ? 'run' : 'runs'}`
                            : 'Runs loading'}
                        </span>
                        <button disabled={!canRun} onClick={() => onUse(dataset)}>
                          <ProductIcon name="play" width={14} height={14} />
                          Evaluate
                        </button>
                      </div>
                    </article>
                  )
                })}
            </div>
          </section>
        ))}
      {visible.length > 0 && (
        <nav className={styles.pagination} aria-label="Dataset pages">
          <span>
            {currentPage * PAGE_SIZE + 1}–{Math.min((currentPage + 1) * PAGE_SIZE, visible.length)}{' '}
            of {number(visible.length)} {visible.length === 1 ? 'dataset' : 'datasets'}
          </span>
          <div>
            <button
              disabled={currentPage === 0}
              onClick={() => setPage(currentPage - 1)}
              aria-label="Previous dataset page"
            >
              <ProductIcon name="chevron-left" width={16} height={16} />
              Previous
            </button>
            <button
              disabled={(currentPage + 1) * PAGE_SIZE >= visible.length}
              onClick={() => setPage(currentPage + 1)}
              aria-label="Next dataset page"
            >
              Next
              <ProductIcon name="chevron-right" width={16} height={16} />
            </button>
          </div>
        </nav>
      )}
    </section>
  )
}
