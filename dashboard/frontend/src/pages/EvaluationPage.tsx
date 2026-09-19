import { useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canRunEvaluation, canWriteEvaluation } from '../utils/accessControl'
import { benchApi } from '../components/sr-bench/api'
import { active } from '../components/sr-bench/model'
import RunComposer from '../components/sr-bench/RunComposer'
import RunDetails from '../components/sr-bench/RunDetails'
import RunComparison from '../components/sr-bench/RunComparison'
import ReplayComposer from '../components/sr-bench/ReplayComposer'
import RunList from '../components/sr-bench/RunList'
import DatasetInventory from '../components/sr-bench/DatasetInventory'
import type { Catalog, Dataset, Run, Target } from '../components/sr-bench/types'
import styles from '../components/sr-bench/SrBench.module.css'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'

type Inventory = 'catalog' | 'datasets' | 'targets' | 'runs'

export default function EvaluationPage() {
  const { user } = useAuth()
  const { serverReadonly, isLoading: settingsLoading } = useReadonly()
  const canRun =
    !settingsLoading && !serverReadonly && canRunEvaluation(user) && canWriteEvaluation(user)
  const [search, setSearch] = useSearchParams()
  const selectedID = search.get('run')
  const view = search.get('view') ?? (search.has('model') ? 'new' : 'runs')
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [readErrors, setReadErrors] = useState<Partial<Record<Inventory, string>>>({})
  const [loaded, setLoaded] = useState<Partial<Record<Inventory, boolean>>>({})
  const [loading, setLoading] = useState(true)
  const [revision, setRevision] = useState(0)
  const [lastRead, setLastRead] = useState<string | null>(null)
  const refresh = useCallback(() => setRevision((value) => value + 1), [])
  const error = Object.entries(readErrors)
    .filter(([, message]) => message)
    .map(([name, message]) => `${name}: ${message}`)
    .join(' ')

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    setLoading(true)
    async function read<T>(name: Inventory, response: Promise<T>, accept: (value: T) => void) {
      try {
        const value = await response
        if (!controller.signal.aborted) {
          accept(value)
          setLoaded((previous) => ({ ...previous, [name]: true }))
          setReadErrors((previous) => ({ ...previous, [name]: '' }))
        }
        return value
      } catch (cause) {
        if (!controller.signal.aborted)
          setReadErrors((previous) => ({
            ...previous,
            [name]:
              cause instanceof Error ? cause.message : 'sr-bench service could not be reached.',
          }))
        throw cause
      }
    }
    async function load() {
      try {
        const responses = await Promise.allSettled([
          read('catalog', benchApi.catalog(controller.signal), setCatalog),
          read('datasets', benchApi.datasets(controller.signal), (value) =>
            setDatasets(value.datasets),
          ),
          read('targets', benchApi.targets(controller.signal), (value) =>
            setTargets(value.targets),
          ),
          read('runs', benchApi.runs(controller.signal), (value) => {
            setRuns(value.runs)
            setLastRead(new Date().toLocaleTimeString())
          }),
        ])
        if (controller.signal.aborted) return
        const failure = responses.find((response) => response.status === 'rejected')
        // Retry reads after a disconnect and discover runs started from the CLI.
        // This timer never submits, resumes or retries an evaluation.
        const running =
          responses[3].status === 'fulfilled' &&
          responses[3].value.runs.some((run) => active(run.status))
        timer = setTimeout(() => void load(), failure || running ? 5000 : 15000)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    void load()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [revision])

  return (
    <section className={styles.page} aria-label="sr-bench workspace">
      <header className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>
            <ProductIcon name="evaluation" /> Evaluation
          </p>
          <h1>
            sr-bench <span>1.0</span>
          </h1>
          <p>
            Measure capability, cost and speed. Improve Balance against your strongest single model.
          </p>
        </div>
        <div className={styles.actions}>
          <button onClick={refresh} disabled={loading}>
            <ProductIcon name="refresh" />
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
          {view !== 'new' && (
            <button className={styles.primary} onClick={() => setSearch({ view: 'new' })}>
              <ProductIcon name="plus" />
              Create evaluation
            </button>
          )}
        </div>
      </header>
      <nav className={styles.tabs} aria-label="Evaluation views">
        {(
          [
            ['runs', 'Runs', 'list'],
            ['compare', 'Compare iterations', 'chart'],
            ['datasets', 'Datasets', 'database'],
            ['catalog', 'Benchmarks', 'evaluation'],
          ] as const
        ).map(([key, label, icon]) => (
          <button
            key={key}
            aria-current={view === key ? 'page' : undefined}
            onClick={() => setSearch({ view: key })}
          >
            <ProductIcon name={icon} />
            {label}
            {key === 'runs' && loaded.runs ? ` (${runs.length})` : ''}
          </button>
        ))}
      </nav>
      {lastRead && (
        <p className={styles.muted} role="status">
          Last synchronized {lastRead} · Read-only updates continue automatically.
        </p>
      )}
      {error && (
        <div className={styles.error} role="alert">
          <strong>sr-bench needs attention.</strong> {error}
          <p>
            Check that the sr-bench service is running and configured for this Dashboard, then
            refresh. Existing runs are owned by the service and continue independently of this page.
          </p>
        </div>
      )}
      {!catalog && loading && <ProductLoadingState compact label="Loading sr-bench…" />}
      {(view === 'new' || (view === 'runs' && selectedID)) && (
        <button className={styles.backLink} onClick={() => setSearch({ view: 'runs' })}>
          <ProductIcon name="arrow-left" />
          Back to runs
        </button>
      )}
      {view === 'new' && catalog && loaded.datasets && loaded.targets && (
        <RunComposer
          catalog={catalog}
          datasets={datasets}
          targets={targets}
          canRun={canRun}
          actorID={user?.id ?? ''}
          key={`${user?.id ?? ''}:${search.get('dataset') ?? 'new'}`}
          initialModel={search.get('model') ?? undefined}
          initialDataset={search.get('dataset') ?? undefined}
          onStarted={(run) => {
            setRuns((previous) => [run, ...previous.filter((item) => item.id !== run.id)])
            setSearch({ view: 'runs', run: run.id })
            refresh()
          }}
        />
      )}
      {view === 'runs' && (
        <>
          {!selectedID &&
            (loaded.runs ? (
              <RunList
                runs={runs}
                selectedID={selectedID}
                onSelect={(id) => setSearch({ view: 'runs', run: id })}
              />
            ) : readErrors.runs ? (
              <p role="status">Run inventory is unavailable.</p>
            ) : (
              <ProductLoadingState compact label="Loading evaluation runs…" />
            ))}
          {selectedID && (
            <RunDetails
              key={`${user?.id ?? ''}:${selectedID}`}
              id={selectedID}
              section={
                ['results', 'questions', 'calls', 'evidence', 'recipe'].includes(
                  search.get('section') ?? '',
                )
                  ? search.get('section')!
                  : 'results'
              }
              onSectionChange={(section) => {
                const next = new URLSearchParams(search)
                next.set('section', section)
                setSearch(next)
              }}
              actorID={user?.id ?? ''}
              canRun={canRun}
              onChanged={refresh}
              onRecovered={(run) => {
                setSearch({ view: 'runs', run: run.id })
                refresh()
              }}
            />
          )}
        </>
      )}
      {view === 'datasets' && loaded.datasets && (
        <DatasetInventory
          datasets={datasets}
          runs={runs}
          runsLoaded={!!loaded.runs}
          canRun={canRun}
          selectedDatasetId={search.get('dataset') ?? ''}
          onSelectDataset={(id) => setSearch({ view: 'datasets', dataset: id })}
          onBackToDatasets={() => setSearch({ view: 'datasets' })}
          onUse={(dataset) => setSearch({ view: 'new', dataset: dataset.id })}
        />
      )}
      {view === 'datasets' &&
        !loaded.datasets &&
        (readErrors.datasets ? (
          <p role="status">Dataset inventory is unavailable.</p>
        ) : (
          <ProductLoadingState compact label="Loading prepared datasets…" />
        ))}
      {view === 'new' && (!catalog || !loaded.datasets || !loaded.targets) && (
        <ProductLoadingState
          compact
          label="Waiting for the benchmark catalog, prepared datasets and configured targets."
        />
      )}
      {view === 'compare' &&
        !loaded.runs &&
        (readErrors.runs ? (
          <p role="status">Run inventory is unavailable.</p>
        ) : (
          <ProductLoadingState compact label="Loading evaluation runs…" />
        ))}
      {view === 'compare' && loaded.runs && (
        <>
          <RunComparison runs={runs} />
          <details className={styles.panel}>
            <summary>Reuse saved answers for diagnostic replay</summary>
            <ReplayComposer
              runs={runs}
              canRun={canRun}
              onCreated={(run) => {
                setSearch({ view: 'runs', run: run.id })
                refresh()
              }}
            />
          </details>
        </>
      )}
      {view === 'catalog' && catalog && (
        <section className={styles.panel}>
          <h2>Benchmark catalog</h2>
          <p>
            Adapters share the same targets, run ledger and reporting. A selected subset is reported
            with its own scope; it is not a full sr-bench score.
          </p>
          <div className={styles.catalog}>
            {catalog.benchmarks.map((benchmark) => (
              <article key={benchmark.id}>
                <span className={styles.badge}>{benchmark.kind}</span>
                <h3>{benchmark.title}</h3>
                <p>{benchmark.description ?? benchmark.id}</p>
                {benchmark.source_url && (
                  <a href={benchmark.source_url} target="_blank" rel="noreferrer">
                    Benchmark source ↗
                  </a>
                )}
              </article>
            ))}
          </div>
        </section>
      )}
    </section>
  )
}
