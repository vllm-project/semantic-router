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
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [revision, setRevision] = useState(0)
  const [lastRead, setLastRead] = useState<string | null>(null)
  const refresh = useCallback(() => setRevision((value) => value + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    setLoading(true)
    async function load() {
      try {
        const responses = await Promise.allSettled([
          benchApi.catalog(controller.signal),
          benchApi.datasets(controller.signal),
          benchApi.targets(controller.signal),
          benchApi.runs(controller.signal),
        ])
        if (controller.signal.aborted) return
        if (responses[0].status === 'fulfilled') setCatalog(responses[0].value)
        if (responses[1].status === 'fulfilled') setDatasets(responses[1].value.datasets)
        if (responses[2].status === 'fulfilled') setTargets(responses[2].value.targets)
        if (responses[3].status === 'fulfilled') {
          setRuns(responses[3].value.runs)
          setLastRead(new Date().toLocaleTimeString())
        }
        const failure = responses.find((response) => response.status === 'rejected')
        setError(
          failure?.status === 'rejected'
            ? failure.reason instanceof Error
              ? failure.reason.message
              : 'sr-bench service could not be reached.'
            : '',
        )
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
          <p className={styles.eyebrow}>Evaluation</p>
          <h1>
            sr-bench <span>1.0</span>
          </h1>
          <p>
            Measure capability, cost and speed. Improve Balance against your strongest single model.
          </p>
        </div>
        <button onClick={refresh} disabled={loading}>
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </header>
      <nav className={styles.tabs} aria-label="Evaluation views">
        {[
          ['runs', 'Runs'],
          ['new', 'Create evaluation'],
          ['datasets', 'Datasets'],
          ['compare', 'Compare iterations'],
          ['catalog', 'Benchmarks'],
        ].map(([key, label]) => (
          <button
            key={key}
            aria-current={view === key ? 'page' : undefined}
            onClick={() => setSearch({ view: key })}
          >
            {label}
            {key === 'runs' ? ` (${runs.length})` : ''}
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
      {!catalog && loading && <p role="status">Loading sr-bench…</p>}
      {view === 'new' && catalog && (
        <RunComposer
          catalog={catalog}
          datasets={datasets}
          targets={targets}
          canRun={canRun}
          key={search.get('dataset') ?? 'new'}
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
          <RunList
            runs={runs}
            selectedID={selectedID}
            onSelect={(id) => setSearch({ view: 'runs', run: id })}
          />
          {selectedID && (
            <RunDetails
              key={selectedID}
              id={selectedID}
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
      {view === 'datasets' && (
        <DatasetInventory
          datasets={datasets}
          runs={runs}
          canRun={canRun}
          onUse={(dataset) => setSearch({ view: 'new', dataset: dataset.id })}
        />
      )}
      {view === 'compare' && (
        <>
          <RunComparison runs={runs} />
          <ReplayComposer
            runs={runs}
            canRun={canRun}
            onCreated={(run) => {
              setSearch({ view: 'runs', run: run.id })
              refresh()
            }}
          />
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
