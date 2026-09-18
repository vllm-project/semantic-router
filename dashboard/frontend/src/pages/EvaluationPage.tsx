import { useCallback, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canRunEvaluation, canWriteEvaluation } from '../utils/accessControl'
import { benchApi } from '../components/sr-bench/api'
import { active, number } from '../components/sr-bench/model'
import RunComposer from '../components/sr-bench/RunComposer'
import RunDetails from '../components/sr-bench/RunDetails'
import RunComparison from '../components/sr-bench/RunComparison'
import ReplayComposer from '../components/sr-bench/ReplayComposer'
import type { Catalog, Dataset, Run, Target } from '../components/sr-bench/types'
import styles from '../components/sr-bench/SrBench.module.css'

export default function EvaluationPage() {
  const { user } = useAuth()
  const { serverReadonly, isLoading: settingsLoading } = useReadonly()
  const canRun =
    !settingsLoading && !serverReadonly && canRunEvaluation(user) && canWriteEvaluation(user)
  const [search, setSearch] = useSearchParams()
  const selectedID = search.get('run')
  const view = search.get('view') ?? (selectedID ? 'runs' : 'new')
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [revision, setRevision] = useState(0)
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
        if (responses[3].status === 'fulfilled') setRuns(responses[3].value.runs)
        const failure = responses.find((response) => response.status === 'rejected')
        setError(
          failure?.status === 'rejected'
            ? failure.reason instanceof Error
              ? failure.reason.message
              : 'sr-bench service could not be reached.'
            : '',
        )
        if (
          responses[3].status === 'fulfilled' &&
          responses[3].value.runs.some((run) => active(run.status))
        )
          timer = setTimeout(() => void load(), 5000)
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
    <main className={styles.page}>
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
          ['new', 'Create evaluation'],
          ['runs', 'Runs'],
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
          initialModel={search.get('model') ?? undefined}
          onStarted={(run) => {
            setRuns((previous) => [run, ...previous.filter((item) => item.id !== run.id)])
            setSearch({ view: 'runs', run: run.id })
            refresh()
          }}
        />
      )}
      {view === 'runs' && (
        <>
          <section className={styles.panel}>
            <div className={styles.sectionHeading}>
              <h2>Evaluation runs</h2>
              <span>{runs.filter((run) => active(run.status)).length} active</span>
            </div>
            {runs.length ? (
              <div className={styles.tableScroll}>
                <table>
                  <thead>
                    <tr>
                      <th>Run</th>
                      <th>Mode / profile</th>
                      <th>Status</th>
                      <th>Completed / total</th>
                      <th>Failures</th>
                      <th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr key={run.id} aria-selected={run.id === selectedID}>
                        <td>
                          <button
                            className={styles.linkButton}
                            onClick={() => setSearch({ view: 'runs', run: run.id })}
                          >
                            {run.manifest.name}
                          </button>
                          <small>{run.id}</small>
                        </td>
                        <td>
                          {run.manifest.mode} / {run.manifest.profile}
                        </td>
                        <td>{run.status}</td>
                        <td>
                          {number(run.progress.completed)} / {number(run.progress.total)}
                        </td>
                        <td>{number(run.progress.failed)}</td>
                        <td>{new Date(run.created_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>No runs yet. Create an evaluation using a prepared dataset.</p>
            )}
          </section>
          {selectedID && (
            <RunDetails key={selectedID} id={selectedID} canRun={canRun} onChanged={refresh} />
          )}
        </>
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
    </main>
  )
}
