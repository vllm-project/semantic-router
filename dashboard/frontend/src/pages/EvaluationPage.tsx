import { useCallback, useEffect, useRef, useState } from 'react'
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
import RunList, { type RunFilters } from '../components/sr-bench/RunList'
import DatasetInventory from '../components/sr-bench/DatasetInventory'
import ExperimentWorkspace from '../components/sr-bench/ExperimentWorkspace'
import type {
  Catalog,
  Dataset,
  ExperimentRunContext,
  Run,
  Target,
} from '../components/sr-bench/types'
import styles from '../components/sr-bench/SrBench.module.css'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'

type Inventory = 'catalog' | 'datasets' | 'targets' | 'runs'

export default function EvaluationPage() {
  const { user } = useAuth()
  const { serverReadonly, isLoading: settingsLoading } = useReadonly()
  const canWrite = !settingsLoading && !serverReadonly && canWriteEvaluation(user)
  const canRun = canWrite && canRunEvaluation(user)
  const [search, setSearch] = useSearchParams()
  const pendingSearch = useRef(search)
  useEffect(() => {
    pendingSearch.current = search
  }, [search])
  const selectedID = search.get('run')
  const experimentID = search.get('experiment')
  const experimentRoute: Record<string, string> = experimentID ? { experiment: experimentID } : {}
  const openComposer = (mode: 'live' | 'preview', baseline = search.get('baseline') ?? '') => {
    const role =
      mode === 'preview'
        ? 'preview'
        : !baseline
          ? 'baseline'
          : runs.find((run) => run.id === baseline)?.manifest.profile === 'standard'
            ? 'validation'
            : 'candidate'
    setSearch({
      view: mode === 'preview' ? 'preview' : 'new',
      ...experimentRoute,
      ...(baseline ? { baseline } : {}),
      ...(experimentID ? { role } : {}),
    })
  }
  const runFilters: RunFilters = {
    query: search.get('q') ?? '',
    status: ['active', 'completed', 'failed', 'interrupted', 'cancelled'].includes(
      search.get('status') ?? '',
    )
      ? search.get('status')!
      : 'all',
    mode: ['live', 'preview', 'replay'].includes(search.get('mode') ?? '')
      ? search.get('mode')!
      : 'all',
    profile: ['smoke', 'quick', 'standard'].includes(search.get('profile') ?? '')
      ? search.get('profile')!
      : 'all',
    page: Math.max(0, Math.floor(Number(search.get('page')) || 0)),
  }
  const openRun = (id?: string) => {
    const next = new URLSearchParams(search)
    next.set('view', 'runs')
    next.delete('section')
    if (id) next.set('run', id)
    else next.delete('run')
    setSearch(next)
  }
  const updateRunFilters = (patch: Partial<RunFilters>) => {
    // Router setters do not queue functional updates. Keep same-event patches
    // together until navigation renders; external navigation resyncs above.
    const next = new URLSearchParams(pendingSearch.current)
    for (const [field, value] of Object.entries(patch)) {
      const key = field === 'query' ? 'q' : field
      if (!value || value === 'all') next.delete(key)
      else next.set(key, String(value))
    }
    pendingSearch.current = next
    setSearch(next, { replace: true })
  }
  const requestedView = search.get('view') ?? (search.has('model') ? 'new' : 'runs')
  const view = ['runs', 'experiments', 'compare', 'datasets', 'new', 'preview'].includes(
    requestedView,
  )
    ? requestedView
    : 'runs'
  const creating = view === 'new' || view === 'preview'
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
          {view !== 'preview' && (
            <button onClick={() => openComposer('preview')}>
              <ProductIcon name="decision" />
              Preview routing
            </button>
          )}
          {view !== 'new' && (
            <button className={styles.primary} onClick={() => openComposer('live')}>
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
            ['experiments', 'Experiments', 'evaluation'],
            ['compare', 'Compare iterations', 'chart'],
            ['datasets', 'Datasets', 'database'],
          ] as const
        ).map(([key, label, icon]) => (
          <button
            key={key}
            aria-current={view === key ? 'page' : undefined}
            onClick={() => setSearch({ view: key, ...experimentRoute })}
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
      {experimentID && view !== 'experiments' && (
        <button
          className={styles.backLink}
          onClick={() => setSearch({ view: 'experiments', experiment: experimentID })}
        >
          <ProductIcon name="arrow-left" />
          Back to experiment
        </button>
      )}
      {!experimentID && (creating || (view === 'runs' && selectedID)) && (
        <button className={styles.backLink} onClick={() => openRun()}>
          <ProductIcon name="arrow-left" />
          Back to runs
        </button>
      )}
      {creating && catalog && loaded.datasets && loaded.targets && (
        <RunComposer
          catalog={catalog}
          datasets={datasets}
          targets={targets}
          canRun={canRun}
          actorID={user?.id ?? ''}
          key={`${user?.id ?? ''}:${view}:${search.get('dataset') ?? 'new'}:${search.get('baseline') ?? ''}:${search.get('experiment') ?? ''}:${search.get('role') ?? ''}`}
          mode={view === 'preview' ? 'preview' : 'live'}
          baselineID={search.get('baseline') ?? undefined}
          experiment={
            search.get('experiment')
              ? {
                  id: search.get('experiment')!,
                  role:
                    view === 'preview'
                      ? 'preview'
                      : (search.get('role') as ExperimentRunContext['role']) ||
                        (search.has('baseline') ? 'candidate' : 'baseline'),
                }
              : undefined
          }
          initialModel={search.get('model') ?? undefined}
          initialDataset={search.get('dataset') ?? undefined}
          onStarted={(run) => {
            setRuns((previous) => [run, ...previous.filter((item) => item.id !== run.id)])
            setSearch({ view: 'runs', run: run.id, ...experimentRoute })
            refresh()
          }}
        />
      )}
      {view === 'experiments' && (
        <ExperimentWorkspace
          key={`${user?.id ?? ''}:${search.get('experiment') ?? 'all'}`}
          id={search.get('experiment') ?? undefined}
          actorID={user?.id ?? ''}
          runs={runs}
          canWrite={canWrite}
          canRun={canRun}
          onSelect={(id) =>
            setSearch(id ? { view: 'experiments', experiment: id } : { view: 'experiments' })
          }
          onOpenRun={openRun}
          onCandidate={(baseline, experiment, mode, role) =>
            setSearch({
              view: mode === 'preview' ? 'preview' : 'new',
              ...(baseline ? { baseline } : {}),
              experiment,
              role,
            })
          }
          onCompare={(baseline) => setSearch({ view: 'compare', baseline, ...experimentRoute })}
        />
      )}
      {view === 'runs' && (
        <>
          {!selectedID &&
            (loaded.runs ? (
              <RunList
                runs={runs}
                selectedID={selectedID}
                filters={runFilters}
                onFilters={updateRunFilters}
                onSelect={openRun}
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
                setSearch({ view: 'runs', run: run.id, ...experimentRoute })
                refresh()
              }}
              onCandidate={(baseline, mode) => openComposer(mode, baseline)}
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
          onSelectDataset={(id) => setSearch({ view: 'datasets', dataset: id, ...experimentRoute })}
          onBackToDatasets={() => setSearch({ view: 'datasets', ...experimentRoute })}
          onUse={(dataset) =>
            setSearch({
              view: 'new',
              dataset: dataset.id,
              ...experimentRoute,
              ...(experimentID ? { role: 'baseline' } : {}),
            })
          }
        />
      )}
      {view === 'datasets' &&
        !loaded.datasets &&
        (readErrors.datasets ? (
          <p role="status">Dataset inventory is unavailable.</p>
        ) : (
          <ProductLoadingState compact label="Loading prepared datasets…" />
        ))}
      {creating && (!catalog || !loaded.datasets || !loaded.targets) && (
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
          <RunComparison key={user?.id ?? ''} runs={runs} />
          <details className={styles.panel}>
            <summary>Estimate a routing change</summary>
            <ReplayComposer
              key={user?.id ?? ''}
              actorID={user?.id ?? ''}
              canRun={canRun}
              onCreated={(run) => {
                setSearch({ view: 'runs', run: run.id, ...experimentRoute })
                refresh()
              }}
            />
          </details>
        </>
      )}
    </section>
  )
}
