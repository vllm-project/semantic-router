import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Database, FlaskConical, Plus } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useReadonly } from '../contexts/ReadonlyContext'
import { canRunEvaluation, canWriteEvaluation } from '../utils/accessControl'
import {
  Button,
  IconTile,
  PageContainer,
  PageHeader,
  StatusDot,
  Tabs,
  TabsList,
  TabsTrigger,
  Tooltip,
} from '../ui'
import { benchApi } from '../components/sr-bench/api'
import { active } from '../components/sr-bench/model'
import { canReuseBaseline } from '../components/sr-bench/baselineReuse'
import RunsView from '../components/sr-bench/RunsView'
import RunSheet from '../components/sr-bench/RunSheet'
import CompareSheet from '../components/sr-bench/CompareSheet'
import DatasetsView from '../components/sr-bench/DatasetsView'
import NewEvaluationSheet, { type EvaluationMode } from '../components/sr-bench/NewEvaluationSheet'
import { rerunSource } from '../components/sr-bench/rerunSource'
import { Notice } from '../components/sr-bench/benchUi'
import {
  parseRunFilters,
  runFilterEntries,
  runSection,
  type RunFilters,
} from '../components/sr-bench/runPresentation'
import type {
  Catalog,
  Dataset,
  ExperimentRunContext,
  Run,
  Target,
} from '../components/sr-bench/types'

type Inventory = 'catalog' | 'datasets' | 'targets' | 'runs'

const COMPOSER_VIEWS: Record<string, EvaluationMode> = {
  new: 'live',
  preview: 'preview',
  estimate: 'estimate',
}

// Start evaluation and the run sheet's primary action (Re-run or Evaluate
// candidate) share the same footer spot. A repeated Start click that lands
// after the new run opens must not reopen the composer for that run.
const START_SETTLE_MS = 1500

function engineLabel(version?: string) {
  const match = version?.match(/^sr-bench-(.+)$/)
  return match ? `sr-bench ${match[1]}` : (version ?? 'sr-bench')
}

export default function EvaluationPage() {
  const { user, refreshSession } = useAuth()
  const {
    serverReadonly,
    isLoading: settingsLoading,
    settingsError,
    refreshSettings,
  } = useReadonly()
  const writeDisabledReason = settingsLoading
    ? 'Checking preparation access…'
    : settingsError
      ? settingsError
      : serverReadonly
        ? 'Preparation is disabled in read-only mode.'
        : !canWriteEvaluation(user)
          ? 'View only. Ask an administrator for dataset preparation access.'
          : null
  const canWrite = writeDisabledReason === null
  const canRun = canWrite && canRunEvaluation(user)
  const [search, setSearch] = useSearchParams()
  const pendingSearch = useRef(search)
  useEffect(() => {
    pendingSearch.current = search
  }, [search])
  const requestedView = search.get('view') ?? (search.has('model') ? 'new' : 'runs')
  const view = [
    'runs',
    'experiments',
    'compare',
    'datasets',
    'new',
    'preview',
    'estimate',
  ].includes(requestedView)
    ? requestedView
    : 'runs'
  const tab = view === 'datasets' ? 'datasets' : 'runs'
  const shownTab = useRef(tab)
  shownTab.current = tab
  const composerMode = COMPOSER_VIEWS[view]
  const experimentID = search.get('experiment') ?? undefined
  const experimentRoute: Record<string, string> = experimentID ? { experiment: experimentID } : {}
  const runFilters = parseRunFilters(search)

  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [targets, setTargets] = useState<Target[]>([])
  const [runs, setRuns] = useState<Run[]>([])
  const [readErrors, setReadErrors] = useState<Partial<Record<Inventory, string>>>({})
  const [loaded, setLoaded] = useState<Partial<Record<Inventory, boolean>>>({})
  const [revision, setRevision] = useState(0)
  const [lastRead, setLastRead] = useState<string | null>(null)
  const [anyActive, setAnyActive] = useState(false)
  const refresh = useCallback(() => setRevision((value) => value + 1), [])
  const error = Object.entries(readErrors)
    .filter(([, message]) => message)
    .map(([name, message]) => `${name}: ${message}`)
    .join(' ')

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
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
      const responses = await Promise.allSettled([
        read('catalog', benchApi.catalog(controller.signal), setCatalog),
        read('datasets', benchApi.datasets(controller.signal), (value) =>
          setDatasets(value.datasets),
        ),
        read('targets', benchApi.targets(controller.signal), (value) => setTargets(value.targets)),
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
      setAnyActive(running)
      timer = setTimeout(() => void load(), failure || running ? 5000 : 15000)
    }
    void load()
    return () => {
      controller.abort()
      if (timer) clearTimeout(timer)
    }
  }, [revision])

  const navigate = (params: Record<string, string>) => setSearch(params)
  const updateRunFilters = (patch: Partial<RunFilters>) => {
    // Router setters do not queue functional updates. Keep same-event patches
    // together until navigation renders; external navigation resyncs above.
    const next = new URLSearchParams(pendingSearch.current)
    for (const [key, value] of runFilterEntries(patch)) {
      if (value === null) next.delete(key)
      else next.set(key, value)
    }
    pendingSearch.current = next
    setSearch(next, { replace: true })
  }
  const openRun = (id?: string) => {
    const next = new URLSearchParams(search)
    next.set('view', 'runs')
    next.delete('section')
    for (const key of [
      'baseline',
      'candidate',
      'role',
      'rerun',
      'dataset',
      'model',
      'comparison_scope',
    ])
      next.delete(key)
    if (id) next.set('run', id)
    else next.delete('run')
    setSearch(next)
  }
  const composerRole = (mode: 'live' | 'preview', baseline: string) =>
    mode === 'preview'
      ? 'preview'
      : !baseline
        ? 'baseline'
        : runs.find((run) => run.id === baseline)?.manifest.profile === 'standard'
          ? 'validation'
          : 'candidate'
  const openComposer = (mode: 'live' | 'preview', baseline = '') => {
    const role = composerRole(mode, baseline)
    navigate({
      view: mode === 'preview' ? 'preview' : 'new',
      ...experimentRoute,
      ...(baseline ? { baseline } : {}),
      ...(experimentID ? { role } : {}),
    })
  }
  const openCompare = (ids: string[]) => {
    const selected = ids.flatMap((id) => runs.filter((run) => run.id === id))
    const reference = selected.find(
      (run) =>
        canReuseBaseline(run) && run.manifest.targets.every((target) => target.kind === 'single'),
    )
    const next = new URLSearchParams({ view: 'compare', ...experimentRoute })
    if (reference) next.set('baseline', reference.id)
    for (const id of ids) if (id !== reference?.id) next.append('candidate', id)
    setSearch(next)
  }
  const closeOverlay = () => navigate({ view: 'runs', ...experimentRoute })
  const startedRun = useRef<{ id: string; at: number } | null>(null)
  const settling = (id: string) =>
    startedRun.current?.id === id && Date.now() - startedRun.current.at < START_SETTLE_MS
  const rerun = search.get('rerun')
  const rerunRun = rerun ? runs.find((run) => run.id === rerun) : undefined
  const composerExperiment: ExperimentRunContext | undefined = experimentID
    ? {
        id: experimentID,
        role:
          view === 'preview'
            ? 'preview'
            : (search.get('role') as ExperimentRunContext['role']) ||
              (search.has('baseline') ? 'candidate' : 'baseline'),
      }
    : undefined
  const composerOpen = !!composerMode && (!rerun || !!rerunRun || !loaded.runs)

  return (
    <PageContainer>
      <section aria-label="sr-bench workspace" className="min-w-0">
        <PageHeader
          icon={<IconTile icon={FlaskConical} tone="orange" size="lg" />}
          title="Evaluations"
          description="Benchmark recipes and the models they route to on the same frozen questions, then compare quality, cost and routing."
          actions={
            <>
              <Tooltip
                content={
                  error
                    ? 'Some sr-bench reads failed; retrying every 5 s.'
                    : lastRead
                      ? `Synced ${lastRead} · updates every ${anyActive ? '5' : '15'} s`
                      : 'Connecting to sr-bench…'
                }
              >
                <span
                  role="status"
                  aria-label={`${engineLabel(catalog?.version)} ${error ? 'needs attention' : lastRead ? `synced ${lastRead}` : 'connecting'}`}
                  className="inline-flex h-8 items-center gap-1.5 rounded-full border border-border bg-surface px-2.5 text-xs font-medium text-foreground/85 shadow-xs"
                >
                  <StatusDot
                    tone={error ? 'warning' : lastRead ? 'success' : 'neutral'}
                    pulse={!lastRead && !error}
                  />
                  {engineLabel(catalog?.version)}
                </span>
              </Tooltip>
              {tab === 'datasets' && (
                <Button
                  variant="outline"
                  onClick={() => navigate({ view: 'datasets', prepare: '1', ...experimentRoute })}
                >
                  <Database aria-hidden="true" />
                  Prepare dataset
                </Button>
              )}
              <Button onClick={() => openComposer('live')}>
                <Plus aria-hidden="true" />
                New evaluation
              </Button>
            </>
          }
        >
          <Tabs
            value={tab}
            onValueChange={(value) => {
              // Tabs activate on mousedown and again on focus; push one history entry.
              if (value === shownTab.current) return
              shownTab.current = value
              navigate({ view: value, ...experimentRoute })
            }}
            className="gap-0"
          >
            <TabsList aria-label="Evaluation views">
              <TabsTrigger value="runs">
                Runs
                {loaded.runs ? (
                  <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
                    {runs.length}
                  </span>
                ) : null}
              </TabsTrigger>
              <TabsTrigger value="datasets">
                Datasets
                {loaded.datasets ? (
                  <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
                    {datasets.length}
                  </span>
                ) : null}
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </PageHeader>
        {error && (
          <Notice tone="danger" role="alert" className="mb-4 mt-0">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <p className="min-w-0 flex-1">
                <strong>sr-bench needs attention.</strong> {error} Existing runs are owned by the
                service and continue independently of this page; reads retry automatically.
              </p>
              <Button variant="outline" size="xs" onClick={refresh}>
                Retry now
              </Button>
            </div>
          </Notice>
        )}
        {tab === 'runs' ? (
          <RunsView
            runs={runs}
            loaded={!!loaded.runs}
            readError={readErrors.runs}
            catalog={catalog}
            filters={runFilters}
            onFilters={updateRunFilters}
            experimentID={experimentID}
            onExperiment={(id) =>
              navigate(id ? { view: 'runs', experiment: id } : { view: 'runs' })
            }
            actorID={user?.id ?? ''}
            canWrite={canWrite}
            canRun={canRun}
            openRunID={search.get('run') ?? undefined}
            onOpenRun={openRun}
            onCompare={openCompare}
            onRerun={(run) =>
              navigate({
                view: run.manifest.mode === 'preview' ? 'preview' : 'new',
                rerun: run.id,
                ...experimentRoute,
              })
            }
            onNewEvaluation={() => openComposer('live')}
            onCandidate={(baseline, experiment, mode, role) =>
              navigate({
                view: mode === 'preview' ? 'preview' : 'new',
                ...(baseline ? { baseline } : {}),
                experiment,
                role,
              })
            }
            onChanged={refresh}
          />
        ) : (
          <DatasetsView
            key={user?.id ?? ''}
            datasets={datasets}
            loaded={!!loaded.datasets}
            readError={readErrors.datasets}
            runs={runs}
            runsLoaded={!!loaded.runs}
            catalog={catalog}
            canRun={canRun}
            writeDisabledReason={writeDisabledReason}
            accessRefreshing={settingsLoading}
            onRefreshAccess={() => {
              refreshSettings()
              void refreshSession()
            }}
            selectedDatasetID={search.get('dataset') ?? undefined}
            onSelectDataset={(id) =>
              navigate({ view: 'datasets', ...(id ? { dataset: id } : {}), ...experimentRoute })
            }
            prepareOpen={search.get('prepare') === '1'}
            onPrepareOpen={(open) =>
              navigate({ view: 'datasets', ...(open ? { prepare: '1' } : {}), ...experimentRoute })
            }
            onUse={(dataset) =>
              navigate({
                view: 'new',
                dataset: dataset.id,
                ...experimentRoute,
                ...(experimentID ? { role: 'baseline' } : {}),
              })
            }
            onInventoryChanged={refresh}
          />
        )}
      </section>
      <RunSheet
        id={tab === 'runs' && !composerMode && view !== 'compare' ? search.get('run') : null}
        onClose={() => openRun()}
        section={runSection(search.get('section'))}
        onSectionChange={(section) => {
          const next = new URLSearchParams(search)
          next.set('section', section)
          setSearch(next)
        }}
        actorID={user?.id ?? ''}
        canRun={canRun}
        onChanged={refresh}
        onRecovered={(run) => {
          navigate({ view: 'runs', run: run.id, ...experimentRoute })
          refresh()
        }}
        onCandidate={(baseline, mode) => {
          if (!settling(baseline)) openComposer(mode, baseline)
        }}
        onCompare={(run) => openCompare([run.id])}
        onRerun={(run) => {
          if (settling(run.id)) return
          navigate({
            view: run.manifest.mode === 'preview' ? 'preview' : 'new',
            rerun: run.id,
            ...experimentRoute,
          })
        }}
      />
      <NewEvaluationSheet
        open={composerOpen}
        onOpenChange={(open) => !open && closeOverlay()}
        mode={composerMode ?? 'live'}
        onModeChange={
          rerun
            ? undefined
            : (mode) => {
                const next = new URLSearchParams(search)
                next.set('view', mode === 'live' ? 'new' : mode)
                if (experimentID)
                  next.set(
                    'role',
                    composerRole(
                      mode === 'preview' ? 'preview' : 'live',
                      search.get('baseline') ?? '',
                    ),
                  )
                setSearch(next, { replace: true })
              }
        }
        allowEstimate={!experimentID}
        catalog={catalog}
        datasets={datasets}
        targets={targets}
        ready={!!catalog && !!loaded.datasets && !!loaded.targets}
        canRun={canRun}
        actorID={user?.id ?? ''}
        baselineID={search.get('baseline') ?? undefined}
        experiment={composerExperiment}
        initialModel={search.get('model') ?? undefined}
        initialDataset={search.get('dataset') ?? undefined}
        rerun={rerunRun ? rerunSource(rerunRun) : undefined}
        onStarted={(run) => {
          setRuns((previous) => [run, ...previous.filter((item) => item.id !== run.id)])
          startedRun.current = { id: run.id, at: Date.now() }
          navigate({ view: 'runs', run: run.id, ...experimentRoute })
          refresh()
        }}
      />
      <CompareSheet
        open={view === 'compare'}
        onClose={closeOverlay}
        runs={runs}
        loaded={!!loaded.runs}
        readError={readErrors.runs}
        actorID={user?.id ?? ''}
      />
    </PageContainer>
  )
}
