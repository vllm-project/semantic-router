import { useMemo, useState } from 'react'
import {
  Copy,
  FlaskConical,
  GitCompareArrows,
  ListChecks,
  PanelRightOpen,
  Plus,
  RotateCw,
  Search,
  TestTubes,
} from 'lucide-react'
import {
  Badge,
  BulkBar,
  Button,
  DataTable,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  InputGroup,
  StatusDot,
  Tooltip,
  cn,
  toast,
  type Column,
} from '../../ui'
import BenchPagination from './BenchPagination'
import ExperimentPanel from './ExperimentPanel'
import { ExperimentCreateDialog, ExperimentFilter } from './ExperimentFilter'
import { useExperimentCreation } from './useExperimentCreation'
import { Notice } from './benchUi'
import { catalogBenchmarkTitle, profileTitle } from './datasetPresentation'
import { BenchmarkChips, RunStatusCell, TargetSummary } from './evalShared'
import type { Experiment } from './experimentApi'
import { canReuseBaseline } from './baselineReuse'
import {
  RUN_MODES,
  RUN_PROFILES,
  RUN_STATUSES,
  filterRuns,
  relativeTime,
  runBenchmarks,
  runBestScore,
  runCost,
  runModeLabel,
  isSupportingRole,
  runStatusLabel,
  sequenceExperiment,
  type RunFilters,
} from './runPresentation'
import useExperimentMembers from './useExperimentMembers'
import type { Catalog, ExperimentRunContext, Run } from './types'

const PAGE_SIZE = 20

interface RunRow {
  id: string
  run?: Run
  label?: string
  hypothesis?: string
}

export interface RunsViewProps {
  runs: Run[]
  loaded: boolean
  readError?: string
  catalog: Catalog | null
  filters: RunFilters
  onFilters: (patch: Partial<RunFilters>) => void
  experimentID?: string
  onExperiment: (id?: string) => void
  actorID: string
  canWrite: boolean
  canRun: boolean
  openRunID?: string
  onOpenRun: (id: string) => void
  onCompare: (ids: string[]) => void
  onRerun: (run: Run) => void
  onNewEvaluation: () => void
  onCandidate: (
    baseline: string,
    experiment: string,
    mode: 'live' | 'preview',
    role: ExperimentRunContext['role'],
  ) => void
  onChanged: () => void
}

export default function RunsView({
  runs,
  loaded,
  readError,
  catalog,
  filters,
  onFilters,
  experimentID,
  onExperiment,
  actorID,
  canWrite,
  canRun,
  openRunID,
  onOpenRun,
  onCompare,
  onRerun,
  onNewEvaluation,
  onCandidate,
  onChanged,
}: RunsViewProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [experiment, setExperiment] = useState<Experiment | null>(null)
  const [creating, setCreating] = useState(false)
  const [showSupporting, setShowSupporting] = useState(false)
  const creation = useExperimentCreation(actorID, (id) => onExperiment(id))
  const membership = useExperimentMembers(experimentID)
  const experimentName = experiment?.id === experimentID ? experiment?.name : undefined

  const filtered = useMemo(() => filterRuns(runs, filters), [runs, filters])
  const rows: RunRow[] = useMemo(() => {
    if (!experimentID) return filtered.map((run) => ({ id: run.id, run }))
    const visible = new Set(filtered.map((run) => run.id))
    const unfiltered =
      !filters.query &&
      !filters.status.length &&
      !filters.mode.length &&
      !filters.profile.length &&
      !filters.benchmark.length
    return sequenceExperiment(membership.members, runs)
      .filter((item) => showSupporting || !isSupportingRole(item.member.role))
      .filter((item) => (item.run ? visible.has(item.run.id) : unfiltered))
      .map((item) => ({
        id: item.member.run_id,
        run: item.run,
        label: item.label,
        hypothesis: item.member.hypothesis,
      }))
  }, [experimentID, filtered, filters, membership.members, runs, showSupporting])
  const supporting = experimentID
    ? membership.members.filter((member) => isSupportingRole(member.role)).length
    : 0
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const page = experimentID ? 0 : Math.min(filters.page, pages - 1)
  const pageRows = experimentID ? rows : rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const running = (experimentID ? runs.filter((run) => membership.ids.has(run.id)) : runs).filter(
    (run) => run.status === 'running',
  ).length
  const onlyRunning = filters.status.length === 1 && filters.status[0] === 'running'
  const filtersActive =
    !!filters.query ||
    filters.status.length > 0 ||
    filters.mode.length > 0 ||
    filters.profile.length > 0 ||
    filters.benchmark.length > 0
  const showScore = runs.some((run) => runBestScore(run))
  const showCost = runs.some((run) => runCost(run) !== null)
  const benchmarkIDs = [...new Set(runs.flatMap(runBenchmarks))]
  const benchmarkLabel = (id: string) => catalogBenchmarkTitle(id, catalog?.benchmarks)
  const count = (predicate: (run: Run) => boolean) => runs.filter(predicate).length

  const columns: Column<RunRow>[] = [
    {
      id: 'run',
      header: 'Run',
      cell: ({ run, label, hypothesis, id }) => (
        <div className="flex min-w-0 max-w-[300px] flex-col gap-0.5 xl:max-w-[360px]">
          <span className="flex min-w-0 items-center gap-1.5">
            {label ? (
              <Tooltip content={hypothesis || label}>
                <Badge tone="outline" size="sm" className="shrink-0">
                  {label}
                </Badge>
              </Tooltip>
            ) : null}
            <button
              type="button"
              className="min-w-0 truncate rounded-sm text-left text-[13px] font-medium text-foreground outline-none hover:underline focus-visible:ring-[3px] focus-visible:ring-ring/40"
              onClick={(event) => {
                event.stopPropagation()
                onOpenRun(id)
              }}
            >
              {run?.manifest.name ?? 'Saved run'}
            </button>
            {run && run.manifest.mode !== 'live' ? (
              <Badge tone="violet" size="sm" className="shrink-0">
                {runModeLabel(run.manifest.mode)}
              </Badge>
            ) : null}
          </span>
          <span className="truncate font-mono text-[11px] text-subtle-foreground">
            {run
              ? `${run.id} · ${profileTitle(run.manifest.profile)} · ${relativeTime(run.created_at)}`
              : 'Open this run to view its status'}
          </span>
        </div>
      ),
    },
    {
      id: 'benchmarks',
      header: 'Benchmarks',
      hideBelow: 'lg',
      cell: ({ run }) =>
        run ? <BenchmarkChips ids={runBenchmarks(run)} catalog={catalog} /> : null,
    },
    {
      id: 'targets',
      header: 'Targets',
      hideBelow: 'md',
      cell: ({ run }) => (run ? <TargetSummary targets={run.manifest.targets} /> : null),
    },
    {
      id: 'status',
      header: 'Status',
      cell: ({ run }) => (run ? <RunStatusCell run={run} /> : <span>—</span>),
    },
    ...(showScore
      ? [
          {
            id: 'score',
            header: 'Best score',
            align: 'right',
            hideBelow: 'sm',
            cell: ({ run }) => {
              const best = run && runBestScore(run)
              if (!best) return <span className="text-subtle-foreground">—</span>
              return (
                <span className="flex flex-col items-end">
                  <span className="font-mono text-[13px] font-semibold tabular">
                    {best.score.toFixed(1)}
                  </span>
                  <span className="max-w-[140px] truncate text-[11px] text-subtle-foreground">
                    {best.target.split('/').pop()}
                  </span>
                </span>
              )
            },
          } satisfies Column<RunRow>,
        ]
      : []),
    ...(showCost
      ? [
          {
            id: 'cost',
            header: 'Cost',
            align: 'right',
            hideBelow: 'lg',
            cell: ({ run }) => {
              const cost = run ? runCost(run) : null
              return cost === null ? (
                <span className="text-subtle-foreground">—</span>
              ) : (
                <span className="font-mono text-[13px] tabular">${cost.toFixed(2)}</span>
              )
            },
          } satisfies Column<RunRow>,
        ]
      : []),
  ]

  const toolbar = (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <InputGroup
        type="search"
        aria-label="Search runs"
        placeholder="Search runs or targets…"
        value={filters.query}
        onChange={(event) => onFilters({ query: event.target.value, page: 0 })}
        leading={<Search aria-hidden="true" />}
        containerClassName="w-full sm:w-64"
        className="h-8"
      />
      <FilterPill
        label="Status"
        options={RUN_STATUSES.map((value) => ({
          value,
          label: runStatusLabel(value),
          count: count((run) => run.status === value),
        }))}
        selected={filters.status}
        onChange={(status) => onFilters({ status, page: 0 })}
      />
      <FilterPill
        label="Benchmark"
        options={benchmarkIDs.map((value) => ({
          value,
          label: benchmarkLabel(value),
          count: count((run) => runBenchmarks(run).includes(value)),
        }))}
        selected={filters.benchmark}
        onChange={(benchmark) => onFilters({ benchmark, page: 0 })}
      />
      <FilterPill
        label="Profile"
        options={RUN_PROFILES.map((value) => ({
          value,
          label: profileTitle(value),
          count: count((run) => run.manifest.profile === value),
        }))}
        selected={filters.profile}
        onChange={(profile) => onFilters({ profile, page: 0 })}
      />
      <FilterPill
        label="Type"
        options={RUN_MODES.map((value) => ({
          value,
          label: runModeLabel(value),
          count: count((run) => run.manifest.mode === value),
        }))}
        selected={filters.mode}
        onChange={(mode) => onFilters({ mode, page: 0 })}
      />
      <ExperimentFilter
        selectedID={experimentID}
        selectedName={experimentName}
        canWrite={canWrite}
        onSelect={(id) => {
          setSelected(new Set())
          onExperiment(id)
        }}
        onCreate={() => setCreating(true)}
      />
      {filtersActive ? (
        <Button
          variant="ghost"
          size="sm"
          onClick={() =>
            onFilters({ query: '', status: [], mode: [], profile: [], benchmark: [], page: 0 })
          }
        >
          Reset
        </Button>
      ) : null}
      <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
        {running > 0 ? (
          <button
            type="button"
            aria-pressed={onlyRunning}
            onClick={() => onFilters({ status: onlyRunning ? [] : ['running'], page: 0 })}
            className={cn(
              'inline-flex h-7 items-center gap-1.5 rounded-full px-2.5 font-medium text-foreground/85 transition-colors hover:bg-accent',
              onlyRunning && 'bg-info-soft text-info-foreground',
            )}
          >
            <StatusDot tone="info" pulse />
            {running} running
          </button>
        ) : null}
        {loaded ? (
          <span className="tabular" role="status">
            {experimentID
              ? `${rows.length} ${rows.length === 1 ? 'run' : 'runs'} in this experiment`
              : `${filtered.length} of ${runs.length} ${runs.length === 1 ? 'run' : 'runs'}`}
          </span>
        ) : null}
      </div>
    </div>
  )

  const empty = !loaded ? null : runs.length === 0 && !experimentID ? (
    <EmptyState
      icon={FlaskConical}
      tone="orange"
      title="No evaluations yet"
      description="Benchmark a recipe against the models it routes to, then compare quality, cost and routing on the same questions."
      actions={
        <Button size="sm" onClick={onNewEvaluation}>
          <Plus aria-hidden="true" /> New evaluation
        </Button>
      }
    />
  ) : experimentID && membership.complete && membership.members.length === 0 ? (
    <EmptyState
      compact
      icon={TestTubes}
      tone="orange"
      title="No runs in this experiment yet"
      description="Start with your single models, or add results you already have."
    />
  ) : experimentID && !membership.complete ? null : (
    <EmptyState
      compact
      icon={ListChecks}
      title="No runs match these filters"
      description="Change the search or filters to see more runs."
      actions={
        <Button
          variant="outline"
          size="sm"
          onClick={() =>
            onFilters({ query: '', status: [], mode: [], profile: [], benchmark: [], page: 0 })
          }
        >
          Clear filters
        </Button>
      }
    />
  )

  return (
    <section aria-label="Evaluation runs" className="min-w-0">
      {toolbar}
      {(creation.pendingRequest || creation.storageError) && !creating && (
        <Notice tone="warning" className="mt-0">
          <strong>An experiment creation is waiting for confirmation</strong>
          <p>
            {creation.storageError ||
              `“${creation.pendingRequest?.name}” was submitted but its response was not confirmed.`}
          </p>
          {canWrite && !creation.storageError ? (
            <div>
              <Button variant="outline" size="xs" onClick={() => setCreating(true)}>
                Check creation request
              </Button>
            </div>
          ) : null}
        </Notice>
      )}
      {experimentID ? (
        <ExperimentPanel
          key={`${actorID}:${experimentID}`}
          id={experimentID}
          actorID={actorID}
          runs={runs}
          membership={membership}
          canWrite={canWrite}
          canRun={canRun}
          onLoaded={setExperiment}
          onOpenRun={onOpenRun}
          onCandidate={(baseline, mode, role) => onCandidate(baseline, experimentID, mode, role)}
          onCompare={(baseline) => onCompare([baseline])}
          onDeleted={() => onExperiment(undefined)}
          onChanged={onChanged}
        />
      ) : null}
      {!loaded && readError ? (
        <div className="rounded-xl border border-border bg-card shadow-xs">
          <EmptyState
            compact
            icon={ListChecks}
            title="Run inventory is unavailable"
            description="Runs keep executing on the sr-bench service. This list reconnects automatically."
          />
        </div>
      ) : (
        <DataTable
          data={pageRows}
          columns={columns}
          getRowId={(row) => row.id}
          loading={!loaded || (!!experimentID && membership.loading)}
          onRowClick={(row) => onOpenRun(row.id)}
          activeRowId={openRunID}
          selectable
          selected={selected}
          onSelectedChange={setSelected}
          rowActions={({ id, run }) => (
            <>
              <DropdownMenuItem onSelect={() => onOpenRun(id)}>
                <PanelRightOpen aria-hidden="true" /> Open run
              </DropdownMenuItem>
              {run && run.manifest.mode === 'live' && canReuseBaseline(run) ? (
                <DropdownMenuItem onSelect={() => onCompare([id])}>
                  <GitCompareArrows aria-hidden="true" /> Compare with…
                </DropdownMenuItem>
              ) : null}
              {run && canRun && run.manifest.mode !== 'replay' ? (
                <DropdownMenuItem onSelect={() => onRerun(run)}>
                  <RotateCw aria-hidden="true" /> Re-run
                </DropdownMenuItem>
              ) : null}
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onSelect={() => {
                  void navigator.clipboard?.writeText(id).catch(() => undefined)
                  toast.success('Run ID copied')
                }}
              >
                <Copy aria-hidden="true" /> Copy run ID
              </DropdownMenuItem>
            </>
          )}
          empty={empty}
        />
      )}
      {experimentID && supporting > 0 && membership.complete ? (
        <Button
          variant="ghost"
          size="sm"
          className="mt-2"
          aria-expanded={showSupporting}
          onClick={() => setShowSupporting((value) => !value)}
        >
          {showSupporting ? 'Hide' : 'Show'} supporting checks ({supporting})
        </Button>
      ) : null}
      {!experimentID && (
        <BenchPagination
          label="Runs"
          total={rows.length}
          page={page}
          pageSize={PAGE_SIZE}
          onChange={(next) => onFilters({ page: next })}
        />
      )}
      <BulkBar count={selected.size} onClear={() => setSelected(new Set())}>
        <Tooltip
          content={
            selected.size < 2
              ? 'Select a reference run and at least one run to compare'
              : 'Compare the selected runs on the same questions'
          }
        >
          <span>
            <Button
              size="sm"
              disabled={selected.size < 2}
              onClick={() => {
                onCompare([...selected])
                setSelected(new Set())
              }}
            >
              <GitCompareArrows aria-hidden="true" /> Compare
            </Button>
          </span>
        </Tooltip>
      </BulkBar>
      <ExperimentCreateDialog open={creating} onOpenChange={setCreating} creation={creation} />
    </section>
  )
}
