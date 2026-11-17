import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Bot, Copy, ListTree, PanelRightOpen } from 'lucide-react'

import {
  Badge,
  Button,
  Card,
  DataTable,
  DataToolbar,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  StatusLabel,
  Tooltip,
  toast,
  type Column,
  type FilterOption,
} from '../ui'

import { compactSessionId, displaySessionId } from './insightsAgentIdentity'
import { formatLatency, useNow } from './insightsDisplayFormat'
import {
  InsightsLoadError,
  InsightsUnavailable,
  RefreshButton,
  RelativeTime,
} from './InsightsDisplay'
import { HarnessLogo } from './InsightsHarness'
import {
  API_CLIENT_LABEL,
  harnessInfo,
  normalizeHarnessId,
  sortHarnessIds,
} from './insightsHarness'
import { ModelChip, modelText, useModelLabels } from './insightsModelLabelSupport'
import { insightsUnavailableReason, type InsightsUnavailableReason } from './insightsPageApi'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'
import { getSessionRequestsPath } from './insightsReviewSupport'
import {
  countText,
  formatRunCost,
  getRunPath,
  RUN_MARKER_GLYPHS,
  RUN_MARKER_LABELS,
  RUN_OUTCOME_LABELS,
  RUN_STATUS_TONES,
  runTitle,
} from './insightsRunSupport'
import type { RunListResponse, RunSummary } from './insightsRunTypes'

const RUN_CHUNK = 100
const API_FILTER = 'api'

export function RunStatus({ run }: { run: Pick<RunSummary, 'outcome'> }) {
  const status = run.outcome.status
  return (
    <StatusLabel tone={RUN_STATUS_TONES[status] ?? 'neutral'} pulse={status === 'in_progress'}>
      {RUN_OUTCOME_LABELS[status] ?? status}
    </StatusLabel>
  )
}

/** A run's events worth a look, each a glyph and its count. */
export function RunEvents({ events }: { events?: Record<string, number> }) {
  const entries = Object.entries(events ?? {})
  if (entries.length === 0) return null
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {entries.map(([marker, count]) => (
        <Badge
          key={marker}
          tone={
            marker === 'failure' || marker === 'block'
              ? 'danger'
              : marker === 'guard' || marker === 'tool_failure'
                ? 'warning'
                : 'outline'
          }
          size="sm"
          title={`${RUN_MARKER_LABELS[marker] ?? marker}: ${count}`}
          data-marker={marker}
        >
          <span aria-hidden>{RUN_MARKER_GLYPHS[marker] ?? '•'}</span>
          {count}
        </Badge>
      ))}
    </span>
  )
}

function runHarness(run: RunSummary): string {
  return normalizeHarnessId(run.harness) || API_FILTER
}

function useRuns(search: string) {
  const [state, setState] = useState<{
    runs: RunSummary[]
    total: number
    loading: boolean
    error: string | null
    unavailable?: InsightsUnavailableReason
  }>({ runs: [], total: 0, loading: true, error: null })
  const abortRef = useRef<AbortController | null>(null)

  const load = useCallback(
    async (offset: number) => {
      abortRef.current?.abort()
      const abort = new AbortController()
      abortRef.current = abort
      setState((current) => ({ ...current, loading: true }))
      const query = new URLSearchParams({ limit: String(RUN_CHUNK), offset: String(offset) })
      if (search.trim()) query.set('search', search.trim())
      try {
        const list = await fetchAbortableInsightsJSON<RunListResponse>(
          `/api/router/api/v1/observability/replays/runs?${query.toString()}`,
          'agent runs',
          abort.signal,
        )
        setState((current) => ({
          runs: offset === 0 ? (list.data ?? []) : [...current.runs, ...(list.data ?? [])],
          total: list.total,
          loading: false,
          error: null,
        }))
      } catch (error) {
        if (isAbortError(error)) return
        const unavailable = insightsUnavailableReason(error)
        setState({
          runs: [],
          total: 0,
          loading: false,
          error: unavailable ? null : error instanceof Error ? error.message : 'Unknown error',
          unavailable,
        })
      }
    },
    [search],
  )

  useEffect(() => {
    void load(0)
    return () => abortRef.current?.abort()
  }, [load])

  return { ...state, reload: () => void load(0), loadMore: () => void load(state.runs.length) }
}

function copy(value: string, label: string) {
  void navigator.clipboard?.writeText(value).catch(() => undefined)
  toast.success(`${label} copied`, { description: value })
}

// One row per agent run, the session a harness names: its title, harness,
// agents, models, what it cost and took, and how it ended.
export default function InsightsRunsView() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const labels = useModelLabels()
  const now = useNow(30_000)
  const search = params.get('q') ?? ''
  const [draft, setDraft] = useState(search)
  const harness = (params.get('harness') ?? '').split(',').filter(Boolean)
  const status = (params.get('status') ?? '').split(',').filter(Boolean)
  const { runs, total, loading, error, unavailable, reload, loadMore } = useRuns(search)

  const setParam = (name: string, value: string) =>
    setParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (value) next.set(name, value)
        else next.delete(name)
        return next
      },
      { replace: true },
    )

  const timer = useRef<number>(undefined)
  useEffect(() => () => window.clearTimeout(timer.current), [])
  const onSearch = (value: string) => {
    setDraft(value)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setParam('q', value.trim()), 300)
  }

  const harnessCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const run of runs) counts.set(runHarness(run), (counts.get(runHarness(run)) ?? 0) + 1)
    return counts
  }, [runs])
  const harnessOptions: FilterOption[] = [
    ...sortHarnessIds([...harnessCounts.keys(), ...harness].filter((id) => id !== API_FILTER)).map(
      (id) => ({
        value: id,
        label: harnessInfo(id)?.label ?? id,
        icon: <HarnessLogo harness={id} size={14} />,
        count: harnessCounts.get(id) ?? 0,
      }),
    ),
    ...(harnessCounts.has(API_FILTER) || harness.includes(API_FILTER)
      ? [
          {
            value: API_FILTER,
            label: API_CLIENT_LABEL,
            icon: <HarnessLogo size={14} />,
            count: harnessCounts.get(API_FILTER) ?? 0,
          },
        ]
      : []),
  ]
  const statusOptions: FilterOption[] = Object.entries(RUN_OUTCOME_LABELS).map(
    ([value, label]) => ({
      value,
      label,
      count: runs.filter((run) => run.outcome.status === value).length,
    }),
  )

  const rows = runs
    .filter((run) => harness.length === 0 || harness.includes(runHarness(run)))
    .filter((run) => status.length === 0 || status.includes(run.outcome.status))

  const columns: Column<RunSummary>[] = [
    {
      id: 'session',
      header: 'Session',
      className: 'w-full max-w-0',
      sortValue: (run) => runTitle(run),
      cell: (run) => {
        const info = harnessInfo(run.harness)
        return (
          <div className="flex min-w-0 items-center gap-3">
            <HarnessLogo harness={run.harness} size={16} framed />
            <div className="flex min-w-0 flex-col gap-0.5">
              <span
                className="truncate font-medium"
                title={run.session_name && run.prompt ? run.prompt : undefined}
              >
                {runTitle(run)}
              </span>
              <span className="truncate text-xs text-subtle-foreground">
                {info?.label ?? API_CLIENT_LABEL} ·{' '}
                <span className="font-mono text-[11.5px]" title={displaySessionId(run.session_id)}>
                  {compactSessionId(run.session_id)}
                </span>
              </span>
            </div>
          </div>
        )
      },
    },
    {
      id: 'agents',
      header: 'Agents',
      align: 'right',
      sortValue: (run) => run.agents.length,
      cell: (run) => (
        <Tooltip
          content={run.agents
            .map(
              (agent) =>
                `${agent.label}${agent.kind ? ` (${agent.kind})` : ''} · ${countText(agent.calls, 'call')}`,
            )
            .join('  ·  ')}
        >
          <span className="inline-flex items-center gap-1.5 text-[13px]">
            <Bot className="size-3.5 text-subtle-foreground" />
            <span className="tabular">{run.agents.length}</span>
          </span>
        </Tooltip>
      ),
    },
    {
      id: 'calls',
      header: 'Calls',
      align: 'right',
      sortValue: (run) => run.calls,
      cell: (run) => <span className="font-mono text-xs tabular">{run.calls}</span>,
    },
    {
      id: 'models',
      header: 'Models',
      hideBelow: 'lg',
      cell: (run) => (
        <Tooltip
          content={
            run.models
              .map(
                (model) => `${modelText(labels, model.name)} · ${countText(model.value, 'call')}`,
              )
              .join(', ') || 'No model answered'
          }
        >
          <span className="flex min-w-0 max-w-[220px] flex-col gap-0.5">
            {run.models.slice(0, 1).map((model) => (
              <ModelChip key={model.name} name={model.name} className="text-[13px]" />
            ))}
            {run.models.length > 1 ? (
              <span className="text-[11px] text-subtle-foreground">
                +{run.models.length - 1} more
              </span>
            ) : null}
            {run.models.length === 0 ? (
              <span className="text-xs text-subtle-foreground">—</span>
            ) : null}
          </span>
        </Tooltip>
      ),
    },
    {
      id: 'cost',
      header: 'Cost',
      align: 'right',
      hideBelow: 'md',
      sortValue: (run) => run.cost.by_currency?.[0]?.total ?? -1,
      cell: (run) => <span className="font-mono text-xs tabular">{formatRunCost(run.cost)}</span>,
    },
    {
      id: 'duration',
      header: 'Duration',
      align: 'right',
      hideBelow: 'sm',
      sortValue: (run) => run.duration_ms,
      cell: (run) => (
        <span className="font-mono text-xs tabular text-muted-foreground">
          {formatLatency(run.duration_ms)}
        </span>
      ),
    },
    {
      id: 'status',
      header: 'Status',
      width: '150px',
      cell: (run) => (
        <span className="flex flex-col items-start gap-1">
          <RunStatus run={run} />
          <RunEvents events={run.events} />
        </span>
      ),
    },
    {
      id: 'started',
      header: 'Started',
      width: '96px',
      hideBelow: 'md',
      sortValue: (run) => Date.parse(run.started_at),
      cell: (run) => (
        <RelativeTime iso={run.started_at} now={now} className="text-xs text-muted-foreground" />
      ),
    },
  ]

  if (unavailable) {
    return (
      <Card className="border-dashed shadow-none">
        <InsightsUnavailable reason={unavailable} noun="agent runs" onRetry={reload} />
      </Card>
    )
  }

  return (
    <>
      {error ? (
        <InsightsLoadError title="Agent runs could not load" message={error} onRetry={reload} />
      ) : null}
      <DataToolbar
        search={draft}
        onSearchChange={onSearch}
        placeholder="Search sessions…"
        filters={
          <>
            <FilterPill
              label="Harness"
              options={harnessOptions}
              selected={harness}
              onChange={(next) => setParam('harness', next.join(','))}
            />
            <FilterPill
              label="Status"
              options={statusOptions}
              selected={status}
              onChange={(next) => setParam('status', next.join(','))}
            />
          </>
        }
        actions={
          <>
            <span className="hidden text-xs text-muted-foreground xl:inline">
              Sessions agent harnesses ran through the gateway
            </span>
            <RefreshButton onClick={reload} loading={loading && runs.length > 0} />
          </>
        }
      />
      <DataTable
        data={rows}
        columns={columns}
        getRowId={(run) => run.session_id}
        loading={loading && runs.length === 0}
        onRowClick={(run) => navigate(getRunPath(run.session_id))}
        rowActions={(run) => (
          <>
            <DropdownMenuItem onSelect={() => navigate(getRunPath(run.session_id))}>
              <PanelRightOpen />
              Open run
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => navigate(getSessionRequestsPath(run.session_id))}>
              <ListTree />
              Show its requests
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onSelect={() => copy(run.session_id, 'Session ID')}>
              <Copy />
              Copy session ID
            </DropdownMenuItem>
          </>
        )}
        empty={
          runs.length === 0 && !search ? (
            <EmptyState
              compact
              icon={Bot}
              tone="blue"
              title="No agent runs yet"
              description="Point Claude Code, Codex, OpenCode or DeepSeek Harness at the gateway and each session shows up here as one run, its subagents under it."
              actions={
                <Button size="sm" variant="outline" asChild>
                  <a
                    href="https://vllm-sr.ai/docs/installation/agent-harness"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Connect an agent harness
                  </a>
                </Button>
              }
            />
          ) : (
            <EmptyState
              compact
              icon={Bot}
              tone="blue"
              title="No agent runs match"
              description="Clear the search or the filters to see every run."
              actions={
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setDraft('')
                    setParams({ tab: 'runs' }, { replace: true })
                  }}
                >
                  Clear filters
                </Button>
              }
            />
          )
        }
      />
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-subtle-foreground">
        <span className="tabular">
          {rows.length.toLocaleString()} of {total.toLocaleString()} run{total === 1 ? '' : 's'}
          {harness.length || status.length ? ' · filters read the runs loaded so far' : ''}
        </span>
        {runs.length < total ? (
          <Button variant="outline" size="xs" onClick={loadMore} loading={loading}>
            Load older runs
          </Button>
        ) : null}
      </div>
    </>
  )
}
