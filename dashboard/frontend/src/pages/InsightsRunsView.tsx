import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Bot, Copy, Info, ListTree, PanelRightOpen } from 'lucide-react'

import {
  Button,
  Card,
  DataTable,
  DataToolbar,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  Segmented,
  StatusLabel,
  Tooltip,
  cn,
  toast,
  type Column,
  type FilterOption,
} from '../ui'
import { formatInsightsCount } from '../utils/insightsCost'

import { runEventCounts } from './insightsAgentEvents'
import { AgentEventChips } from './InsightsAgentEvents'
import { compactSessionId, displaySessionId } from './insightsAgentIdentity'
import { formatLatency, useNow } from './insightsDisplayFormat'
import {
  InsightsLoadError,
  InsightsUnavailable,
  LiveToggle,
  Pager,
  RelativeTime,
} from './InsightsDisplay'
import { HarnessLogo } from './InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo, PRIMARY_HARNESSES, sortHarnessIds } from './insightsHarness'
import { readPage, readPageSize, writePage } from './insightsListData'
import { ModelChip, modelText, useModelLabels } from './insightsModelLabelSupport'
import { RANGE_TEXT, REQUEST_RANGES } from './insightsRequestsData'
import { getSessionRequestsPath } from './insightsReviewSupport'
import {
  API_RUN_FILTER,
  hasRunClientFilters,
  readRunFilters,
  RUN_WINDOW_SIZE,
  runHarnessValue,
  useRunsData,
  writeRunFilters,
  type RunFilters,
} from './insightsRunsData'
import {
  countText,
  formatRunCost,
  getRunPath,
  oneLine,
  RUN_OUTCOME_LABELS,
  RUN_STATUS_TONES,
  runHeadline,
} from './insightsRunSupport'
import type { RunSummary } from './insightsRunTypes'

export function RunStatus({ run }: { run: Pick<RunSummary, 'outcome'> }) {
  const status = run.outcome.status
  return (
    <StatusLabel tone={RUN_STATUS_TONES[status] ?? 'neutral'} pulse={status === 'in_progress'}>
      {RUN_OUTCOME_LABELS[status] ?? status}
    </StatusLabel>
  )
}

/** A run's events worth a look, in the shared event vocabulary, each with its count. */
export function RunEvents({ events }: { events?: Record<string, number> }) {
  const counts = runEventCounts({ events })
  if (counts.length === 0) return null
  return <AgentEventChips events={counts} compact />
}

function copy(value: string, label: string) {
  void navigator.clipboard?.writeText(value).catch(() => undefined)
  toast.success(`${label} copied`, { description: value })
}

function countBy<T>(rows: T[], valueOf: (row: T) => string) {
  const counts = new Map<string, number>()
  for (const row of rows) counts.set(valueOf(row), (counts.get(valueOf(row)) ?? 0) + 1)
  return counts
}

// The task a run was given, when its title is the session's name, for the row's tooltip.
function taskHint(run: RunSummary): string | undefined {
  const task = run.prompt?.trim()
  return run.session_name && task ? oneLine(task, 280) : undefined
}

// One row per agent run, the session a harness names: its title, harness,
// agents, models, what it cost and took, and how it ended. The runs load
// when this tab opens, a page at a time, bounded by the time range.
export default function InsightsRunsView() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const labels = useModelLabels()
  const now = useNow(30_000)
  const filters = useMemo(() => readRunFilters(params), [params])
  const page = readPage(params)
  const pageSize = readPageSize(params)
  const [draft, setDraft] = useState(filters.search)
  const [live, setLive] = useState(true)
  const [held, setHeld] = useState(false)
  const [facets, setFacets] = useState(false)
  const data = useRunsData({ filters, page, pageSize, live, held, withWindow: facets })
  const clientMode = hasRunClientFilters(filters)
  const wantFacets = () => setFacets(true)

  const update = (patch: Partial<RunFilters>) =>
    setParams((current) => writeRunFilters(current, patch), { replace: true })

  const timer = useRef<number>(undefined)
  useEffect(() => () => window.clearTimeout(timer.current), [])
  useEffect(() => {
    setDraft((current) => (current.trim() === filters.search ? current : filters.search))
  }, [filters.search])
  const onSearch = (value: string) => {
    setDraft(value)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => update({ search: value.trim() }), 300)
  }

  const pages = Math.max(1, Math.ceil(data.total / pageSize))
  useEffect(() => {
    if (!data.loading && !data.stale && data.total > 0 && page > pages) {
      setParams((current) => writePage(current, pages), { replace: true })
    }
  }, [data.loading, data.stale, data.total, page, pages, setParams])

  const counted = data.windowLoaded
  const harnessCounts = useMemo(() => countBy(data.window, runHarnessValue), [data.window])
  const statusCounts = useMemo(
    () => countBy(data.window, (run) => run.outcome.status),
    [data.window],
  )
  const harnessOptions: FilterOption[] = useMemo(() => {
    const seen = counted ? [...harnessCounts.keys()] : [...PRIMARY_HARNESSES, API_RUN_FILTER]
    const shown = new Set([...seen, ...filters.harness])
    const options: FilterOption[] = sortHarnessIds(
      [...shown].filter((id) => id !== API_RUN_FILTER),
    ).map((id) => ({
      value: id,
      label: harnessInfo(id)?.label ?? id,
      icon: <HarnessLogo harness={id} size={14} />,
      count: counted ? (harnessCounts.get(id) ?? 0) : undefined,
    }))
    if (shown.has(API_RUN_FILTER)) {
      options.push({
        value: API_RUN_FILTER,
        label: API_CLIENT_LABEL,
        icon: <HarnessLogo size={14} />,
        count: counted ? (harnessCounts.get(API_RUN_FILTER) ?? 0) : undefined,
      })
    }
    return options
  }, [counted, filters.harness, harnessCounts])
  const statusOptions: FilterOption[] = Object.entries(RUN_OUTCOME_LABELS).map(
    ([value, label]) => ({
      value,
      label,
      count: counted ? (statusCounts.get(value) ?? 0) : undefined,
    }),
  )

  const columns: Column<RunSummary>[] = [
    {
      id: 'session',
      header: 'Session',
      className: 'w-full max-w-0',
      sortValue: (run) => runHeadline(run),
      cell: (run) => {
        const info = harnessInfo(run.harness)
        return (
          <div className="flex min-w-0 items-center gap-3">
            <HarnessLogo harness={run.harness} size={16} framed />
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="truncate font-medium" title={taskHint(run)}>
                {runHeadline(run)}
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
          content={
            run.agents.length > 6
              ? `${countText(run.agents.length, 'agent')}: the main agent and ${countText(run.agents.length - 1, 'subagent')}`
              : run.agents
                  .map(
                    (agent) =>
                      `${agent.label}${agent.kind ? ` (${agent.kind})` : ''} · ${countText(agent.calls, 'call')}`,
                  )
                  .join('  ·  ')
          }
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
      cell: (run) => (
        <span className="font-mono text-xs tabular">{run.calls.toLocaleString()}</span>
      ),
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
      id: 'active',
      header: 'Last call',
      width: '96px',
      hideBelow: 'md',
      sortValue: (run) => Date.parse(run.ended_at),
      cell: (run) => (
        <RelativeTime iso={run.ended_at} now={now} className="text-xs text-muted-foreground" />
      ),
    },
  ]

  if (data.unavailable) {
    return (
      <Card className="border-dashed shadow-none">
        <InsightsUnavailable reason={data.unavailable} noun="agent runs" onRetry={data.refresh} />
      </Card>
    )
  }

  const range = RANGE_TEXT[filters.range]
  const quiet = !data.loading && !data.error && data.total === 0 && !filters.search && !clientMode
  const clientNote = !clientMode
    ? null
    : !counted
      ? `Harness and status filters run in your browser over the latest ${RUN_WINDOW_SIZE} runs ${range}.`
      : data.windowTotal > data.window.length
        ? `Harness and status filters run in your browser over the latest ${formatInsightsCount(data.window.length)} of ${formatInsightsCount(data.windowTotal)} runs ${range}. Narrow the time range to cover them all.`
        : `Harness and status filters run in your browser over all ${formatInsightsCount(data.windowTotal)} runs ${range}.`

  return (
    <>
      {data.error ? (
        <InsightsLoadError
          title="Agent runs could not load"
          message={data.error}
          onRetry={data.refresh}
        />
      ) : null}
      <DataToolbar
        search={draft}
        onSearchChange={onSearch}
        placeholder="Search sessions…"
        filters={
          <div
            role="group"
            aria-label="Harness and status filters"
            className="flex items-center gap-2"
            onPointerOver={wantFacets}
            onFocus={wantFacets}
          >
            <FilterPill
              label="Harness"
              options={harnessOptions}
              selected={filters.harness}
              onChange={(harness) => update({ harness })}
            />
            <FilterPill
              label="Status"
              options={statusOptions}
              selected={filters.status}
              onChange={(status) => update({ status })}
            />
          </div>
        }
        actions={
          <>
            <Segmented
              aria-label="Time range"
              value={filters.range}
              onValueChange={(next) => update({ range: next })}
              options={REQUEST_RANGES}
            />
            <LiveToggle
              live={live}
              held={held}
              every="every minute"
              onLiveChange={(next) => {
                setLive(next)
                if (next) data.refresh()
              }}
            />
          </>
        }
      />
      {clientNote ? (
        <p className="mb-3 flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
          <Info className="size-3.5 shrink-0 text-subtle-foreground" />
          <span className="min-w-0">{clientNote}</span>
        </p>
      ) : null}
      <div
        onPointerEnter={() => setHeld(true)}
        onPointerLeave={() => setHeld(false)}
        aria-busy={data.stale || undefined}
        className={cn('transition-opacity duration-200', data.stale && 'opacity-60')}
      >
        <DataTable
          data={data.rows}
          columns={columns}
          getRowId={(run) => run.session_id}
          loading={data.loading && data.rows.length === 0}
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
            quiet && filters.range !== 'all' ? (
              <EmptyState
                compact
                icon={Bot}
                tone="blue"
                title={`No agent runs ${range}`}
                description="No agent harness made a call through the gateway in this range. Older runs are under a longer range."
                actions={
                  <Button size="sm" variant="outline" onClick={() => update({ range: 'all' })}>
                    Show all time
                  </Button>
                }
              />
            ) : quiet ? (
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
                description="Clear the search or the filters, or widen the time range."
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
      </div>
      <Pager
        page={Math.min(page, pages)}
        pageSize={pageSize}
        total={data.total}
        noun="run"
        note={
          filters.range === 'all' || data.total === 0
            ? undefined
            : `Calls, cost and duration count what each run did ${range}.`
        }
        onPage={(next) => setParams((current) => writePage(current, next))}
        onPageSize={(size) => setParams((current) => writePage(current, 1, size))}
      />
    </>
  )
}
