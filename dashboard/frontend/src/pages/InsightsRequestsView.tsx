import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  AlertTriangle,
  Bot,
  ChartColumn,
  CircleDollarSign,
  Copy,
  Filter,
  GitBranch,
  ListTree,
  MessagesSquare,
  PanelRightOpen,
  PiggyBank,
  Workflow,
  X,
  Zap,
} from 'lucide-react'

import {
  Button,
  Card,
  DataTable,
  DataToolbar,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  Label,
  ProviderLogo,
  Segmented,
  Skeleton,
  StatusDot,
  Switch,
  Tooltip,
  cn,
  toast,
  type Column,
  type FilterOption,
} from '../ui'
import {
  formatInsightsCompactCount,
  formatInsightsCost,
  formatInsightsCount,
  formatInsightsShare,
} from '../utils/insightsCost'

import {
  compactIdentifier,
  compactSessionId,
  displaySessionId,
  housekeepingDetail,
  REQUEST_CLASSES,
  REQUEST_CLASS_LABELS,
  requestClassOf,
  spawnsSubagent,
  subagentKindLabel,
  traceToolNames,
  type RequestClass,
} from './insightsAgentIdentity'
import { modelProviderId, useInsightsConfig } from './insightsConfigFacts'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { formatLatency, formatTokenCount, useNow } from './insightsDisplayFormat'
import {
  InsightsLoadError,
  InsightsUnavailable,
  LifecycleStatus,
  LiveToggle,
  Pager,
  RefreshButton,
  RelativeTime,
} from './InsightsDisplay'
import { HarnessLogo, ToolChips } from './InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo, sortHarnessIds } from './insightsHarness'
import { ModelChip, modelText, useModelLabels } from './insightsModelLabelSupport'
import {
  formatInsightsDecisionName,
  getInsightsCostUnavailableReason,
  getInsightsRecordPath,
  hasCompleteCostData,
} from './insightsPageSupport'
import type { InsightsAggregateResponse, InsightsRecord } from './insightsPageTypes'
import {
  API_CLIENT_FILTER,
  countFacet,
  hasAnyFilter,
  hasClientFilters,
  readRequestFilters,
  recordClientFilterValue,
  REQUESTS_PAGE_SIZE,
  useRecentRuns,
  useRequestsData,
  writeRequestFilters,
  type RequestFilters,
  type RequestRange,
} from './insightsRequestsData'
import { compactID } from './insightsReviewSupport'
import { countText, getRunPath } from './insightsRunSupport'
import type { RunSummary } from './insightsRunTypes'

const InsightsCharts = lazy(() => import('../components/InsightsCharts'))

const BREAKDOWN_KEY = 'vsr-requests-breakdown'

function readBreakdown(): boolean {
  try {
    return window.localStorage.getItem(BREAKDOWN_KEY) === '1'
  } catch {
    return false
  }
}

function copyText(value: string, label: string) {
  void navigator.clipboard?.writeText(value).catch(() => undefined)
  toast.success(`${label} copied`, { description: value })
}

// ---------------------------------------------------------------- summary

function SummaryTile({
  label,
  icon,
  value,
  hint,
  tone,
}: {
  label: string
  icon: ReactNode
  value: ReactNode
  hint?: ReactNode
  tone?: 'negative'
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5 px-4 py-3">
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground [&_svg]:size-3.5">
        {icon}
        {label}
      </span>
      <span
        className={cn(
          'truncate text-lg font-semibold leading-none tracking-[-0.02em] tabular',
          tone === 'negative' && 'text-destructive-foreground',
        )}
      >
        {value}
      </span>
      {hint ? <span className="truncate text-[11.5px] text-subtle-foreground">{hint}</span> : null}
    </div>
  )
}

function spendOf(aggregate: InsightsAggregateResponse) {
  const summary = aggregate.summary
  const groups = summary.by_currency?.length ? summary.by_currency : [summary]
  const lead = groups[0]
  return {
    spend: formatInsightsCost(lead.total_spend ?? lead.actual_spend, lead.currency, {
      total: true,
    }),
    saved:
      lead.cost_record_count > 0
        ? formatInsightsCost(lead.total_saved, lead.currency, { total: true })
        : 'N/A',
    savedShare:
      lead.cost_record_count > 0 && lead.baseline_spend > 0
        ? `${formatInsightsShare(lead.total_saved, lead.baseline_spend)} of the baseline`
        : 'No priced baseline yet',
    negative: lead.cost_record_count > 0 && lead.total_saved < 0,
    more:
      groups.length > 1
        ? `+${groups.length - 1} more currenc${groups.length > 2 ? 'ies' : 'y'}`
        : undefined,
  }
}

function RequestsSummary({
  aggregate,
  loading,
  breakdown,
  onBreakdown,
  clientFiltered,
}: {
  aggregate: InsightsAggregateResponse | null
  loading: boolean
  breakdown: boolean
  onBreakdown: (open: boolean) => void
  clientFiltered: boolean
}) {
  if (!aggregate) {
    if (!loading) return null
    return (
      <Card className="mb-4 grid grid-cols-2 gap-px overflow-hidden sm:grid-cols-3 lg:grid-cols-5">
        {Array.from({ length: 5 }, (_, index) => (
          <div key={index} className="flex flex-col gap-2 px-4 py-3">
            <Skeleton className="h-3 w-20" />
            <Skeleton className="h-5 w-24" />
          </div>
        ))}
      </Card>
    )
  }
  const lifecycle = aggregate.lifecycle
  const failed = lifecycle.failed + lifecycle.aborted
  const money = spendOf(aggregate)
  const tokens = aggregate.token_volume
  return (
    <section aria-label="Requests summary" className="mb-4">
      <Card className="grid grid-cols-2 gap-px overflow-hidden bg-border sm:grid-cols-3 xl:grid-cols-[repeat(5,minmax(0,1fr))_auto] [&>*]:bg-card">
        <SummaryTile
          label="Requests"
          icon={<Zap className="text-chart-1" />}
          value={formatInsightsCount(aggregate.record_count)}
          hint={
            lifecycle.in_progress > 0
              ? `${formatInsightsCount(lifecycle.in_progress)} running`
              : `${formatInsightsCount(lifecycle.completed)} completed`
          }
        />
        <SummaryTile
          label="Estimated spend"
          icon={<CircleDollarSign className="text-chart-6" />}
          value={money.spend}
          hint={money.more ?? 'Recorded tokens × configured rates'}
        />
        <SummaryTile
          label="Saved by routing"
          icon={<PiggyBank className="text-chart-2" />}
          value={money.saved}
          hint={money.savedShare}
          tone={money.negative ? 'negative' : undefined}
        />
        <SummaryTile
          label="Tokens"
          icon={<MessagesSquare className="text-chart-4" />}
          value={formatInsightsCompactCount(tokens.total_tokens)}
          hint={`${formatInsightsCompactCount(tokens.input_tokens)} in · ${formatInsightsCompactCount(tokens.output_tokens)} out`}
        />
        <SummaryTile
          label="Failed"
          icon={<AlertTriangle className="text-chart-5" />}
          value={
            aggregate.record_count > 0 ? formatInsightsShare(failed, aggregate.record_count) : '0%'
          }
          hint={`${formatInsightsCount(failed)} failed or aborted`}
          tone={failed > 0 ? 'negative' : undefined}
        />
        <div className="flex items-center justify-end px-3 py-2 max-sm:col-span-full">
          <Button
            variant={breakdown ? 'secondary' : 'ghost'}
            size="sm"
            aria-expanded={breakdown}
            aria-controls="requests-breakdown"
            onClick={() => onBreakdown(!breakdown)}
          >
            <ChartColumn />
            Breakdown
          </Button>
        </div>
      </Card>
      {clientFiltered ? (
        <p className="mt-1.5 text-[11.5px] text-subtle-foreground">
          Totals cover every client and request class the other filters select.
        </p>
      ) : null}
      {breakdown ? (
        <div id="requests-breakdown" className="mt-4">
          <Suspense fallback={<Skeleton className="h-48 w-full rounded-xl" />}>
            <InsightsCharts aggregate={aggregate} />
          </Suspense>
        </div>
      ) : null}
    </section>
  )
}

// ---------------------------------------------------------------- cells

function Dim({ on, children }: { on: boolean; children: ReactNode }) {
  return <div className={cn('min-w-0', on && 'opacity-55')}>{children}</div>
}

function classLine(record: InsightsRecord, requestClass: RequestClass, detail?: string) {
  const turn = `turn ${record.turn_index + 1}`
  switch (requestClass) {
    case 'main':
      return `main agent · ${turn}`
    case 'subagent': {
      const kind = subagentKindLabel(record.agent?.kind)
      return `subagent${kind ? ` · ${kind}` : ''} · ${turn}`
    }
    case 'housekeeping':
      return `housekeeping${detail ? ` · ${detail.toLowerCase().split(' · ')[0]}` : ''}`
    case 'review':
      return 'approval review'
    case 'router':
      return 'router call'
    default:
      return record.session_id ? `conversation · ${turn}` : 'no session'
  }
}

function ClientCell({
  record,
  requestClass,
  detail,
}: {
  record: InsightsRecord
  requestClass: RequestClass
  detail?: string
}) {
  const info = harnessInfo(record.agent?.harness)
  const spawned = spawnsSubagent(record)
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <span className="relative">
        <HarnessLogo harness={record.agent?.harness} size={16} framed />
        {requestClass === 'subagent' ? (
          <span
            aria-hidden
            className="absolute -bottom-1 -right-1 grid size-3.5 place-items-center rounded-full border border-border bg-surface text-subtle-foreground"
          >
            <GitBranch className="size-2.5" />
          </span>
        ) : null}
      </span>
      <div className="flex min-w-0 flex-col gap-0.5 max-xl:hidden">
        <span className="truncate text-[13px] font-medium">{info?.label ?? API_CLIENT_LABEL}</span>
        <span className="flex min-w-0 items-center gap-1 text-xs text-subtle-foreground">
          <span className="truncate" data-request-class={requestClass}>
            {classLine(record, requestClass, detail)}
          </span>
          {spawned ? (
            <Tooltip content="This call started a subagent">
              <Workflow aria-label="Started a subagent" className="size-3 shrink-0 text-chart-3" />
            </Tooltip>
          ) : null}
        </span>
      </div>
    </div>
  )
}

function SessionCell({
  record,
  run,
  onFilter,
}: {
  record: InsightsRecord
  run?: RunSummary
  onFilter: (session: string) => void
}) {
  const session = record.session_id
  if (!session) {
    return (
      <div className="flex min-w-0 flex-col gap-0.5">
        <span
          className="truncate font-mono text-[12px] text-foreground/85"
          title={record.request_id || record.id}
        >
          {compactIdentifier(record.request_id || record.id, 10, 6)}
        </span>
        <span className="text-xs text-subtle-foreground">
          {record.recipe || 'default'} · no session
        </span>
      </div>
    )
  }
  const title = run?.session_name || record.session_name
  const shown = displaySessionId(session)
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation()
          onFilter(session)
        }}
        title={`Show only this session · ${shown}`}
        className="w-fit max-w-full truncate rounded-sm text-left text-[13px] text-foreground decoration-border-strong underline-offset-4 hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
      >
        {title ? title : <span className="font-mono text-[12px]">{compactSessionId(session)}</span>}
      </button>
      <span className="flex min-w-0 items-center gap-1.5 text-xs text-subtle-foreground">
        <span className="truncate font-mono text-[11.5px]">
          {title ? compactSessionId(session) : record.recipe || 'default'}
        </span>
        {title ? (
          <>
            <span aria-hidden>·</span>
            <span className="shrink-0">{record.recipe || 'default'}</span>
          </>
        ) : null}
      </span>
    </div>
  )
}

function CostCell({ record }: { record: InsightsRecord }) {
  const labels = useModelLabels()
  if (!hasCompleteCostData(record)) {
    const reason = getInsightsCostUnavailableReason(record, labels)
    if (typeof record.actual_cost === 'number' && record.currency) {
      return (
        <Tooltip content={reason ?? 'Baseline not recorded'}>
          <span className="font-mono text-xs tabular">
            {formatInsightsCost(record.actual_cost, record.currency)}
          </span>
        </Tooltip>
      )
    }
    return (
      <Tooltip content={reason ?? 'No price recorded'}>
        <span className="text-xs text-subtle-foreground">—</span>
      </Tooltip>
    )
  }
  const saved = record.cost_savings ?? 0
  return (
    <Tooltip
      content={`Saved ${formatInsightsCost(saved, record.currency)} against ${modelText(labels, record.baseline_model ?? '')}`}
    >
      <span className="font-mono text-xs tabular">
        {formatInsightsCost(record.actual_cost, record.currency)}
      </span>
    </Tooltip>
  )
}

// ---------------------------------------------------------------- filters

function single(selected: string[], current: string): string {
  return selected.filter((value) => value !== current).pop() ?? ''
}

const RANGE_OPTIONS: { value: RequestRange; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: '1h', label: '1h' },
  { value: '24h', label: '24h' },
  { value: '7d', label: '7d' },
]

function ActiveChip({
  label,
  value,
  title,
  onClear,
  action,
}: {
  label: string
  value: string
  title?: string
  onClear: () => void
  action?: ReactNode
}) {
  return (
    <span
      className="inline-flex h-7 max-w-full items-center gap-1.5 rounded-md border border-border bg-surface pl-2 pr-1 text-xs shadow-xs"
      title={title}
    >
      <span className="text-subtle-foreground">{label}</span>
      <span className="truncate font-medium text-foreground">{value}</span>
      {action}
      <button
        type="button"
        aria-label={`Clear ${label.toLowerCase()} filter`}
        onClick={onClear}
        className="grid size-5 place-items-center rounded text-subtle-foreground hover:bg-accent hover:text-foreground"
      >
        <X className="size-3.5" />
      </button>
    </span>
  )
}

// ---------------------------------------------------------------- view

export default function InsightsRequestsView() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const filters = useMemo(() => readRequestFilters(params), [params])
  const page = Math.max(1, Number(params.get('page')) || 1)
  const { agentRules, modelProviders } = useInsightsConfig()
  const labels = useModelLabels()
  const [live, setLive] = useState(false)
  const [held, setHeld] = useState(false)
  const [breakdown, setBreakdown] = useState(readBreakdown)
  const [searchDraft, setSearchDraft] = useState(filters.search)
  const [runsKey, setRunsKey] = useState(0)
  const now = useNow(live ? 5_000 : 30_000)

  const data = useRequestsData({ filters, page, live, held, rules: agentRules })
  // new requests can open sessions, so the run titles follow the window
  const runs = useRecentRuns(
    data.window.some((record) => record.session_id),
    `${runsKey}:${data.window[0]?.id ?? ''}`,
  )
  const clientMode = hasClientFilters(filters)

  const update = useCallback(
    (patch: Partial<RequestFilters>) =>
      setParams((current) => writeRequestFilters(current, patch), { replace: true }),
    [setParams],
  )

  // the search box writes the URL after a pause, as the old list did
  const searchTimer = useRef<number>(undefined)
  useEffect(() => () => window.clearTimeout(searchTimer.current), [])
  useEffect(() => {
    setSearchDraft((draft) => (draft.trim() === filters.search ? draft : filters.search))
  }, [filters.search])
  const onSearch = (value: string) => {
    setSearchDraft(value)
    window.clearTimeout(searchTimer.current)
    searchTimer.current = window.setTimeout(() => update({ search: value.trim() }), 300)
  }

  const setBreakdownOpen = (open: boolean) => {
    setBreakdown(open)
    try {
      window.localStorage.setItem(BREAKDOWN_KEY, open ? '1' : '0')
    } catch {
      // the choice lasts for this page only
    }
  }

  const refresh = () => {
    data.refresh()
    setRunsKey((key) => key + 1)
  }

  const harnessCounts = useMemo(
    () => countFacet(data.window, recordClientFilterValue),
    [data.window],
  )
  const classCounts = useMemo(
    () => countFacet(data.window, (record) => requestClassOf(record, agentRules)),
    [agentRules, data.window],
  )
  const harnessOptions: FilterOption[] = useMemo(() => {
    const ids = sortHarnessIds(
      [...harnessCounts.keys(), ...filters.harness].filter((id) => id !== API_CLIENT_FILTER),
    )
    const options = ids.map((id) => ({
      value: id,
      label: harnessInfo(id)?.label ?? id,
      icon: <HarnessLogo harness={id} size={14} />,
      count: harnessCounts.get(id) ?? 0,
    }))
    if (harnessCounts.has(API_CLIENT_FILTER) || filters.harness.includes(API_CLIENT_FILTER)) {
      options.push({
        value: API_CLIENT_FILTER,
        label: API_CLIENT_LABEL,
        icon: <HarnessLogo harness={undefined} size={14} />,
        count: harnessCounts.get(API_CLIENT_FILTER) ?? 0,
      })
    }
    return options
  }, [filters.harness, harnessCounts])
  const classOptions: FilterOption[] = REQUEST_CLASSES.filter(
    (value) => classCounts.has(value) || filters.classes.includes(value),
  ).map((value) => ({
    value,
    label: REQUEST_CLASS_LABELS[value],
    count: classCounts.get(value) ?? 0,
  }))

  const aggregate = data.aggregate
  const columns: Column<InsightsRecord>[] = [
    {
      id: 'time',
      header: 'Time',
      width: '88px',
      sortValue: (row) => Date.parse(row.timestamp),
      cell: (row) => (
        <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
          {now - Date.parse(row.timestamp) < 15_000 ? <StatusDot tone="info" /> : null}
          <RelativeTime iso={row.timestamp} now={now} />
        </span>
      ),
    },
    {
      id: 'client',
      header: 'Client',
      className: 'xl:max-w-[192px]',
      headerClassName: 'xl:w-[192px]',
      cell: (row) => {
        const requestClass = requestClassOf(row, agentRules)
        return (
          <Dim on={requestClass === 'housekeeping'}>
            <ClientCell
              record={row}
              requestClass={requestClass}
              detail={housekeepingDetail(row, agentRules)}
            />
          </Dim>
        )
      },
    },
    {
      id: 'session',
      header: 'Session',
      className: 'w-full max-w-0 min-w-[180px]',
      cell: (row) => (
        <Dim on={requestClassOf(row, agentRules) === 'housekeeping'}>
          <SessionCell
            record={row}
            run={row.session_id ? runs.get(row.session_id) : undefined}
            onFilter={(session) => update({ session })}
          />
        </Dim>
      ),
    },
    {
      id: 'decision',
      header: 'Decision',
      className: 'max-[1399px]:hidden',
      headerClassName: 'max-[1399px]:hidden',
      cell: (row) =>
        row.decision ? (
          <InsightsDecisionBadge
            decision={row.decision}
            label={formatInsightsDecisionName(row.decision)}
            className="max-w-[150px]"
          />
        ) : (
          <span className="text-xs text-subtle-foreground">Unresolved</span>
        ),
    },
    {
      id: 'model',
      header: 'Model',
      cell: (row) => (
        <Dim on={requestClassOf(row, agentRules) === 'housekeeping'}>
          <ModelChip
            name={row.selected_model}
            fallback="No backend"
            className="max-w-[170px] text-[13px]"
          />
        </Dim>
      ),
    },
    {
      id: 'tools',
      header: 'Tools',
      hideBelow: '2xl',
      cell: (row) => {
        const names = traceToolNames(row.tool_trace)
        return names.length ? (
          <ToolChips names={names} max={2} />
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        )
      },
    },
    {
      id: 'latency',
      header: 'Latency',
      align: 'right',
      hideBelow: 'md',
      sortValue: (row) => row.duration_ms ?? -1,
      cell: (row) => (
        <span className="font-mono text-xs tabular">{formatLatency(row.duration_ms)}</span>
      ),
    },
    {
      id: 'tokens',
      header: 'Tokens',
      align: 'right',
      hideBelow: '2xl',
      sortValue: (row) => row.total_tokens ?? -1,
      cell: (row) =>
        typeof row.prompt_tokens === 'number' || typeof row.completion_tokens === 'number' ? (
          <Tooltip
            content={`${(row.prompt_tokens ?? 0).toLocaleString()} in · ${(row.completion_tokens ?? 0).toLocaleString()} out`}
          >
            <span className="font-mono text-xs tabular">
              {formatTokenCount(row.prompt_tokens)}
              <span className="text-subtle-foreground"> / </span>
              {formatTokenCount(row.completion_tokens)}
            </span>
          </Tooltip>
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        ),
    },
    {
      id: 'cost',
      header: 'Cost',
      align: 'right',
      hideBelow: 'xl',
      sortValue: (row) => row.actual_cost ?? -1,
      cell: (row) => <CostCell record={row} />,
    },
    {
      id: 'status',
      header: 'Status',
      width: '76px',
      cell: (row) => (
        <span className="inline-flex items-center gap-1">
          <LifecycleStatus record={row} />
          {row.from_cache ? (
            <Tooltip content="Answered from the Router cache">
              <span className="rounded-full bg-info-soft px-1.5 text-[10.5px] font-medium leading-4 text-info-foreground">
                cache
              </span>
            </Tooltip>
          ) : null}
        </span>
      ),
    },
  ]

  const unfilteredEmpty =
    !data.loading && !hasAnyFilter(filters) && data.total === 0 && !data.unavailable && !data.error
  const windowNote =
    clientMode && data.windowTotal > data.window.length
      ? `Client and class filters read the newest ${formatInsightsCount(data.window.length)} of ${formatInsightsCount(data.windowTotal)} matching requests.`
      : undefined

  if (data.unavailable) {
    return (
      <Card className="border-dashed shadow-none">
        <InsightsUnavailable reason={data.unavailable} onRetry={refresh} />
      </Card>
    )
  }

  const sessionRun = filters.session ? runs.get(filters.session) : undefined

  return (
    <>
      {data.error ? (
        <InsightsLoadError title="Requests could not load" message={data.error} onRetry={refresh} />
      ) : null}
      <RequestsSummary
        aggregate={
          aggregate && (aggregate.record_count > 0 || hasAnyFilter(filters)) ? aggregate : null
        }
        loading={data.loading}
        breakdown={breakdown}
        onBreakdown={setBreakdownOpen}
        clientFiltered={clientMode}
      />
      <DataToolbar
        search={searchDraft}
        onSearchChange={onSearch}
        placeholder="Search request, recipe or session id…"
        filters={
          <>
            <FilterPill
              label="Client"
              options={harnessOptions}
              selected={filters.harness}
              onChange={(harness) => update({ harness })}
            />
            <FilterPill
              label="Class"
              options={classOptions}
              selected={filters.classes}
              onChange={(classes) => update({ classes: classes as RequestClass[] })}
            />
            <FilterPill
              label="Decision"
              options={(aggregate?.available_decisions ?? []).map((value) => ({
                value,
                label: formatInsightsDecisionName(value),
              }))}
              selected={filters.decision ? [filters.decision] : []}
              onChange={(next) => update({ decision: single(next, filters.decision) })}
            />
            <FilterPill
              label="Model"
              options={(aggregate?.available_models ?? []).map((value) => {
                const provider = modelProviderId(value, modelProviders)
                return {
                  value,
                  label: modelText(labels, value),
                  icon: provider ? <ProviderLogo provider={provider} size={14} /> : undefined,
                }
              })}
              selected={filters.model ? [filters.model] : []}
              onChange={(next) => update({ model: single(next, filters.model) })}
            />
            {(aggregate?.available_recipes.length ?? 0) > 1 || filters.recipe ? (
              <FilterPill
                label="Recipe"
                options={(aggregate?.available_recipes ?? []).map((value) => ({
                  value,
                  label: value,
                }))}
                selected={filters.recipe ? [filters.recipe] : []}
                onChange={(next) => update({ recipe: single(next, filters.recipe) })}
              />
            ) : null}
            <FilterPill
              label="Cache"
              options={[
                { value: 'cached', label: 'Router cache hits' },
                { value: 'streamed', label: 'Streamed answers' },
              ]}
              selected={filters.cache === 'all' ? [] : [filters.cache]}
              onChange={(next) =>
                update({ cache: (single(next, filters.cache) || 'all') as RequestFilters['cache'] })
              }
            />
          </>
        }
        actions={
          <>
            <Segmented
              aria-label="Time range"
              value={filters.range}
              onValueChange={(range) => update({ range })}
              options={RANGE_OPTIONS}
            />
            <LiveToggle live={live} onLiveChange={setLive} held={held} />
            <RefreshButton onClick={refresh} loading={data.loading && data.rows.length > 0} />
          </>
        }
      />
      <div className="mb-3 flex min-h-7 flex-wrap items-center gap-2">
        {filters.session ? (
          <ActiveChip
            label="Session"
            value={sessionRun?.session_name || compactSessionId(filters.session)}
            title={displaySessionId(filters.session)}
            onClear={() => update({ session: '' })}
            action={
              <Button
                variant="ghost"
                size="xs"
                className="h-5 px-1.5"
                onClick={() => navigate(getRunPath(filters.session))}
              >
                <Bot />
                Open run
              </Button>
            }
          />
        ) : null}
        {filters.turn ? (
          <ActiveChip
            label="Codex turn"
            value={compactID(filters.turn)}
            title={filters.turn}
            onClear={() => update({ turn: '' })}
          />
        ) : null}
        {!filters.session && !filters.turn && data.window.length > 0 ? (
          <span className="text-xs text-subtle-foreground tabular">
            {countText(
              new Set(data.window.map((record) => record.session_id).filter(Boolean)).size,
              'session',
            )}{' '}
            from {countText(harnessCounts.size, 'client')} in the newest{' '}
            {formatInsightsCount(data.window.length)} requests
          </span>
        ) : null}
        <span className="ml-auto flex items-center gap-2">
          <Switch
            id="requests-hide-housekeeping"
            size="sm"
            checked={filters.hideHousekeeping}
            onCheckedChange={(checked) => update({ hideHousekeeping: checked })}
          />
          <Label
            htmlFor="requests-hide-housekeeping"
            className="whitespace-nowrap text-[13px] font-normal text-muted-foreground"
          >
            Hide housekeeping
            {classCounts.get('housekeeping') ? (
              <span className="ml-1 font-mono text-[11px] text-subtle-foreground">
                {classCounts.get('housekeeping')}
              </span>
            ) : null}
          </Label>
        </span>
      </div>
      <div onPointerEnter={() => setHeld(true)} onPointerLeave={() => setHeld(false)}>
        <DataTable
          data={data.rows}
          columns={columns}
          getRowId={(row) => row.id}
          loading={data.loading && data.rows.length === 0}
          onRowClick={(row) => navigate(getInsightsRecordPath(row.id))}
          rowActions={(row) => (
            <>
              <DropdownMenuItem onSelect={() => navigate(getInsightsRecordPath(row.id))}>
                <PanelRightOpen />
                Open details
              </DropdownMenuItem>
              {row.session_id ? (
                <>
                  <DropdownMenuItem
                    onSelect={() => navigate(getRunPath(row.session_id ?? '', row.id))}
                  >
                    <Bot />
                    Open agent run
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => update({ session: row.session_id })}>
                    <Filter />
                    Only this session
                  </DropdownMenuItem>
                </>
              ) : null}
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={() => copyText(row.id, 'Record ID')}>
                <Copy />
                Copy record ID
              </DropdownMenuItem>
              {row.session_id ? (
                <DropdownMenuItem onSelect={() => copyText(row.session_id ?? '', 'Session ID')}>
                  <Copy />
                  Copy session ID
                </DropdownMenuItem>
              ) : null}
            </>
          )}
          empty={
            unfilteredEmpty ? (
              <EmptyState
                compact
                icon={ListTree}
                tone="blue"
                title="No requests recorded yet"
                description="Send a request through the gateway, or point Claude Code, Codex, OpenCode or DeepSeek Harness at it, and it appears here with its decision, model and cost."
                actions={
                  <Button size="sm" variant="outline" onClick={() => navigate('/playground')}>
                    Open the playground
                  </Button>
                }
              />
            ) : (
              <EmptyState
                compact
                icon={ListTree}
                tone="blue"
                title="No requests match"
                description="Widen the time range or clear the filters. New requests appear as soon as the router records them."
                actions={
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setSearchDraft('')
                      setParams({}, { replace: true })
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
        page={page}
        pageSize={REQUESTS_PAGE_SIZE}
        total={data.total}
        noun="request"
        note={windowNote}
        onPage={(next) =>
          setParams((current) => {
            const nextParams = new URLSearchParams(current)
            if (next <= 1) nextParams.delete('page')
            else nextParams.set('page', String(next))
            return nextParams
          })
        }
      />
    </>
  )
}
