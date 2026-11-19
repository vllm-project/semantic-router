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
  Info,
  ListTree,
  MessagesSquare,
  PanelRightOpen,
  Scale,
  RefreshCw,
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
  sessionParam,
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
  RelativeTime,
  SummaryStrip,
  SummaryTile,
} from './InsightsDisplay'
import { moreCurrenciesLabel, spendSummary } from './homeSupport'
import { recordEvents } from './insightsAgentEvents'
import { AgentEventChips } from './InsightsAgentEvents'
import { HarnessLogo, ToolChips } from './InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo } from './insightsHarness'
import { readPage, readPageSize, writePage } from './insightsListData'
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
  harnessFilterValues,
  hasAnyFilter,
  hasClientFilters,
  RANGE_TEXT,
  readRequestFilters,
  REQUEST_RANGES,
  REQUEST_WINDOW_SIZE,
  useAnyRecord,
  useRequestsData,
  writeRequestFilters,
  type RequestFilters,
} from './insightsRequestsData'
import { compactID } from './insightsReviewSupport'
import { useSessionRun } from './insightsRunsData'
import { getRunPath, oneLine, runHeadline } from './insightsRunSupport'
import { SessionChoice } from './InsightsSessionChoice'
import { useSessionScope } from './insightsSessionScope'

const InsightsCharts = lazy(() => import('../components/InsightsCharts'))
const InsightsSessionsView = lazy(() => import('./InsightsSessionsView'))

const BREAKDOWN_KEY = 'vsr-requests-breakdown'
const VIEW_KEY = 'vsr-requests-view'

type RequestsViewMode = 'requests' | 'sessions'

const VIEW_OPTIONS: { value: RequestsViewMode; label: string }[] = [
  { value: 'requests', label: 'Requests' },
  { value: 'sessions', label: 'Sessions' },
]

function readViewMode(params: URLSearchParams): RequestsViewMode {
  const value = params.get('view')
  if (value === 'sessions' || value === 'requests') return value
  try {
    return window.localStorage.getItem(VIEW_KEY) === 'sessions' ? 'sessions' : 'requests'
  } catch {
    return 'requests'
  }
}

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

const SUMMARY_GRID = 'grid-cols-2 sm:grid-cols-3 xl:grid-cols-[repeat(5,minmax(0,1fr))_auto]'

function spendOf(aggregate: InsightsAggregateResponse) {
  const lead = spendSummary(aggregate)
  const compared = lead.saved !== undefined
  const difference = compared ? -(lead.saved ?? 0) : undefined
  return {
    spend: formatInsightsCost(lead.spend, lead.currency, { total: true }),
    difference:
      difference === undefined
        ? '—'
        : `${difference > 0 ? '+' : ''}${formatInsightsCost(difference, lead.currency, { total: true })}`,
    differenceHint:
      compared && (lead.baseline ?? 0) > 0
        ? `${formatInsightsShare(Math.abs(lead.saved ?? 0), lead.baseline ?? 0)} ${(lead.saved ?? 0) >= 0 ? 'below' : 'above'} the reference · cost only`
        : 'Cost only, not a quality comparison',
    differenceWhy: compared
      ? undefined
      : 'No request in range has both a priced cost and a priced reference model',
    more: moreCurrenciesLabel(lead.otherCurrencies),
  }
}

function SummarySkeleton() {
  return (
    <SummaryStrip label="Loading the totals" className={SUMMARY_GRID}>
      {Array.from({ length: 5 }, (_, index) => (
        <div key={index} className="flex flex-col gap-1.5 px-4 py-3">
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-[18px] w-24" />
          <Skeleton className="h-3 w-28" />
        </div>
      ))}
      <span aria-hidden className="max-xl:hidden" />
    </SummaryStrip>
  )
}

// The totals of every request the server filters select, loaded on their own:
// the list never waits for them.
function RequestsSummary({
  aggregate,
  loading,
  error,
  onRetry,
  breakdown,
  onBreakdown,
  clientFiltered,
  harnessCounts,
}: {
  aggregate: InsightsAggregateResponse | null
  loading: boolean
  error: string | null
  onRetry: () => void
  breakdown: boolean
  onBreakdown: (open: boolean) => void
  clientFiltered: boolean
  harnessCounts: Record<string, number> | null
}) {
  if (!aggregate) {
    if (error && !loading) {
      return (
        <section aria-label="Requests summary" className="mb-4">
          <Card className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 shadow-none">
            <span className="flex min-w-0 items-center gap-2 text-[13px] text-muted-foreground">
              <AlertTriangle className="size-4 shrink-0 text-warning" />
              <span className="min-w-0 truncate" title={error}>
                The totals could not load: {error}
              </span>
            </span>
            <Button size="xs" variant="outline" onClick={onRetry}>
              <RefreshCw />
              Retry
            </Button>
          </Card>
        </section>
      )
    }
    if (!loading) return null
    return (
      <section aria-label="Requests summary" aria-busy className="mb-4">
        <SummarySkeleton />
      </section>
    )
  }
  const lifecycle = aggregate.lifecycle
  const failed = lifecycle.failed + lifecycle.aborted
  const money = spendOf(aggregate)
  const tokens = aggregate.token_volume
  return (
    <section aria-label="Requests summary" className="mb-4">
      <SummaryStrip label="Totals" className={SUMMARY_GRID}>
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
          label="Cost vs reference"
          icon={<Scale className="text-chart-2" />}
          value={
            money.differenceWhy ? (
              <Tooltip content={money.differenceWhy}>
                <span>{money.difference}</span>
              </Tooltip>
            ) : (
              money.difference
            )
          }
          hint={money.differenceHint}
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
      </SummaryStrip>
      {clientFiltered ? (
        <p className="mt-1.5 text-[11.5px] text-subtle-foreground">
          Totals cover every request class the other filters select.
        </p>
      ) : null}
      {breakdown ? (
        <div id="requests-breakdown" className="mt-4">
          <Suspense fallback={<Skeleton className="h-48 w-full rounded-xl" />}>
            <InsightsCharts aggregate={aggregate} harnessCounts={harnessCounts} />
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
          <AgentEventChips events={recordEvents(record)} compact />
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

// A session reads by the name its harness gave it when this record carries
// one, else by its id; the run page holds the rest.
function SessionCell({
  record,
  onFilter,
}: {
  record: InsightsRecord
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
  const title = record.session_name ? oneLine(record.session_name) : ''
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
      className="inline-flex h-7 min-w-0 max-w-full items-center gap-1.5 rounded-md border border-border bg-surface pl-2 pr-1 text-xs shadow-xs"
      title={title}
    >
      <span className="shrink-0 text-subtle-foreground">{label}</span>
      <span className="min-w-0 max-w-[420px] truncate font-medium text-foreground">{value}</span>
      {action}
      <button
        type="button"
        aria-label={`Clear ${label.toLowerCase()} filter`}
        onClick={onClear}
        className="grid size-5 shrink-0 place-items-center rounded text-subtle-foreground hover:bg-accent hover:text-foreground"
      >
        <X className="size-3.5" />
      </button>
    </span>
  )
}

/** What the class filters read, said whenever one is on. */
function windowNote(data: {
  windowLoaded: boolean
  window: InsightsRecord[]
  windowTotal: number
  range: RequestFilters['range']
}): string {
  const range = RANGE_TEXT[data.range]
  if (!data.windowLoaded) {
    return `Class filters run in your browser over the newest ${REQUEST_WINDOW_SIZE} requests ${range}.`
  }
  if (data.windowTotal > data.window.length) {
    return `Class filters run in your browser over the newest ${formatInsightsCount(data.window.length)} of ${formatInsightsCount(data.windowTotal)} requests ${range}. Narrow the time range or add a filter to cover them all.`
  }
  return `Class filters run in your browser over all ${formatInsightsCount(data.windowTotal)} requests ${range}.`
}

// ---------------------------------------------------------------- view

export default function InsightsRequestsView() {
  const navigate = useNavigate()
  const [params, setParams] = useSearchParams()
  const filters = useMemo(() => readRequestFilters(params), [params])
  const page = readPage(params)
  const pageSize = readPageSize(params)
  const { agentRules, modelProviders } = useInsightsConfig()
  const labels = useModelLabels()
  const [live, setLive] = useState(true)
  const [held, setHeld] = useState(false)
  // the client and class counts load once a filter control is reached for
  const [facets, setFacets] = useState(false)
  const [breakdown, setBreakdown] = useState(readBreakdown)
  const view = readViewMode(params)
  const [searchDraft, setSearchDraft] = useState(filters.search)
  const now = useNow(live ? 5_000 : 30_000)

  // a session filter may name the id its harness sent; the reads match the full id
  const sessionScope = useSessionScope(filters.session)
  const scopedFilters = useMemo(
    () => (filters.session ? { ...filters, session: sessionScope.id } : filters),
    [filters, sessionScope.id],
  )
  const data = useRequestsData({
    filters: scopedFilters,
    page,
    pageSize,
    live,
    held,
    rules: agentRules,
    withFacets: facets || breakdown,
    paused: sessionScope.resolving,
  })
  const clientMode = hasClientFilters(filters)
  const narrowed = hasAnyFilter(filters)
  const sessionRun = useSessionRun(
    sessionScope.resolving ? undefined : scopedFilters.session || undefined,
  )
  const quiet =
    !data.loading && !narrowed && data.total === 0 && !data.listError && !data.unavailable
  const anyRecord = useAnyRecord(quiet && filters.range !== 'all')
  const onboarding = quiet && (filters.range === 'all' || anyRecord === false)
  const wantFacets = () => setFacets(true)

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

  // a page past the end, as when the records it held expired, goes to the last one
  const pages = Math.max(1, Math.ceil(data.total / pageSize))
  useEffect(() => {
    if (!data.loading && !data.stale && data.total > 0 && page > pages) {
      setParams((current) => writePage(current, pages), { replace: true })
    }
  }, [data.loading, data.stale, data.total, page, pages, setParams])

  const setView = (next: RequestsViewMode) => {
    setParams(
      (current) => {
        const updated = writePage(current, 1)
        if (next === 'requests') updated.delete('view')
        else updated.set('view', next)
        return updated
      },
      { replace: true },
    )
    try {
      window.localStorage.setItem(VIEW_KEY, next)
    } catch {
      // the choice lasts for this page only
    }
  }

  const setBreakdownOpen = (open: boolean) => {
    setBreakdown(open)
    try {
      window.localStorage.setItem(BREAKDOWN_KEY, open ? '1' : '0')
    } catch {
      // the choice lasts for this page only
    }
  }

  const classCounts = useMemo(
    () => countFacet(data.window, (record) => requestClassOf(record, agentRules)),
    [agentRules, data.window],
  )
  const counted = data.windowLoaded
  // The Router counts the harnesses over every request the other filters select.
  const harnessOptions: FilterOption[] = useMemo(
    () =>
      harnessFilterValues(data.harnessCounts, filters.harness).map(({ value, count }) => {
        const harness = value === API_CLIENT_FILTER ? undefined : value
        return {
          value,
          label: harnessInfo(harness)?.label ?? API_CLIENT_LABEL,
          icon: <HarnessLogo harness={harness} size={14} />,
          count,
        }
      }),
    [data.harnessCounts, filters.harness],
  )
  const classOptions: FilterOption[] = REQUEST_CLASSES.filter(
    (value) => !counted || classCounts.has(value) || filters.classes.includes(value),
  ).map((value) => ({
    value,
    label: REQUEST_CLASS_LABELS[value],
    count: counted ? (classCounts.get(value) ?? 0) : undefined,
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
            onFilter={(session) => update({ session: sessionParam(session) })}
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
            fallback={row.decision ? 'No model call' : 'No backend'}
            className="max-w-[170px] text-[13px]"
          />
          {row.requested_model ? (
            <Tooltip
              content={`The client asked for ${row.requested_model}; the listener routed it to ${row.original_model ?? 'its route'}.`}
            >
              <span className="block max-w-[170px] truncate text-[11.5px] text-subtle-foreground">
                requested {row.requested_model}
              </span>
            </Tooltip>
          ) : null}
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

  if (data.unavailable) {
    return (
      <Card className="border-dashed shadow-none">
        <InsightsUnavailable reason={data.unavailable} onRetry={data.refresh} />
      </Card>
    )
  }

  const clearFilters = () => {
    setSearchDraft('')
    setParams({}, { replace: true })
  }

  return (
    <>
      {data.listError ? (
        <InsightsLoadError
          title="Requests could not load"
          message={data.listError}
          onRetry={data.refresh}
        />
      ) : null}
      {onboarding ? null : (
        <RequestsSummary
          aggregate={aggregate}
          loading={data.aggregateLoading}
          error={data.aggregateError}
          onRetry={data.refresh}
          breakdown={breakdown}
          onBreakdown={setBreakdownOpen}
          clientFiltered={clientMode}
          harnessCounts={data.harnessCounts}
        />
      )}
      <DataToolbar
        search={searchDraft}
        onSearchChange={onSearch}
        placeholder="Search request, recipe or session id…"
        filters={
          <>
            <div
              role="group"
              aria-label="Client and class filters"
              className="flex items-center gap-2"
              onPointerOver={wantFacets}
              onFocus={wantFacets}
            >
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
            </div>
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
              aria-label="Requests view"
              value={view}
              onValueChange={(next) => setView(next as RequestsViewMode)}
              options={VIEW_OPTIONS}
            />
            <Segmented
              aria-label="Time range"
              value={filters.range}
              onValueChange={(range) => update({ range })}
              options={REQUEST_RANGES}
            />
            <LiveToggle
              live={live}
              held={held}
              onLiveChange={(next) => {
                setLive(next)
                if (next) data.refresh()
              }}
            />
          </>
        }
      />
      <div className="mb-3 flex min-h-7 flex-wrap items-center gap-2">
        {filters.session ? (
          <ActiveChip
            label="Session"
            value={sessionRun ? runHeadline(sessionRun) : compactSessionId(filters.session)}
            title={displaySessionId(filters.session)}
            onClear={() => update({ session: '' })}
            action={
              <Button
                variant="ghost"
                size="xs"
                className="h-5 shrink-0 px-1.5"
                onClick={() => navigate(getRunPath(scopedFilters.session))}
              >
                <Bot />
                Open run
              </Button>
            }
          />
        ) : null}
        <SessionChoice scope={sessionScope} />
        {filters.turn ? (
          <ActiveChip
            label="Codex turn"
            value={compactID(filters.turn)}
            title={filters.turn}
            onClear={() => update({ turn: '' })}
          />
        ) : null}
        {clientMode ? (
          <p className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
            <Info className="size-3.5 shrink-0 text-subtle-foreground" />
            <span className="min-w-0">
              {windowNote({
                windowLoaded: data.windowLoaded,
                window: data.window,
                windowTotal: data.windowTotal,
                range: filters.range,
              })}
            </span>
          </p>
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
            {counted && classCounts.get('housekeeping') ? (
              <span className="ml-1 font-mono text-[11px] text-subtle-foreground">
                {classCounts.get('housekeeping')}
              </span>
            ) : null}
          </Label>
        </span>
      </div>
      {view === 'sessions' ? (
        <Suspense fallback={<Skeleton className="h-48 w-full rounded-xl" />}>
          <InsightsSessionsView
            filters={scopedFilters}
            live={live}
            paused={sessionScope.resolving}
          />
        </Suspense>
      ) : (
        <>
          <div
            onPointerEnter={() => setHeld(true)}
            onPointerLeave={() => setHeld(false)}
            aria-busy={data.stale || undefined}
            className={cn('transition-opacity duration-200', data.stale && 'opacity-60')}
          >
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
                      <DropdownMenuItem
                        onSelect={() => update({ session: sessionParam(row.session_id ?? '') })}
                      >
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
                    <DropdownMenuItem
                      onSelect={() => copyText(displaySessionId(row.session_id), 'Session ID')}
                    >
                      <Copy />
                      Copy session ID
                    </DropdownMenuItem>
                  ) : null}
                </>
              )}
              empty={
                onboarding ? (
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
                ) : quiet ? (
                  <EmptyState
                    compact
                    icon={ListTree}
                    tone="blue"
                    title={`No requests ${RANGE_TEXT[filters.range]}`}
                    description="Older requests are under a longer range; new ones appear here as soon as the router records them."
                    actions={
                      <Button size="sm" variant="outline" onClick={() => update({ range: 'all' })}>
                        Show all time
                      </Button>
                    }
                  />
                ) : data.listError ? (
                  <EmptyState
                    compact
                    icon={ListTree}
                    tone="slate"
                    title="No requests to show"
                    description="The list could not load. Retry above, or check the router on the Status page."
                  />
                ) : (
                  <EmptyState
                    compact
                    icon={ListTree}
                    tone="blue"
                    title="No requests match"
                    description={
                      clientMode
                        ? 'None of the newest requests match the class filters. Widen the time range or clear the filters.'
                        : 'Widen the time range or clear the filters. New requests appear as soon as the router records them.'
                    }
                    actions={
                      <Button size="sm" variant="outline" onClick={clearFilters}>
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
            noun="request"
            onPage={(next) => setParams((current) => writePage(current, next))}
            onPageSize={(size) => setParams((current) => writePage(current, 1, size))}
          />
        </>
      )}
    </>
  )
}
