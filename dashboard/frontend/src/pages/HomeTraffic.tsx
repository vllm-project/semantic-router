import type { ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  ArrowUpRight,
  CircleDollarSign,
  Gauge,
  GitBranch,
  ListTree,
  Lock,
  MessagesSquare,
  PiggyBank,
  RefreshCw,
  Rocket,
  ServerOff,
  Zap,
} from 'lucide-react'

import {
  Badge,
  Button,
  Card,
  CardAction,
  CardDescription,
  CardHeader,
  CardTitle,
  EmptyState,
  IconTile,
  Progress,
  Skeleton,
  StatCard,
  StatusDot,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableSkeleton,
  Tooltip,
  cn,
  type Tone,
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
  requestClassOf,
  type AgentRuleIndex,
} from './insightsAgentIdentity'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { formatLatency } from './insightsDisplayFormat'
import { RelativeTime } from './InsightsDisplay'
import { HarnessLogo, RequestClassBadge } from './InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo } from './insightsHarness'
import { ModelChip } from './insightsModelLabelSupport'
import type { InsightsUnavailableReason } from './insightsPageApi'
import {
  formatInsightsDecisionName,
  getInsightsLifecyclePresentation,
  getInsightsRecordPath,
} from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'
import type { RequestsData } from './insightsRequestsData'
import type { RunSummary } from './insightsRunTypes'
import {
  bucketTraffic,
  HOME_RANGE_LABEL,
  latencyStats,
  spendSummary,
  type HomeRange,
} from './homeSupport'

const UNAVAILABLE: Record<
  InsightsUnavailableReason,
  { title: string; description: string; tone: Tone; icon: typeof ListTree }
> = {
  replay_off: {
    title: 'Traffic metrics need Router Replay',
    description:
      'Turn on the router_replay plugin for a recipe, then send a request through the gateway.',
    tone: 'blue',
    icon: ListTree,
  },
  router_unreachable: {
    title: 'Traffic is unavailable while the router is unreachable',
    description: 'Requests, latency and spend appear here once the router API answers again.',
    tone: 'slate',
    icon: ServerOff,
  },
  forbidden: {
    title: 'Your role cannot read Router Replay',
    description: 'Traffic metrics need the replay.read permission. Ask an admin for access.',
    tone: 'slate',
    icon: Lock,
  },
}

function KpiLink({ to, label, children }: { to: string; label: string; children: ReactNode }) {
  return (
    <Link
      to={to}
      aria-label={label}
      className="group block rounded-xl outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40 [&>[data-slot=card]]:h-full [&>[data-slot=card]]:transition-[border-color,box-shadow,transform] [&>[data-slot=card]]:duration-200 hover:[&>[data-slot=card]]:-translate-y-px hover:[&>[data-slot=card]]:border-border-strong hover:[&>[data-slot=card]]:shadow-md"
    >
      {children}
    </Link>
  )
}

function TrafficNotice({
  reason,
  error,
  onRetry,
}: {
  reason?: InsightsUnavailableReason
  error?: string | null
  onRetry: () => void
}) {
  const navigate = useNavigate()
  const copy = reason
    ? UNAVAILABLE[reason]
    : {
        title: 'Traffic could not load',
        description: error ?? 'The replay API did not answer.',
        tone: 'slate' as Tone,
        icon: ServerOff,
      }
  return (
    <Card className="flex flex-col gap-4 px-5 py-5 sm:col-span-2 sm:flex-row sm:items-center xl:col-span-4">
      <IconTile icon={copy.icon} tone={copy.tone} size="lg" />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="text-[15px] font-semibold tracking-[-0.01em]">{copy.title}</span>
        <span className="text-[13px] text-muted-foreground">{copy.description}</span>
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        {reason === 'replay_off' ? (
          <Button size="sm" onClick={() => navigate('/routing/recipes')}>
            Open recipes
          </Button>
        ) : null}
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RefreshCw />
          Try again
        </Button>
      </div>
    </Card>
  )
}

/** Requests, latency, spend and savings over the range, from Router Replay. */
export function HomeKpis({
  data,
  range,
  now,
}: {
  data: RequestsData
  range: HomeRange
  now: number
}) {
  if (data.unavailable || (data.error && !data.aggregate)) {
    return (
      <section
        aria-label="Traffic"
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4"
      >
        <TrafficNotice reason={data.unavailable} error={data.error} onRetry={data.refresh} />
      </section>
    )
  }
  const aggregate = data.aggregate
  if (!aggregate) {
    return (
      <section
        aria-label="Traffic"
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4"
      >
        {[0, 1, 2, 3].map((index) => (
          <Card key={index} className="flex h-[138px] flex-col gap-3 px-4 pt-4">
            <Skeleton className="h-3.5 w-24" />
            <Skeleton className="h-7 w-28" />
            <Skeleton className="h-3 w-36" />
          </Card>
        ))}
      </section>
    )
  }

  const series = bucketTraffic(data.window, range, data.windowTotal, now)
  const latency = latencyStats(data.window)
  const sampledNote =
    data.windowTotal > data.window.length
      ? `newest ${formatInsightsCount(data.window.length)}`
      : HOME_RANGE_LABEL[range]
  const money = spendSummary(aggregate)
  const failed = aggregate.lifecycle.failed + aggregate.lifecycle.aborted
  const savedShare =
    money.saved !== undefined && money.baseline ? (money.saved / money.baseline) * 100 : undefined
  const perThousand =
    aggregate.record_count > 0 && money.currency
      ? `${formatInsightsCost((money.spend / aggregate.record_count) * 1000, money.currency)} per 1k requests`
      : 'Recorded tokens × configured rates'
  const rangeQuery = `?range=${range}`

  return (
    <section aria-label="Traffic" className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <KpiLink to={`/observe/requests${rangeQuery}`} label="Requests: open Requests">
        <StatCard
          label="Requests"
          icon={<Zap className="size-4 text-chart-1" />}
          value={aggregate.record_count}
          format={(value) => formatInsightsCompactCount(Math.round(value))}
          hint={
            failed > 0
              ? `${formatInsightsCount(failed)} failed · ${HOME_RANGE_LABEL[range]}`
              : `All public models · ${HOME_RANGE_LABEL[range]}`
          }
          spark={series?.requests}
          sparkColor="var(--chart-1)"
        />
      </KpiLink>
      <KpiLink to={`/observe/requests${rangeQuery}`} label="p95 latency: open Requests">
        <StatCard
          label="p95 latency"
          icon={<Gauge className="size-4 text-chart-3" />}
          value={latency.p95 ?? 0}
          format={(value) => (latency.p95 === undefined ? '—' : formatLatency(value))}
          hint={
            latency.p50 === undefined
              ? 'No completed request yet'
              : `p50 ${formatLatency(latency.p50)} · ${sampledNote}`
          }
          spark={series?.p95}
          sparkColor="var(--chart-3)"
        />
      </KpiLink>
      <KpiLink to={`/observe/requests${rangeQuery}`} label="Spend: open Requests">
        <StatCard
          label="Spend"
          icon={<CircleDollarSign className="size-4 text-chart-6" />}
          value={money.spend}
          format={(value) => formatInsightsCost(value, money.currency, { total: true })}
          hint={
            money.otherCurrencies > 0
              ? `+${money.otherCurrencies} more currenc${money.otherCurrencies > 1 ? 'ies' : 'y'}`
              : perThousand
          }
          spark={series?.cost}
          sparkColor="var(--chart-6)"
        />
      </KpiLink>
      <Card className="relative overflow-hidden" data-testid="home-saved">
        <div className="pointer-events-none absolute -right-10 -top-10 size-40 rounded-full bg-[radial-gradient(closest-side,var(--glow-gold),transparent)]" />
        <div className="relative flex h-full flex-col gap-3 px-4 pb-4 pt-4">
          <span className="flex items-center gap-2 text-[13px] font-medium text-muted-foreground">
            <PiggyBank className="size-4 text-chart-2" />
            Saved by routing
          </span>
          <div className="flex items-baseline gap-2">
            <span
              className={cn(
                'text-[26px] font-semibold leading-none tracking-[-0.03em] tabular',
                money.saved !== undefined && money.saved < 0 && 'text-destructive-foreground',
              )}
            >
              {money.saved === undefined
                ? 'N/A'
                : formatInsightsCost(money.saved, money.currency, { total: true })}
            </span>
            {savedShare !== undefined && savedShare > 0 ? (
              <Badge tone="gold">{savedShare.toFixed(0)}% less</Badge>
            ) : null}
          </div>
          <p className="text-xs text-muted-foreground">
            {money.saved === undefined
              ? 'No request was priced against a baseline yet.'
              : 'Compared with sending each request to its baseline model.'}
          </p>
          {money.saved !== undefined && money.baseline ? (
            <div className="mt-auto">
              <div className="mb-1.5 flex justify-between text-[11px] text-subtle-foreground tabular">
                <span>
                  Actual {formatInsightsCost(money.spend, money.currency, { total: true })}
                </span>
                <span>
                  Baseline {formatInsightsCost(money.baseline, money.currency, { total: true })}
                </span>
              </div>
              <Progress
                value={Math.min(100, (money.spend / money.baseline) * 100)}
                tone="neutral"
                aria-label={`${formatInsightsShare(money.spend, money.baseline)} of the baseline`}
                className="[&>div]:bg-chart-2"
              />
            </div>
          ) : null}
        </div>
      </Card>
    </section>
  )
}

function requestTitle(record: InsightsRecord, run?: RunSummary): string {
  const title = run?.session_name || record.session_name
  if (title) return title
  if (record.session_id) return compactSessionId(record.session_id)
  return compactIdentifier(record.request_id || record.id, 10, 6)
}

function RequestCell({
  record,
  run,
  rules,
  now,
}: {
  record: InsightsRecord
  run?: RunSummary
  rules?: AgentRuleIndex
  now: number
}) {
  const requestClass = requestClassOf(record, rules)
  const info = harnessInfo(record.agent?.harness)
  const lifecycle = getInsightsLifecyclePresentation(record)
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <span className="relative shrink-0">
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
      <div className="flex min-w-0 flex-col gap-0.5">
        <Link
          to={getInsightsRecordPath(record.id)}
          onClick={(event) => event.stopPropagation()}
          className="truncate rounded-sm text-[13px] font-medium text-foreground outline-none hover:text-primary focus-visible:ring-[3px] focus-visible:ring-ring/40"
          title={record.session_id ?? record.request_id ?? record.id}
        >
          {requestTitle(record, run)}
        </Link>
        <span className="flex min-w-0 items-center gap-1.5 text-xs text-subtle-foreground">
          {lifecycle.errored ? (
            <Tooltip content={lifecycle.label}>
              <span className="flex" aria-label={lifecycle.label}>
                <StatusDot tone="danger" />
              </span>
            </Tooltip>
          ) : null}
          <RelativeTime iso={record.timestamp} now={now} className="shrink-0" />
          <span aria-hidden>·</span>
          <span className="shrink-0">{info?.label ?? API_CLIENT_LABEL}</span>
          {requestClass === 'api' ? null : (
            <RequestClassBadge
              requestClass={requestClass}
              kind={record.agent?.kind}
              className="min-w-0 max-w-[140px] shrink"
            />
          )}
        </span>
      </div>
    </div>
  )
}

/** The newest recorded requests, each with the agent that sent it and the route it took. */
export function HomeRecentRequests({
  data,
  runs,
  rules,
  range,
  now,
}: {
  data: RequestsData
  runs: Map<string, RunSummary>
  rules?: AgentRuleIndex
  range: HomeRange
  now: number
}) {
  const navigate = useNavigate()
  const rows = data.rows.slice(0, 7)
  let body: ReactNode
  if (data.unavailable || (data.error && rows.length === 0)) {
    const copy = data.unavailable ? UNAVAILABLE[data.unavailable] : undefined
    body = (
      <EmptyState
        compact
        icon={copy?.icon ?? ServerOff}
        tone={copy?.tone ?? 'slate'}
        title={copy ? 'No recorded requests to show' : 'Requests could not load'}
        description={
          data.unavailable === 'replay_off'
            ? 'Each request appears here with the agent that sent it, its decision and its model once Router Replay records it.'
            : data.unavailable === 'forbidden'
              ? copy?.description
              : data.unavailable
                ? 'They load again as soon as the router API answers.'
                : data.error
        }
      />
    )
  } else if (data.loading && rows.length === 0) {
    body = <TableSkeleton rows={6} columns={4} />
  } else if (rows.length === 0) {
    body = (
      <EmptyState
        compact
        icon={MessagesSquare}
        tone="blue"
        title={`No requests in the ${HOME_RANGE_LABEL[range]}`}
        description="Point an app or a coding agent at the gateway, or send a message from the playground, and its route shows up here."
        actions={
          <>
            <Button size="sm" onClick={() => navigate('/get-started')}>
              <Rocket />
              Connect an app
            </Button>
            <Button size="sm" variant="outline" onClick={() => navigate('/playground')}>
              Open the playground
            </Button>
          </>
        }
      />
    )
  } else {
    body = (
      <Table>
        <TableHeader className="bg-surface-subtle/70">
          <TableRow>
            <TableHead>Request</TableHead>
            <TableHead className="hidden @2xl:table-cell">Decision</TableHead>
            <TableHead>Model</TableHead>
            <TableHead className="hidden text-right @3xl:table-cell">Latency</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((record) => (
            <TableRow
              key={record.id}
              className="cursor-pointer"
              onClick={() => navigate(getInsightsRecordPath(record.id))}
            >
              <TableCell className="max-w-[280px]">
                <RequestCell
                  record={record}
                  run={record.session_id ? runs.get(record.session_id) : undefined}
                  rules={rules}
                  now={now}
                />
              </TableCell>
              <TableCell className="hidden @2xl:table-cell">
                {record.decision ? (
                  <InsightsDecisionBadge
                    decision={record.decision}
                    label={formatInsightsDecisionName(record.decision)}
                    className="max-w-[150px]"
                  />
                ) : (
                  <span className="text-xs text-subtle-foreground">Unresolved</span>
                )}
              </TableCell>
              <TableCell>
                <ModelChip
                  name={record.selected_model}
                  fallback="No backend"
                  className="max-w-[200px] text-[13px]"
                />
              </TableCell>
              <TableCell className="hidden text-right font-mono text-xs tabular @3xl:table-cell">
                {formatLatency(record.duration_ms)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    )
  }

  return (
    <Card className="@container overflow-hidden xl:col-span-2" data-testid="home-recent-requests">
      <CardHeader className="pb-3">
        <CardTitle>Recent requests</CardTitle>
        <CardDescription>
          {data.aggregate
            ? `${formatInsightsCount(data.aggregate.record_count)} in the ${HOME_RANGE_LABEL[range]}, from Router Replay`
            : 'From Router Replay'}
        </CardDescription>
        <CardAction>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/observe/requests?range=${range}`)}
          >
            View all
            <ArrowUpRight />
          </Button>
        </CardAction>
      </CardHeader>
      {body}
    </Card>
  )
}
