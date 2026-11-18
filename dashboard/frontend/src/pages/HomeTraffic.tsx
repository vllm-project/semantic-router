import type { ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  ArrowUpRight,
  CircleDollarSign,
  Gauge,
  GitBranch,
  MessagesSquare,
  PiggyBank,
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
import {
  formatInsightsDecisionName,
  getInsightsLifecyclePresentation,
  getInsightsRecordPath,
} from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'
import {
  carriedP95,
  HOME_RANGE_LABEL,
  HOME_RECENT_LIMIT,
  moreCurrenciesLabel,
  needsTrafficWindow,
  spendSummary,
  TRAFFIC_UNAVAILABLE,
  type HomeRange,
  type TrafficView,
} from './homeSupport'
import type { HomeTraffic } from './homeTraffic'

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

/** A metric card whose value is still on its way, with its label already in place. */
function PendingStat({ label, icon }: { label: string; icon: ReactNode }) {
  return (
    <Card aria-busy="true" className="flex flex-col gap-3 px-4 pb-3 pt-4">
      <span className="flex items-center gap-2 text-[13px] font-medium text-muted-foreground">
        {icon}
        {label}
      </span>
      <Skeleton className="h-[26px] w-24" />
      <Skeleton className="h-4 w-36" />
    </Card>
  )
}

/**
 * Requests, latency, spend and savings over the range, from Router Replay.
 * Without replay data the traffic chart explains why, so the row stays out.
 */
export function HomeKpis({
  traffic,
  view,
  range,
}: {
  traffic: HomeTraffic
  view: TrafficView | null
  range: HomeRange
}) {
  if (traffic.unavailable || (traffic.error && !traffic.aggregate)) return null
  const aggregate = traffic.aggregate
  if (!aggregate) {
    return (
      <section
        aria-label="Key metrics"
        aria-busy="true"
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

  // A sparkline covering only the newest records would draw a false start.
  const series = view?.coversRange ? view.points : undefined
  const latency = view?.latency ?? {}
  // Without a series, latency waits for the newest records while the rest shows.
  const latencyPending = !view && needsTrafficWindow(aggregate) && !traffic.windowError
  const sampledNote =
    view && !view.coversRange
      ? `newest ${formatInsightsCount(view.sampled ?? 0)}`
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
    <section
      aria-label="Key metrics"
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4"
    >
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
          spark={series?.map((point) => point.requests)}
          sparkColor="var(--chart-1)"
        />
      </KpiLink>
      <KpiLink to={`/observe/requests${rangeQuery}`} label="p95 latency: open Requests">
        {latencyPending ? (
          <PendingStat label="p95 latency" icon={<Gauge className="size-4 text-chart-3" />} />
        ) : (
          <StatCard
            label="p95 latency"
            icon={<Gauge className="size-4 text-chart-3" />}
            value={latency.p95 ?? 0}
            format={(value) => (latency.p95 === undefined ? '—' : formatLatency(value))}
            hint={
              latency.p50 !== undefined
                ? `p50 ${formatLatency(latency.p50)} · ${sampledNote}`
                : !view && traffic.windowError
                  ? 'The newest requests did not load'
                  : 'No completed request yet'
            }
            spark={series ? carriedP95(series) : undefined}
            sparkColor="var(--chart-3)"
          />
        )}
      </KpiLink>
      <KpiLink to={`/observe/requests${rangeQuery}`} label="Spend: open Requests">
        <StatCard
          label="Spend"
          icon={<CircleDollarSign className="size-4 text-chart-6" />}
          value={money.spend}
          format={(value) => formatInsightsCost(value, money.currency, { total: true })}
          hint={moreCurrenciesLabel(money.otherCurrencies) ?? perThousand}
          spark={series?.map((point) => point.cost)}
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

function requestTitle(record: InsightsRecord, sessionTitle?: string): string {
  const title = sessionTitle || record.session_name
  if (title) return title
  if (record.session_id) return compactSessionId(record.session_id)
  return compactIdentifier(record.request_id || record.id, 10, 6)
}

function RequestCell({
  record,
  sessionTitle,
  rules,
  now,
}: {
  record: InsightsRecord
  sessionTitle?: string
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
          {requestTitle(record, sessionTitle)}
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
  traffic,
  rules,
  range,
  now,
}: {
  traffic: HomeTraffic
  rules?: AgentRuleIndex
  range: HomeRange
  now: number
}) {
  const navigate = useNavigate()
  const rows = traffic.rows.slice(0, HOME_RECENT_LIMIT)
  let body: ReactNode
  if (traffic.unavailable || (traffic.rowsError && rows.length === 0)) {
    const copy = traffic.unavailable ? TRAFFIC_UNAVAILABLE[traffic.unavailable] : undefined
    body = (
      <EmptyState
        compact
        icon={copy?.icon ?? ServerOff}
        tone={copy?.tone ?? 'slate'}
        title={copy ? 'No recorded requests to show' : 'Requests could not load'}
        description={
          traffic.unavailable === 'replay_off'
            ? 'Each request appears here with the agent that sent it, its decision and its model once Router Replay records it.'
            : traffic.unavailable === 'forbidden'
              ? copy?.description
              : traffic.unavailable
                ? 'They load again as soon as the router API answers.'
                : traffic.rowsError
        }
        actions={
          traffic.unavailable ? undefined : (
            <Button size="sm" variant="outline" onClick={traffic.refresh}>
              Try again
            </Button>
          )
        }
      />
    )
  } else if (traffic.rowsLoading && rows.length === 0) {
    body = <TableSkeleton rows={6} columns={5} />
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
            <TableHead className="hidden text-right @xl:table-cell">Cost</TableHead>
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
                  sessionTitle={record.session_id ? traffic.titles[record.session_id] : undefined}
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
              <TableCell className="hidden text-right font-mono text-xs tabular @xl:table-cell">
                {typeof record.actual_cost === 'number' && record.currency
                  ? formatInsightsCost(record.actual_cost, record.currency)
                  : '—'}
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
          {traffic.aggregate
            ? `${formatInsightsCount(traffic.aggregate.record_count)} in the ${HOME_RANGE_LABEL[range]}, from Router Replay`
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
