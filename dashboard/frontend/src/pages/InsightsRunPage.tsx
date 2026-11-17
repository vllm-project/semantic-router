import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  ArrowLeft,
  Bot,
  Clock,
  CornerDownRight,
  ListTree,
  MoreHorizontal,
  SearchX,
  ShieldCheck,
  User,
} from 'lucide-react'

import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CopyButton,
  DataTable,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  EmptyState,
  IconTile,
  MetaItem,
  MiniBar,
  MiniStat,
  PageContainer,
  PageHeader,
  Skeleton,
  cn,
  toast,
  type Column,
} from '../ui'
import { formatDateTime } from '../utils/dateTime'
import { formatInsightsCost } from '../utils/insightsCost'

import { displaySessionId, subagentKindLabel } from './insightsAgentIdentity'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { formatLatency, formatTokenCount } from './insightsDisplayFormat'
import { InsightsLoadError, InsightsUnavailable, RefreshButton } from './InsightsDisplay'
import { HarnessName } from './InsightsHarness'
import { InsightsDataProviders, ModelChip } from './insightsModelLabelSupport'
import {
  fetchInsightsJSON,
  InsightsRequestError,
  insightsUnavailableReason,
  type InsightsUnavailableReason,
} from './insightsPageApi'
import { formatInsightsDecisionName, getInsightsRecordPath } from './insightsPageSupport'
import { getReviewsPath, getSessionRequestsPath } from './insightsReviewSupport'
import { RunCallDetails, RunDiagram, RunLegend, RunSegmentPanel } from './InsightsRunDiagram'
import { RunEvents, RunStatus } from './InsightsRunsView'
import {
  buildRunLanes,
  countText,
  describeCallStatus,
  formatRunCost,
  RUN_MARKER_GLYPHS,
  RUN_MARKER_LABELS,
  runTitle,
  type RunLane,
  type RunSegment,
} from './insightsRunSupport'
import type { RunCall, RunResponse } from './insightsRunTypes'

const RUNS_TAB = '/observe/requests?tab=runs'

function useRun(sessionId: string) {
  const [state, setState] = useState<{
    response: RunResponse | null
    error: string | null
    unavailable?: InsightsUnavailableReason
    notFound: boolean
    loading: boolean
  }>({ response: null, error: null, notFound: false, loading: true })
  const load = useCallback(async () => {
    if (!sessionId) {
      setState({ response: null, error: null, notFound: true, loading: false })
      return
    }
    setState((current) => ({ ...current, loading: true }))
    try {
      const query = new URLSearchParams({ session_id: sessionId })
      const response = await fetchInsightsJSON<RunResponse>(
        `/api/router/api/v1/observability/replays/run?${query.toString()}`,
        'agent run',
      )
      setState({ response, error: null, notFound: false, loading: false })
    } catch (error) {
      const reason = insightsUnavailableReason(error)
      const notFound = error instanceof InsightsRequestError && error.status === 404
      setState({
        response: null,
        error: reason ? null : error instanceof Error ? error.message : 'Unknown error',
        unavailable: notFound ? undefined : reason,
        notFound,
        loading: false,
      })
    }
  }, [sessionId])
  useEffect(() => {
    void load()
  }, [load])
  return { ...state, load }
}

// One agent run, its models read by their labels.
export default function InsightsRunPage() {
  return (
    <InsightsDataProviders>
      <AgentRun />
    </InsightsDataProviders>
  )
}

function BackLink() {
  return (
    <Link
      to={RUNS_TAB}
      className="mb-4 inline-flex items-center gap-1.5 text-[13px] text-muted-foreground transition-colors hover:text-foreground"
    >
      <ArrowLeft className="size-3.5" />
      Agent runs
    </Link>
  )
}

/** The run's agents as a tree: the main agent, then each subagent under the agent that spawned it. */
function AgentTree({
  lanes,
  active,
  onSelect,
}: {
  lanes: RunLane[]
  active?: string
  onSelect: (label?: string) => void
}) {
  const most = Math.max(1, ...lanes.map((lane) => lane.agent.calls))
  return (
    <ul className="flex flex-col" aria-label="Agents">
      {lanes.map(({ agent, depth }) => {
        const main = depth === 0 && agent.label === 'main'
        const selected = active === agent.label
        const kind = subagentKindLabel(agent.kind)
        return (
          <li key={agent.label}>
            <button
              type="button"
              aria-pressed={selected}
              onClick={() => onSelect(selected ? undefined : agent.label)}
              className={cn(
                'flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-surface-subtle focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40',
                selected && 'bg-primary-soft/60 hover:bg-primary-soft/70',
              )}
              style={{ paddingLeft: `${0.625 + depth * 1.1}rem` }}
            >
              <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-md border border-border bg-surface text-subtle-foreground">
                {main ? <User className="size-3.5" /> : <CornerDownRight className="size-3.5" />}
              </span>
              <span className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="flex min-w-0 items-center gap-1.5">
                  <span className="truncate font-mono text-[12.5px] font-medium">
                    {agent.label}
                  </span>
                  {kind ? (
                    <Badge tone="outline" size="sm">
                      {kind}
                    </Badge>
                  ) : null}
                  <span className="ml-auto shrink-0 font-mono text-[11.5px] text-muted-foreground tabular">
                    {countText(agent.calls, 'call')}
                  </span>
                </span>
                <span className="flex min-w-0 items-center gap-1.5 text-xs text-subtle-foreground">
                  <span className="truncate">
                    {main ? 'Main agent' : `Spawned by ${agent.parent || 'main'}`}
                    {agent.id
                      ? ` · #${agent.id.slice(0, 8)}`
                      : agent.id === ''
                        ? ' · id hidden'
                        : ''}
                  </span>
                  <span className="ml-auto shrink-0 font-mono tabular">
                    {formatRunCost(agent.cost)}
                  </span>
                </span>
                {agent.models?.length ? (
                  <span className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
                    {agent.models.map((model) => (
                      <ModelChip key={model} name={model} size={12} />
                    ))}
                  </span>
                ) : null}
                <MiniBar
                  value={agent.calls}
                  max={most}
                  tone={main ? 'blue' : 'violet'}
                  className="mt-0.5 h-1"
                />
              </span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}

function CallMarkers({ markers }: { markers?: string[] }) {
  if (!markers?.length) return <span className="text-xs text-subtle-foreground">—</span>
  return (
    <span className="inline-flex gap-1.5 text-xs">
      {markers.map((marker) => (
        <span key={marker} title={RUN_MARKER_LABELS[marker] ?? marker} data-marker={marker}>
          {RUN_MARKER_GLYPHS[marker] ?? '•'}
        </span>
      ))}
    </span>
  )
}

function RunSkeleton() {
  return (
    <div className="flex flex-col gap-4" role="status" aria-label="Loading the run">
      <Skeleton className="h-11 w-80" />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {Array.from({ length: 6 }, (_, index) => (
          <Skeleton key={index} className="h-[68px] rounded-lg" />
        ))}
      </div>
      <Skeleton className="h-64 rounded-xl" />
    </div>
  )
}

// One agent run: its agents as a tree and as lanes on a time axis, a step's
// details on click, and every call in a table.
function AgentRun() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const sessionId = searchParams.get('session') ?? ''
  const { response, error, unavailable, notFound, loading, load } = useRun(sessionId)
  const lanes = useMemo(() => (response ? buildRunLanes(response) : []), [response])
  const [segmentKey, setSegmentKey] = useState<string>()
  const [callId, setCallId] = useState<string | undefined>(searchParams.get('step') ?? undefined)
  const [agent, setAgent] = useState<string>()

  const segments = useMemo(() => lanes.flatMap((lane) => lane.segments), [lanes])
  const segment =
    segments.find((entry) => entry.key === segmentKey) ??
    segments.find((entry) => entry.calls.some((call) => call.record_id === callId))
  const call =
    segment?.calls.find((entry) => entry.record_id === callId) ??
    (segment?.calls.length === 1 ? segment.calls[0] : undefined)
  const selectSegment = (next: RunSegment) => {
    setSegmentKey(next.key)
    setCallId(next.calls.length === 1 ? next.calls[0].record_id : undefined)
  }
  const selectCall = (next: RunCall) => {
    setSegmentKey(segments.find((entry) => entry.calls.includes(next))?.key)
    setCallId(next.record_id)
  }

  const calls = useMemo(
    () => (response?.calls ?? []).filter((entry) => !agent || entry.agent === agent),
    [agent, response],
  )
  const callIndex = useMemo(
    () => new Map((response?.calls ?? []).map((entry, index) => [entry.record_id, index + 1])),
    [response],
  )

  if (!sessionId || notFound) {
    return (
      <PageContainer>
        <BackLink />
        <Card className="border-dashed shadow-none">
          <EmptyState
            icon={SearchX}
            tone="blue"
            title={sessionId ? 'No run under this session' : 'Pick an agent run'}
            description={
              sessionId
                ? `Replay holds no requests for session ${displaySessionId(sessionId)}. Its records may have expired, or Replay was off when it ran.`
                : 'Open a run from the Agent runs list to see its agents, their calls and what they cost.'
            }
            actions={<Button onClick={() => navigate(RUNS_TAB)}>Back to agent runs</Button>}
          />
        </Card>
      </PageContainer>
    )
  }

  if (unavailable) {
    return (
      <PageContainer>
        <BackLink />
        <Card className="border-dashed shadow-none">
          <InsightsUnavailable reason={unavailable} noun="agent runs" onRetry={() => void load()} />
        </Card>
      </PageContainer>
    )
  }

  if (!response) {
    return (
      <PageContainer>
        <BackLink />
        {error ? (
          <InsightsLoadError
            title="The run could not load"
            message={error}
            onRetry={() => void load()}
          />
        ) : null}
        {loading ? <RunSkeleton /> : null}
      </PageContainer>
    )
  }

  const run = response.run
  const subagents = run.agents.filter((entry) => entry.label !== 'main').length
  const shownSession = displaySessionId(run.session_id)
  const columns: Column<RunCall>[] = [
    {
      id: 'index',
      header: '#',
      width: '44px',
      cell: (entry) => (
        <span className="font-mono text-xs text-subtle-foreground tabular">
          {callIndex.get(entry.record_id)}
        </span>
      ),
    },
    {
      id: 'time',
      header: 'Time',
      width: '96px',
      sortValue: (entry) => Date.parse(entry.timestamp),
      cell: (entry) => (
        <span
          className="font-mono text-xs text-muted-foreground tabular"
          title={formatDateTime(entry.timestamp)}
        >
          {new Date(entry.timestamp).toLocaleTimeString()}
        </span>
      ),
    },
    {
      id: 'agent',
      header: 'Agent',
      cell: (entry) => (
        <span className="inline-flex items-center gap-1.5 font-mono text-[12px]">
          {entry.agent === 'main' ? (
            <User className="size-3 text-subtle-foreground" />
          ) : (
            <CornerDownRight className="size-3 text-subtle-foreground" />
          )}
          {entry.agent}
        </span>
      ),
    },
    {
      id: 'decision',
      header: 'Decision',
      hideBelow: 'lg',
      cell: (entry) =>
        entry.decision ? (
          <InsightsDecisionBadge
            decision={entry.decision}
            label={formatInsightsDecisionName(entry.decision)}
          />
        ) : (
          <span className="text-xs text-subtle-foreground">
            {entry.request_kind ? entry.request_kind.replace(/_/g, ' ') : '—'}
          </span>
        ),
    },
    {
      id: 'model',
      header: 'Model',
      className: 'max-w-0 w-full',
      cell: (entry) => (
        <span className="flex min-w-0 items-center gap-1.5">
          <ModelChip name={entry.model} fallback="No model" className="text-[13px]" />
          {entry.previous_model ? (
            <span
              className="truncate text-[11px] text-subtle-foreground"
              title={`Switched from ${entry.previous_model}`}
            >
              from {entry.previous_model}
            </span>
          ) : null}
          {entry.hops?.length ? (
            <Badge tone="outline" size="sm" title="Looper hops this call made">
              {countText(entry.hops.length, 'hop')}
            </Badge>
          ) : null}
        </span>
      ),
    },
    {
      id: 'status',
      header: 'Status',
      cell: (entry) => (
        <span
          className={cn(
            'font-mono text-xs',
            entry.lifecycle_state === 'failed' || (entry.response_status ?? 0) >= 400
              ? 'text-destructive-foreground'
              : 'text-muted-foreground',
          )}
        >
          {describeCallStatus(entry)}
        </span>
      ),
    },
    {
      id: 'latency',
      header: 'Latency',
      align: 'right',
      hideBelow: 'md',
      sortValue: (entry) => entry.duration_ms,
      cell: (entry) => (
        <span className="font-mono text-xs tabular">{formatLatency(entry.duration_ms)}</span>
      ),
    },
    {
      id: 'tokens',
      header: 'Tokens',
      align: 'right',
      hideBelow: 'lg',
      cell: (entry) => (
        <span className="font-mono text-xs tabular">
          {formatTokenCount(entry.prompt_tokens)}
          <span className="text-subtle-foreground"> / </span>
          {formatTokenCount(entry.completion_tokens)}
        </span>
      ),
    },
    {
      id: 'cost',
      header: 'Cost',
      align: 'right',
      hideBelow: 'md',
      sortValue: (entry) => entry.cost ?? -1,
      cell: (entry) => (
        <span className="font-mono text-xs tabular">
          {entry.cost !== undefined ? formatInsightsCost(entry.cost, entry.currency) : '—'}
        </span>
      ),
    },
    {
      id: 'events',
      header: 'Events',
      width: '72px',
      cell: (entry) => <CallMarkers markers={entry.markers} />,
    },
  ]

  return (
    <PageContainer key={run.session_id}>
      <BackLink />
      <PageHeader
        icon={<IconTile icon={Bot} tone="blue" size="lg" />}
        title={
          <span className="inline-flex flex-wrap items-center gap-2">
            <span title={run.prompt}>{runTitle(run)}</span>
            <RunStatus run={run} />
          </span>
        }
        description={`${countText(run.calls, 'call')} by ${subagents === 0 ? 'the main agent' : `the main agent and ${countText(subagents, 'subagent')}`}, over ${formatLatency(run.duration_ms)}.`}
        meta={
          <>
            <MetaItem>
              <HarnessName harness={run.harness} size={13} />
            </MetaItem>
            <MetaItem label="Session">
              <span className="inline-flex items-center gap-1">
                <span className="font-mono" title={run.session_id}>
                  {shownSession}
                </span>
                <CopyButton value={run.session_id} label="Copy session ID" className="size-6" />
              </span>
            </MetaItem>
            <MetaItem icon={<Clock />}>{formatDateTime(run.started_at)}</MetaItem>
            {run.events ? <RunEvents events={run.events} /> : null}
          </>
        }
        actions={
          <>
            <Button
              variant="outline"
              onClick={() => navigate(getSessionRequestsPath(run.session_id))}
            >
              <ListTree />
              View requests
            </Button>
            <RefreshButton onClick={() => void load()} loading={loading} />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="icon" aria-label="More actions">
                  <MoreHorizontal />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuItem onSelect={() => navigate(getReviewsPath(run.session_id))}>
                  <ShieldCheck />
                  Approval reviews
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() => {
                    void navigator.clipboard?.writeText(window.location.href).catch(() => undefined)
                    toast.success('Link copied')
                  }}
                >
                  <ListTree />
                  Copy link
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </>
        }
      />
      {error ? (
        <InsightsLoadError
          title="The run could not refresh"
          message={error}
          onRetry={() => void load()}
        />
      ) : null}

      <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <MiniStat label="Model calls" value={run.calls} />
        <MiniStat
          label="Agents"
          value={run.agents.length}
          hint={subagents ? countText(subagents, 'subagent') : 'Main agent only'}
        />
        <MiniStat label="Duration" value={formatLatency(run.duration_ms)} />
        <MiniStat
          label="Cost"
          value={formatRunCost(run.cost)}
          hint={
            run.cost.by_currency?.some((entry) => entry.router_call_spend > 0)
              ? 'Router calls included'
              : undefined
          }
        />
        <MiniStat
          label="Tokens"
          value={
            <span>
              {formatTokenCount(run.cost.prompt_tokens)}
              <span className="text-subtle-foreground"> / </span>
              {formatTokenCount(run.cost.completion_tokens)}
            </span>
          }
          hint="Input / output"
        />
        <MiniStat
          label="Tool results"
          value={run.outcome.tool_results}
          hint={run.outcome.tool_failures ? `${run.outcome.tool_failures} failed` : 'None failed'}
        />
      </div>

      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card>
          <CardHeader>
            <CardTitle>Timeline</CardTitle>
            <CardDescription>
              One lane per agent; a block is a stretch of calls on one model. Click one for its
              calls.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <RunDiagram
              run={run}
              lanes={lanes}
              selected={segment?.key}
              activeAgent={agent}
              onSelect={selectSegment}
            />
            <RunLegend run={run} />
            {segment || call ? (
              <div className="border-t border-border pt-4">
                {segment ? (
                  <RunSegmentPanel segment={segment} call={call} onSelectCall={selectCall} />
                ) : call ? (
                  <RunCallDetails call={call} />
                ) : null}
              </div>
            ) : null}
          </CardContent>
        </Card>
        <div className="flex min-w-0 flex-col gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Agents</CardTitle>
              <CardDescription>Click an agent to follow its calls</CardDescription>
            </CardHeader>
            <CardContent className="px-2.5 pt-2">
              <AgentTree lanes={lanes} active={agent} onSelect={setAgent} />
            </CardContent>
          </Card>
          {run.outcome.reported?.length ? (
            <Card>
              <CardHeader>
                <CardTitle>Reported outcomes</CardTitle>
                <CardDescription>
                  Outcomes clients reported for the run&apos;s calls
                </CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-1.5 pt-3 text-[13px]">
                {run.outcome.reported.map(({ record_id: recordId, outcome }, index) => (
                  <Link
                    key={`${recordId}-${index}`}
                    to={getInsightsRecordPath(recordId)}
                    className="flex items-center justify-between gap-2 rounded-md px-2 py-1 hover:bg-surface-subtle"
                  >
                    <span className="truncate">{outcome.source}</span>
                    <Badge tone="neutral" size="sm">
                      {outcome.verdict}
                    </Badge>
                  </Link>
                ))}
              </CardContent>
            </Card>
          ) : null}
        </div>
        <section aria-label="Calls" className="flex min-w-0 flex-col gap-2 xl:col-span-2">
          <div className="flex items-end justify-between gap-3">
            <div>
              <h2 className="text-[15px] font-semibold tracking-[-0.012em]">Calls</h2>
              <p className="text-[13px] text-muted-foreground">
                {agent
                  ? `${countText(calls.length, 'call')} by ${agent}`
                  : 'Every routed call of the run, in time order'}
              </p>
            </div>
            {agent ? (
              <Button variant="ghost" size="sm" onClick={() => setAgent(undefined)}>
                Show every agent
              </Button>
            ) : null}
          </div>
          <DataTable
            data={calls}
            columns={columns}
            getRowId={(entry) => entry.record_id}
            activeRowId={call?.record_id}
            onRowClick={(entry) => navigate(getInsightsRecordPath(entry.record_id))}
          />
        </section>
      </div>
    </PageContainer>
  )
}
