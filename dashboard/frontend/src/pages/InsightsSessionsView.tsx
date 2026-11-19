import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Bot, ChevronRight, ExternalLink, GitBranch, KeyRound, ListTree } from 'lucide-react'

import { Badge, Button, Card, EmptyState, Skeleton, Tooltip, cn } from '../ui'

import { callEvents, runEventCounts } from './insightsAgentEvents'
import { AgentEventChips } from './InsightsAgentEvents'
import { REQUEST_CLASS_LABELS } from './insightsAgentIdentity'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { formatLatency, formatTokenCount, useNow } from './insightsDisplayFormat'
import { InsightsLoadError, InsightsUnavailable, Pager, RelativeTime } from './InsightsDisplay'
import { HarnessLogo } from './InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo, HARNESS_SIGNALS } from './insightsHarness'
import { readPage, readPageSize, writePage } from './insightsListData'
import { ModelChip } from './insightsModelLabelSupport'
import { fetchInsightsJSON } from './insightsPageApi'
import { formatInsightsDecisionName, getInsightsRecordPath } from './insightsPageSupport'
import { API_CLIENT_FILTER, type RequestFilters } from './insightsRequestsData'
import { API_RUN_FILTER, useRunsData, type RunFilters } from './insightsRunsData'
import { RunStatus } from './InsightsRunsView'
import {
  countText,
  formatRunCost,
  formatRunDuration,
  getRunPath,
  runAgentTreeText,
  runHeadline,
} from './insightsRunSupport'
import type { RunCall, RunResponse, RunSummary } from './insightsRunTypes'
import {
  agentTypeText,
  buildAgentTree,
  harnessSignalRows,
  HARNESS_SIGNAL_LABELS,
  isDerivedSession,
  type AgentNode,
  type SignalCell,
} from './insightsSessionTree'

/** Calls an expanded agent lists before "Show all". */
const AGENT_CALL_LIMIT = 200

function runFiltersOf(filters: RequestFilters): RunFilters {
  return {
    search: filters.session || filters.search,
    range: filters.range,
    harness: filters.harness.map((value) => (value === API_CLIENT_FILTER ? API_RUN_FILTER : value)),
    status: [],
  }
}

// ---------------------------------------------------------------- signals by harness

function SignalMark({ cell }: { cell: SignalCell }) {
  switch (cell.state) {
    case 'seen':
      return (
        <span className="font-mono text-xs tabular text-foreground" data-signal="seen">
          ✓ {cell.count.toLocaleString()}
        </span>
      )
    case 'derived':
      return (
        <Tooltip content="The harness sends no session id; the router keys a conversation by how it starts.">
          <span className="text-xs text-muted-foreground" data-signal="derived">
            router-keyed
          </span>
        </Tooltip>
      )
    case 'never':
      return (
        <Tooltip content="This harness's requests do not carry this signal.">
          <span className="text-xs text-subtle-foreground" data-signal="never">
            not sent
          </span>
        </Tooltip>
      )
    default:
      return (
        <Tooltip content="Not seen in the sessions loaded for this range.">
          <span className="text-xs text-subtle-foreground" data-signal="unseen">
            —
          </span>
        </Tooltip>
      )
  }
}

/** Which agent signals each harness gave in the loaded sessions. */
export function HarnessSignalsCard({ runs }: { runs: RunSummary[] }) {
  const rows = useMemo(() => harnessSignalRows(runs), [runs])
  if (rows.length === 0) return null
  return (
    <Card className="mb-3 overflow-hidden px-0 py-0 shadow-none" aria-label="Signals by harness">
      <details open className="group">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-2.5 text-[13px] font-medium">
          <span className="flex items-center gap-2">
            <ChevronRight className="size-3.5 text-subtle-foreground transition-transform group-open:rotate-90" />
            Signals by harness
          </span>
          <span className="text-xs font-normal text-subtle-foreground">
            What each harness told the router in {countText(runs.length, 'loaded session')}
          </span>
        </summary>
        <div className="overflow-x-auto border-t border-border">
          <table className="w-full text-left text-xs" aria-label="Signals by harness">
            <thead className="text-subtle-foreground">
              <tr>
                <th scope="col" className="px-4 py-2 font-medium">
                  Harness
                </th>
                {HARNESS_SIGNALS.map((signal) => (
                  <th key={signal} scope="col" className="px-3 py-2 font-medium whitespace-nowrap">
                    {HARNESS_SIGNAL_LABELS[signal]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr
                  key={row.harness || 'api'}
                  className="border-t border-border"
                  data-harness={row.harness || 'api_client'}
                >
                  <th scope="row" className="px-4 py-2 font-medium">
                    <span className="flex items-center gap-2 whitespace-nowrap">
                      <HarnessLogo harness={row.harness || undefined} size={14} />
                      {harnessInfo(row.harness)?.label ?? API_CLIENT_LABEL}
                      <span className="font-normal text-subtle-foreground">
                        {row.sessions.toLocaleString()}
                      </span>
                    </span>
                  </th>
                  {HARNESS_SIGNALS.map((signal) => (
                    <td key={signal} className="px-3 py-2">
                      <SignalMark cell={row.cells[signal]} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Card>
  )
}

// ---------------------------------------------------------------- one run as a tree

function callClass(call: RunCall, agentLabel: string): string {
  switch (call.request_class) {
    case 'compaction':
      return 'Compaction'
    case 'auxiliary':
      return 'Housekeeping'
    case 'workflow':
      return 'Workflow'
    default:
      if (call.request_kind === 'session_title') return 'Title'
      return agentLabel === 'main' ? REQUEST_CLASS_LABELS.main : REQUEST_CLASS_LABELS.subagent
  }
}

const clock = (iso: string) =>
  new Date(iso).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })

function TurnRow({
  call,
  index,
  agentLabel,
}: {
  call: RunCall
  index: number
  agentLabel: string
}) {
  const navigate = useNavigate()
  const events = callEvents(call)
  const tokens =
    typeof call.prompt_tokens === 'number' || typeof call.completion_tokens === 'number'
      ? `${formatTokenCount(call.prompt_tokens)} / ${formatTokenCount(call.completion_tokens)}`
      : '—'
  return (
    <li
      className="grid grid-cols-[64px_36px_minmax(96px,auto)_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 border-t border-border/60 px-3 py-1.5 text-xs first:border-t-0 max-lg:grid-cols-[64px_36px_minmax(0,1fr)]"
      data-turn={call.record_id}
    >
      <span className="font-mono text-subtle-foreground tabular">{clock(call.timestamp)}</span>
      <span className="font-mono text-subtle-foreground tabular">#{index + 1}</span>
      <span className="truncate text-muted-foreground">{callClass(call, agentLabel)}</span>
      <span className="flex min-w-0 flex-wrap items-center gap-1.5 max-lg:col-span-full">
        {call.decision ? (
          <InsightsDecisionBadge
            decision={call.decision}
            label={formatInsightsDecisionName(call.decision)}
            className="max-w-[140px]"
          />
        ) : null}
        <ModelChip
          name={call.model}
          fallback={call.decision ? 'No model call' : 'No backend'}
          className="max-w-[180px]"
        />
        <AgentEventChips events={events} />
      </span>
      <span className="flex items-center justify-end gap-3 font-mono tabular text-muted-foreground max-lg:col-span-full max-lg:justify-start">
        <span>{formatLatency(call.duration_ms)}</span>
        <Tooltip content="Input / output tokens">
          <span>{tokens}</span>
        </Tooltip>
        <Button
          variant="ghost"
          size="xs"
          className="h-6 px-1.5"
          aria-label={`Open request ${index + 1} of ${agentLabel}`}
          onClick={() => navigate(getInsightsRecordPath(call.record_id))}
        >
          <ExternalLink />
        </Button>
      </span>
      {(call.hops?.length ?? 0) > 0 ? (
        <details className="col-span-full pl-[112px] text-subtle-foreground max-lg:pl-0">
          <summary className="cursor-pointer">
            {countText(call.hops?.length ?? 0, 'router hop')}
          </summary>
          <ul className="mt-1 flex flex-col gap-0.5">
            {call.hops?.map((hop) => (
              <li key={hop.record_id} className="flex items-center gap-2">
                <ModelChip name={hop.model} fallback="No backend" className="max-w-[180px]" />
                <span className="font-mono tabular">{formatLatency(hop.duration_ms)}</span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </li>
  )
}

function AgentBranch({ node }: { node: AgentNode }) {
  const [all, setAll] = useState(false)
  const { agent } = node
  const main = agent.label === 'main'
  const type = agentTypeText(agent)
  const shown = all ? node.calls : node.calls.slice(0, AGENT_CALL_LIMIT)
  return (
    <li className="flex flex-col gap-1" data-agent={agent.label}>
      <div className="flex flex-wrap items-center gap-2 text-[13px]">
        {main ? (
          <Bot className="size-3.5 text-muted-foreground" aria-hidden />
        ) : (
          <GitBranch className="size-3.5 text-chart-3" aria-hidden />
        )}
        <span className="font-medium">
          {main ? 'Main agent' : agent.label.replace(/^subagent/, 'Subagent')}
        </span>
        {type ? <Badge tone="violet">{type}</Badge> : null}
        {agent.parent && !main ? (
          <span className="text-xs text-subtle-foreground">
            started by {agent.parent === 'main' ? 'the main agent' : agent.parent}
          </span>
        ) : null}
        <span className="text-xs text-subtle-foreground">
          {countText(agent.calls, 'call')}
          {agent.models?.length
            ? ` · ${agent.models.length === 1 ? '1 model' : `${agent.models.length} models`}`
            : ''}
          {` · ${formatRunCost(agent.cost)}`}
        </span>
      </div>
      <ol
        className="ml-[7px] rounded-md border border-border bg-surface"
        aria-label={`${main ? 'Main agent' : agent.label} turns`}
      >
        {shown.map((call, index) => (
          <TurnRow key={call.record_id} call={call} index={index} agentLabel={agent.label} />
        ))}
      </ol>
      {!all && node.calls.length > AGENT_CALL_LIMIT ? (
        <Button variant="ghost" size="xs" className="w-fit" onClick={() => setAll(true)}>
          Show all {node.calls.length.toLocaleString()} turns
        </Button>
      ) : null}
      {node.children.length > 0 ? (
        <ul
          className="ml-3 flex flex-col gap-3 border-l border-dashed border-border-strong pl-4 pt-1"
          aria-label={`Agents started by ${agent.label}`}
        >
          {node.children.map((child) => (
            <AgentBranch key={child.agent.label} node={child} />
          ))}
        </ul>
      ) : null}
    </li>
  )
}

function SessionTree({ sessionId }: { sessionId: string }) {
  const [state, setState] = useState<{ run?: RunResponse; error?: string }>({})
  const load = useCallback(() => {
    setState({})
    const query = new URLSearchParams({ session_id: sessionId })
    fetchInsightsJSON<RunResponse>(
      `/api/router/api/v1/observability/replays/run?${query.toString()}`,
      'agent run',
    )
      .then((run) => setState({ run }))
      .catch((error: unknown) =>
        setState({ error: error instanceof Error ? error.message : 'Unknown error' }),
      )
  }, [sessionId])
  useEffect(() => load(), [load])
  const tree = useMemo(() => (state.run ? buildAgentTree(state.run) : []), [state.run])
  if (state.error) {
    return (
      <InsightsLoadError title="This session could not load" message={state.error} onRetry={load} />
    )
  }
  if (!state.run) {
    return (
      <div className="flex flex-col gap-2 px-1 py-2" aria-busy>
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-16 w-full" />
      </div>
    )
  }
  return (
    <ul className="flex flex-col gap-3" aria-label="Agents of this session">
      {tree.map((node) => (
        <AgentBranch key={node.agent.label} node={node} />
      ))}
    </ul>
  )
}

// ---------------------------------------------------------------- the list

function SessionRow({
  run,
  open,
  onToggle,
  now,
}: {
  run: RunSummary
  open: boolean
  onToggle: () => void
  now: number
}) {
  const navigate = useNavigate()
  const info = harnessInfo(run.harness)
  const derived = isDerivedSession(run.session_id)
  const tokens = run.cost.prompt_tokens + run.cost.completion_tokens
  const cost = formatRunCost(run.cost)
  const events = runEventCounts(run)
  const panel = `session-${run.session_id.replace(/[^a-zA-Z0-9_-]/g, '_')}`
  return (
    <li className="rounded-xl border border-border bg-card shadow-xs" data-session={run.session_id}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2.5">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls={panel}
          className="flex min-w-0 flex-1 items-center gap-3 rounded-md text-left focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
        >
          <ChevronRight
            className={cn(
              'size-4 shrink-0 text-subtle-foreground transition-transform',
              open && 'rotate-90',
            )}
            aria-hidden
          />
          <HarnessLogo harness={run.harness} size={18} framed />
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="flex min-w-0 items-center gap-2">
              <span className="truncate text-[13px] font-medium">{runHeadline(run)}</span>
              {derived ? (
                <Tooltip content="The harness sends no session id; the router keyed this conversation by how it starts.">
                  <Badge tone="outline" className="gap-1">
                    <KeyRound className="size-3" aria-hidden />
                    router-keyed
                  </Badge>
                </Tooltip>
              ) : null}
            </span>
            <span className="flex flex-wrap items-center gap-x-1.5 text-xs text-subtle-foreground">
              <span>{info?.label ?? API_CLIENT_LABEL}</span>
              <span aria-hidden>·</span>
              <RelativeTime iso={run.started_at} now={now} />
              <span aria-hidden>·</span>
              <span>{formatRunDuration(run.duration_ms)}</span>
              <span aria-hidden>·</span>
              <span>{countText(run.calls, 'call')}</span>
              <span aria-hidden>·</span>
              <span>{runAgentTreeText(run)}</span>
            </span>
          </span>
        </button>
        <span className="flex flex-wrap items-center gap-2">
          {run.models.slice(0, 2).map((model) => (
            <ModelChip key={model.name} name={model.name} className="max-w-[160px] text-xs" />
          ))}
        </span>
        <AgentEventChips events={events} compact />
        <span className="flex items-center gap-3 font-mono text-xs tabular text-muted-foreground">
          <Tooltip
            content={`${run.cost.prompt_tokens.toLocaleString()} in · ${run.cost.completion_tokens.toLocaleString()} out`}
          >
            <span>{formatTokenCount(tokens)} tok</span>
          </Tooltip>
          {cost === 'no price' ? (
            <Tooltip content="No call of this session has a recorded price.">
              <span className="text-subtle-foreground">—</span>
            </Tooltip>
          ) : (
            <span>{cost}</span>
          )}
        </span>
        <RunStatus run={run} />
        <Button
          variant="ghost"
          size="xs"
          className="h-6 px-1.5"
          aria-label="Open this agent run"
          onClick={() => navigate(getRunPath(run.session_id))}
        >
          <ExternalLink />
        </Button>
      </div>
      {open ? (
        <div id={panel} className="border-t border-border px-3 py-3">
          {events.length > 0 ? <AgentEventChips events={events} className="mb-3" /> : null}
          <SessionTree sessionId={run.session_id} />
        </div>
      ) : null}
    </li>
  )
}

/**
 * The Requests page's Sessions view: one row per harness session (an agent
 * run), each opening into its agents, the main agent first and every subagent
 * under the agent that started it, and each agent's turns.
 */
export default function InsightsSessionsView({
  filters,
  live,
}: {
  filters: RequestFilters
  live: boolean
}) {
  const [params, setParams] = useSearchParams()
  const page = readPage(params)
  const pageSize = readPageSize(params)
  const now = useNow(live ? 5_000 : 30_000)
  const runFilters = useMemo(() => runFiltersOf(filters), [filters])
  const data = useRunsData({
    filters: runFilters,
    page,
    pageSize,
    live,
    held: false,
    withWindow: true,
  })
  const [open, setOpen] = useState<Set<string>>(() => new Set())
  const toggle = (session: string) =>
    setOpen((current) => {
      const next = new Set(current)
      if (next.has(session)) next.delete(session)
      else next.add(session)
      return next
    })
  const signalRuns = data.windowLoaded ? data.window : data.rows

  if (data.unavailable) {
    return (
      <Card className="border-dashed shadow-none">
        <InsightsUnavailable reason={data.unavailable} onRetry={data.refresh} />
      </Card>
    )
  }
  return (
    <section aria-label="Sessions">
      {data.error ? (
        <InsightsLoadError
          title="Sessions could not load"
          message={data.error}
          onRetry={data.refresh}
        />
      ) : null}
      <HarnessSignalsCard runs={signalRuns} />
      {data.loading ? (
        <div className="flex flex-col gap-2" aria-busy>
          {Array.from({ length: 4 }, (_, index) => (
            <Skeleton key={index} className="h-14 w-full rounded-xl" />
          ))}
        </div>
      ) : data.rows.length === 0 ? (
        <EmptyState
          compact
          icon={ListTree}
          tone="blue"
          title="No sessions match"
          description="Agent harnesses that name a session, and conversations the router keys itself, appear here. Widen the time range or clear the filters."
        />
      ) : (
        <ul className={cn('flex flex-col gap-2 transition-opacity', data.stale && 'opacity-60')}>
          {data.rows.map((run) => (
            <SessionRow
              key={run.session_id}
              run={run}
              open={open.has(run.session_id)}
              onToggle={() => toggle(run.session_id)}
              now={now}
            />
          ))}
        </ul>
      )}
      <Pager
        page={page}
        pageSize={pageSize}
        total={data.total}
        noun="session"
        onPage={(next) => setParams((current) => writePage(current, next))}
        onPageSize={(size) => setParams((current) => writePage(current, 1, size))}
      />
    </section>
  )
}
