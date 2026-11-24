import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CornerDownRight, Crosshair, MessageSquareText, PanelRightOpen, User } from 'lucide-react'

import {
  Badge,
  DataTable,
  DataToolbar,
  DropdownMenuItem,
  EmptyState,
  FilterPill,
  MiniBar,
  StatusDot,
  cn,
  type Column,
  type FilterOption,
} from '../ui'
import { formatDateTime } from '../utils/dateTime'

import { AGENT_EVENT_INFO, AGENT_EVENTS, callEvents } from './insightsAgentEvents'
import { AgentEventChips, AgentEventIcon } from './InsightsAgentEvents'
import { subagentKindLabel } from './insightsAgentIdentity'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import { formatLatency, formatTokenCount } from './insightsDisplayFormat'
import { Pager, SummaryStrip, SummaryTile } from './InsightsDisplay'
import { DEFAULT_PAGE_SIZE, pageOf } from './insightsListData'
import { ModelChip } from './insightsModelLabelSupport'
import { formatInsightsDecisionName, getInsightsRecordPath } from './insightsPageSupport'
import {
  callCostText,
  countText,
  countVerdicts,
  describeCallStatus,
  filterRunCalls,
  formatRunCost,
  formatRunDuration,
  OUTCOME_VERDICTS,
  orderRunAgents,
  outcomeWord,
  RUN_OUTCOME_LABELS,
  RUN_STATUS_TONES,
  runCurrency,
  verdictLabel,
  type CallsView,
} from './insightsRunSupport'
import type { RunAgent, RunCall, RunReportedOutcome, RunSummary } from './insightsRunTypes'

// A call's events as the Timeline marks them, the ones the table is filtered
// by first and solid.
function CallEvents({ call, filtered }: { call: RunCall; filtered: string[] }) {
  const events = callEvents(call)
  if (!events.length) return <span className="text-xs text-subtle-foreground">—</span>
  return <AgentEventChips events={events} emphasize={filtered} />
}

function AgentName({ label }: { label: string }) {
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5 font-mono text-[12px]">
      {label === 'main' ? (
        <User className="size-3 shrink-0 text-subtle-foreground" />
      ) : (
        <CornerDownRight className="size-3 shrink-0 text-subtle-foreground" />
      )}
      <span className="truncate">{label}</span>
    </span>
  )
}

// ---------------------------------------------------------------- calls

// Every call of the run in time order, a page at a time: the run API returns
// them all, so the filters and pages run in the browser.
export function RunCallsPanel({
  run,
  calls,
  numbers,
  view,
  onView,
  activeId,
  onShow,
}: {
  run: RunSummary
  calls: RunCall[]
  numbers: Map<string, number>
  view: CallsView
  onView: (patch: Partial<CallsView>) => void
  activeId?: string
  onShow: (call: RunCall) => void
}) {
  const navigate = useNavigate()
  const filtered = filterRunCalls(calls, view)
  const shown = pageOf(filtered, view.page, view.size)
  const agentOptions: FilterOption[] = orderRunAgents(run.agents).map(({ agent }) => ({
    value: agent.label,
    label: agent.label,
    count: agent.calls,
  }))
  const currency = runCurrency(run.cost)
  const markerOptions: FilterOption[] = AGENT_EVENTS.filter(
    (event) => run.events?.[event] || view.markers.includes(event),
  ).map((event) => ({
    value: event,
    label: AGENT_EVENT_INFO[event].label,
    icon: <AgentEventIcon event={event} className="size-3.5" />,
    count: run.events?.[event] ?? 0,
  }))
  const columns: Column<RunCall>[] = [
    {
      id: 'index',
      header: '#',
      width: '56px',
      cell: (entry) => (
        <span className="font-mono text-xs text-subtle-foreground tabular">
          {numbers.get(entry.record_id)}
        </span>
      ),
    },
    {
      id: 'time',
      header: 'Time',
      width: '96px',
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
      className: 'max-w-[160px]',
      cell: (entry) => <AgentName label={entry.agent} />,
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
            className="max-w-[160px]"
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
          <ModelChip name={entry.model} fallback="No model" className="min-w-0 text-[13px]" />
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
            'whitespace-nowrap font-mono text-xs',
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
      cell: (entry) => (
        <span className="font-mono text-xs tabular">{formatLatency(entry.duration_ms)}</span>
      ),
    },
    {
      id: 'tokens',
      header: 'Tokens',
      align: 'right',
      hideBelow: 'xl',
      cell: (entry) => (
        <span className="whitespace-nowrap font-mono text-xs tabular">
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
      cell: (entry) => (
        <span className="font-mono text-xs tabular">{callCostText(entry, currency)}</span>
      ),
    },
    {
      id: 'events',
      header: 'Events',
      cell: (entry) => <CallEvents call={entry} filtered={view.markers} />,
    },
  ]
  return (
    <div className="flex flex-col">
      <DataToolbar
        filters={
          <>
            <FilterPill
              label="Agent"
              options={agentOptions}
              selected={view.agents}
              onChange={(agents) => onView({ agents, page: 1 })}
            />
            {markerOptions.length > 0 ? (
              <FilterPill
                label="Events"
                options={markerOptions}
                selected={view.markers}
                onChange={(markers) => onView({ markers, page: 1 })}
              />
            ) : null}
          </>
        }
      />
      <DataTable
        data={shown.rows}
        columns={columns}
        getRowId={(entry) => entry.record_id}
        activeRowId={activeId}
        onRowClick={(entry) => navigate(getInsightsRecordPath(entry.record_id))}
        rowActions={(entry) => (
          <>
            <DropdownMenuItem onSelect={() => navigate(getInsightsRecordPath(entry.record_id))}>
              <PanelRightOpen />
              Open the request
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => onShow(entry)}>
              <Crosshair />
              Show in the timeline
            </DropdownMenuItem>
          </>
        )}
        empty={
          <EmptyState
            compact
            icon={MessageSquareText}
            tone="blue"
            title="No calls match"
            description="No call of this run matches the agent and event filters."
          />
        }
      />
      <Pager
        page={shown.page}
        pageSize={view.size}
        total={filtered.length}
        noun="call"
        onPage={(page) => onView({ page })}
        onPageSize={(size) => onView({ size, page: 1 })}
      />
    </div>
  )
}

// ---------------------------------------------------------------- agents

// The run's agents as a tree, a page at a time: the main agent, then each
// subagent under the agent that spawned it. A row follows the agent's calls.
export function RunAgentsPanel({
  run,
  onFollow,
}: {
  run: RunSummary
  onFollow: (label: string) => void
}) {
  const [page, setPage] = useState(1)
  const [size, setSize] = useState(DEFAULT_PAGE_SIZE)
  const tree = orderRunAgents(run.agents)
  const most = Math.max(1, ...run.agents.map((agent) => agent.calls))
  const shown = pageOf(tree, page, size)
  const columns: Column<{ agent: RunAgent; depth: number }>[] = [
    {
      id: 'agent',
      header: 'Agent',
      className: 'max-w-0 w-full',
      cell: ({ agent, depth }) => {
        const main = depth === 0 && agent.label === 'main'
        const kind = subagentKindLabel(agent.kind)
        return (
          <span
            className="flex min-w-0 items-start gap-2.5"
            style={{ paddingLeft: `${depth * 1.1}rem` }}
          >
            <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-md border border-border bg-surface text-subtle-foreground">
              {main ? <User className="size-3.5" /> : <CornerDownRight className="size-3.5" />}
            </span>
            <span className="flex min-w-0 flex-col gap-0.5">
              <span className="flex min-w-0 items-center gap-1.5">
                <span className="truncate font-mono text-[12.5px] font-medium">{agent.label}</span>
                {kind ? (
                  <Badge tone="outline" size="sm">
                    {kind}
                  </Badge>
                ) : null}
              </span>
              <span className="truncate text-xs text-subtle-foreground">
                {main ? 'Main agent' : `Spawned by ${agent.parent || 'main'}`}
                {agent.id ? ` · #${agent.id.slice(0, 8)}` : agent.id === '' ? ' · id hidden' : ''}
              </span>
            </span>
          </span>
        )
      },
    },
    {
      id: 'calls',
      header: 'Calls',
      align: 'right',
      width: '140px',
      cell: ({ agent, depth }) => (
        <span className="flex flex-col items-end gap-1">
          <span className="font-mono text-xs tabular">{agent.calls.toLocaleString()}</span>
          <MiniBar
            value={agent.calls}
            max={most}
            tone={depth === 0 ? 'blue' : 'violet'}
            className="h-1 w-20"
          />
        </span>
      ),
    },
    {
      id: 'models',
      header: 'Models',
      hideBelow: 'lg',
      cell: ({ agent }) =>
        agent.models?.length ? (
          <span className="flex min-w-0 max-w-[240px] flex-col gap-0.5">
            <ModelChip name={agent.models[0]} size={12} className="text-[13px]" />
            {agent.models.length > 1 ? (
              <span className="text-[11px] text-subtle-foreground">
                +{agent.models.length - 1} more
              </span>
            ) : null}
          </span>
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        ),
    },
    {
      id: 'cost',
      header: 'Cost',
      align: 'right',
      hideBelow: 'md',
      cell: ({ agent }) => (
        <span className="whitespace-nowrap font-mono text-xs tabular">
          {formatRunCost(agent.cost)}
        </span>
      ),
    },
    {
      id: 'active',
      header: 'Active',
      align: 'right',
      hideBelow: 'sm',
      cell: ({ agent }) => {
        const first = Date.parse(agent.first_call_at)
        const last = Date.parse(agent.last_call_at)
        if (!Number.isFinite(first) || !Number.isFinite(last)) {
          return <span className="text-xs text-subtle-foreground">—</span>
        }
        return (
          <span
            className="whitespace-nowrap font-mono text-xs text-muted-foreground tabular"
            title={`${formatDateTime(agent.first_call_at)} to ${formatDateTime(agent.last_call_at)}`}
          >
            {formatRunDuration(Math.max(0, last - first))}
          </span>
        )
      },
    },
  ]
  return (
    <div className="flex flex-col">
      <DataTable
        data={shown.rows}
        columns={columns}
        getRowId={({ agent }) => agent.label}
        onRowClick={({ agent }) => onFollow(agent.label)}
        rowActions={({ agent }) => (
          <DropdownMenuItem onSelect={() => onFollow(agent.label)}>
            <Crosshair />
            Follow its calls
          </DropdownMenuItem>
        )}
      />
      <Pager
        page={shown.page}
        pageSize={size}
        total={tree.length}
        noun="agent"
        onPage={setPage}
        onPageSize={(next) => {
          setSize(next)
          setPage(1)
        }}
      />
    </div>
  )
}

// ---------------------------------------------------------------- outcomes

type ReportedRow = { record_id: string; outcome: RunReportedOutcome; key: string }

// How the run ended and what was reported about it: the final answer, the
// tool results its calls carried, and the outcomes clients, evaluators and
// operators reported through the outcome API, one row per report.
export function RunOutcomesPanel({
  run,
  calls,
  numbers,
}: {
  run: RunSummary
  calls: RunCall[]
  numbers: Map<string, number>
}) {
  const navigate = useNavigate()
  const [page, setPage] = useState(1)
  const [size, setSize] = useState(DEFAULT_PAGE_SIZE)
  const reported: ReportedRow[] = (run.outcome.reported ?? []).map((entry, index) => ({
    ...entry,
    key: `${entry.record_id}-${index}`,
  }))
  const agentOf = new Map(calls.map((call) => [call.record_id, call.agent]))
  const verdicts = countVerdicts(reported)
  const finalAnswer = run.outcome.final_answer_record_id
  const shown = pageOf(reported, page, size)
  const columns: Column<ReportedRow>[] = [
    {
      id: 'call',
      header: 'Call',
      width: '72px',
      cell: (row) => (
        <span className="font-mono text-xs text-subtle-foreground tabular">
          {numbers.get(row.record_id) ? `#${numbers.get(row.record_id)}` : '—'}
        </span>
      ),
    },
    {
      id: 'agent',
      header: 'Agent',
      hideBelow: 'md',
      cell: (row) =>
        agentOf.get(row.record_id) ? (
          <AgentName label={agentOf.get(row.record_id) ?? ''} />
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        ),
    },
    {
      id: 'source',
      header: 'Reported by',
      cell: (row) => <span className="text-[13px]">{outcomeWord(row.outcome.source)}</span>,
    },
    {
      id: 'target',
      header: 'About',
      hideBelow: 'sm',
      cell: (row) => (
        <span className="text-[13px] text-muted-foreground">{outcomeWord(row.outcome.target)}</span>
      ),
    },
    {
      id: 'verdict',
      header: 'Verdict',
      cell: (row) => (
        <Badge tone={OUTCOME_VERDICTS[row.outcome.verdict]?.tone ?? 'neutral'} size="sm">
          {verdictLabel(row.outcome.verdict)}
        </Badge>
      ),
    },
    {
      id: 'reason',
      header: 'Reason',
      className: 'max-w-0 w-full',
      cell: (row) =>
        row.outcome.reason ? (
          <span className="block truncate text-xs text-muted-foreground" title={row.outcome.reason}>
            {row.outcome.reason}
          </span>
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        ),
    },
    {
      id: 'reported',
      header: 'Reported',
      align: 'right',
      hideBelow: 'lg',
      cell: (row) =>
        row.outcome.timestamp ? (
          <span
            className="whitespace-nowrap font-mono text-xs text-muted-foreground tabular"
            title={formatDateTime(row.outcome.timestamp)}
          >
            {new Date(row.outcome.timestamp).toLocaleTimeString()}
          </span>
        ) : (
          <span className="text-xs text-subtle-foreground">—</span>
        ),
    },
  ]
  return (
    <div className="flex flex-col gap-4">
      <SummaryStrip label="Outcome" className="grid-cols-1 sm:grid-cols-3">
        <SummaryTile
          label="Ended"
          value={
            <span className="inline-flex items-center gap-2">
              <StatusDot
                tone={RUN_STATUS_TONES[run.outcome.status] ?? 'neutral'}
                pulse={run.outcome.status === 'in_progress'}
              />
              {RUN_OUTCOME_LABELS[run.outcome.status] ?? run.outcome.status}
            </span>
          }
          hint={
            finalAnswer && numbers.get(finalAnswer)
              ? `Final answer on call #${numbers.get(finalAnswer)}`
              : 'No final answer recorded'
          }
        />
        <SummaryTile
          label="Tool results"
          value={run.outcome.tool_results.toLocaleString()}
          hint={
            run.outcome.tool_failures
              ? `${run.outcome.tool_failures.toLocaleString()} failed`
              : 'None failed'
          }
          tone={run.outcome.tool_failures ? 'negative' : undefined}
        />
        <SummaryTile
          label="Reported outcomes"
          value={reported.length.toLocaleString()}
          hint={
            verdicts.length
              ? verdicts
                  .map(({ verdict, count }) => `${count} ${verdictLabel(verdict).toLowerCase()}`)
                  .join(' · ')
              : 'None reported'
          }
        />
      </SummaryStrip>
      {reported.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border">
          <EmptyState
            compact
            icon={MessageSquareText}
            tone="blue"
            title="No outcomes reported for this run"
            description="Clients, evaluators and operators report how a call went through the Router outcome API. Each report shows here with its verdict and reason."
          />
        </div>
      ) : (
        <div className="flex flex-col">
          <DataTable
            data={shown.rows}
            columns={columns}
            getRowId={(row) => row.key}
            onRowClick={(row) => navigate(getInsightsRecordPath(row.record_id))}
          />
          <Pager
            page={shown.page}
            pageSize={size}
            total={reported.length}
            noun="reported outcome"
            onPage={setPage}
            onPageSize={(next) => {
              setSize(next)
              setPage(1)
            }}
          />
        </div>
      )}
    </div>
  )
}
