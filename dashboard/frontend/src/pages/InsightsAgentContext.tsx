import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowUpRight, Workflow } from 'lucide-react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle, CopyButton } from '../ui'

import {
  agentHashText,
  compactIdentifier,
  displaySessionId,
  housekeepingDetail,
  isSubagentRecord,
  requestClassOf,
  runAgentFor,
  sessionActionLabel,
  sessionReasonText,
  spawnsSubagent,
  subagentKindLabel,
  traceToolNames,
  type AgentRuleIndex,
} from './insightsAgentIdentity'
import { HarnessLogo, RequestClassBadge, ToolChips } from './InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo } from './insightsHarness'
import { ModelChip } from './insightsModelLabelSupport'
import type { InsightsRecord } from './insightsPageTypes'
import { getSessionRequestsPath, getTurnPath } from './insightsReviewSupport'
import { getRunPath } from './insightsRunSupport'
import type { RunSummary } from './insightsRunTypes'

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[92px_minmax(0,1fr)] gap-3 py-2">
      <dt className="text-xs text-subtle-foreground">{label}</dt>
      <dd className="min-w-0 text-[13px] text-foreground">{children}</dd>
    </div>
  )
}

const LINK = 'inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline'

/** Who sent a request: the harness, its session, the agent of the run, its turn, the tools it used and how session routing treated it. */
export default function InsightsAgentContext({
  record,
  run,
  turnCount,
  rules,
}: {
  record: InsightsRecord
  run?: RunSummary | null
  turnCount?: number
  rules?: AgentRuleIndex
}) {
  const info = harnessInfo(record.agent?.harness)
  const requestClass = requestClassOf(record, rules)
  const runAgent = runAgentFor(run, record)
  const subagent = isSubagentRecord(record)
  const title = run?.session_name || record.session_name
  const tools = traceToolNames(record.tool_trace)
  const diagnostics = record.route_diagnostics
  const action = sessionActionLabel(diagnostics?.session_action)
  const reason = sessionReasonText(diagnostics?.session_reason)
  const detail = housekeepingDetail(record, rules)
  const session = record.session_id
  const kind = subagentKindLabel(record.agent?.kind ?? runAgent?.kind)
  const parent = subagent
    ? runAgent?.parent ||
      (record.agent?.parent_id ? `agent ${agentHashText(record.agent.parent_id)}` : 'main')
    : undefined

  return (
    <Card role="region" aria-label="Agent">
      <CardHeader>
        <div className="flex min-w-0 items-center gap-2.5">
          <HarnessLogo harness={record.agent?.harness} size={18} framed />
          <div className="flex min-w-0 flex-col gap-1">
            <CardTitle className="truncate">{info?.label ?? API_CLIENT_LABEL}</CardTitle>
            <CardDescription className="text-xs">
              {info
                ? 'Agent harness that sent this request'
                : 'No harness or agent identity on this request'}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-2">
        <dl className="divide-y divide-border">
          <Row label="Class">
            <span className="flex flex-col items-start gap-1">
              <RequestClassBadge requestClass={requestClass} kind={record.agent?.kind} />
              {detail ? <span className="text-xs text-muted-foreground">{detail}</span> : null}
            </span>
          </Row>
          {session ? (
            <Row label="Session">
              <span className="flex min-w-0 flex-col gap-1">
                {title ? <span className="font-medium">{title}</span> : null}
                <span className="flex min-w-0 items-center gap-1">
                  <span
                    className="truncate font-mono text-[12px] text-muted-foreground"
                    title={session}
                  >
                    {displaySessionId(session)}
                  </span>
                  <CopyButton value={session} label="Copy session ID" className="size-6" />
                </span>
                <span className="flex flex-wrap gap-x-3 gap-y-1">
                  <Link className={LINK} to={getRunPath(session, record.id)}>
                    Open agent run
                    <ArrowUpRight className="size-3" />
                  </Link>
                  <Link className={LINK} to={getSessionRequestsPath(session)}>
                    Session&apos;s requests
                  </Link>
                </span>
              </span>
            </Row>
          ) : null}
          {info ? (
            <Row label="Agent">
              <span className="flex flex-col gap-0.5">
                <span className="font-mono text-[12.5px]">
                  {runAgent?.label ?? (subagent ? 'subagent' : 'main')}
                  {kind ? <span className="font-sans text-muted-foreground"> · {kind}</span> : null}
                </span>
                {subagent ? (
                  <span className="text-xs text-muted-foreground">
                    Spawned by {parent}
                    {record.agent?.id !== undefined ? ` · ${agentHashText(record.agent.id)}` : ''}
                  </span>
                ) : (
                  <span className="text-xs text-muted-foreground">The run&apos;s main agent</span>
                )}
                {spawnsSubagent(record) ? (
                  <span className="inline-flex items-center gap-1 text-xs text-chart-3">
                    <Workflow className="size-3" />
                    Started a subagent on this call
                  </span>
                ) : null}
              </span>
            </Row>
          ) : null}
          <Row label="Turn">
            <span className="tabular">
              {record.turn_index + 1}
              {turnCount && turnCount > 1 ? (
                <span className="text-muted-foreground"> of {turnCount}</span>
              ) : null}
            </span>
          </Row>
          {record.thread_id || record.turn_id || record.parent_turn_id ? (
            <Row label="Codex turn">
              <span className="flex flex-col gap-0.5 font-mono text-[12px]">
                {record.turn_id ? (
                  <Link
                    className="text-primary hover:underline"
                    to={getTurnPath(record.turn_id)}
                    title={record.turn_id}
                  >
                    {compactIdentifier(record.turn_id)}
                  </Link>
                ) : null}
                {record.parent_turn_id ? (
                  <span className="text-muted-foreground" title={record.parent_turn_id}>
                    from turn{' '}
                    <Link
                      className="text-primary hover:underline"
                      to={getTurnPath(record.parent_turn_id)}
                    >
                      {compactIdentifier(record.parent_turn_id)}
                    </Link>
                  </span>
                ) : null}
                {record.thread_id ? (
                  <span className="text-muted-foreground" title={record.thread_id}>
                    thread {compactIdentifier(record.thread_id)}
                  </span>
                ) : null}
              </span>
            </Row>
          ) : null}
          {tools.length > 0 ? (
            <Row label="Tools">
              <ToolChips names={tools} max={8} />
            </Row>
          ) : null}
          {action || reason ? (
            <Row label="Session routing">
              <span className="flex flex-col gap-0.5">
                {action ? <span>{action}</span> : null}
                {reason ? <span className="text-xs text-muted-foreground">{reason}</span> : null}
                {diagnostics?.previous_model &&
                diagnostics.previous_model !== record.selected_model ? (
                  <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                    from <ModelChip name={diagnostics.previous_model} size={12} />
                  </span>
                ) : null}
              </span>
            </Row>
          ) : null}
          {record.conversation_id ? (
            <Row label="Conversation">
              <span className="flex min-w-0 items-center gap-1">
                <span
                  className="truncate font-mono text-[12px] text-muted-foreground"
                  title={record.conversation_id}
                >
                  {compactIdentifier(record.conversation_id, 10, 8)}
                </span>
                <CopyButton
                  value={record.conversation_id}
                  label="Copy conversation ID"
                  className="size-6"
                />
              </span>
            </Row>
          ) : null}
        </dl>
      </CardContent>
    </Card>
  )
}
