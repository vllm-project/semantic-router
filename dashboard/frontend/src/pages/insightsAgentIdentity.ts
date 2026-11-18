import { API_CLIENT_LABEL, harnessInfo, recordHarnessId } from './insightsHarness'
import type { InsightsRecord, ToolTrace } from './insightsPageTypes'
import type { RunAgent, RunSummary } from './insightsRunTypes'

export const SESSION_TITLE_REQUEST_KIND = 'session_title'
const APPROVAL_REVIEW_REQUEST_KIND = 'codex_approval_review'

/**
 * What a request was to the agent that sent it, read from the record's own
 * fields: the main agent's or a subagent's call, a harness's housekeeping
 * (its session title, or a compaction or side request an agent rule names),
 * an approval review, a call the Router made itself for a client request, or
 * a request from a plain API client.
 */
export type RequestClass = 'main' | 'subagent' | 'housekeeping' | 'review' | 'router' | 'api'

export const REQUEST_CLASSES: readonly RequestClass[] = [
  'main',
  'subagent',
  'housekeeping',
  'review',
  'router',
  'api',
]

export const REQUEST_CLASS_LABELS: Record<RequestClass, string> = {
  main: 'Main agent',
  subagent: 'Subagent',
  housekeeping: 'Housekeeping',
  review: 'Approval review',
  router: 'Router call',
  api: 'API call',
}

/** An agent signal rule as the configuration defines it. */
export interface AgentRuleFacts {
  role?: string
  harness?: string
  types?: string[]
  request_classes?: string[]
}

/** Agent rules by recipe, then by rule name. */
export type AgentRuleIndex = Map<string, Map<string, AgentRuleFacts>>

const HOUSEKEEPING_REQUEST_CLASSES = new Set(['compaction', 'auxiliary'])

/**
 * The matched agent rule that marks a request as housekeeping: one that
 * names only compaction and auxiliary requests. The Router records the rule a
 * request matched, not its request class.
 */
export function housekeepingRule(
  record: Pick<InsightsRecord, 'signals' | 'recipe'>,
  rules?: AgentRuleIndex,
): string | undefined {
  const matched = record.signals?.agent
  if (!rules || !Array.isArray(matched)) return undefined
  const recipeRules = rules.get(record.recipe || 'default')
  return matched.find((name) => {
    const classes = recipeRules?.get(name)?.request_classes ?? []
    return classes.length > 0 && classes.every((value) => HOUSEKEEPING_REQUEST_CLASSES.has(value))
  })
}

export function isSubagentRecord(record: Pick<InsightsRecord, 'agent'>): boolean {
  return typeof record.agent?.id === 'string'
}

export function requestClassOf(
  record: Pick<
    InsightsRecord,
    'request_kind' | 'parent_request_id' | 'agent' | 'signals' | 'recipe'
  >,
  rules?: AgentRuleIndex,
): RequestClass {
  if (record.request_kind === APPROVAL_REVIEW_REQUEST_KIND) return 'review'
  if (record.request_kind === SESSION_TITLE_REQUEST_KIND) return 'housekeeping'
  if (record.parent_request_id) return 'router'
  if (housekeepingRule(record, rules)) return 'housekeeping'
  if (isSubagentRecord(record)) return 'subagent'
  if (recordHarnessId(record)) return 'main'
  return 'api'
}

/** What kind of housekeeping a call is, for its label's detail. */
export function housekeepingDetail(
  record: Pick<InsightsRecord, 'request_kind' | 'signals' | 'recipe'>,
  rules?: AgentRuleIndex,
): string | undefined {
  if (record.request_kind === SESSION_TITLE_REQUEST_KIND) return 'Session title'
  const rule = housekeepingRule(record, rules)
  return rule ? `Compaction or side request · rule ${rule}` : undefined
}

const SUBAGENT_KIND_LABELS: Record<string, string> = {
  collab_spawn: 'spawned agent',
  review: 'review child',
}

export function subagentKindLabel(kind?: string): string | undefined {
  if (!kind) return undefined
  return SUBAGENT_KIND_LABELS[kind] ?? kind
}

/** A short form of an agent identity hash, which redaction may have emptied. */
export function agentHashText(id?: string): string {
  if (id === undefined) return ''
  return id === '' ? 'hidden' : `#${id.slice(0, 8)}`
}

/** The run agent a record's agent identity is: main, or the subagent with its id. */
export function runAgentFor(
  run: Pick<RunSummary, 'agents'> | null | undefined,
  record: Pick<InsightsRecord, 'agent'>,
): RunAgent | undefined {
  if (!run) return undefined
  const id = record.agent?.id
  if (id === undefined) return run.agents.find((agent) => agent.label === 'main')
  return id ? run.agents.find((agent) => agent.id === id) : undefined
}

/**
 * Router-scoped session ids carry the API key scope they were keyed under, as
 * key_<hash>:<id>. People read the id the harness sent; links and filters keep
 * the whole value.
 */
export function displaySessionId(sessionId?: string): string {
  return (sessionId ?? '').replace(/^key_[^:]*:/, '')
}

/** A long id as its head and tail. */
export function compactIdentifier(value: string, head = 8, tail = 6): string {
  return value.length <= head + tail + 1 ? value : `${value.slice(0, head)}…${value.slice(-tail)}`
}

/** A session for a narrow column: the id the harness sent, shortened. */
export function compactSessionId(sessionId?: string): string {
  return compactIdentifier(displaySessionId(sessionId))
}

export function traceToolNames(trace?: ToolTrace | null): string[] {
  if (!trace) return []
  const names = new Set<string>()
  for (const name of trace.tool_names ?? []) if (name) names.add(name)
  for (const step of trace.steps ?? []) if (step.tool_name) names.add(step.tool_name)
  return [...names]
}

/** The tools a harness spawns a subagent with: Claude Code's Agent or Task, OpenCode's task and Codex's spawn_agent. */
const SUBAGENT_SPAWN_TOOLS = new Set(['agent', 'task', 'spawn_agent'])

export function spawnsSubagent(record: Pick<InsightsRecord, 'tool_trace'>): boolean {
  return traceToolNames(record.tool_trace).some((name) =>
    SUBAGENT_SPAWN_TOOLS.has(name.toLowerCase()),
  )
}

/** What session protection did with a request's model, as words. */
export const SESSION_ACTION_LABELS: Record<string, string> = {
  establish_baseline: 'Picked the session model',
  stay: 'Kept the session model',
  switch: 'Switched model',
  hard_lock: 'Held the model',
  none: 'Not applied',
}

/** A one-word form of the action, before a model. */
export const SESSION_ACTION_WORDS: Record<string, string> = {
  establish_baseline: 'picked',
  stay: 'kept',
  switch: 'switched',
  hard_lock: 'held',
}

export const SESSION_REASON_TEXT: Record<string, string> = {
  new_session: 'New session',
  new_conversation: 'New conversation',
  session_scope_protect: 'Session protection held the model',
  previous_model_not_in_candidates: 'The previous model is not a candidate here',
  previous_candidate_not_in_candidates: 'The previous model is not a candidate here',
  switch_allowed: 'A better model was allowed',
  stay_has_best_adjusted_score: 'The current model still scores best',
  rescue_underpowered_model: 'Moved up from a model that fell short',
  idle_reset: 'The session was idle, so it started over',
  tool_loop: 'Inside a tool call loop',
  observe_only: 'Observed only',
  missing_identity: 'No session identity on the request',
}

export function sessionActionLabel(action?: string): string | undefined {
  if (!action) return undefined
  return SESSION_ACTION_LABELS[action] ?? action.replace(/_/g, ' ')
}

/** A recorded reason as words; redaction empties one it keeps. */
export function sessionReasonText(reason?: string): string | undefined {
  if (reason === undefined) return undefined
  if (reason === '') return 'Hidden from this viewer'
  return SESSION_REASON_TEXT[reason] ?? reason.replace(/_/g, ' ')
}

export function recordClientLabel(record: Pick<InsightsRecord, 'agent'>): string {
  return harnessInfo(record.agent?.harness)?.label ?? API_CLIENT_LABEL
}
