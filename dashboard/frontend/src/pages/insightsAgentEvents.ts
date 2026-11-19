import type { InsightsRecord } from './insightsPageTypes'
import type { RunCall, RunSummary } from './insightsRunTypes'

/**
 * The events coding-agent harnesses share, as the Router marks a run's calls,
 * in the order the Requests page lists them: what started, what was kept
 * short, what the agent did, and what went wrong.
 */
export const AGENT_EVENTS = [
  'subagent_start',
  'compaction',
  'housekeeping',
  'tool_loop',
  'image',
  'block',
  'guard',
  'failure',
  'tool_failure',
  'switch',
  'escalation',
  'fallback',
] as const

export type AgentEvent = (typeof AGENT_EVENTS)[number]

export type AgentEventTone = 'violet' | 'info' | 'neutral' | 'teal' | 'warning' | 'danger'

export interface AgentEventInfo {
  label: string
  /** The words of a count, as in "2 compactions". */
  noun: [singular: string, plural: string]
  tone: AgentEventTone
  description: string
  /**
   * Routine events, which most calls of an agent carry, read as quiet marks;
   * the rest are worth a labelled badge wherever a call is listed.
   */
  quiet?: boolean
}

export const AGENT_EVENT_INFO: Record<AgentEvent, AgentEventInfo> = {
  subagent_start: {
    label: 'Subagent',
    noun: ['subagent', 'subagents'],
    tone: 'violet',
    description: 'The first call of a subagent the agent started',
  },
  compaction: {
    label: 'Compaction',
    noun: ['compaction', 'compactions'],
    tone: 'info',
    description: 'The harness summarized the conversation to keep its context short',
  },
  housekeeping: {
    label: 'Housekeeping',
    noun: ['housekeeping call', 'housekeeping calls'],
    tone: 'neutral',
    description: 'A session title or another side request of the harness',
    quiet: true,
  },
  tool_loop: {
    label: 'Tool loop',
    noun: ['tool turn', 'tool turns'],
    tone: 'neutral',
    description: 'The call carried tool results back to the model',
    quiet: true,
  },
  image: {
    label: 'Image',
    noun: ['image turn', 'image turns'],
    tone: 'teal',
    description: 'The newest turn gave the model a picture to read',
  },
  block: {
    label: 'Guard refusal',
    noun: ['guard refusal', 'guard refusals'],
    tone: 'danger',
    description: 'A guard refused the request; no model answered it',
  },
  guard: {
    label: 'Guard flag',
    noun: ['guard flag', 'guard flags'],
    tone: 'warning',
    description: 'A guard flagged the request: jailbreak, PII or hallucination',
  },
  failure: {
    label: 'Failed',
    noun: ['failure', 'failures'],
    tone: 'danger',
    description: 'The call failed or got an error status',
  },
  tool_failure: {
    label: 'Tool failed',
    noun: ['tool failure', 'tool failures'],
    tone: 'warning',
    description: 'The call carried a failed tool result',
  },
  switch: {
    label: 'Model switch',
    noun: ['model switch', 'model switches'],
    tone: 'neutral',
    description: "Another model than the agent's previous call answered",
  },
  escalation: {
    label: 'Escalated',
    noun: ['escalation', 'escalations'],
    tone: 'warning',
    description: 'A later attempt than the first answered',
  },
  fallback: {
    label: 'Fell back',
    noun: ['fallback', 'fallbacks'],
    tone: 'warning',
    description: "Another model answered after the first one's failure",
  },
}

/**
 * The events the run Timeline marks on its events strip, compaction first:
 * what a reader looks for in a long run.
 */
export const TIMELINE_EVENTS = [
  'compaction',
  'subagent_start',
  'image',
  'block',
  'guard',
  'failure',
  'switch',
] as const satisfies readonly AgentEvent[]

export type TimelineEvent = (typeof TIMELINE_EVENTS)[number]

const KNOWN = new Set<string>(AGENT_EVENTS)
const ON_TIMELINE = new Set<string>(TIMELINE_EVENTS)

export function isAgentEvent(value: string): value is AgentEvent {
  return KNOWN.has(value)
}

export function isTimelineEvent(value: string): value is TimelineEvent {
  return ON_TIMELINE.has(value)
}

/** A call's events in the page's order, its guard flag left out when the guard also refused it. */
export function callEvents(call: Pick<RunCall, 'markers'>): AgentEvent[] {
  const markers = new Set((call.markers ?? []).filter(isAgentEvent))
  if (markers.has('block')) markers.delete('guard')
  return AGENT_EVENTS.filter((event) => markers.has(event))
}

/** A run's event counts in the page's order, without zero counts. */
export function runEventCounts(run: Pick<RunSummary, 'events'>): [AgentEvent, number][] {
  const events = run.events ?? {}
  return AGENT_EVENTS.flatMap((event) =>
    (events[event] ?? 0) > 0 ? [[event, events[event] as number] as [AgentEvent, number]] : [],
  )
}

export function eventCountText(event: AgentEvent, count: number): string {
  const [one, many] = AGENT_EVENT_INFO[event].noun
  return `${count.toLocaleString()} ${count === 1 ? one : many}`
}

type RecordEventFields = Pick<
  InsightsRecord,
  | 'agent'
  | 'image_count'
  | 'request_kind'
  | 'jailbreak_detected'
  | 'pii_detected'
  | 'pii_blocked'
  | 'hallucination_detected'
  | 'selection_method'
  | 'selected_model'
  | 'lifecycle_state'
  | 'response_status'
>

/**
 * The events a record shows by itself in the flat list: what its class and
 * fields say, read as the Router marks a run's calls. A run marks the rest,
 * such as a subagent's first call or a model switch.
 */
export function recordEvents(record: Partial<RecordEventFields>): AgentEvent[] {
  const events: AgentEvent[] = []
  const requestClass = record.agent?.request_class
  if (requestClass === 'compaction') events.push('compaction')
  else if (requestClass === 'auxiliary' || record.request_kind === 'session_title')
    events.push('housekeeping')
  if ((record.image_count ?? 0) > 0) events.push('image')
  const flagged = Boolean(record.jailbreak_detected || record.pii_detected)
  // A guard decision that answers with its fast response refuses with a 200 and calls no model.
  const refused =
    Boolean(record.pii_blocked) ||
    (flagged && (record.selection_method === 'fast_response' || !record.selected_model)) ||
    ((flagged || Boolean(record.hallucination_detected)) &&
      (record.response_status ?? 0) >= 400 &&
      (record.response_status ?? 0) < 500)
  if (refused) events.push('block')
  else if (flagged || record.hallucination_detected) events.push('guard')
  if (!refused && (record.lifecycle_state === 'failed' || (record.response_status ?? 0) >= 400))
    events.push('failure')
  return events
}
