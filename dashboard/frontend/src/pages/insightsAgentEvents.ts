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
  },
  tool_loop: {
    label: 'Tool loop',
    noun: ['tool turn', 'tool turns'],
    tone: 'teal',
    description: 'The call carried tool results back to the model',
  },
  image: {
    label: 'Image',
    noun: ['image turn', 'image turns'],
    tone: 'info',
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

const KNOWN = new Set<string>(AGENT_EVENTS)

export function isAgentEvent(value: string): value is AgentEvent {
  return KNOWN.has(value)
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

/**
 * The events a record shows by itself in the flat list: what its class and
 * fields say. A run marks the rest, such as a subagent's first call.
 */
export function recordEvents(
  record: Pick<InsightsRecord, 'agent' | 'image_count' | 'request_kind'>,
): AgentEvent[] {
  const events: AgentEvent[] = []
  const requestClass = record.agent?.request_class
  if (requestClass === 'compaction') events.push('compaction')
  else if (requestClass === 'auxiliary' || record.request_kind === 'session_title')
    events.push('housekeeping')
  if ((record.image_count ?? 0) > 0) events.push('image')
  return events
}
