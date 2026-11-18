import {
  HARNESS_NEVER_SENDS,
  HARNESS_SIGNALS,
  normalizeHarnessId,
  sortHarnessIds,
  type HarnessSignal,
} from './insightsHarness'
import type { RunAgent, RunCall, RunResponse, RunSummary } from './insightsRunTypes'

/** One agent of a run with its own calls in time order and the agents it started. */
export interface AgentNode {
  agent: RunAgent
  depth: number
  calls: RunCall[]
  children: AgentNode[]
}

const byTime = (left: RunCall, right: RunCall) =>
  Date.parse(left.timestamp) - Date.parse(right.timestamp)

/**
 * A run as a tree: the main agent first, each subagent under the agent that
 * started it, and an agent whose parent the run does not hold at the top.
 */
export function buildAgentTree(run: Pick<RunResponse, 'run' | 'calls'>): AgentNode[] {
  const agents = run.run.agents
  const labels = new Set(agents.map((agent) => agent.label))
  const calls = new Map<string, RunCall[]>()
  for (const call of [...run.calls].sort(byTime)) {
    const own = calls.get(call.agent) ?? []
    own.push(call)
    calls.set(call.agent, own)
  }
  const node = (agent: RunAgent, depth: number): AgentNode => ({
    agent,
    depth,
    calls: calls.get(agent.label) ?? [],
    children: agents
      .filter((child) => child.parent === agent.label && child.label !== agent.label)
      .map((child) => node(child, depth + 1)),
  })
  const roots = agents.filter((agent) => !agent.parent || !labels.has(agent.parent))
  roots.sort((left, right) => (left.label === 'main' ? -1 : right.label === 'main' ? 1 : 0))
  return roots.map((agent) => node(agent, 0))
}

/** The type or kind a subagent shows: its harness's type, else Codex's kind. */
export function agentTypeText(agent: Pick<RunAgent, 'type' | 'kind'>): string | undefined {
  return agent.type || agent.kind || undefined
}

/**
 * A session the Router keyed itself, from how the conversation starts,
 * because the client sent no session id: cc-start-, or cursor-start- and
 * openclaw-start- for those harnesses.
 */
export function isDerivedSession(sessionId: string): boolean {
  return (
    /(^|:)(cc|cursor|openclaw)-start-[0-9a-f]{8,}$/.test(sessionId) ||
    /(^|:)cc-[0-9a-f]{8,}$/.test(sessionId)
  )
}

export type SignalCell =
  | { state: 'seen'; count: number }
  | { state: 'derived'; count: number }
  | { state: 'never' }
  | { state: 'unseen' }

export interface HarnessSignalRow {
  /** The harness id, or "" for plain API clients. */
  harness: string
  sessions: number
  cells: Record<HarnessSignal, SignalCell>
}

function runSignals(run: RunSummary): Partial<Record<HarnessSignal, number>> {
  const events = run.events ?? {}
  const subagents = run.agents.filter((agent) => agent.label !== 'main')
  return {
    session: isDerivedSession(run.session_id) ? 0 : 1,
    subagents: subagents.length,
    subagent_type: subagents.filter((agent) => agentTypeText(agent)).length,
    request_class: (events.compaction ?? 0) + (events.housekeeping ?? 0),
    compaction: events.compaction ?? 0,
    housekeeping: events.housekeeping ?? 0,
    images: events.image ?? 0,
    tool_loops: events.tool_loop ?? 0,
  }
}

/**
 * Which agent signals each harness gave in these runs: seen with a count, a
 * session the Router derived itself, a signal the harness never sends, or not
 * seen yet. Harnesses come in the page's order, plain API clients last.
 */
export function harnessSignalRows(runs: RunSummary[]): HarnessSignalRow[] {
  const groups = new Map<string, RunSummary[]>()
  for (const run of runs) {
    const harness = normalizeHarnessId(run.harness)
    groups.set(harness, [...(groups.get(harness) ?? []), run])
  }
  const ids = sortHarnessIds([...groups.keys()].filter(Boolean))
  if (groups.has('')) ids.push('')
  return ids.map((harness) => {
    const group = groups.get(harness) ?? []
    const never = new Set(HARNESS_NEVER_SENDS[harness] ?? [])
    const totals = new Map<HarnessSignal, number>()
    let derived = 0
    for (const run of group) {
      const signals = runSignals(run)
      if (isDerivedSession(run.session_id)) derived += 1
      for (const signal of HARNESS_SIGNALS)
        totals.set(signal, (totals.get(signal) ?? 0) + (signals[signal] ?? 0))
    }
    const cells = Object.fromEntries(
      HARNESS_SIGNALS.map((signal): [HarnessSignal, SignalCell] => {
        const count = totals.get(signal) ?? 0
        if (signal === 'session' && count === 0 && derived > 0)
          return [signal, { state: 'derived', count: derived }]
        if (count > 0) return [signal, { state: 'seen', count }]
        return [signal, never.has(signal) ? { state: 'never' } : { state: 'unseen' }]
      }),
    ) as Record<HarnessSignal, SignalCell>
    return { harness, sessions: group.length, cells }
  })
}

export const HARNESS_SIGNAL_LABELS: Record<HarnessSignal, string> = {
  session: 'Session',
  subagents: 'Subagents',
  subagent_type: 'Subagent type',
  request_class: 'Request class',
  compaction: 'Compaction',
  housekeeping: 'Title / housekeeping',
  images: 'Images',
  tool_loops: 'Tool loops',
}
