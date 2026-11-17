import { formatInsightsCost } from '../utils/insightsCost'

import { compactID } from './insightsReviewSupport'
import type { RunAgent, RunCall, RunCost, RunResponse, RunSummary } from './insightsRunTypes'

export const RUN_MARKER_LABELS: Record<string, string> = {
  switch: 'Model switched',
  escalation: 'Escalated',
  guard: 'Guard flagged',
  block: 'Blocked',
  fallback: 'Fell back',
  failure: 'Failed',
  tool_failure: 'Tool failed',
}

// A marker's one-character glyph in the diagram, its label in the title.
export const RUN_MARKER_GLYPHS: Record<string, string> = {
  switch: '⇄',
  escalation: '↑',
  guard: '⚑',
  block: '⊘',
  fallback: '↺',
  failure: '✕',
  tool_failure: '⚠',
}

export const RUN_OUTCOME_LABELS: Record<string, string> = {
  answered: 'Answered',
  failed: 'Failed',
  aborted: 'Abandoned',
  in_progress: 'Running',
  unknown: 'No final answer',
}

export const HARNESS_LABELS: Record<string, string> = {
  claude_code: 'Claude Code',
  codex: 'Codex',
  opencode: 'OpenCode',
}

// One color per model, in the order the run used them most: the hues the
// session routing history gives its agents.
const MODEL_HUES = ['#5cc8b8', '#a395e8', '#e0a84f', '#e8798f', '#7aa7e8', '#c9cbd0']

export function modelHue(models: { name: string }[], model?: string): string {
  const index = models.findIndex((entry) => entry.name === model)
  return index < 0 ? '#8a94a6' : MODEL_HUES[index % MODEL_HUES.length]
}

export function getRunPath(sessionId: string, step?: string) {
  const query = new URLSearchParams({ session: sessionId })
  if (step) query.set('step', step)
  return `/insights/run?${query.toString()}`
}

export function formatRunCost(cost: RunCost): string {
  const amounts = (cost.by_currency ?? []).map((entry) =>
    formatInsightsCost(entry.total, entry.currency),
  )
  if (amounts.length === 0) return 'no price'
  const unpriced = cost.unpriced_records + cost.unpriced_router_calls
  return amounts.join(' + ') + (unpriced > 0 ? ` (${unpriced} unpriced)` : '')
}

export function formatRunDuration(ms: number): string {
  if (ms < 1_000) return `${ms}ms`
  const seconds = ms / 1_000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${Math.round(seconds % 60)}s`
}

// A run reads by the name its harness gave the session, else its main
// agent's first prompt, else its session.
export function runTitle(run: RunSummary): string {
  return run.session_name || run.prompt?.trim() || `Session ${compactID(run.session_id)}`
}

export function countText(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`
}

export function runAgentTreeText(run: RunSummary): string {
  const subagents = run.agents.filter((agent) => agent.label !== 'main').length
  if (subagents === 0) return 'main only'
  return `main, ${countText(subagents, 'subagent')}`
}

export interface RunSegment {
  key: string
  agent: string
  model: string
  calls: RunCall[]
  // The run-wide index of the segment's last call.
  last: number
  // Percent of the run's span where the segment starts, and how much it covers.
  left: number
  width: number
  markers: string[]
}

export interface RunLane {
  agent: RunAgent
  depth: number
  segments: RunSegment[]
  // Percent of the run's span where the agent's first call starts.
  spawnAt?: number
}

// A call's start and end, its Looper hops included: a Looper call's record
// starts once its hops are done.
function callSpan(call: RunCall): [number, number] {
  let begin = Date.parse(call.timestamp)
  let end = begin + call.duration_ms
  for (const hop of call.hops ?? []) {
    const [hopBegin, hopEnd] = callSpan(hop)
    begin = Math.min(begin, hopBegin)
    end = Math.max(end, hopEnd)
  }
  return [begin, end]
}

// The run's agents in tree order, each after the agent that spawned it.
function orderAgents(agents: RunAgent[]): { agent: RunAgent; depth: number }[] {
  const labels = new Set(agents.map((agent) => agent.label))
  const ordered: { agent: RunAgent; depth: number }[] = []
  const visit = (parent: string | undefined, depth: number) => {
    for (const agent of agents) {
      const root = !agent.parent || !labels.has(agent.parent)
      if ((parent === undefined && root) || (parent !== undefined && agent.parent === parent)) {
        ordered.push({ agent, depth })
        visit(agent.label, depth + 1)
      }
    }
  }
  visit(undefined, 0)
  return ordered
}

// buildRunLanes lays a run out as one lane per agent, each a sequence of
// segments: calls of the agent on one model that no other agent's call came
// between, with the markers of the calls in it, so a run of hundreds of calls
// reads as a few blocks and an agent waiting on its subagent shows a gap.
export function buildRunLanes(response: RunResponse): RunLane[] {
  const start = Date.parse(response.run.started_at)
  const span = Math.max(Date.parse(response.run.ended_at) - start, 1)
  const percent = (time: number) => Math.min(100, Math.max(0, ((time - start) / span) * 100))
  const lanes = orderAgents(response.run.agents).map(({ agent, depth }) => ({
    agent,
    depth,
    segments: [] as RunSegment[],
  }))
  const byLabel = new Map(lanes.map((lane) => [lane.agent.label, lane]))
  response.calls.forEach((call, index) => {
    const lane = byLabel.get(call.agent)
    if (!lane) return
    const [begin, end] = callSpan(call)
    const model = call.model ?? ''
    let segment = lane.segments[lane.segments.length - 1]
    if (!segment || segment.model !== model || segment.last !== index - 1) {
      segment = {
        key: call.record_id,
        agent: call.agent,
        model,
        calls: [],
        last: index,
        left: percent(begin),
        width: 0,
        markers: [],
      }
      lane.segments.push(segment)
    }
    segment.calls.push(call)
    segment.last = index
    segment.width = Math.max(segment.width, percent(end) - segment.left)
    for (const marker of call.markers ?? []) {
      if (!segment.markers.includes(marker)) segment.markers.push(marker)
    }
  })
  return lanes.map((lane) =>
    lane.depth > 0 && lane.segments.length > 0 ? { ...lane, spawnAt: lane.segments[0].left } : lane,
  )
}

export function describeCallStatus(call: RunCall): string {
  if (call.lifecycle_state === 'in_progress') return 'running'
  const status = call.response_status ? ` ${call.response_status}` : ''
  return `${call.lifecycle_state}${status}`
}
