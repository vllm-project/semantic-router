import type { BadgeProps, StatusTone } from '../ui'
import { formatInsightsCost } from '../utils/insightsCost'

import { displaySessionId } from './insightsAgentIdentity'
import { compactID } from './insightsReviewSupport'
import type { RunAgent, RunCall, RunCost, RunReportedOutcome, RunSummary } from './insightsRunTypes'

export const RUN_MARKER_LABELS: Record<string, string> = {
  subagent_start: 'Subagent started',
  compaction: 'Compaction',
  housekeeping: 'Housekeeping',
  tool_loop: 'Tool loop',
  image: 'Image',
  switch: 'Model switched',
  escalation: 'Escalated',
  guard: 'Guard flagged',
  block: 'Guard refusal',
  fallback: 'Fell back',
  failure: 'Failed',
  tool_failure: 'Tool failed',
}

// A marker's one-character glyph in the diagram, its label in the title.
export const RUN_MARKER_GLYPHS: Record<string, string> = {
  subagent_start: '⑂',
  compaction: '⇊',
  housekeeping: '#',
  tool_loop: '↻',
  image: '▣',
  switch: '⇄',
  escalation: '↑',
  guard: '⚑',
  block: '⊘',
  fallback: '↺',
  failure: '✕',
  tool_failure: '⚠',
}

// How a run ended, as a status: only state is colored.
export const RUN_STATUS_TONES: Record<string, StatusTone> = {
  answered: 'success',
  in_progress: 'info',
  failed: 'danger',
  aborted: 'neutral',
  unknown: 'neutral',
}

export const RUN_OUTCOME_LABELS: Record<string, string> = {
  answered: 'Answered',
  failed: 'Failed',
  aborted: 'Abandoned',
  in_progress: 'Running',
  unknown: 'Ended',
}

// The verdicts the outcome API takes, each with its label and how it reads.
export const OUTCOME_VERDICTS: Record<string, { label: string; tone: BadgeProps['tone'] }> = {
  good_fit: { label: 'Good fit', tone: 'success' },
  underpowered: { label: 'Underpowered', tone: 'warning' },
  overprovisioned: { label: 'Overprovisioned', tone: 'info' },
  failed: { label: 'Failed', tone: 'danger' },
}

const OUTCOME_WORDS: Record<string, string> = {
  user: 'User',
  agent: 'Agent',
  eval: 'Evaluation',
  operator: 'Operator',
  provider: 'Provider',
  router: 'Router',
  model: 'Model',
  route: 'Route',
  policy: 'Policy',
  stability: 'Stability',
}

/** An outcome source or target as a word: "eval" reads "Evaluation". */
export function outcomeWord(value: string): string {
  return OUTCOME_WORDS[value] ?? value.replace(/_/g, ' ')
}

export function verdictLabel(verdict: string): string {
  return OUTCOME_VERDICTS[verdict]?.label ?? verdict.replace(/_/g, ' ')
}

/** How many reported outcomes each verdict has, in the order the outcome API lists verdicts. */
export function countVerdicts(
  reported: { outcome: Pick<RunReportedOutcome, 'verdict'> }[],
): { verdict: string; count: number }[] {
  const counts = new Map<string, number>()
  for (const { outcome } of reported) {
    counts.set(outcome.verdict, (counts.get(outcome.verdict) ?? 0) + 1)
  }
  const known = Object.keys(OUTCOME_VERDICTS)
  return [...counts]
    .map(([verdict, count]) => ({ verdict, count }))
    .sort(
      (left, right) =>
        (known.indexOf(left.verdict) + 1 || known.length + 1) -
          (known.indexOf(right.verdict) + 1 || known.length + 1) ||
        left.verdict.localeCompare(right.verdict),
    )
}

// One color per model, in the order the run used them most.
const MODEL_HUES = [
  'var(--chart-4)',
  'var(--chart-3)',
  'var(--chart-2)',
  'var(--chart-5)',
  'var(--chart-1)',
  'var(--chart-7)',
]

export function modelHue(models: { name: string }[], model?: string): string {
  const index = models.findIndex((entry) => entry.name === model)
  return index < 0 ? 'var(--chart-8)' : MODEL_HUES[index % MODEL_HUES.length]
}

export function getRunPath(sessionId: string, step?: string) {
  const query = new URLSearchParams({ session: sessionId })
  if (step) query.set('step', step)
  return `/observe/runs?${query.toString()}`
}

export function formatRunCost(cost: RunCost): string {
  const entries = cost.by_currency ?? []
  const charged = entries.filter((entry) => entry.total !== 0)
  const amounts = (charged.length > 0 ? charged : entries.slice(0, 1)).map((entry) =>
    formatInsightsCost(entry.total, entry.currency),
  )
  if (amounts.length === 0) return 'no price'
  // An unpriced Router call leaves its record unpriced or partly priced, so
  // the records count it already.
  const unpriced = cost.unpriced_records
  return amounts.join(' + ') + (unpriced > 0 ? ` (${unpriced} unpriced)` : '')
}

export function formatRunDuration(ms: number): string {
  if (ms < 1_000) return `${Math.round(ms)}ms`
  const seconds = ms / 1_000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}

// A run reads by the name its harness gave the session, else its main
// agent's first prompt, else its session.
export function runTitle(run: Pick<RunSummary, 'session_name' | 'prompt' | 'session_id'>): string {
  return (
    run.session_name ||
    run.prompt?.trim() ||
    `Session ${compactID(displaySessionId(run.session_id))}`
  )
}

/** Text on one line, its whitespace collapsed and cut at max characters. */
export function oneLine(text: string, max = 120): string {
  const line = text.replace(/\s+/g, ' ').trim()
  return line.length > max ? `${line.slice(0, max - 1).trimEnd()}…` : line
}

/** A run's title on one line, for headers, rows and chips; the task reads in full elsewhere. */
export function runHeadline(run: Pick<RunSummary, 'session_name' | 'prompt' | 'session_id'>) {
  return oneLine(runTitle(run))
}

export function countText(count: number, noun: string): string {
  return `${count.toLocaleString()} ${noun}${count === 1 ? '' : 's'}`
}

export function runAgentTreeText(run: RunSummary): string {
  const subagents = run.agents.filter((agent) => agent.label !== 'main').length
  if (subagents === 0) return 'main only'
  return `main, ${countText(subagents, 'subagent')}`
}

/**
 * A call's start and end, its Looper hops included: a Looper call's record
 * starts once its hops are done.
 */
export function callSpan(call: RunCall): [number, number] {
  let begin = Date.parse(call.timestamp)
  let end = begin + call.duration_ms
  for (const hop of call.hops ?? []) {
    const [hopBegin, hopEnd] = callSpan(hop)
    begin = Math.min(begin, hopBegin)
    end = Math.max(end, hopEnd)
  }
  return [begin, end]
}

/** The run's agents in tree order, each after the agent that spawned it, with its depth. */
export function orderRunAgents(agents: RunAgent[]): { agent: RunAgent; depth: number }[] {
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

/** What the run page's calls table shows: the agents and events it keeps, and its page. */
export interface CallsView {
  agents: string[]
  markers: string[]
  page: number
  size: number
}

/** The calls the agent and event filters keep, in time order. */
export function filterRunCalls(calls: RunCall[], view: Pick<CallsView, 'agents' | 'markers'>) {
  return calls.filter(
    (call) =>
      (view.agents.length === 0 || view.agents.includes(call.agent)) &&
      (view.markers.length === 0 ||
        (call.markers ?? []).some((marker) => view.markers.includes(marker))),
  )
}

export function describeCallStatus(call: RunCall): string {
  if (call.lifecycle_state === 'in_progress') return 'running'
  const status = call.response_status ? ` ${call.response_status}` : ''
  return `${call.lifecycle_state}${status}`
}
