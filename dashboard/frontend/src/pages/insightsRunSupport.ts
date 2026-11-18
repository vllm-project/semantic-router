import { formatInsightsCost } from '../utils/insightsCost'

import { compactID } from './insightsReviewSupport'
import type { RunCall, RunCost, RunSummary } from './insightsRunTypes'

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
  deepseek_harness: 'DeepSeek Harness',
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
  // An unpriced Router call leaves its record unpriced or partly priced, so
  // the records count it already.
  const unpriced = cost.unpriced_records
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

export function describeCallStatus(call: RunCall): string {
  if (call.lifecycle_state === 'in_progress') return 'running'
  const status = call.response_status ? ` ${call.response_status}` : ''
  return `${call.lifecycle_state}${status}`
}
