import { useEffect, useState } from 'react'

import { formatInsightsCost } from '../utils/insightsCost'

import { countText, formatRunCost, formatRunDuration } from './insightsRunSupport'
import type { RunAgent, RunCall, RunResponse, RunToolCall } from './insightsRunTypes'

// A gap with no call in flight that is longer than this shows as idle.
const RUN_IDLE_MS = 30_000

// One agent's lifeline: main first, then each subagent after the agent that
// spawned it.
export interface RunColumn {
  agent: RunAgent
  depth: number
}

interface RowSpan {
  key: string
  // When the row's first call started and its last call ended, in ms.
  begin: number
  end: number
}

// Consecutive turns of one agent on one model and decision, collapsed.
export interface RunTurnRow extends RowSpan {
  kind: 'turns'
  column: number
  agent: string
  model: string
  decision: string
  calls: RunCall[]
  markers: string[]
}

// Consecutive side requests, such as a title, a compaction or a quota check:
// none is a turn of its agent, so they keep off its lifeline.
export interface RunSideRow extends RowSpan {
  kind: 'side'
  calls: RunCall[]
  markers: string[]
}

// A message between two agents, as in a sequence diagram: a subagent's start
// from the call that spawned it, or its result back to the call that took it.
export interface RunArrowRow extends RowSpan {
  kind: 'spawn' | 'return'
  from: number
  to: number
  agent: string
  parent: string
  label: string
  failed?: boolean
  // The parent's call the arrow leaves from or returns to.
  call?: string
}

export interface RunIdleRow extends RowSpan {
  kind: 'idle'
}

export type RunRow = RunTurnRow | RunSideRow | RunArrowRow | RunIdleRow

// A row a click opens: its calls and their details.
export type RunCallsRow = RunTurnRow | RunSideRow

export interface RunSequence {
  columns: RunColumn[]
  rows: RunRow[]
}

// A call's start and end, its Looper hops included: a Looper call's record
// starts once its hops are done.
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

// The run's agents in tree order, each after the agent that spawned it.
function orderAgents(agents: RunAgent[]): RunColumn[] {
  const labels = new Set(agents.map((agent) => agent.label))
  const ordered: RunColumn[] = []
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

function addMarkers(row: { markers: string[] }, call: RunCall) {
  for (const marker of call.markers ?? []) {
    if (!row.markers.includes(marker)) row.markers.push(marker)
  }
}

function joins(row: RunRow, begin: number): boolean {
  return begin - row.end <= RUN_IDLE_MS
}

// Idle dividers where no call was in flight for longer than RUN_IDLE_MS.
function withIdleGaps(rows: RunRow[]): RunRow[] {
  const result: RunRow[] = []
  let last: number | undefined
  for (const row of rows) {
    if (last !== undefined && row.begin - last > RUN_IDLE_MS) {
      result.push({ kind: 'idle', key: `idle-${row.key}`, begin: last, end: row.begin })
    }
    result.push(row)
    last = Math.max(last ?? row.end, row.end)
  }
  return result
}

function spawnArrow(
  columns: RunColumn[],
  parent: number,
  child: number,
  begin: number,
): RunArrowRow {
  const { agent } = columns[child]
  return {
    kind: 'spawn',
    key: `spawn-${agent.label}`,
    begin,
    end: begin,
    from: parent,
    to: child,
    agent: agent.label,
    parent: columns[parent].agent.label,
    // what it was asked where Replay kept it, else the kind its harness sent
    label: agent.task || agent.kind || `spawns ${agent.label}`,
    call: agent.spawned_after_record_id,
  }
}

// The arrows of the subagents that returned, by the parent's call that took
// each result.
function returnArrows(columns: RunColumn[], column: Map<string, number>) {
  const returns = new Map<string, RunArrowRow[]>()
  columns.forEach(({ agent }, child) => {
    const call = agent.returned_to_record_id
    if (!call) return
    const failed = agent.return_status === 'failed'
    const label = [
      failed ? 'failed' : 'returned',
      countText(agent.calls, 'call'),
      formatRunCost(agent.cost),
    ]
    const arrow: RunArrowRow = {
      kind: 'return',
      key: `return-${agent.label}`,
      begin: 0,
      end: 0,
      from: child,
      to: column.get(agent.parent ?? '') ?? 0,
      agent: agent.label,
      parent: agent.parent ?? columns[0].agent.label,
      label: label.join(' · '),
      failed,
      call,
    }
    returns.set(call, [...(returns.get(call) ?? []), arrow])
  })
  return returns
}

// buildRunSequence lays a run out top to bottom in time order: one row per
// burst of an agent's turns on one model and decision, side requests in rows
// of their own that never split the turns around them, an arrow where a
// subagent starts and one where its result comes back, and a divider
// wherever the run sat idle.
export function buildRunSequence(response: RunResponse): RunSequence {
  const columns = orderAgents(response.run.agents)
  const column = new Map(columns.map((entry, index) => [entry.agent.label, index]))
  const returns = returnArrows(columns, column)
  const rows: RunRow[] = []
  let turns: RunTurnRow | undefined
  let side: RunSideRow | undefined
  const started = new Set<string>()
  const calls = [...response.calls].sort((a, b) => callSpan(a)[0] - callSpan(b)[0])
  for (const call of calls) {
    const index = column.get(call.agent)
    if (index === undefined) continue
    const [begin, end] = callSpan(call)
    if (call.request_kind) {
      if (!side || rows[rows.length - 1] !== side || !joins(side, begin)) {
        side = { kind: 'side', key: `side-${call.record_id}`, begin, end, calls: [], markers: [] }
        rows.push(side)
      }
      side.calls.push(call)
      side.end = Math.max(side.end, end)
      addMarkers(side, call)
      continue
    }
    const model = call.model ?? ''
    const decision = call.decision ?? ''
    const same = turns?.agent === call.agent && turns.model === model && turns.decision === decision
    // A call that takes a subagent's result starts a row, under its arrow.
    const returning = returns.get(call.record_id)
    if (!turns || !same || !joins(turns, begin) || returning) {
      const { agent, depth } = columns[index]
      if (depth > 0 && !started.has(agent.label)) {
        rows.push(spawnArrow(columns, column.get(agent.parent ?? '') ?? 0, index, begin))
      }
      for (const arrow of returning ?? []) rows.push({ ...arrow, begin, end: begin })
      turns = {
        kind: 'turns',
        key: call.record_id,
        begin,
        end,
        column: index,
        agent: call.agent,
        model,
        decision,
        calls: [],
        markers: [],
      }
      rows.push(turns)
    }
    started.add(call.agent)
    turns.calls.push(call)
    turns.end = Math.max(turns.end, end)
    addMarkers(turns, call)
  }
  return { columns, rows: withIdleGaps(rows) }
}

// What a row's calls cost, per currency, marked as a list price when any of
// it is the catalog's estimate; empty when none has a price.
export function formatRowCost(calls: RunCall[]): string {
  const totals = new Map<string, number>()
  let listPrice = false
  for (const call of calls) {
    if (call.cost === undefined) continue
    const currency = call.currency || 'USD'
    totals.set(currency, (totals.get(currency) ?? 0) + call.cost)
    listPrice ||= call.pricing_source === 'catalog_list_price'
  }
  const text = [...totals]
    .map(([currency, total]) => formatInsightsCost(total, currency))
    .join(' + ')
  return text && listPrice ? `${text} list price` : text
}

// The kinds of a row of side requests, as "2 auxiliary, compaction".
export function sideKinds(calls: RunCall[]): string {
  const counts = new Map<string, number>()
  for (const call of calls) {
    const kind = call.request_kind ?? 'side'
    counts.set(kind, (counts.get(kind) ?? 0) + 1)
  }
  return [...counts].map(([kind, count]) => (count > 1 ? `${count} ${kind}` : kind)).join(', ')
}

// Under this width the lifelines turn into one indented list.
const NARROW_RUN_LAYOUT = '(max-width: 640px)'

export function useNarrowRunLayout(): boolean {
  const [narrow, setNarrow] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(NARROW_RUN_LAYOUT).matches,
  )
  useEffect(() => {
    const media = window.matchMedia(NARROW_RUN_LAYOUT)
    const update = () => setNarrow(media.matches)
    update()
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  return narrow
}

// A tool call as a chip reads it: its name and the argument a person knows
// it by, when Replay captured one.
export function toolLabel(tool: RunToolCall): string {
  return tool.summary ? `${tool.name} · ${tool.summary}` : tool.name
}

// How long a tool call took on the client: the plugin's run when it
// reported one, else the gateway's window.
export function toolDuration(tool: RunToolCall): number | undefined {
  if (tool.started_at && tool.ended_at) {
    return Date.parse(tool.ended_at) - Date.parse(tool.started_at)
  }
  return tool.client_ms
}

// A tool call as a line under its call: "Bash · pytest -q · 13.0s · ok".
export function toolLine(tool: RunToolCall): string {
  const duration = toolDuration(tool)
  return [
    toolLabel(tool),
    duration === undefined ? '' : formatRunDuration(duration),
    toolStatusText(tool.status),
  ]
    .filter(Boolean)
    .join(' · ')
}

export function toolStatusText(status?: string): string {
  if (status === 'succeeded') return 'ok'
  if (status === 'failed') return 'failed'
  return 'no result yet'
}
