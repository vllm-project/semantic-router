import { callSpan, orderRunAgents } from './insightsRunSupport'
import type { RunAgent, RunCall, RunResponse } from './insightsRunTypes'

/** The window sizes the timeline offers, in calls; a run with fewer shows them all. */
export const TIMELINE_SIZES = [50, 100, 200] as const
export const TIMELINE_SIZE = 100

/** Calls of one agent on one model, drawn as one block. */
export interface TimelineBlock {
  key: string
  agent: string
  model: string
  calls: RunCall[]
  markers: string[]
  /** Percent of the axis where the block starts, and how much of it the block covers. */
  left: number
  width: number
}

export interface TimelineLane {
  agent: RunAgent
  depth: number
  blocks: TimelineBlock[]
  /** Percent of the axis where a subagent made its first call of the run, when that is in view. */
  spawnAt?: number
}

/** An idle stretch the axis folds to a narrow mark. */
export interface TimelineBreak {
  left: number
  width: number
  ms: number
}

export interface RunTimeline {
  lanes: TimelineLane[]
  breaks: TimelineBreak[]
  /** The first and last instants on the axis, in ms since the epoch. */
  begin: number
  end: number
  /** The time the axis draws to scale: its span less the folded idle stretches. */
  activeMs: number
  /** The run-wide indexes of the first call in view and of the one after the last. */
  from: number
  to: number
}

// A gap folds when it is at least a minute and a tenth of the window's span;
// folded gaps share at most a third of the axis.
const MIN_IDLE_MS = 60_000
const IDLE_SHARE = 0.1
const BREAK_WIDTH = 3
const BREAKS_SHARE = 30
// Blocks of one agent on one model closer than this, in percent of the axis, read as one.
const MERGE_GAP = 0.6

/** The idle stretches between the calls of a window long enough to fold. */
export function idleGaps(spans: [number, number][], minimum: number): [number, number][] {
  const sorted = [...spans].sort((left, right) => left[0] - right[0])
  const gaps: [number, number][] = []
  let reach = sorted[0]?.[1] ?? 0
  for (const [begin, end] of sorted.slice(1)) {
    if (begin - reach >= minimum) gaps.push([reach, begin])
    reach = Math.max(reach, end)
  }
  return gaps
}

/**
 * buildRunTimeline lays the calls from..to of a run out on one axis, a lane
 * per agent in tree order for the agents that made a call in view. A block is
 * a stretch of one agent's calls on one model: it grows while no other
 * agent's call came between, or while the gap to the next is too thin to
 * see, so a window of hundreds of calls reads as a few dozen blocks, and an
 * agent waiting on its subagent still shows the wait. Idle stretches fold so
 * the work around them stays wide enough to read, and no block spans one.
 */
export function buildRunTimeline(response: RunResponse, from: number, to: number): RunTimeline {
  const calls = response.calls.slice(from, to)
  const spans = calls.map(callSpan)
  if (calls.length === 0) {
    return { lanes: [], breaks: [], begin: 0, end: 0, activeMs: 0, from, to: from }
  }
  const begin = Math.min(...spans.map(([start]) => start))
  const end = Math.max(begin + 1, ...spans.map(([, finish]) => finish))
  const folded = idleGaps(spans, Math.max(MIN_IDLE_MS, (end - begin) * IDLE_SHARE))
  const foldedMs = folded.reduce((sum, [start, finish]) => sum + finish - start, 0)
  const activeMs = Math.max(1, end - begin - foldedMs)
  const breakWidth = folded.length ? Math.min(BREAK_WIDTH, BREAKS_SHARE / folded.length) : 0
  const scale = (100 - folded.length * breakWidth) / activeMs
  const place = (time: number) => {
    let offset = 0
    let cursor = begin
    for (const [start, finish] of folded) {
      if (time <= start) break
      if (time < finish) {
        return offset + (start - cursor) * scale + ((time - start) / (finish - start)) * breakWidth
      }
      offset += (start - cursor) * scale + breakWidth
      cursor = finish
    }
    return Math.min(100, Math.max(0, offset + (time - cursor) * scale))
  }

  const firstCall = new Map<string, number>()
  response.calls.forEach((call, index) => {
    if (!firstCall.has(call.agent)) firstCall.set(call.agent, index)
  })
  const tree = orderRunAgents(response.run.agents)
  const known = new Set(tree.map(({ agent }) => agent.label))
  const lanes = new Map<string, TimelineLane>()
  const lastIndex = new Map<TimelineBlock, number>()
  const lastEnd = new Map<TimelineBlock, number>()
  const folds = (after: number, before: number) =>
    folded.some(([start, finish]) => start >= after && finish <= before)
  calls.forEach((call, offset) => {
    if (!known.has(call.agent)) return
    const index = from + offset
    let lane = lanes.get(call.agent)
    if (!lane) {
      const entry = tree.find(({ agent }) => agent.label === call.agent)!
      lane = { agent: entry.agent, depth: entry.depth, blocks: [] }
      lanes.set(call.agent, lane)
    }
    const [start, finish] = spans[offset]
    const left = place(start)
    const right = Math.max(left, place(finish))
    const model = call.model ?? ''
    const last = lane.blocks[lane.blocks.length - 1]
    if (
      last &&
      last.model === model &&
      (lastIndex.get(last) === index - 1 || left - (last.left + last.width) < MERGE_GAP) &&
      !folds(lastEnd.get(last) ?? start, start)
    ) {
      last.calls.push(call)
      last.width = Math.max(last.left + last.width, right) - last.left
      for (const marker of call.markers ?? []) {
        if (!last.markers.includes(marker)) last.markers.push(marker)
      }
      lastIndex.set(last, index)
      lastEnd.set(last, Math.max(lastEnd.get(last) ?? finish, finish))
      return
    }
    const block: TimelineBlock = {
      key: call.record_id,
      agent: call.agent,
      model,
      calls: [call],
      markers: [...new Set(call.markers ?? [])],
      left,
      width: right - left,
    }
    lane.blocks.push(block)
    lastIndex.set(block, index)
    lastEnd.set(block, finish)
  })

  const ordered = tree.flatMap(({ agent }) => {
    const lane = lanes.get(agent.label)
    if (!lane) return []
    const first = firstCall.get(agent.label) ?? -1
    return lane.depth > 0 && first >= from && first < to
      ? [{ ...lane, spawnAt: place(callSpan(response.calls[first])[0]) }]
      : [lane]
  })
  return {
    lanes: ordered,
    breaks: folded.map(([start, finish]) => ({
      left: place(start),
      width: breakWidth,
      ms: finish - start,
    })),
    begin,
    end,
    activeMs,
    from,
    to: from + calls.length,
  }
}
