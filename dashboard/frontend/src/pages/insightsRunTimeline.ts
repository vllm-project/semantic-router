import {
  callEvents,
  isTimelineEvent,
  TIMELINE_EVENTS,
  type TimelineEvent,
} from './insightsAgentEvents'
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

/** One event of one call in view, where the events strip marks it. */
export interface TimelineMark {
  key: string
  event: TimelineEvent
  call: RunCall
  /** Percent of the axis where the call starts. */
  at: number
}

/** A stretch of the axis drawn to scale, between folded idle stretches. */
export interface TimelineSegment {
  start: number
  end: number
  left: number
  width: number
}

export interface RunTimeline {
  lanes: TimelineLane[]
  breaks: TimelineBreak[]
  /** The calls' events the strip above the lanes marks, in time order. */
  marks: TimelineMark[]
  segments: TimelineSegment[]
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
    return {
      lanes: [],
      breaks: [],
      marks: [],
      segments: [],
      begin: 0,
      end: 0,
      activeMs: 0,
      from,
      to: from,
    }
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
  const marks: TimelineMark[] = []
  calls.forEach((call, offset) => {
    if (!known.has(call.agent)) return
    for (const event of callEvents(call).filter(isTimelineEvent)) {
      marks.push({ key: `${call.record_id}-${event}`, event, call, at: place(spans[offset][0]) })
    }
  })
  marks.sort(
    (left, right) =>
      left.at - right.at ||
      TIMELINE_EVENTS.indexOf(left.event) - TIMELINE_EVENTS.indexOf(right.event),
  )
  const segments: TimelineSegment[] = []
  let cursor = begin
  for (const [start, finish] of [...folded, [end, end] as [number, number]]) {
    if (start > cursor) {
      const left = place(cursor)
      segments.push({ start: cursor, end: start, left, width: place(start) - left })
    }
    cursor = finish
  }
  return {
    lanes: ordered,
    breaks: folded.map(([start, finish]) => ({
      left: place(start),
      width: breakWidth,
      ms: finish - start,
    })),
    marks,
    segments,
    begin,
    end,
    activeMs,
    from,
    to: from + calls.length,
  }
}

const SECOND = 1_000
const MINUTE = 60 * SECOND
const HOUR = 60 * MINUTE
/** The steps the axis ticks at: round clock times. */
const TICK_STEPS = [
  SECOND,
  2 * SECOND,
  5 * SECOND,
  10 * SECOND,
  15 * SECOND,
  30 * SECOND,
  MINUTE,
  2 * MINUTE,
  5 * MINUTE,
  10 * MINUTE,
  15 * MINUTE,
  30 * MINUTE,
  HOUR,
  2 * HOUR,
  3 * HOUR,
  6 * HOUR,
  12 * HOUR,
  24 * HOUR,
]

export interface TimelineTick {
  /** Percent of the axis. */
  at: number
  time: number
  /** Whether the step is under a minute, so the label needs its seconds. */
  seconds: boolean
}

/**
 * Round clock times along the parts of the axis drawn to scale, at the finest
 * step that keeps about `most` of them; folded idle stretches get none.
 */
export function timelineTicks(
  timeline: Pick<RunTimeline, 'segments' | 'activeMs'>,
  most: number,
): TimelineTick[] {
  if (timeline.segments.length === 0 || most < 1) return []
  const step =
    TICK_STEPS.find((candidate) => timeline.activeMs / candidate <= most) ??
    TICK_STEPS[TICK_STEPS.length - 1]
  // Steps of an hour and more tick on the local clock's hours.
  const offset = step >= HOUR ? new Date().getTimezoneOffset() * MINUTE : 0
  const ticks: TimelineTick[] = []
  for (const segment of timeline.segments) {
    const span = segment.end - segment.start
    if (span <= 0) continue
    for (
      let time = Math.ceil((segment.start - offset) / step) * step + offset;
      time <= segment.end;
      time += step
    ) {
      ticks.push({
        at: segment.left + ((time - segment.start) / span) * segment.width,
        time,
        seconds: step < MINUTE,
      })
    }
  }
  return ticks
}

/** The time at a position of the folded axis: idle stretches keep their own length. */
export function timeAt(
  timeline: Pick<RunTimeline, 'begin' | 'breaks' | 'activeMs'>,
  position: number,
): number {
  const breaks = [...timeline.breaks].sort((a, b) => a.left - b.left)
  const folded = breaks.reduce((sum, gap) => sum + gap.width, 0)
  const msPerPoint = folded < 100 ? timeline.activeMs / (100 - folded) : 0
  let time = timeline.begin
  let at = 0
  for (const gap of breaks) {
    if (position <= gap.left) break
    time += (gap.left - at) * msPerPoint
    const inside = Math.min(position, gap.left + gap.width) - gap.left
    time += gap.width > 0 ? (inside / gap.width) * gap.ms : 0
    at = gap.left + gap.width
  }
  if (position > at) time += (position - at) * msPerPoint
  return time
}
