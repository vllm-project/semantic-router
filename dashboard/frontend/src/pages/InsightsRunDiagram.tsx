import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, ArrowUpRight, Bot, ChevronDown, ChevronUp, GitBranch } from 'lucide-react'

import { Badge, cn, Segmented, Tooltip } from '../ui'

import {
  AGENT_EVENT_INFO,
  callEvents,
  eventCountText,
  TIMELINE_EVENTS,
  type TimelineEvent,
} from './insightsAgentEvents'
import { AgentEventChips, AgentEventMarker } from './InsightsAgentEvents'
import { sessionActionLabel, sessionReasonText } from './insightsAgentIdentity'
import { formatTokenCount } from './insightsDisplayFormat'
import { ModelName, modelLabel, modelText, useModelLabels } from './insightsModelLabelSupport'
import { getInsightsRecordPath } from './insightsPageSupport'
import {
  callCostText,
  countText,
  describeCallStatus,
  formatRunCost,
  formatRunDuration,
  modelHue,
  outcomeWord,
  runCurrency,
  verdictLabel,
} from './insightsRunSupport'
import {
  timeAt,
  timelineTicks,
  type RunTimeline,
  type TimelineBlock,
  type TimelineBreak,
  type TimelineMark,
} from './insightsRunTimeline'
import type { RunCall, RunSummary, RunToolCall } from './insightsRunTypes'

function modelTitle(labels: Record<string, string>, model: string): string {
  return model ? modelText(labels, model) : 'no model'
}

const clock = (time: number, seconds = true) =>
  new Date(time).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    ...(seconds ? { second: '2-digit' } : {}),
    hourCycle: 'h23',
  })

/** Lanes shown before the rest fold behind a toggle. */
const LANE_LIMIT = 6
const ZOOMS = ['1', '2', '4', '8'] as const
type Zoom = (typeof ZOOMS)[number]
/** Heights of the strip, a lane and the axis: the label column keeps the same rows. */
const STRIP_PX = 34
const LANE_PX = 44
const AXIS_PX = 24
/** Marks closer than this, in pixels, share one marker with a count. */
const MARK_GAP_PX = 24
/** Width of one character of the 10.5px axis text and of the 11px marker labels. */
const AXIS_CHAR_PX = 6.4
const LABEL_CHAR_PX = 6.3
/** The track width assumed before it is measured. */
const TRACK_DEFAULT_PX = 960

function blockSpan(block: TimelineBlock): [number, number] {
  const starts = block.calls.map((call) => Date.parse(call.timestamp))
  const ends = block.calls.map((call) => Date.parse(call.timestamp) + (call.duration_ms ?? 0))
  return [Math.min(...starts), Math.max(...ends)]
}

function blockTotals(block: TimelineBlock, currency?: string) {
  let input = 0
  let output = 0
  let cost = 0
  let priced = false
  let unpriced = 0
  for (const call of block.calls) {
    input += call.prompt_tokens ?? 0
    output += call.completion_tokens ?? 0
    if (call.cost !== undefined && (!currency || call.currency === currency)) {
      cost += call.cost
      priced = true
    } else {
      unpriced += 1
    }
  }
  return { input, output, cost: priced ? cost : undefined, unpriced }
}

function BlockCard({
  block,
  labels,
  currency,
}: {
  block: TimelineBlock
  labels: Record<string, string>
  currency?: string
}) {
  const [start, end] = blockSpan(block)
  const totals = blockTotals(block, currency)
  const events = [...new Set(block.calls.flatMap((call) => callEvents(call)))].filter(
    (event) => !AGENT_EVENT_INFO[event].quiet,
  )
  return (
    <div className="flex min-w-48 max-w-72 flex-col gap-1.5 py-0.5 text-xs">
      <strong className="text-[13px] font-semibold">{modelTitle(labels, block.model)}</strong>
      <span className="opacity-80">
        {block.agent === 'main' ? 'Main agent' : block.agent} ·{' '}
        {countText(block.calls.length, 'call')} · {clock(start)}–{clock(end)}
      </span>
      <span className="font-mono tabular">
        {formatTokenCount(totals.input)} in · {formatTokenCount(totals.output)} out
        {totals.cost !== undefined && currency
          ? ` · ${callCostText({ cost: totals.cost, currency }, currency)}`
          : ''}
        {totals.unpriced && totals.cost !== undefined ? ` · ${totals.unpriced} unpriced` : ''}
      </span>
      {events.length ? (
        <span className="flex flex-wrap gap-x-2.5 gap-y-0.5 opacity-90">
          {events.map((event) => (
            <span key={event} className="inline-flex items-center gap-1">
              <AgentEventMarker event={event} size="sm" />
              {AGENT_EVENT_INFO[event].label}
            </span>
          ))}
        </span>
      ) : null}
    </div>
  )
}

function Hatch({ gap }: { gap: TimelineBreak }) {
  return (
    <span
      aria-hidden
      className="absolute inset-y-0 border-x border-dashed border-border-strong/60 bg-[repeating-linear-gradient(135deg,var(--border)_0_1px,transparent_1px_6px)]"
      style={{ left: `${gap.left}%`, width: `${gap.width}%` }}
    />
  )
}

interface AxisLabel {
  key: string
  at: number
  text: string
  kind: 'tick' | 'edge' | 'idle'
  align: 'start' | 'center' | 'end'
}

// The first and last instants in view, each idle stretch's length, then round
// clock ticks, each kept only where it clears the labels already placed.
function axisLabels(timeline: RunTimeline, trackPx: number): AxisLabel[] {
  const width = trackPx > 0 ? trackPx : TRACK_DEFAULT_PX
  const ticks = timelineTicks(timeline, Math.max(2, Math.floor(width / 90)))
  const seconds = ticks.length === 0 || ticks[0].seconds
  const candidates: AxisLabel[] = [
    { key: 'start', at: 0, text: clock(timeline.begin), kind: 'edge', align: 'start' },
    { key: 'end', at: 100, text: clock(timeAt(timeline, 100)), kind: 'edge', align: 'end' },
    ...timeline.breaks.map((gap) => ({
      key: `idle-${gap.left}`,
      at: gap.left + gap.width / 2,
      text: `${formatRunDuration(gap.ms)} idle`,
      kind: 'idle' as const,
      align: 'center' as const,
    })),
    ...ticks.map((tick) => ({
      key: `tick-${tick.time}`,
      at: tick.at,
      text: clock(tick.time, seconds),
      kind: 'tick' as const,
      align: 'center' as const,
    })),
  ]
  const span = (label: AxisLabel): [number, number] => {
    const size = label.text.length * AXIS_CHAR_PX
    const x = (label.at / 100) * width
    if (label.align === 'start') return [x, x + size]
    if (label.align === 'end') return [x - size, x]
    return [x - size / 2, x + size / 2]
  }
  const placed: AxisLabel[] = []
  for (const label of candidates) {
    const [from, to] = span(label)
    if (from < 0 || to > width) continue
    const clear = placed.every((other) => {
      const [start, end] = span(other)
      return to + 10 <= start || from >= end + 10
    })
    if (clear) placed.push(label)
  }
  return placed
}

/** One kind of event at one place: marks of that kind too close to tell apart share it. */
interface MarkCluster {
  key: string
  /** Percent of the axis where its first event happened. */
  at: number
  /** Percent of the axis where its marker sits, nudged clear of its neighbors. */
  left: number
  lead: TimelineMark
  marks: TimelineMark[]
  labelled: boolean
}

/** A marker's width on the strip, its count badge included. */
const markerWidth = (cluster: Pick<MarkCluster, 'lead' | 'labelled' | 'marks'>) =>
  (cluster.labelled ? 26 + AGENT_EVENT_INFO[cluster.lead.event].label.length * LABEL_CHAR_PX : 20) +
  (cluster.marks.length > 1 ? 8 : 0) +
  2

/** Each kind's marks merged where they are closer than gap pixels. */
function mergeByKind(marks: TimelineMark[], gap: number, px: (at: number) => number) {
  const clusters: MarkCluster[] = []
  const lastOfKind = new Map<string, MarkCluster>()
  for (const mark of marks) {
    const last = lastOfKind.get(mark.event)
    if (last && px(mark.at - last.at) < gap) {
      last.marks.push(mark)
      continue
    }
    const cluster = {
      key: mark.key,
      at: mark.at,
      left: mark.at,
      lead: mark,
      marks: [mark],
      labelled: false,
    }
    clusters.push(cluster)
    lastOfKind.set(mark.event, cluster)
  }
  return clusters.sort(
    (a, b) =>
      a.at - b.at || TIMELINE_EVENTS.indexOf(a.lead.event) - TIMELINE_EVENTS.indexOf(b.lead.event),
  )
}

/**
 * The strip's markers: an event kind's marks closer than a marker merge into
 * one with a count, more coarsely while the markers would not fit the track,
 * and markers of different kinds are nudged apart rather than merged, so
 * every kind in view keeps its own marker at any zoom. A compaction reads its
 * label in words first; any other single event where its neighbors leave room.
 */
function layoutMarks(marks: TimelineMark[], trackPx: number): MarkCluster[] {
  const width = trackPx > 0 ? trackPx : TRACK_DEFAULT_PX
  const px = (percent: number) => (percent / 100) * width
  const needed = (list: MarkCluster[]) =>
    list.reduce((sum, cluster) => sum + markerWidth(cluster), 0)
  let gap = MARK_GAP_PX
  let clusters = mergeByKind(marks, gap, px)
  while (needed(clusters) > width - 4 && gap < width) {
    gap *= 2
    clusters = mergeByKind(marks, gap, px)
  }
  let spare = width - 4 - needed(clusters)
  const byPriority = [...clusters].sort(
    (a, b) => TIMELINE_EVENTS.indexOf(a.lead.event) - TIMELINE_EVENTS.indexOf(b.lead.event),
  )
  for (const cluster of byPriority) {
    if (cluster.marks.length > 1) continue
    const extra = markerWidth({ ...cluster, labelled: true }) - markerWidth(cluster)
    const index = clusters.indexOf(cluster)
    const next = clusters[index + 1]
    const room = (next ? px(next.at - cluster.at) : width - px(cluster.at)) - 12
    const fits =
      cluster.lead.event === 'compaction' || room >= markerWidth({ ...cluster, labelled: true })
    if (fits && spare >= extra && px(cluster.at) + extra + 20 <= width) {
      cluster.labelled = true
      spare -= extra
    }
  }
  // Left to right, then right to left: each marker clear of the one before it, all inside the track.
  const spans = clusters.map((cluster) => {
    const size = markerWidth(cluster)
    const center = cluster.labelled ? px(cluster.at) - 10 + size / 2 : px(cluster.at)
    return { size, center }
  })
  for (let index = 1; index < spans.length; index += 1) {
    const before = spans[index - 1]
    const span = spans[index]
    span.center = Math.max(span.center, before.center + (before.size + span.size) / 2 + 2)
  }
  for (let index = spans.length - 1; index >= 0; index -= 1) {
    const span = spans[index]
    const after = spans[index + 1]
    const limit = after
      ? after.center - (after.size + span.size) / 2 - 2
      : width - span.size / 2 - 2
    span.center = Math.max(span.size / 2 + 2, Math.min(span.center, limit))
  }
  clusters.forEach((cluster, index) => {
    const { size, center } = spans[index]
    cluster.left = ((cluster.labelled ? center - size / 2 + 10 : center) / width) * 100
  })
  return clusters
}

function MarkCard({
  cluster,
  numbers,
  labels,
}: {
  cluster: MarkCluster
  numbers?: Map<string, number>
  labels: Record<string, string>
}) {
  const { lead } = cluster
  const info = AGENT_EVENT_INFO[lead.event]
  const call = lead.call
  const number = numbers?.get(call.record_id)
  const agent = call.agent === 'main' ? 'Main agent' : call.agent
  const counts = TIMELINE_EVENTS.map(
    (event) => [event, cluster.marks.filter((mark) => mark.event === event).length] as const,
  ).filter(([, count]) => count > 0)
  const firsts = cluster.marks.map((mark) => numbers?.get(mark.call.record_id) ?? 0)
  return (
    <div className="flex w-64 flex-col gap-1.5 py-0.5 text-xs">
      <span className="flex items-center gap-2">
        <AgentEventMarker event={lead.event} labelled />
        {cluster.marks.length > 1 ? (
          <span className="opacity-80">and {cluster.marks.length - 1} more here</span>
        ) : null}
      </span>
      <span className="leading-snug opacity-90">{info.description}.</span>
      <span className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 font-normal">
        <span className="opacity-70">Call</span>
        <span className="truncate">
          {number ? `#${number.toLocaleString()} · ` : ''}
          {agent}
        </span>
        <span className="opacity-70">At</span>
        <span>{clock(Date.parse(call.timestamp))}</span>
        <span className="opacity-70">Model</span>
        <span className="truncate">
          {lead.event === 'switch' && call.previous_model
            ? `${modelLabel(labels, call.previous_model)} → ${modelLabel(labels, call.model ?? '') || 'no model'}`
            : modelLabel(labels, call.model ?? '') || 'No model call'}
        </span>
        <span className="opacity-70">Tokens</span>
        <span className="font-mono tabular">
          {formatTokenCount(call.prompt_tokens)} in · {formatTokenCount(call.completion_tokens)} out
        </span>
      </span>
      {counts.length > 1 || cluster.marks.length > 1 ? (
        <span className="flex flex-wrap gap-x-2.5 gap-y-1 border-t border-background/20 pt-1.5">
          {counts.map(([event, count]) => (
            <span key={event} className="inline-flex items-center gap-1">
              <AgentEventMarker event={event} size="sm" />
              {eventCountText(event, count)}
            </span>
          ))}
          {firsts.every(Boolean) ? (
            <span className="opacity-70">
              calls #{Math.min(...firsts).toLocaleString()}–#{Math.max(...firsts).toLocaleString()}
            </span>
          ) : null}
        </span>
      ) : null}
      <span className="opacity-60">Click for the call’s card</span>
    </div>
  )
}

/** The events strip: one marker per event, or per events too close to tell apart, with a hover card each. */
function EventsStrip({
  clusters,
  numbers,
  labels,
  selectedCall,
  onSelectCall,
}: {
  clusters: MarkCluster[]
  numbers?: Map<string, number>
  labels: Record<string, string>
  selectedCall?: string
  onSelectCall?: (call: RunCall) => void
}) {
  return (
    <div className="relative" style={{ height: STRIP_PX }} role="list" aria-label="Run events">
      {clusters.map((cluster) =>
        Math.abs(cluster.left - cluster.at) > 0.4 ? (
          <span
            key={`${cluster.key}-stem`}
            aria-hidden
            className="absolute bottom-0 h-1.5 w-px bg-border-strong"
            style={{ left: `${cluster.at}%` }}
          />
        ) : null,
      )}
      {clusters.map((cluster) => {
        const info = AGENT_EVENT_INFO[cluster.lead.event]
        const selected = cluster.marks.some((mark) => mark.call.record_id === selectedCall)
        const number = numbers?.get(cluster.lead.call.record_id)
        const name =
          cluster.marks.length > 1
            ? eventCountText(cluster.lead.event, cluster.marks.length)
            : number
              ? `${info.label}, call ${number}`
              : info.label
        return (
          <span
            key={cluster.key}
            role="listitem"
            className="absolute top-[7px]"
            style={{
              left: `${cluster.left}%`,
              transform: cluster.labelled ? 'translateX(-10px)' : 'translateX(-50%)',
            }}
          >
            <Tooltip content={<MarkCard cluster={cluster} numbers={numbers} labels={labels} />}>
              <button
                type="button"
                aria-label={name}
                data-count={cluster.marks.length}
                onClick={() => onSelectCall?.(cluster.lead.call)}
                className={cn(
                  'rounded-full outline-none transition-transform hover:scale-110 focus-visible:ring-[3px] focus-visible:ring-ring/50',
                  selected && 'ring-2 ring-foreground ring-offset-1 ring-offset-card',
                )}
              >
                <AgentEventMarker
                  event={cluster.lead.event}
                  labelled={cluster.labelled}
                  count={cluster.marks.length}
                  className="shadow-[0_0_0_2px_var(--card),0_1px_3px_oklch(0_0_0/0.18)]"
                />
              </button>
            </Tooltip>
          </span>
        )
      })}
    </div>
  )
}

/** The whole run at a glance, with the window the lanes show. */
function Overview({
  run,
  timeline,
  view,
  onJump,
}: {
  run: RunSummary
  timeline: RunTimeline
  view: { left: number; width: number }
  onJump: (position: number) => void
}) {
  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label="Run overview"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(view.left)}
      className="relative h-6 cursor-pointer overflow-hidden rounded-md bg-surface-subtle ring-1 ring-inset ring-border"
      onClick={(event) => {
        const box = event.currentTarget.getBoundingClientRect()
        onJump(((event.clientX - box.left) / box.width) * 100)
      }}
      onKeyDown={(event) => {
        if (event.key === 'ArrowLeft')
          onJump(Math.max(0, view.left + view.width / 2 - view.width / 4))
        if (event.key === 'ArrowRight')
          onJump(Math.min(100, view.left + view.width / 2 + view.width / 4))
      }}
    >
      {timeline.breaks.map((gap) => (
        <Hatch key={gap.left} gap={gap} />
      ))}
      {timeline.lanes.map((lane, index) =>
        lane.blocks.map((block) => (
          <span
            key={block.key}
            aria-hidden
            className="absolute rounded-[2px] opacity-90"
            style={{
              left: `${block.left}%`,
              width: `max(2px, ${block.width}%)`,
              top: `${3 + (index % 4) * 4}px`,
              height: '4px',
              background: modelHue(run.models, block.model),
            }}
          />
        )),
      )}
      {timeline.marks
        .filter((mark) => mark.event === 'compaction')
        .map((mark) => (
          <span
            key={mark.key}
            aria-hidden
            className="absolute inset-y-0 w-0.5 bg-info"
            style={{ left: `${mark.at}%` }}
          />
        ))}
      <span
        aria-hidden
        className="absolute inset-y-0 rounded-md border-2 border-primary/70 bg-primary/10"
        style={{ left: `${view.left}%`, width: `${view.width}%` }}
      />
    </div>
  )
}

interface RunDiagramProps {
  run: RunSummary
  timeline: RunTimeline
  selected?: string
  /** The call whose card is open, its marker ringed. */
  selectedCall?: string
  /** The agents the calls table follows; the other lanes dim. */
  agents?: string[]
  /** The run-wide number of each call, by record id. */
  numbers?: Map<string, number>
  onSelect: (block: TimelineBlock) => void
  onSelectCall?: (call: RunCall) => void
  onAgent?: (label: string) => void
}

// A lane per agent that made a call in view, in tree order, its calls as
// blocks on one axis under a strip of the calls' events; a compaction also
// rules a line through every lane. Idle stretches fold to a hatched mark.
// Long runs zoom, with an overview of the whole run and its focus window, and
// long agent lists fold after the first lanes.
export function RunDiagram({
  run,
  timeline,
  selected,
  selectedCall,
  agents = [],
  numbers,
  onSelect,
  onSelectCall,
  onAgent,
}: RunDiagramProps) {
  const labels = useModelLabels()
  const [zoom, setZoom] = useState<Zoom>('1')
  const [allLanes, setAllLanes] = useState(false)
  const [view, setView] = useState({ left: 0, width: 100 })
  const [trackPx, setTrackPx] = useState(0)
  const track = useRef<HTMLDivElement>(null)
  const scale = Number(zoom)
  const lanes = allLanes ? timeline.lanes : timeline.lanes.slice(0, LANE_LIMIT)
  const hidden = timeline.lanes.length - lanes.length
  const currency = runCurrency(run.cost)
  const widthPx = (trackPx || TRACK_DEFAULT_PX) * scale
  const clusters = useMemo(() => layoutMarks(timeline.marks, widthPx), [timeline.marks, widthPx])
  const compactions = timeline.marks.filter((mark) => mark.event === 'compaction')
  const ticks = axisLabels(timeline, widthPx)

  const measure = useCallback(() => {
    const node = track.current
    if (!node || node.scrollWidth === 0) return
    setView({
      left: (node.scrollLeft / node.scrollWidth) * 100,
      width: (node.clientWidth / node.scrollWidth) * 100,
    })
    setTrackPx(node.clientWidth)
  }, [])
  useEffect(measure, [measure, scale, timeline])
  useEffect(() => {
    const node = track.current
    if (!node || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [measure])

  const jump = (position: number) => {
    const node = track.current
    if (!node) return
    node.scrollTo({
      left: (position / 100) * node.scrollWidth - node.clientWidth / 2,
      behavior: 'smooth',
    })
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="min-w-0 flex-1">
          {scale > 1 ? <Overview run={run} timeline={timeline} view={view} onJump={jump} /> : null}
        </div>
        <Segmented
          size="sm"
          aria-label="Timeline zoom"
          value={zoom}
          onValueChange={setZoom}
          options={ZOOMS.map((value) => ({ value, label: value === '1' ? 'Fit' : `${value}×` }))}
        />
      </div>
      <div className="grid grid-cols-[minmax(0,112px)_minmax(0,1fr)] gap-x-3 sm:grid-cols-[minmax(0,184px)_minmax(0,1fr)]">
        <div className="flex flex-col" role="list" aria-label="Run agents">
          <div
            className="flex items-center text-[10.5px] font-semibold uppercase tracking-[0.06em] text-subtle-foreground"
            style={{ height: STRIP_PX }}
            aria-hidden
          >
            Events
          </div>
          {lanes.map((lane) => {
            const followed = agents.includes(lane.agent.label)
            const main = lane.agent.label === 'main'
            const type = lane.agent.type || lane.agent.kind
            const figures = `${countText(lane.agent.calls, 'call')} · ${formatRunCost(lane.agent.cost)}`
            const content = (
              <>
                <span className="flex min-w-0 items-center gap-1.5">
                  {main ? (
                    <Bot className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
                  ) : (
                    <GitBranch className="size-3.5 shrink-0 text-chart-3" aria-hidden />
                  )}
                  <strong className="truncate text-[12.5px] font-semibold">
                    {main ? 'Main agent' : lane.agent.label.replace(/^subagent/, 'Subagent')}
                  </strong>
                </span>
                <span
                  className="truncate pl-5 text-[11px] text-subtle-foreground tabular"
                  title={type ? `${type} · ${figures}` : figures}
                >
                  {type ? (
                    <span className="font-medium text-muted-foreground">{type} · </span>
                  ) : null}
                  {figures}
                </span>
              </>
            )
            return (
              <div
                key={lane.agent.label}
                role="listitem"
                className={cn(
                  'flex items-center transition-opacity',
                  agents.length > 0 && !followed && 'opacity-45',
                )}
                style={{ height: LANE_PX, paddingLeft: `${lane.depth * 0.75}rem` }}
              >
                {onAgent ? (
                  <button
                    type="button"
                    aria-pressed={followed}
                    title={`Follow ${lane.agent.label}'s calls`}
                    onClick={() => onAgent(lane.agent.label)}
                    className={cn(
                      'flex w-full min-w-0 flex-col rounded-md px-2 py-1 text-left transition-colors hover:bg-surface-subtle focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40',
                      followed && 'bg-primary-soft/60 hover:bg-primary-soft/70',
                    )}
                  >
                    {content}
                  </button>
                ) : (
                  <div className="flex min-w-0 flex-col px-2">{content}</div>
                )}
              </div>
            )
          })}
        </div>
        <div
          ref={track}
          onScroll={measure}
          className="min-w-0 overflow-x-auto overscroll-x-contain"
        >
          <div className="relative" style={{ width: `${scale * 100}%` }}>
            <EventsStrip
              clusters={clusters}
              numbers={numbers}
              labels={labels}
              selectedCall={selectedCall}
              onSelectCall={onSelectCall}
            />
            <div className="relative" role="list" aria-label="Run timeline">
              {ticks
                .filter((label) => label.kind === 'tick')
                .map((label) => (
                  <span
                    key={label.key}
                    aria-hidden
                    className="absolute inset-y-0 w-px bg-border/70"
                    style={{ left: `${label.at}%` }}
                  />
                ))}
              {lanes.map((lane) => (
                <div
                  key={lane.agent.label}
                  role="listitem"
                  className={cn(
                    'relative flex items-center transition-opacity',
                    agents.length > 0 && !agents.includes(lane.agent.label) && 'opacity-45',
                  )}
                  style={{ height: LANE_PX }}
                >
                  <div className="relative h-[30px] w-full rounded-lg bg-surface-subtle/70 ring-1 ring-inset ring-border/70">
                    {timeline.breaks.map((gap) => (
                      <Hatch key={gap.left} gap={gap} />
                    ))}
                    {lane.spawnAt !== undefined ? (
                      <span
                        className="absolute -inset-y-1 w-px border-l-2 border-dotted border-chart-3"
                        style={{ left: `${lane.spawnAt}%` }}
                        title={`Started by ${lane.agent.parent === 'main' || !lane.agent.parent ? 'the main agent' : lane.agent.parent}`}
                      />
                    ) : null}
                    {lane.blocks.map((block) => (
                      <Tooltip
                        key={block.key}
                        content={<BlockCard block={block} labels={labels} currency={currency} />}
                      >
                        <button
                          type="button"
                          data-selected={block.key === selected}
                          aria-label={`${modelTitle(labels, block.model)}, ${countText(block.calls.length, 'call')}`}
                          className="absolute inset-y-[3px] flex items-center justify-center overflow-hidden rounded-md px-1 text-[10.5px] text-white shadow-xs outline-none transition-[box-shadow,filter] hover:brightness-110 focus-visible:ring-[3px] focus-visible:ring-ring/50 data-[selected=true]:ring-2 data-[selected=true]:ring-foreground data-[selected=true]:ring-offset-1 data-[selected=true]:ring-offset-card"
                          style={{
                            left: `${Math.min(block.left, 99.4)}%`,
                            width: `${block.width}%`,
                            minWidth: '6px',
                            background: modelHue(run.models, block.model),
                          }}
                          onClick={() => onSelect(block)}
                        >
                          {block.calls.length > 1 ? (
                            <span className="font-mono font-semibold tabular drop-shadow-sm">
                              {block.calls.length}
                            </span>
                          ) : null}
                        </button>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              ))}
              {compactions.map((mark) => (
                <span
                  key={mark.key}
                  aria-hidden
                  data-compaction-rule
                  className="pointer-events-none absolute -top-2 bottom-0 border-l-2 border-dashed border-info/80"
                  style={{ left: `${mark.at}%` }}
                />
              ))}
            </div>
            <div
              className="relative font-mono text-[10.5px] text-subtle-foreground tabular"
              style={{ height: AXIS_PX }}
              aria-hidden
            >
              {ticks.map((label) => (
                <span
                  key={label.key}
                  className={cn(
                    'absolute top-1 flex flex-col items-center whitespace-nowrap',
                    label.align === 'start' && 'left-0 items-start',
                    label.align === 'end' && 'right-0 items-end',
                    label.align === 'center' && '-translate-x-1/2',
                    label.kind === 'idle' && 'top-1.5 font-sans italic',
                  )}
                  style={label.align === 'center' ? { left: `${label.at}%` } : undefined}
                >
                  {label.kind === 'tick' ? (
                    <i className="-mt-1 mb-0.5 h-1 w-px bg-border-strong" />
                  ) : null}
                  {label.text}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
      {timeline.lanes.length > LANE_LIMIT ? (
        <button
          type="button"
          onClick={() => setAllLanes((value) => !value)}
          className="inline-flex w-fit items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-primary hover:bg-primary-soft/50"
        >
          {allLanes ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
          {allLanes ? 'Show fewer agents' : `Show ${hidden} more agent${hidden === 1 ? '' : 's'}`}
        </button>
      ) : null}
    </div>
  )
}

/** What a newcomer needs to read the Timeline: each model's color and each marker in words. */
export function RunLegend({
  run,
  timeline,
  onFind,
}: {
  run: RunSummary
  timeline: RunTimeline
  /** Moves the Timeline to the next call with this event, when given. */
  onFind?: (event: TimelineEvent) => void
}) {
  const events = TIMELINE_EVENTS.filter((event) => (run.events?.[event] ?? 0) > 0)
  return (
    <div
      className="flex flex-col gap-2 border-t border-border pt-3 text-xs text-muted-foreground"
      aria-label="Timeline legend"
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
        <span className="text-[10.5px] font-semibold uppercase tracking-[0.06em] text-subtle-foreground">
          Models
        </span>
        {run.models.map((model) => (
          <span key={model.name} className="inline-flex items-center gap-1.5">
            <i
              className="size-2.5 rounded-[3px]"
              style={{ background: modelHue(run.models, model.name) }}
            />
            <ModelName name={model.name} />
          </span>
        ))}
        <span className="inline-flex items-center gap-1.5">
          <i
            className="h-2.5 w-3.5 rounded-[2px] bg-[repeating-linear-gradient(135deg,var(--border-strong)_0_1px,transparent_1px_4px)]"
            aria-hidden
          />
          Idle time, folded
        </span>
        <span className="tabular text-subtle-foreground">
          {formatRunDuration(timeline.activeMs)} of activity in view
        </span>
      </div>
      {events.length ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <span className="text-[10.5px] font-semibold uppercase tracking-[0.06em] text-subtle-foreground">
            Events
          </span>
          {events.map((event) => {
            const info = AGENT_EVENT_INFO[event]
            const count = run.events?.[event] ?? 0
            const content = (
              <>
                <AgentEventMarker event={event} size="sm" />
                <span className="text-foreground/85">{info.label}</span>
                <span className="font-mono text-subtle-foreground tabular">
                  {count.toLocaleString()}
                </span>
              </>
            )
            return (
              <Tooltip key={event} content={info.description}>
                {onFind ? (
                  <button
                    type="button"
                    data-legend={event}
                    aria-label={`${eventCountText(event, count)}: show the next one`}
                    onClick={() => onFind(event)}
                    className="-mx-1 inline-flex items-center gap-1.5 rounded-md px-1 py-0.5 transition-colors hover:bg-surface-subtle focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
                  >
                    {content}
                  </button>
                ) : (
                  <span data-legend={event} className="inline-flex items-center gap-1.5">
                    {content}
                  </span>
                )}
              </Tooltip>
            )
          })}
          {onFind ? (
            <span className="text-subtle-foreground">Click an event to show the next one.</span>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}

/** A signal as a person reads it: its type and its rule. */
function signalParts(signal: string): [string, string] {
  const at = signal.indexOf(':')
  return at < 0
    ? ['', signal]
    : [signal.slice(0, at).replace(/_/g, ' '), signal.slice(at + 1).replace(/_/g, ' ')]
}

function Section({
  title,
  action,
  className,
  children,
}: {
  title: string
  action?: ReactNode
  className?: string
  children: ReactNode
}) {
  return (
    <section
      className={cn(
        'flex min-w-0 flex-col gap-2.5 rounded-xl border border-border bg-surface/60 p-3.5',
        className,
      )}
      aria-label={title}
    >
      <header className="flex items-center justify-between gap-3">
        <h4 className="text-[11px] font-semibold uppercase tracking-[0.06em] text-subtle-foreground">
          {title}
        </h4>
        {action}
      </header>
      {children}
    </section>
  )
}

function Row({ label, children, title }: { label: string; children: ReactNode; title?: string }) {
  return (
    <div className="flex min-w-0 items-baseline justify-between gap-3 text-[13px]">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate text-right" title={title}>
        {children}
      </span>
    </div>
  )
}

const statusTone = (call: RunCall) =>
  call.lifecycle_state === 'completed' && (call.response_status ?? 200) < 400
    ? 'success'
    : call.lifecycle_state === 'in_progress'
      ? 'info'
      : 'danger'

// One call as a card: how it was routed and why beside what it used and cost
// over how it ended, with internal codes in words and the raw value on hover.
export function RunCallDetails({ call, currency }: { call: RunCall; currency?: string }) {
  const labels = useModelLabels()
  const method = [call.selection_method, call.session_action].filter(Boolean).join(' · ')
  const reason = call.reason === '' ? 'Hidden from this viewer' : call.reason
  const events = callEvents(call)
  const cost = callCostText(call, currency)
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <Section title="Routing" className="lg:row-span-2">
        <div className="flex min-w-0 flex-wrap items-center gap-1.5 text-[13px]">
          {call.previous_model ? (
            <>
              <span className="text-muted-foreground">
                {modelLabel(labels, call.previous_model)}
              </span>
              <ArrowRight className="size-3.5 text-subtle-foreground" aria-label="switched to" />
            </>
          ) : null}
          <strong className="font-semibold">
            <ModelName name={call.model} />
          </strong>
        </div>
        <Row label="Decision">{call.decision || '—'}</Row>
        <Row label="Effort">{call.effort || '—'}</Row>
        <Row label="Selection" title={method || undefined}>
          {sessionActionLabel(call.session_action) ??
            (call.selection_method ? call.selection_method.replace(/_/g, ' ') : '—')}
        </Row>
        {reason !== undefined ? (
          <Row label="Why" title={call.reason}>
            {sessionReasonText(reason) ?? reason}
          </Row>
        ) : null}
        {call.signals?.length ? (
          <div className="mt-auto flex flex-wrap gap-1 pt-0.5" aria-label="Signals">
            {call.signals.map((signal) => {
              const [type, rule] = signalParts(signal)
              return (
                <span
                  key={signal}
                  title={signal}
                  className="inline-flex max-w-full items-center gap-1 truncate rounded-md bg-secondary px-1.5 py-0.5 text-[11.5px] text-secondary-foreground"
                >
                  {type ? <span className="text-subtle-foreground">{type}</span> : null}
                  {rule}
                </span>
              )
            })}
          </div>
        ) : null}
      </Section>
      <Section title="Usage">
        <div className="grid grid-cols-2 gap-2">
          <div className="flex flex-col">
            <span className="text-[11px] text-subtle-foreground">Input</span>
            <span className="font-mono text-lg font-semibold tabular">
              {formatTokenCount(call.prompt_tokens)}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-[11px] text-subtle-foreground">Output</span>
            <span className="font-mono text-lg font-semibold tabular">
              {formatTokenCount(call.completion_tokens)}
            </span>
          </div>
        </div>
        <Row
          label="Cost"
          title={
            call.cost !== undefined && cost === '—'
              ? 'Priced in another unit than the run’s total, so left out of it'
              : undefined
          }
        >
          {call.cost === undefined
            ? 'No price'
            : cost +
              (cost !== '—' && call.pricing_source === 'catalog_list_price' ? ' list price' : '')}
        </Row>
        <Row label="Duration">{formatRunDuration(call.duration_ms)}</Row>
        {(call.router_calls ?? []).map((entry) => (
          <Row
            key={`${entry.kind}-${entry.model ?? ''}`}
            label={`Router: ${entry.kind.replace(/_/g, ' ')}`}
          >
            {entry.calls}× ·{' '}
            {entry.cost !== undefined
              ? callCostText({ cost: entry.cost, currency: entry.currency }, currency)
              : 'no price'}
          </Row>
        ))}
        {call.looper_attempts ? (
          <Row label="Looper">{countText(call.looper_attempts, 'attempt')}</Row>
        ) : null}
      </Section>
      <Section
        title="Outcome"
        action={
          <Link
            to={getInsightsRecordPath(call.record_id)}
            className="inline-flex items-center gap-1 text-[12.5px] font-medium text-primary hover:underline"
          >
            Open the replay record
            <ArrowUpRight className="size-3.5" />
          </Link>
        }
      >
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge tone={statusTone(call)} title={call.terminal_reason}>
            {describeCallStatus(call)}
          </Badge>
          {call.final_answer ? <Badge tone="brand">Final answer</Badge> : null}
        </div>
        <Row label="Tools">
          {call.tool_results
            ? `${call.tool_results} results sent · ${call.tool_failures ?? 0} failed`
            : 'No results sent'}
        </Row>
        {call.hops?.length ? (
          <Row label="Hops">
            {call.hops
              .map(
                (hop) =>
                  `${modelLabel(labels, hop.model) || 'no model'} ${describeCallStatus(hop)}`,
              )
              .join(', ')}
          </Row>
        ) : null}
        {call.outcomes?.length ? (
          <Row
            label="Reported"
            title={call.outcomes
              .map((outcome) => `${outcome.source}: ${outcome.verdict}`)
              .join(', ')}
          >
            {call.outcomes
              .map((outcome) => `${outcomeWord(outcome.source)} · ${verdictLabel(outcome.verdict)}`)
              .join(', ')}
          </Row>
        ) : null}
        {events.length ? <AgentEventChips events={events} /> : null}
      </Section>
    </div>
  )
}

interface RunSegmentPanelProps {
  segment: Pick<TimelineBlock, 'agent' | 'model' | 'calls'>
  call?: RunCall
  /** The run-wide number of each call, by record id. */
  numbers?: Map<string, number>
  /** The run's one currency, which the costs read in. */
  currency?: string
  onSelectCall: (call: RunCall) => void
}

// A block's calls, one per row, and the card of the one picked.
export function RunSegmentPanel({
  segment,
  call,
  numbers,
  currency,
  onSelectCall,
}: RunSegmentPanelProps) {
  const labels = useModelLabels()
  return (
    <aside className="flex flex-col gap-3" aria-label="Step details">
      <header className="flex flex-wrap items-center gap-2">
        <Badge tone="neutral" className="font-semibold">
          {segment.agent === 'main' ? 'Main agent' : segment.agent}
        </Badge>
        <span
          className="inline-flex min-w-0 items-center gap-1.5 text-[13px]"
          title={modelTitle(labels, segment.model)}
        >
          <i className="size-2.5 rounded-[3px] bg-current opacity-60" aria-hidden />
          <ModelName name={segment.model} />
        </span>
        <span className="ml-auto text-xs text-subtle-foreground">
          {countText(segment.calls.length, 'call')}
        </span>
      </header>
      <RunToolList calls={segment.calls} />
      {segment.calls.length > 1 ? (
        <ol className="flex max-h-60 flex-col overflow-y-auto rounded-xl border border-border bg-card">
          {segment.calls.map((entry) => (
            <li key={entry.record_id} className="border-b border-border last:border-0">
              <button
                type="button"
                data-selected={entry.record_id === call?.record_id}
                onClick={() => onSelectCall(entry)}
                className="grid w-full grid-cols-[2.25rem_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 px-3 py-2 text-left text-xs transition-colors hover:bg-surface-subtle data-[selected=true]:bg-primary-soft/60 sm:grid-cols-[2.5rem_4.5rem_minmax(0,1fr)_auto_auto]"
              >
                <span className="font-mono tabular text-subtle-foreground">
                  {numbers?.get(entry.record_id) ? `#${numbers.get(entry.record_id)}` : ''}
                </span>
                <span className="hidden font-mono tabular text-subtle-foreground sm:inline">
                  {clock(Date.parse(entry.timestamp))}
                </span>
                <span className="truncate font-medium">
                  {entry.request_kind || entry.decision || '—'}
                </span>
                <Badge size="sm" tone={statusTone(entry)}>
                  {describeCallStatus(entry)}
                </Badge>
                <AgentEventChips
                  events={callEvents(entry)}
                  className="col-span-full justify-end sm:col-span-1"
                />
              </button>
            </li>
          ))}
        </ol>
      ) : null}
      {call ? (
        <RunCallDetails call={call} currency={currency} />
      ) : (
        <p className="text-[13px] text-muted-foreground">Pick a call for its card.</p>
      )}
    </aside>
  )
}

// How long a tool call took on the client: the tool's own run when a harness
// plugin reported it, else the gateway's window.
function toolDuration(tool: RunToolCall): number | undefined {
  if (tool.timing_source === 'plugin') {
    return tool.started_at && tool.ended_at
      ? Date.parse(tool.ended_at) - Date.parse(tool.started_at)
      : undefined
  }
  return tool.client_ms
}

// A tool call's time as the run reads it. A plugin's report is the tool's
// own run, with the time it waited for approval apart; the gateway's window
// is an estimate, marked ≈, that the calls of one response share.
function toolTimeText(tool: RunToolCall): string {
  const duration = toolDuration(tool)
  if (tool.timing_source !== 'plugin') {
    return duration === undefined ? '' : `≈${formatRunDuration(duration)}`
  }
  let wait = ''
  if (tool.approval_ms !== undefined) {
    wait = `waited ${formatRunDuration(tool.approval_ms)} for approval`
  } else if (tool.approval_requested_at) {
    // A start the harness took before the answer holds the wait.
    wait = duration === undefined ? 'asked for approval' : 'approval wait included'
  }
  return [duration === undefined ? '' : `ran ${formatRunDuration(duration)}`, wait]
    .filter(Boolean)
    .join(', ')
}

function toolStatusText(status?: string): string {
  if (status === 'succeeded') return 'ok'
  if (status === 'failed') return 'failed'
  if (status === 'denied') return 'refused'
  return 'no result yet'
}

// How long a tool call took on the client, and whose clock says so.
function ToolDuration({ tool }: { tool: RunToolCall }) {
  const text = toolTimeText(tool)
  if (!text) return <span />
  const source =
    tool.timing_source === 'plugin'
      ? "As the harness's vSR plugin reported it: the tool's own run, and its approval wait apart"
      : 'Estimated by the gateway: from the response that asked for the call to the request that carried its result, shared by the calls of one response'
  return (
    <span title={source} data-timing={tool.timing_source}>
      {text}
    </span>
  )
}

// Every tool call of a block's turns with its arguments and result.
function RunToolList({ calls }: { calls: RunCall[] }) {
  const rows = calls.flatMap((entry) =>
    (entry.tool_calls ?? []).map((tool, index) => ({ entry, tool, index })),
  )
  if (rows.length === 0) return null
  return (
    <ol
      className="m-0 grid list-none grid-cols-[minmax(0,max-content)_minmax(0,1fr)_auto_auto] gap-x-2 gap-y-1 p-0"
      aria-label="Tool calls"
    >
      {rows.map(({ entry, tool, index }) => (
        <li
          key={`${entry.record_id}-${tool.id || index}`}
          data-status={tool.status}
          className="col-span-full grid grid-cols-subgrid items-baseline text-xs text-muted-foreground data-[status=failed]:[&>span]:text-destructive-foreground"
        >
          <strong className="truncate font-mono font-semibold text-foreground">{tool.name}</strong>
          <code className="truncate">{tool.arguments || tool.summary || 'arguments not kept'}</code>
          <ToolDuration tool={tool} />
          <span>{toolStatusText(tool.status)}</span>
        </li>
      ))}
    </ol>
  )
}
