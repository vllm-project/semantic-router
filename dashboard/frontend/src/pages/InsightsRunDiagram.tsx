import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, ArrowUpRight, ChevronDown, ChevronUp } from 'lucide-react'

import { Badge, cn, Segmented, Tooltip } from '../ui'
import { formatInsightsCost } from '../utils/insightsCost'

import { sessionActionLabel, sessionReasonText } from './insightsAgentIdentity'
import { formatTokenCount } from './insightsDisplayFormat'
import { ModelName, modelLabel, modelText, useModelLabels } from './insightsModelLabelSupport'
import { getInsightsRecordPath } from './insightsPageSupport'
import {
  countText,
  describeCallStatus,
  formatRunCost,
  formatRunDuration,
  modelHue,
  outcomeWord,
  RUN_MARKER_GLYPHS,
  RUN_MARKER_LABELS,
  verdictLabel,
} from './insightsRunSupport'
import {
  timeAt,
  type RunTimeline,
  type TimelineBlock,
  type TimelineBreak,
} from './insightsRunTimeline'
import type { RunCall, RunSummary } from './insightsRunTypes'

const MARKER_TONE: Record<string, string> = {
  failure: 'text-destructive-foreground',
  block: 'text-destructive-foreground',
  tool_failure: 'text-warning-foreground',
  guard: 'text-warning-foreground',
}

function Markers({ markers, className }: { markers?: string[]; className?: string }) {
  return (
    <>
      {(markers ?? []).map((marker) => (
        <span
          key={marker}
          data-marker={marker}
          title={RUN_MARKER_LABELS[marker] ?? marker}
          className={cn(
            'font-semibold leading-none',
            MARKER_TONE[marker] ?? 'text-foreground/80',
            className,
          )}
        >
          {RUN_MARKER_GLYPHS[marker] ?? '•'}
        </span>
      ))}
    </>
  )
}

function modelTitle(labels: Record<string, string>, model: string): string {
  return model ? modelText(labels, model) : 'no model'
}

const clock = (time: number) =>
  new Date(time).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  })

/** Lanes shown before the rest fold behind a toggle. */
const LANE_LIMIT = 6
const ZOOMS = ['1', '2', '4', '8'] as const
type Zoom = (typeof ZOOMS)[number]
const ROW_HEIGHT = 'h-10'

function blockSpan(block: TimelineBlock): [number, number] {
  const starts = block.calls.map((call) => Date.parse(call.timestamp))
  const ends = block.calls.map((call) => Date.parse(call.timestamp) + (call.duration_ms ?? 0))
  return [Math.min(...starts), Math.max(...ends)]
}

function blockTotals(block: TimelineBlock) {
  let input = 0
  let output = 0
  let cost = 0
  let currency: string | undefined
  for (const call of block.calls) {
    input += call.prompt_tokens ?? 0
    output += call.completion_tokens ?? 0
    if (call.cost !== undefined) {
      cost += call.cost
      currency = currency ?? call.currency
    }
  }
  return { input, output, cost, currency }
}

function BlockCard({ block, labels }: { block: TimelineBlock; labels: Record<string, string> }) {
  const [start, end] = blockSpan(block)
  const totals = blockTotals(block)
  return (
    <div className="flex min-w-48 flex-col gap-1.5 text-xs">
      <strong className="text-[13px] font-semibold">{modelTitle(labels, block.model)}</strong>
      <span className="text-muted-foreground">
        {block.agent} · {countText(block.calls.length, 'call')} · {clock(start)}–{clock(end)}
      </span>
      <span className="font-mono tabular">
        {formatTokenCount(totals.input)} in · {formatTokenCount(totals.output)} out
        {totals.currency ? ` · ${formatInsightsCost(totals.cost, totals.currency)}` : ''}
      </span>
      {block.markers.length ? (
        <span className="flex flex-wrap gap-x-2 gap-y-0.5 text-muted-foreground">
          {block.markers.map((marker) => (
            <span key={marker} className="inline-flex items-center gap-1">
              <Markers markers={[marker]} />
              {RUN_MARKER_LABELS[marker] ?? marker}
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
  align: 'start' | 'center' | 'end'
}

/** Width of one character of the axis's 10.5px monospace text. */
const AXIS_CHAR_PX = 6.4
/** The track width assumed before it is measured. */
const AXIS_DEFAULT_PX = 960

// The edge times first, then each idle stretch's length, then the middle
// times, each kept only where it clears the labels already placed.
function axisLabels(timeline: RunTimeline, trackPx: number): AxisLabel[] {
  const width = trackPx > 0 ? trackPx : AXIS_DEFAULT_PX
  const tick = (at: number): AxisLabel => ({
    key: `tick-${at}`,
    at,
    text: clock(timeAt(timeline, at)),
    align: at === 0 ? 'start' : at === 100 ? 'end' : 'center',
  })
  const candidates: AxisLabel[] = [
    tick(0),
    tick(100),
    ...timeline.breaks.map((gap) => ({
      key: `idle-${gap.left}`,
      at: gap.left + gap.width / 2,
      text: `${formatRunDuration(gap.ms)} idle`,
      align: 'center' as const,
    })),
    tick(50),
    tick(25),
    tick(75),
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
      return to + 12 <= start || from >= end + 12
    })
    if (clear) placed.push(label)
  }
  return placed
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
  /** The agents the calls table follows; the other lanes dim. */
  agents?: string[]
  onSelect: (block: TimelineBlock) => void
  onAgent?: (label: string) => void
}

// A lane per agent that made a call in view, in tree order, its calls as
// blocks on one axis; idle stretches fold to a hatched mark. Long runs zoom,
// with an overview of the whole run and its focus window, and long agent
// lists fold after the first lanes.
export function RunDiagram({
  run,
  timeline,
  selected,
  agents = [],
  onSelect,
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
      <div className="grid grid-cols-[minmax(0,148px)_minmax(0,1fr)] gap-x-3 sm:grid-cols-[minmax(0,200px)_minmax(0,1fr)]">
        <div className="flex flex-col" role="list" aria-label="Run agents">
          {lanes.map((lane) => {
            const followed = agents.includes(lane.agent.label)
            const type = lane.agent.type || lane.agent.kind
            const figures = `${countText(lane.agent.calls, 'call')} · ${formatRunCost(lane.agent.cost)}`
            const content = (
              <>
                <span className="flex min-w-0 items-center gap-1.5">
                  {lane.depth > 0 ? (
                    <span aria-hidden className="text-subtle-foreground">
                      ↳
                    </span>
                  ) : null}
                  <strong className="truncate text-[12.5px] font-semibold">
                    {lane.agent.label}
                  </strong>
                </span>
                <span
                  className="truncate text-[11px] text-subtle-foreground tabular"
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
                  ROW_HEIGHT,
                  'flex items-center transition-opacity',
                  agents.length > 0 && !followed && 'opacity-45',
                )}
                style={{ paddingLeft: `${lane.depth * 0.75}rem` }}
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
          <div
            className="relative"
            style={{ width: `${scale * 100}%` }}
            role="list"
            aria-label="Run timeline"
          >
            {lanes.map((lane) => (
              <div
                key={lane.agent.label}
                role="listitem"
                className={cn(
                  ROW_HEIGHT,
                  'relative flex items-center transition-opacity',
                  agents.length > 0 && !agents.includes(lane.agent.label) && 'opacity-45',
                )}
              >
                <div className="relative h-8 w-full rounded-lg bg-surface-subtle/70 ring-1 ring-inset ring-border/70">
                  {timeline.breaks.map((gap) => (
                    <Hatch key={gap.left} gap={gap} />
                  ))}
                  {lane.spawnAt !== undefined ? (
                    <span
                      className="absolute -inset-y-1 w-px border-l-2 border-dotted border-chart-3"
                      style={{ left: `${lane.spawnAt}%` }}
                      title={`Spawned by ${lane.agent.parent ?? 'main'}`}
                    />
                  ) : null}
                  {lane.blocks.map((block) => (
                    <Tooltip key={block.key} content={<BlockCard block={block} labels={labels} />}>
                      <button
                        type="button"
                        data-selected={block.key === selected}
                        aria-label={`${modelTitle(labels, block.model)}, ${countText(block.calls.length, 'call')}`}
                        className="absolute inset-y-1 flex items-center justify-center gap-1 overflow-hidden rounded-md px-1 text-[10.5px] text-white shadow-xs outline-none transition-[box-shadow,filter,transform] hover:brightness-110 focus-visible:ring-[3px] focus-visible:ring-ring/50 data-[selected=true]:ring-2 data-[selected=true]:ring-foreground data-[selected=true]:ring-offset-1 data-[selected=true]:ring-offset-card"
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
                        {block.markers.length ? (
                          <span className="inline-flex gap-0.5 rounded-sm bg-white/85 px-1 py-px text-[10px]">
                            <Markers markers={block.markers.slice(0, 3)} />
                          </span>
                        ) : null}
                      </button>
                    </Tooltip>
                  ))}
                </div>
              </div>
            ))}
            <div
              className="relative h-5 font-mono text-[10.5px] text-subtle-foreground tabular"
              aria-hidden
            >
              {axisLabels(timeline, trackPx * scale).map((label) => (
                <span
                  key={label.key}
                  className={cn(
                    'absolute top-1 whitespace-nowrap',
                    label.align === 'start' && 'left-0',
                    label.align === 'end' && 'right-0',
                    label.align === 'center' && '-translate-x-1/2',
                  )}
                  style={label.align === 'center' ? { left: `${label.at}%` } : undefined}
                >
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

export function RunLegend({ run }: { run: RunSummary }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      {run.models.map((model) => (
        <span key={model.name} className="inline-flex items-center gap-1.5">
          <i
            className="size-2.5 rounded-[3px]"
            style={{ background: modelHue(run.models, model.name) }}
          />
          <ModelName name={model.name} />
        </span>
      ))}
      {Object.entries(RUN_MARKER_LABELS)
        .filter(([marker]) => run.events?.[marker])
        .map(([marker, label]) => (
          <span key={marker} className="inline-flex items-center gap-1">
            <Markers markers={[marker]} /> {label}
          </span>
        ))}
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
export function RunCallDetails({ call }: { call: RunCall }) {
  const labels = useModelLabels()
  const method = [call.selection_method, call.session_action].filter(Boolean).join(' · ')
  const reason = call.reason === '' ? 'Hidden from this viewer' : call.reason
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
        <Row label="Cost">
          {call.cost !== undefined ? formatInsightsCost(call.cost, call.currency) : 'No price'}
        </Row>
        <Row label="Duration">{formatRunDuration(call.duration_ms)}</Row>
        {(call.router_calls ?? []).map((entry) => (
          <Row
            key={`${entry.kind}-${entry.model ?? ''}`}
            label={`Router: ${entry.kind.replace(/_/g, ' ')}`}
          >
            {entry.calls}× ·{' '}
            {entry.cost !== undefined ? formatInsightsCost(entry.cost, entry.currency) : 'no price'}
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
            ? `${call.tool_results} returned · ${call.tool_failures ?? 0} failed`
            : 'None returned'}
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
        {call.markers?.length ? (
          <div className="flex flex-wrap gap-x-2.5 gap-y-1 text-xs text-muted-foreground">
            {call.markers.map((marker) => (
              <span key={marker} className="inline-flex items-center gap-1">
                <Markers markers={[marker]} />
                {RUN_MARKER_LABELS[marker] ?? marker}
              </span>
            ))}
          </div>
        ) : null}
      </Section>
    </div>
  )
}

interface RunSegmentPanelProps {
  segment: Pick<TimelineBlock, 'agent' | 'model' | 'calls'>
  call?: RunCall
  /** The run-wide number of each call, by record id. */
  numbers?: Map<string, number>
  onSelectCall: (call: RunCall) => void
}

// A block's calls, one per row, and the card of the one picked.
export function RunSegmentPanel({ segment, call, numbers, onSelectCall }: RunSegmentPanelProps) {
  const labels = useModelLabels()
  return (
    <aside className="flex flex-col gap-3" aria-label="Step details">
      <header className="flex flex-wrap items-center gap-2">
        <Badge tone="neutral" className="font-semibold">
          {segment.agent}
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
      {segment.calls.length > 1 ? (
        <ol className="flex max-h-60 flex-col overflow-y-auto rounded-xl border border-border bg-card">
          {segment.calls.map((entry) => (
            <li key={entry.record_id} className="border-b border-border last:border-0">
              <button
                type="button"
                data-selected={entry.record_id === call?.record_id}
                onClick={() => onSelectCall(entry)}
                className="grid w-full grid-cols-[2.25rem_minmax(0,1fr)_auto_auto] items-center gap-3 px-3 py-2 text-left text-xs transition-colors hover:bg-surface-subtle data-[selected=true]:bg-primary-soft/60 sm:grid-cols-[2.5rem_4.5rem_minmax(0,1fr)_auto_auto]"
              >
                <span className="font-mono tabular text-subtle-foreground">
                  {numbers?.get(entry.record_id) ? `#${numbers.get(entry.record_id)}` : ''}
                </span>
                <span className="hidden font-mono tabular text-subtle-foreground sm:inline">
                  {clock(Date.parse(entry.timestamp))}
                </span>
                <span className="truncate font-medium">{entry.decision || '—'}</span>
                <Badge size="sm" tone={statusTone(entry)}>
                  {describeCallStatus(entry)}
                </Badge>
                <span className="inline-flex gap-1">
                  <Markers markers={entry.markers} />
                </span>
              </button>
            </li>
          ))}
        </ol>
      ) : null}
      {call ? (
        <RunCallDetails call={call} />
      ) : (
        <p className="text-[13px] text-muted-foreground">Pick a call for its card.</p>
      )}
    </aside>
  )
}
