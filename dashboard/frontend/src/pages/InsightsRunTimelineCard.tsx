import { useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'

import {
  Button,
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui'

import { pageHolding } from './insightsListData'
import { RunDiagram, RunLegend, RunSegmentPanel } from './InsightsRunDiagram'
import { formatRunDuration } from './insightsRunSupport'
import {
  buildRunTimeline,
  TIMELINE_SIZE,
  TIMELINE_SIZES,
  type TimelineBlock,
} from './insightsRunTimeline'
import type { RunCall, RunResponse, RunSummary } from './insightsRunTypes'

const SIZE_ALL = 'all'

interface RunTimelineCardProps {
  run: RunSummary
  response: RunResponse
  /** The run-wide number of each call, by record id. */
  numbers: Map<string, number>
  /** The step a link names. */
  step?: string
  /** The agents the calls table follows. */
  agents: string[]
  onAgent: (label: string) => void
}

// The run's calls as lanes on a time axis that pages through long runs and
// folds idle time, with the calls of a picked block under it.
export function RunTimelineCard({
  run,
  response,
  numbers,
  step,
  agents,
  onAgent,
}: RunTimelineCardProps) {
  const total = response.calls.length
  const [blockKey, setBlockKey] = useState<string>()
  const [callId, setCallId] = useState<string | undefined>(step)
  const [windowSize, setWindowSize] = useState<number>()
  const [windowPage, setWindowPage] = useState<number>()

  // The timeline opens on the window that holds the step a link names, else the first.
  const size = Math.min(windowSize ?? (total > TIMELINE_SIZE ? TIMELINE_SIZE : total), total) || 1
  const pages = Math.max(1, Math.ceil(total / size))
  const stepIndex = step ? (numbers.get(step) ?? 0) - 1 : -1
  const windowAt = Math.min(windowPage ?? pageHolding(stepIndex, size), pages)
  const from = (windowAt - 1) * size
  const to = Math.min(total, from + size)
  const timeline = useMemo(() => buildRunTimeline(response, from, to), [from, response, to])
  const blocks = useMemo(() => timeline.lanes.flatMap((lane) => lane.blocks), [timeline])
  const block =
    blocks.find((entry) => entry.key === blockKey) ??
    blocks.find((entry) => entry.calls.some((one) => one.record_id === callId))
  const call =
    block?.calls.find((entry) => entry.record_id === callId) ??
    (block?.calls.length === 1 ? block.calls[0] : undefined)

  const selectBlock = (next: TimelineBlock) => {
    setBlockKey(next.key)
    setCallId(next.calls.length === 1 ? next.calls[0].record_id : undefined)
  }
  const selectCall = (next: RunCall) => {
    setBlockKey(blocks.find((entry) => entry.calls.includes(next))?.key)
    setCallId(next.record_id)
  }

  return (
    <Card id="run-timeline" className="mb-5 scroll-mt-6">
      <CardHeader>
        <CardTitle>Timeline</CardTitle>
        <CardDescription className="col-start-1">
          One lane per agent; a block is a stretch of calls on one model. Click a block for its
          calls, an agent to follow its calls below.
        </CardDescription>
        {total > TIMELINE_SIZES[0] ? (
          <CardAction className="gap-2 max-sm:col-start-1 max-sm:row-span-1 max-sm:row-start-3 max-sm:justify-self-start">
            <span className="whitespace-nowrap text-xs text-subtle-foreground tabular">
              Calls {(from + 1).toLocaleString()}–{to.toLocaleString()} of {total.toLocaleString()}
            </span>
            <Button
              variant="outline"
              size="icon-xs"
              aria-label="Earlier calls"
              disabled={windowAt <= 1}
              onClick={() => setWindowPage(windowAt - 1)}
            >
              <ChevronLeft />
            </Button>
            <Button
              variant="outline"
              size="icon-xs"
              aria-label="Later calls"
              disabled={windowAt >= pages}
              onClick={() => setWindowPage(windowAt + 1)}
            >
              <ChevronRight />
            </Button>
            <Select
              value={size >= total ? SIZE_ALL : String(size)}
              onValueChange={(value) => {
                const next = value === SIZE_ALL ? total : Number(value)
                setWindowSize(next)
                setWindowPage(pageHolding(from, next))
              }}
            >
              <SelectTrigger size="sm" aria-label="Calls in view" className="h-7 px-2 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                {TIMELINE_SIZES.filter((option) => option < total).map((option) => (
                  <SelectItem key={option} value={String(option)}>
                    {option} calls
                  </SelectItem>
                ))}
                <SelectItem value={SIZE_ALL}>All {total.toLocaleString()} calls</SelectItem>
              </SelectContent>
            </Select>
          </CardAction>
        ) : null}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <RunDiagram
          run={run}
          timeline={timeline}
          selected={block?.key}
          agents={agents}
          onSelect={selectBlock}
          onAgent={onAgent}
        />
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5">
          <RunLegend run={run} />
          <span className="text-xs text-subtle-foreground tabular">
            {formatRunDuration(timeline.activeMs)} of activity in view
            {timeline.breaks.length === 1 ? ' · 1 idle stretch folded' : ''}
            {timeline.breaks.length > 1 ? ` · ${timeline.breaks.length} idle stretches folded` : ''}
          </span>
        </div>
        {block ? (
          <div className="border-t border-border pt-4">
            <RunSegmentPanel
              segment={block}
              call={call}
              numbers={numbers}
              onSelectCall={selectCall}
            />
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
