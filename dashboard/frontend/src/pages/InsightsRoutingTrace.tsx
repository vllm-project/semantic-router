import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { LucideIcon } from 'lucide-react'
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Blend,
  Boxes,
  Check,
  CornerDownLeft,
  Puzzle,
  Radar,
  Split,
} from 'lucide-react'
import { motion } from 'framer-motion'

import type { ViewSection } from '../components/ViewPanel'
import { Badge, IconTile, cn, type Tone } from '../ui'
import { formatDateTime } from '../utils/dateTime'

import { formatLatency, formatTokenCount } from './insightsDisplayFormat'
import { LifecycleStatus, ScoreBar } from './InsightsDisplay'
import { ModelChip } from './insightsModelLabelSupport'
import { formatInsightsDecisionName } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'
import InsightsRecordSection from './InsightsRecordSection'
import { recordedSignalMatches, signalFamilyLabel } from './insightsRecordSignals'

const ease = [0.16, 1, 0.3, 1] as const

function TraceStep({
  index,
  icon,
  tone,
  title,
  meta,
  aside,
  last = false,
  children,
}: {
  index: number
  icon: LucideIcon
  tone: Tone
  title: string
  meta?: ReactNode
  aside?: ReactNode
  last?: boolean
  children?: ReactNode
}) {
  return (
    <motion.li
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.04 + index * 0.04, duration: 0.4, ease }}
      className="relative flex gap-3.5"
    >
      <div className="flex flex-col items-center">
        <IconTile icon={icon} tone={tone} size="sm" />
        {last ? null : <span className="mt-1.5 w-px flex-1 bg-border" />}
      </div>
      <div className={cn('flex min-w-0 flex-1 flex-col', last ? 'pb-0.5' : 'pb-6')}>
        <div className="flex min-h-7 flex-wrap items-center gap-x-2.5 gap-y-1">
          <h3 data-trace-step={title} className="shrink-0 text-[13px] font-semibold">
            {title}
          </h3>
          {meta ? (
            <span className="min-w-0 truncate text-xs text-muted-foreground">{meta}</span>
          ) : null}
          {aside ? <span className="ml-auto flex shrink-0 items-center gap-2">{aside}</span> : null}
        </div>
        {children ? <div className="mt-2.5 flex min-w-0 flex-col gap-2.5">{children}</div> : null}
      </div>
    </motion.li>
  )
}

function Chip({
  label,
  children,
  mono = true,
}: {
  label: string
  children: ReactNode
  mono?: boolean
}) {
  return (
    <span className="inline-flex h-6 max-w-full items-center gap-1.5 rounded-md border border-border bg-surface px-2 text-xs">
      <span className="shrink-0 text-subtle-foreground">{label}</span>
      <span className={cn('truncate', mono && 'font-mono text-[11.5px]')}>{children}</span>
    </span>
  )
}

function Tile({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-lg border border-border bg-surface px-3 py-2">
      <span className="text-[11px] text-subtle-foreground">{label}</span>
      <span className="flex min-h-5 items-center truncate text-[13px]">{children}</span>
    </div>
  )
}

function confidenceOf(record: InsightsRecord, type: string, value: string) {
  const confidences = record.signal_confidences ?? {}
  const found = confidences[`${type}:${value}`] ?? confidences[`systemone:${value}`]
  return typeof found === 'number' && Number.isFinite(found) && found >= 0 && found <= 1
    ? found
    : undefined
}

/** The record's routing, step by step: what came in, which signals matched, what was derived, the decision, the model, the plugins and the answer. */
export default function InsightsRoutingTrace({
  record,
  sections,
}: {
  record: InsightsRecord
  sections: ViewSection[]
}) {
  const section = (title: string) => sections.find((entry) => entry.title === title)
  const sectionIndex = (title: string) => sections.findIndex((entry) => entry.title === title)
  const embedded = (title: string) => {
    const found = section(title)
    return found ? (
      <InsightsRecordSection
        key={title}
        section={found}
        sectionIndex={sectionIndex(title)}
        embedded
      />
    ) : null
  }

  const matches = recordedSignalMatches(record.signals)
  const families = [...new Set(matches.map((match) => match.type))]
  const projections = record.projections ?? []
  const scores = Object.entries(record.projection_scores ?? {})
  const diagnostics = record.route_diagnostics
  const promptHidden = record.prompt === ''
  const confidence =
    record.confidence_score_available === true && typeof record.confidence_score === 'number'
      ? record.confidence_score
      : undefined
  const selectionSections = [
    'Selection Stages',
    'Session Routing',
    'Protection Candidate Scores',
    'Observed Candidate Scores',
    'Adaptation Candidate Scores',
    'Request Capacity',
  ]

  return (
    <ol className="flex flex-col" aria-label="Routing trace">
      <TraceStep
        index={0}
        icon={ArrowDownToLine}
        tone="blue"
        title="Received"
        meta={formatDateTime(record.timestamp)}
        aside={
          record.request_id ? (
            <span className="font-mono text-[11px] text-subtle-foreground">
              {record.request_id}
            </span>
          ) : null
        }
      >
        <div className="flex flex-wrap items-center gap-1.5">
          <Chip label="model">{record.original_model || 'not named'}</Chip>
          <ArrowRight className="size-3 text-subtle-foreground" />
          <Chip label="recipe">{record.recipe || 'default'}</Chip>
          {typeof record.context_token_count === 'number' ? (
            <Chip label="context">{record.context_token_count.toLocaleString()} tokens</Chip>
          ) : null}
          {record.streaming ? (
            <Chip label="stream" mono={false}>
              on
            </Chip>
          ) : null}
          {record.reasoning_effort || record.reasoning_mode ? (
            <Chip label="reasoning">{record.reasoning_effort || record.reasoning_mode}</Chip>
          ) : null}
        </div>
        {record.prompt ? (
          <p className="line-clamp-3 rounded-lg border border-border bg-surface-subtle/60 px-3 py-2 text-[13px] leading-relaxed text-foreground/85">
            {record.prompt}
            {record.prompt_truncated ? (
              <span className="text-subtle-foreground"> … cut at the capture limit</span>
            ) : null}
          </p>
        ) : promptHidden ? (
          <p className="text-xs text-subtle-foreground">The prompt is hidden for your role.</p>
        ) : null}
      </TraceStep>

      <TraceStep
        index={1}
        icon={Radar}
        tone="gold"
        title="Signals matched"
        meta={
          matches.length
            ? `${matches.length} matched in ${families.map(signalFamilyLabel).join(', ')}`
            : 'No signal matches were recorded'
        }
      >
        {matches.length ? (
          <div className="overflow-hidden rounded-lg border border-border">
            {matches.map((match) => {
              const probability = confidenceOf(record, match.type, match.value)
              return (
                <div
                  key={`${match.type}:${match.value}`}
                  className="flex items-center gap-3 border-b border-border px-3 py-2 last:border-0"
                  title={match.value}
                >
                  <Radar className="size-3.5 shrink-0 text-chart-2" />
                  <div className="flex min-w-0 flex-1 items-baseline gap-2">
                    <span className="truncate font-mono text-[12px]">{match.value}</span>
                    <span className="shrink-0 text-[11px] text-subtle-foreground">
                      {signalFamilyLabel(match.type)}
                    </span>
                  </div>
                  {probability !== undefined ? (
                    <>
                      <ScoreBar value={probability} className="w-20 shrink-0" />
                      <span className="w-9 shrink-0 text-right font-mono text-[11.5px] tabular">
                        {probability.toFixed(2)}
                      </span>
                    </>
                  ) : null}
                  <Badge tone="success" size="sm">
                    <Check />
                    Matched
                  </Badge>
                </div>
              )
            })}
          </div>
        ) : null}
        {embedded('Routing Metadata')}
        {embedded('Signal Errors')}
      </TraceStep>

      {projections.length || scores.length || record.projection_trace ? (
        <TraceStep
          index={2}
          icon={Blend}
          tone="teal"
          title="Projections"
          meta={`${projections.length} ${projections.length === 1 ? 'output' : 'outputs'} · ${scores.length} ${scores.length === 1 ? 'score' : 'scores'}`}
        >
          {projections.length || scores.length ? (
            <div className="flex flex-wrap items-center gap-1.5">
              {projections.map((output) => (
                <Badge key={output} tone="teal">
                  {output}
                </Badge>
              ))}
              {scores.map(([name, value]) => (
                <Chip key={name} label={name}>
                  {Number.isFinite(value) ? value.toFixed(2) : '—'}
                </Chip>
              ))}
            </div>
          ) : null}
          {embedded('Projection Trace')}
        </TraceStep>
      ) : null}

      <TraceStep
        index={3}
        icon={Split}
        tone="violet"
        title="Decision matched"
        meta={record.category ? `category ${record.category}` : undefined}
      >
        {record.decision ? (
          <div className="rounded-lg border border-chart-3/25 bg-chart-3/5 px-3.5 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[13px] font-semibold">
                {formatInsightsDecisionName(record.decision)}
              </span>
              <span className="rounded-md bg-chart-3/12 px-1.5 py-0.5 font-mono text-[11px] font-medium text-chart-3">
                P{record.decision_priority}
              </span>
              {record.decision_tier ? (
                <span className="font-mono text-[11px] text-subtle-foreground">
                  tier {record.decision_tier}
                </span>
              ) : null}
              <span className="font-mono text-[11.5px] text-subtle-foreground">
                {record.decision}
              </span>
              <Link
                to={`/routing/decisions?d=${encodeURIComponent(record.decision)}`}
                className="ml-auto inline-flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground"
              >
                Open decision
                <ArrowUpRight className="size-3" />
              </Link>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              {confidence !== undefined
                ? `Matched with ${(confidence * 100).toFixed(1)}% confidence.`
                : 'No confidence score was recorded for this match.'}
              {record.reasoning_mode ? ` Reasoning ${record.reasoning_mode}.` : ''}
            </p>
          </div>
        ) : (
          <p className="text-[13px] text-muted-foreground">
            No decision matched; the recipe&apos;s default answered.
          </p>
        )}
      </TraceStep>

      <TraceStep
        index={4}
        icon={Boxes}
        tone="green"
        title="Model selected"
        meta={
          record.selection_method ? `by ${record.selection_method.replace(/_/g, ' ')}` : undefined
        }
      >
        <div className="flex flex-wrap items-center gap-2 text-[13px]">
          {record.original_model ? (
            <>
              <ModelChip name={record.original_model} className="text-muted-foreground" />
              <ArrowRight className="size-3.5 text-subtle-foreground" />
            </>
          ) : null}
          <span className="font-medium">
            <ModelChip name={record.selected_model} fallback="No backend selected" />
          </span>
          {diagnostics?.previous_model && diagnostics.previous_model !== record.selected_model ? (
            <span className="text-xs text-muted-foreground">
              switched from <ModelChip name={diagnostics.previous_model} size={12} />
            </span>
          ) : null}
        </div>
        {diagnostics?.selection_reasoning ? (
          <p className="text-xs leading-relaxed text-muted-foreground">
            {diagnostics.selection_reasoning}
          </p>
        ) : null}
        {selectionSections.map(embedded)}
      </TraceStep>

      <TraceStep
        index={5}
        icon={Puzzle}
        tone="rose"
        title="Plugins"
        meta="What ran around the model call"
      >
        {embedded('Plugin Status')}
      </TraceStep>

      <TraceStep
        index={6}
        icon={CornerDownLeft}
        tone="slate"
        title="Response"
        meta={
          record.from_cache
            ? 'Answered from the Router cache'
            : record.terminal_reason?.replace(/_/g, ' ')
        }
        last
      >
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Tile label="Status">
            <LifecycleStatus record={record} />
          </Tile>
          <Tile label="Duration">
            <span className="font-mono text-[12px] tabular">
              {formatLatency(record.duration_ms)}
            </span>
          </Tile>
          <Tile label="Output tokens">
            <span className="font-mono text-[12px] tabular">
              {formatTokenCount(record.completion_tokens)}
            </span>
          </Tile>
          <Tile label="Ended">
            <span className="truncate text-[12px]">
              {record.ended_at ? formatDateTime(record.ended_at) : '—'}
            </span>
          </Tile>
        </div>
        {embedded('Outcomes')}
      </TraceStep>
    </ol>
  )
}
