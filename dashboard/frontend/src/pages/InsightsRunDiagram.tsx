import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowUpRight } from 'lucide-react'

import { cn } from '../ui'
import { formatInsightsCost } from '../utils/insightsCost'

import { ModelName, modelLabel, modelText, useModelLabels } from './insightsModelLabelSupport'
import { getInsightsRecordPath } from './insightsPageSupport'
import {
  countText,
  describeCallStatus,
  formatRunCost,
  formatRunDuration,
  modelHue,
  RUN_MARKER_GLYPHS,
  RUN_MARKER_LABELS,
  type RunLane,
  type RunSegment,
} from './insightsRunSupport'
import type { RunCall, RunSummary } from './insightsRunTypes'

const MARKER_TONE: Record<string, string> = {
  failure: 'text-destructive-foreground',
  block: 'text-destructive-foreground',
  tool_failure: 'text-warning-foreground',
  guard: 'text-warning-foreground',
}

function Markers({ markers }: { markers?: string[] }) {
  return (
    <>
      {(markers ?? []).map((marker) => (
        <span
          key={marker}
          data-marker={marker}
          title={RUN_MARKER_LABELS[marker] ?? marker}
          className={cn('font-semibold leading-none', MARKER_TONE[marker] ?? 'text-foreground/80')}
        >
          {RUN_MARKER_GLYPHS[marker] ?? '•'}
        </span>
      ))}
    </>
  )
}

interface RunDiagramProps {
  run: RunSummary
  lanes: RunLane[]
  selected?: string
  activeAgent?: string
  onSelect: (segment: RunSegment) => void
}

// A model's label with its configured name, for a tooltip.
function modelTitle(labels: Record<string, string>, model: string): string {
  return model ? modelText(labels, model) : 'no model'
}

// A lane per agent, in tree order, with its calls as segments on one time axis.
export function RunDiagram({ run, lanes, selected, activeAgent, onSelect }: RunDiagramProps) {
  const labels = useModelLabels()
  return (
    <div className="flex flex-col gap-1.5" role="list" aria-label="Run timeline">
      {lanes.map((lane) => (
        <div
          key={lane.agent.label}
          role="listitem"
          className={cn(
            'grid grid-cols-[minmax(0,170px)_minmax(0,1fr)] items-center gap-3 rounded-md transition-opacity',
            activeAgent && activeAgent !== lane.agent.label && 'opacity-45',
          )}
        >
          <div className="flex min-w-0 flex-col" style={{ paddingLeft: `${lane.depth * 0.9}rem` }}>
            <strong className="truncate font-mono text-[12px] font-medium">
              {lane.depth > 0 ? `↳ ${lane.agent.label}` : lane.agent.label}
            </strong>
            <span className="truncate text-[11px] text-subtle-foreground">
              {[
                lane.agent.kind,
                countText(lane.agent.calls, 'call'),
                formatRunCost(lane.agent.cost),
              ]
                .filter(Boolean)
                .join(' · ')}
            </span>
          </div>
          <div className="relative h-7 rounded-md bg-surface-subtle ring-1 ring-inset ring-border">
            {lane.spawnAt !== undefined ? (
              <span
                className="absolute inset-y-0 w-px border-l border-dashed border-chart-3"
                style={{ left: `${lane.spawnAt}%` }}
                title={`Spawned by ${lane.agent.parent ?? 'main'}`}
              />
            ) : null}
            {lane.segments.map((segment) => (
              <button
                key={segment.key}
                type="button"
                data-selected={segment.key === selected}
                className="absolute inset-y-1 flex min-w-1.5 items-center justify-center gap-0.5 overflow-hidden rounded-[5px] px-0.5 text-[10px] text-primary-foreground shadow-xs outline-none transition-[box-shadow,filter] hover:brightness-110 focus-visible:ring-[3px] focus-visible:ring-ring/50 data-[selected=true]:ring-2 data-[selected=true]:ring-foreground"
                style={{
                  left: `${segment.left}%`,
                  width: `${segment.width}%`,
                  background: modelHue(run.models, segment.model),
                }}
                title={`${modelTitle(labels, segment.model)} · ${countText(segment.calls.length, 'call')}`}
                onClick={() => onSelect(segment)}
              >
                {segment.calls.length > 1 ? (
                  <span className="font-mono font-semibold tabular">{segment.calls.length}</span>
                ) : null}
                <span className="rounded-sm bg-surface/85 px-0.5 empty:hidden">
                  <Markers markers={segment.markers} />
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className="grid grid-cols-[minmax(0,170px)_minmax(0,1fr)] gap-3">
        <span aria-hidden />
        <div className="flex justify-between font-mono text-[10.5px] text-subtle-foreground tabular">
          <span>0s</span>
          <span>{formatRunDuration(Math.round(run.duration_ms / 2))}</span>
          <span>{formatRunDuration(run.duration_ms)}</span>
        </div>
      </div>
    </div>
  )
}

export function RunLegend({ run }: { run: RunSummary }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
      {run.models.map((model) => (
        <span key={model.name} className="inline-flex items-center gap-1.5">
          <i
            className="size-2 rounded-[3px]"
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

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[96px_minmax(0,1fr)] gap-3 py-1.5">
      <dt className="text-xs text-subtle-foreground">{label}</dt>
      <dd className="min-w-0 break-words text-[13px]">{children}</dd>
    </div>
  )
}

function callCost(call: RunCall) {
  const own = call.cost !== undefined ? formatInsightsCost(call.cost, call.currency) : 'no price'
  const router = (call.router_calls ?? [])
    .map(
      (entry) =>
        `${entry.kind} ${entry.calls}× ${entry.cost !== undefined ? formatInsightsCost(entry.cost, entry.currency) : 'no price'}`,
    )
    .join(', ')
  return router ? `${own}; Router calls: ${router}` : own
}

// The details of one call: how it was routed and why, what answered, what it
// cost, how it ended, and the record it came from.
export function RunCallDetails({ call }: { call: RunCall }) {
  const labels = useModelLabels()
  const tokens = `${call.prompt_tokens ?? '-'} in / ${call.completion_tokens ?? '-'} out`
  return (
    <dl className="divide-y divide-border">
      <Field label="Decision">{call.decision || '-'}</Field>
      <Field label="Signals">{call.signals?.join(', ') || 'none matched'}</Field>
      <Field label="Model">
        {call.entrypoint ? `${call.entrypoint} → ` : ''}
        <ModelName name={call.model} />
        {call.previous_model ? ` (was ${modelTitle(labels, call.previous_model)})` : ''}
      </Field>
      <Field label="Effort">{call.effort || '-'}</Field>
      <Field label="Method">
        {[call.selection_method, call.session_action].filter(Boolean).join(' · ') || '-'}
      </Field>
      <Field label="Reason">
        {call.reason === '' ? 'Hidden from this viewer' : call.reason || '-'}
      </Field>
      <Field label="Tokens">{tokens}</Field>
      <Field label="Cost">{callCost(call)}</Field>
      <Field label="Status">
        {describeCallStatus(call)}
        {call.terminal_reason ? ` · ${call.terminal_reason}` : ''}
        {call.final_answer ? ' · final answer' : ''}
      </Field>
      <Field label="Tools">
        {call.tool_results
          ? `${call.tool_results} results, ${call.tool_failures ?? 0} failed`
          : 'none returned'}
      </Field>
      {call.looper_attempts ? (
        <Field label="Looper">{`${call.looper_attempts} attempts`}</Field>
      ) : null}
      {call.hops?.length ? (
        <Field label="Hops">
          {call.hops
            .map(
              (hop) => `${modelLabel(labels, hop.model) || 'no model'} ${describeCallStatus(hop)}`,
            )
            .join(', ')}
        </Field>
      ) : null}
      {call.outcomes?.length ? (
        <Field label="Outcome">
          {call.outcomes.map((outcome) => `${outcome.source}: ${outcome.verdict}`).join(', ')}
        </Field>
      ) : null}
      <Field label="Markers">
        {call.markers?.length ? (
          <span className="inline-flex gap-1.5">
            <Markers markers={call.markers} />
          </span>
        ) : (
          'none'
        )}
      </Field>
      <Field label="Record">
        <Link
          to={getInsightsRecordPath(call.record_id)}
          className="inline-flex items-center gap-1 font-medium text-primary hover:underline"
        >
          Open the replay record
          <ArrowUpRight className="size-3.5" />
        </Link>
      </Field>
    </dl>
  )
}

interface RunSegmentPanelProps {
  segment: RunSegment
  call?: RunCall
  onSelectCall: (call: RunCall) => void
}

// A segment's calls, one per row, and the details of the one picked.
export function RunSegmentPanel({ segment, call, onSelectCall }: RunSegmentPanelProps) {
  const labels = useModelLabels()
  return (
    <aside className="flex flex-col gap-3" aria-label="Step details">
      <header className="flex items-baseline justify-between gap-2">
        <strong
          className="truncate text-[13px] font-semibold"
          title={modelTitle(labels, segment.model)}
        >
          {segment.agent} · {modelLabel(labels, segment.model) || 'no model'}
        </strong>
        <span className="shrink-0 text-xs text-subtle-foreground">
          {segment.calls.length} call{segment.calls.length === 1 ? '' : 's'}
        </span>
      </header>
      {segment.calls.length > 1 ? (
        <ol className="flex max-h-56 flex-col overflow-y-auto rounded-lg border border-border">
          {segment.calls.map((entry) => (
            <li key={entry.record_id} className="border-b border-border last:border-0">
              <button
                type="button"
                data-selected={entry.record_id === call?.record_id}
                onClick={() => onSelectCall(entry)}
                className="grid w-full grid-cols-[72px_minmax(0,1fr)_auto_auto] items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-surface-subtle data-[selected=true]:bg-primary-soft/60"
              >
                <span className="font-mono tabular text-subtle-foreground">
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </span>
                <span className="truncate">{entry.decision || '-'}</span>
                <span className="text-muted-foreground">{describeCallStatus(entry)}</span>
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
        <p className="text-[13px] text-muted-foreground">Pick a call for its details.</p>
      )}
    </aside>
  )
}
