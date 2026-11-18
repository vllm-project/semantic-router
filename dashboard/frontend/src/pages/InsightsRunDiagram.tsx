import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowUpRight } from 'lucide-react'

import { cn } from '../ui'
import { formatInsightsCost } from '../utils/insightsCost'

import { ModelName, modelLabel, modelText, useModelLabels } from './insightsModelLabelSupport'
import { getInsightsRecordPath } from './insightsPageSupport'
import {
  describeCallStatus,
  formatRunDuration,
  modelHue,
  outcomeWord,
  RUN_MARKER_GLYPHS,
  RUN_MARKER_LABELS,
  verdictLabel,
} from './insightsRunSupport'
import { toolDuration, toolStatusText } from './insightsRunSequence'
import type { RunCall, RunSummary, RunToolCall } from './insightsRunTypes'

const MARKER_TONE: Record<string, string> = {
  failure: 'text-destructive-foreground',
  block: 'text-destructive-foreground',
  tool_failure: 'text-warning-foreground',
  guard: 'text-warning-foreground',
}

export function Markers({ markers }: { markers?: string[] }) {
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

// A model's label with its configured name, for a tooltip.
function modelTitle(labels: Record<string, string>, model: string): string {
  return model ? modelText(labels, model) : 'no model'
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
  const listPrice = call.pricing_source === 'catalog_list_price' ? ' list price' : ''
  const own =
    call.cost !== undefined ? formatInsightsCost(call.cost, call.currency) + listPrice : 'no price'
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
      <Field label="Results">
        {call.tool_results
          ? `${call.tool_results} tool results sent, ${call.tool_failures ?? 0} failed`
          : 'no tool results sent'}
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
          {call.outcomes
            .map((outcome) => `${outcomeWord(outcome.source)}: ${verdictLabel(outcome.verdict)}`)
            .join(', ')}
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

interface RunRowDetailsProps {
  calls: RunCall[]
  call?: RunCall
  /** The run-wide number of each call, by record id. */
  numbers?: Map<string, number>
  onSelectCall: (call: RunCall) => void
}

// How long a tool call took on the client, and whose clock says so.
function ToolDuration({ tool }: { tool: RunToolCall }) {
  const duration = toolDuration(tool)
  if (duration === undefined) return <span />
  const source =
    tool.timing_source === 'plugin'
      ? 'Run time the harness reported'
      : 'From the response that asked for it to the request that carried its result, as the gateway saw it'
  return <span title={source}>{formatRunDuration(duration)}</span>
}

// Every tool call of a row's turns with its arguments and result.
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

// A row's tool calls, its calls one per line when there are several, and the
// details of the one picked.
export function RunRowDetails({ calls, call, numbers, onSelectCall }: RunRowDetailsProps) {
  return (
    <div
      className="flex flex-col gap-3 rounded-lg border border-border bg-surface-subtle p-3"
      aria-label="Step details"
    >
      <RunToolList calls={calls} />
      {calls.length > 1 ? (
        <ol className="flex max-h-56 flex-col overflow-y-auto rounded-lg border border-border bg-surface">
          {calls.map((entry) => (
            <li key={entry.record_id} className="border-b border-border last:border-0">
              <button
                type="button"
                data-selected={entry.record_id === call?.record_id}
                onClick={() => onSelectCall(entry)}
                className="grid w-full grid-cols-[40px_72px_minmax(0,1fr)_auto_auto] items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-surface-subtle data-[selected=true]:bg-primary-soft/60"
              >
                <span className="font-mono tabular text-subtle-foreground">
                  {numbers?.get(entry.record_id) ? `#${numbers.get(entry.record_id)}` : ''}
                </span>
                <span className="font-mono tabular text-subtle-foreground">
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </span>
                <span className="truncate">{entry.request_kind || entry.decision || '-'}</span>
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
    </div>
  )
}
