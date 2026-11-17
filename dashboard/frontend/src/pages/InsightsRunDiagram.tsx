import { Link } from 'react-router-dom'

import { formatInsightsCost } from '../utils/insightsCost'

import styles from './InsightsRun.module.css'
import { ModelName, modelLabel, modelText, useModelLabels } from './insightsModelLabelSupport'
import { getInsightsRecordPath } from './insightsPageSupport'
import {
  describeCallStatus,
  modelHue,
  RUN_MARKER_GLYPHS,
  RUN_MARKER_LABELS,
} from './insightsRunSupport'
import { toolStatusText, toolTimeSource, toolTimeText } from './insightsRunSequence'
import type { RunCall, RunSummary, RunToolCall } from './insightsRunTypes'

export function Markers({ markers }: { markers?: string[] }) {
  return (
    <>
      {(markers ?? []).map((marker) => (
        <span
          key={marker}
          className={styles.marker}
          data-marker={marker}
          title={RUN_MARKER_LABELS[marker] ?? marker}
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
    <div className={styles.legend}>
      {run.models.map((model) => (
        <span key={model.name}>
          <i style={{ background: modelHue(run.models, model.name) }} />
          <ModelName name={model.name} />
        </span>
      ))}
      {Object.entries(RUN_MARKER_LABELS).map(([marker, label]) => (
        <span key={marker}>
          <Markers markers={[marker]} /> {label}
        </span>
      ))}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className={styles.field}>
      <dt>{label}</dt>
      <dd>{children}</dd>
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
    <dl className={styles.details}>
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
          {call.outcomes.map((outcome) => `${outcome.source}: ${outcome.verdict}`).join(', ')}
        </Field>
      ) : null}
      <Field label="Markers">
        {call.markers?.length ? <Markers markers={call.markers} /> : 'none'}
      </Field>
      <Field label="Record">
        <Link to={getInsightsRecordPath(call.record_id)}>Open the replay record</Link>
      </Field>
    </dl>
  )
}

interface RunRowDetailsProps {
  calls: RunCall[]
  call?: RunCall
  onSelectCall: (call: RunCall) => void
}

// How long a tool call took on the client, and whose clock says so.
function ToolDuration({ tool }: { tool: RunToolCall }) {
  const text = toolTimeText(tool)
  if (!text) return <span />
  return (
    <span title={toolTimeSource(tool)} data-timing={tool.timing_source}>
      {text}
    </span>
  )
}

// Every tool call of a row's turns with its arguments and result.
function RunToolList({ calls }: { calls: RunCall[] }) {
  const rows = calls.flatMap((entry) =>
    (entry.tool_calls ?? []).map((tool, index) => ({ entry, tool, index })),
  )
  if (rows.length === 0) return null
  return (
    <ol className={styles.toolList} aria-label="Tool calls">
      {rows.map(({ entry, tool, index }) => (
        <li key={`${entry.record_id}-${tool.id || index}`} data-status={tool.status}>
          <strong>{tool.name}</strong>
          <code>{tool.arguments || tool.summary || 'arguments not kept'}</code>
          <ToolDuration tool={tool} />
          <span>{toolStatusText(tool.status)}</span>
        </li>
      ))}
    </ol>
  )
}

// A row's tool calls, its calls one per line when there are several, and the
// details of the one picked.
export function RunRowDetails({ calls, call, onSelectCall }: RunRowDetailsProps) {
  return (
    <div className={styles.panel} aria-label="Step details">
      <RunToolList calls={calls} />
      {calls.length > 1 ? (
        <ol className={styles.steps}>
          {calls.map((entry) => (
            <li key={entry.record_id}>
              <button
                type="button"
                data-selected={entry.record_id === call?.record_id}
                onClick={() => onSelectCall(entry)}
              >
                <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                <span>{entry.request_kind || entry.decision || '-'}</span>
                <span>{describeCallStatus(entry)}</span>
                <Markers markers={entry.markers} />
              </button>
            </li>
          ))}
        </ol>
      ) : null}
      {call ? (
        <RunCallDetails call={call} />
      ) : (
        <p className={styles.muted}>Pick a call for its details.</p>
      )}
    </div>
  )
}
