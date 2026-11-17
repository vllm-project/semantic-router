import { Link } from 'react-router-dom'

import { formatInsightsCost } from '../utils/insightsCost'

import styles from './InsightsRun.module.css'
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

function Markers({ markers }: { markers?: string[] }) {
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

interface RunDiagramProps {
  run: RunSummary
  lanes: RunLane[]
  selected?: string
  onSelect: (segment: RunSegment) => void
}

// A model's label with its configured name, for a tooltip.
function modelTitle(labels: Record<string, string>, model: string): string {
  return model ? modelText(labels, model) : 'no model'
}

// A lane per agent, in tree order, with its calls as segments on one time axis.
export function RunDiagram({ run, lanes, selected, onSelect }: RunDiagramProps) {
  const labels = useModelLabels()
  return (
    <div className={styles.diagram} role="list" aria-label="Run timeline">
      {lanes.map((lane) => (
        <div key={lane.agent.label} className={styles.lane} role="listitem">
          <div className={styles.laneLabel} style={{ paddingLeft: `${lane.depth * 0.9}rem` }}>
            <strong>{lane.depth > 0 ? `↳ ${lane.agent.label}` : lane.agent.label}</strong>
            <span>
              {[
                lane.agent.kind,
                countText(lane.agent.calls, 'call'),
                formatRunCost(lane.agent.cost),
              ]
                .filter(Boolean)
                .join(' · ')}
            </span>
          </div>
          <div className={styles.track}>
            {lane.spawnAt !== undefined ? (
              <span
                className={styles.spawn}
                style={{ left: `${lane.spawnAt}%` }}
                title={`Spawned by ${lane.agent.parent ?? 'main'}`}
              />
            ) : null}
            {lane.segments.map((segment) => (
              <button
                key={segment.key}
                type="button"
                className={styles.segment}
                data-selected={segment.key === selected}
                style={{
                  left: `${segment.left}%`,
                  width: `${segment.width}%`,
                  background: modelHue(run.models, segment.model),
                }}
                title={`${modelTitle(labels, segment.model)} · ${countText(segment.calls.length, 'call')}`}
                onClick={() => onSelect(segment)}
              >
                {segment.calls.length > 1 ? (
                  <span className={styles.count}>{segment.calls.length}</span>
                ) : null}
                <Markers markers={segment.markers} />
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className={styles.axis}>
        <span>0s</span>
        <span>{formatRunDuration(Math.round(run.duration_ms / 2))}</span>
        <span>{formatRunDuration(run.duration_ms)}</span>
      </div>
    </div>
  )
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
            .map((hop) => `${modelLabel(labels, hop.model) || 'no model'} ${describeCallStatus(hop)}`)
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

interface RunSegmentPanelProps {
  segment: RunSegment
  call?: RunCall
  onSelectCall: (call: RunCall) => void
}

// A segment's calls, one per row, and the details of the one picked.
export function RunSegmentPanel({ segment, call, onSelectCall }: RunSegmentPanelProps) {
  const labels = useModelLabels()
  return (
    <aside className={styles.panel} aria-label="Step details">
      <header>
        <strong title={modelTitle(labels, segment.model)}>
          {segment.agent} · {modelLabel(labels, segment.model) || 'no model'}
        </strong>
        <span>
          {segment.calls.length} call{segment.calls.length === 1 ? '' : 's'}
        </span>
      </header>
      {segment.calls.length > 1 ? (
        <ol className={styles.steps}>
          {segment.calls.map((entry) => (
            <li key={entry.record_id}>
              <button
                type="button"
                data-selected={entry.record_id === call?.record_id}
                onClick={() => onSelectCall(entry)}
              >
                <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                <span>{entry.decision || '-'}</span>
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
    </aside>
  )
}
