import type { ReactNode } from 'react'

import type { ViewSection } from '../components/ViewPanel'
import type { InsightsRecord } from './insightsPageTypes'
import type { ReplaySessionPolicy } from './insightsPageRoutingTypes'
import InsightsSelectionTrace from './InsightsSelectionTrace'
import styles from './InsightsPage.module.css'

function metric(value: number | undefined) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : '—'
}

function diagnosticTable(headers: string[], rows: Array<{ key: string; cells: ReactNode[] }>) {
  return (
    <div className={styles.projectionTraceWrap}>
      <table className={styles.projectionTraceTable}>
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              {row.cells.map((value, index) => (
                <td key={headers[index]}>{value}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function protectionCandidates(policy: ReplaySessionPolicy): ViewSection[] {
  const names = [
    ...new Set([
      ...(policy.candidate_models || []),
      ...Object.keys(policy.base_scores || {}),
      ...Object.keys(policy.final_scores || {}),
      ...Object.keys(policy.candidate_traces || {}),
    ]),
  ].sort()
  if (!names.length) return []
  return [
    {
      title:
        policy.mode === 'observe' ? 'Observed Candidate Scores' : 'Protection Candidate Scores',
      fields: [
        {
          label: 'Recorded selector scores and switch adjustments',
          fullWidth: true,
          value: diagnosticTable(
            [
              'Model',
              'Base',
              'Final',
              'Quality gap',
              'Handoff',
              'Cache penalty',
              'Tool-loop penalty',
              'Switch history',
              'Net switch advantage',
            ],
            names.map((model) => {
              const trace = policy.candidate_traces?.[model]
              return {
                key: model,
                cells: [
                  model,
                  metric(policy.base_scores?.[model] ?? trace?.base_score),
                  metric(policy.final_scores?.[model] ?? trace?.final_score),
                  metric(trace?.quality_gap),
                  metric(trace?.handoff_penalty),
                  metric(trace?.prefix_cache_penalty),
                  metric(trace?.tool_loop_penalty),
                  metric(trace?.switch_history_penalty),
                  metric(trace?.net_switch_advantage),
                ],
              }
            }),
          ),
        },
      ],
    },
  ]
}

export function buildRoutingExplanationSections(record: InsightsRecord): ViewSection[] {
  const sections: ViewSection[] = []
  const diagnostics = record.route_diagnostics
  if (diagnostics?.selection_trace) {
    sections.push({
      title: 'Selection Stages',
      fields: [
        {
          label: 'Recorded objective filtering',
          fullWidth: true,
          value: (
            <InsightsSelectionTrace
              trace={diagnostics.selection_trace}
              selectedModel={record.selected_model}
            />
          ),
        },
      ],
    })
  }
  const policy =
    record.learning?.protection ?? record.session_policy ?? record.learning?.protection_preflight
  if (record.session_id || policy) {
    sections.push({
      title: 'Session Routing',
      fields: [
        { label: 'Session', value: record.session_id || 'Not recorded' },
        { label: 'Conversation', value: record.conversation_id || 'Not recorded' },
        { label: 'Turn', value: record.turn_index + 1 },
        {
          label: 'Previous model',
          value: diagnostics?.previous_model || policy?.current_model || 'Not recorded',
        },
        { label: 'Actual selected model', value: record.selected_model || 'No backend selected' },
        { label: 'Protection mode', value: policy?.mode || 'Not applied' },
        { label: 'Protection scope', value: policy?.scope || 'Not recorded' },
        { label: 'Eligible models', value: policy?.candidate_models?.join(', ') || 'Not recorded' },
        {
          label: 'Session identity header',
          value: policy?.identity?.headers?.session || 'Not recorded',
        },
        {
          label: 'Session identity status',
          value: policy?.identity?.session?.status || 'Not recorded',
        },
        {
          label: 'Conversation identity header',
          value: policy?.identity?.headers?.conversation || 'Not recorded',
        },
        {
          label: 'Conversation identity status',
          value: policy?.identity?.conversation?.status || 'Not recorded',
        },
        {
          label: 'Protection applied',
          value: diagnostics?.session_policy_applied === true ? 'Yes' : 'No',
        },
        { label: 'Recorded route action', value: diagnostics?.session_action || 'Not recorded' },
        {
          label: 'Route reason',
          value: diagnostics?.session_reason || policy?.reason || 'Not recorded',
        },
        { label: 'Phase', value: diagnostics?.session_phase || policy?.phase || 'Not recorded' },
        ...(policy
          ? [
              {
                label: policy.mode === 'observe' ? 'Observed model suggestion' : 'Protected model',
                value: policy.selected_model || 'Not recorded',
              },
              {
                label: policy.mode === 'observe' ? 'Observed hold reason' : 'Hold reason',
                value: policy.hard_lock_reason || policy.decision_reason || 'None recorded',
              },
              { label: 'Active tool loop', value: policy.active_tool_loop === true ? 'Yes' : 'No' },
              {
                label: 'Idle time',
                value:
                  policy.idle_known === true ? `${metric(policy.idle_for_seconds)} s` : 'Unknown',
              },
              {
                label: 'Missing session evidence',
                value: policy.missing_signals?.join(', ') || 'None recorded',
              },
            ]
          : []),
      ],
    })
  }
  if (policy) sections.push(...protectionCandidates(policy))

  const adaptation = record.learning?.adaptation
  const scores = adaptation?.scores
  if (scores && Object.keys(scores).length) {
    sections.push({
      title: 'Adaptation Candidate Scores',
      fields: [
        { label: 'Mode', value: adaptation?.mode || 'Not recorded' },
        { label: 'Reason', value: adaptation?.reason || 'Not recorded' },
        {
          label: 'Recorded score decomposition',
          fullWidth: true,
          value: diagnosticTable(
            [
              'Model',
              'Score',
              'Predicted quality',
              'Cost penalty',
              'Reliability penalty',
              'Latency adjustment',
              'Cache adjustment',
            ],
            Object.entries(scores)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([model, score]) => ({
                key: model,
                cells: [
                  model,
                  metric(score.score),
                  metric(score.predicted_quality),
                  metric(score.cost_penalty),
                  metric(score.reliability_penalty),
                  metric(score.latency_adjustment),
                  metric(score.cache_adjustment),
                ],
              })),
          ),
        },
      ],
    })
  }
  if (diagnostics?.request_demand_snapshots?.length) {
    sections.push({
      title: 'Request Capacity',
      fields: [
        {
          label: 'Recorded input estimates and output reserves',
          fullWidth: true,
          value: diagnosticTable(
            [
              'Stage',
              'Model',
              'Input estimate',
              'Output reserve',
              'Total demand',
              'Input source',
              'Reserve source',
            ],
            diagnostics.request_demand_snapshots.map((demand, index) => ({
              key: String(index),
              cells: [
                demand.stage,
                demand.model || 'Unselected',
                demand.prompt_tokens,
                demand.reserved_output_tokens,
                demand.total_demand_known ? demand.total_demand_tokens : 'Unknown',
                demand.counting_source,
                demand.output_reserve_source,
              ],
            })),
          ),
        },
      ],
    })
  }
  if (diagnostics?.signal_errors && Object.keys(diagnostics.signal_errors).length) {
    sections.push({
      title: 'Signal Errors',
      fields: Object.entries(diagnostics.signal_errors).map(([name, error]) => ({
        label: name,
        value: error,
      })),
    })
  }
  return sections
}
