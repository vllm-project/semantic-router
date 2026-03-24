import { Link } from 'react-router-dom'

import type { Column } from '../components/DataTable'
import type { ViewSection } from '../components/ViewModal'
import { formatDate } from '../types/evaluation'

import styles from './MetaRoutingPage.module.css'
import type {
  MetaRoutingAggregateValue,
  MetaRoutingFeedbackDetailResponse,
  MetaRoutingFeedbackSummary,
  MetaRoutingPassTrace,
  MetaRoutingQueryFilters,
} from './metaRoutingPageTypes'

const renderPillRow = (values: string[] | undefined, accent = false) => {
  if (!values || values.length === 0) {
    return <span className={styles.mono}>-</span>
  }
  return (
    <div className={styles.pillRow}>
      {values.map((value) => (
        <span
          key={value}
          className={`${styles.pill} ${accent ? styles.pillAccent : ''}`.trim()}
        >
          {value}
        </span>
      ))}
    </div>
  )
}

export function buildMetaRoutingQueryString(
  filters: MetaRoutingQueryFilters,
  pagination?: { limit: number; offset: number },
) {
  const params = new URLSearchParams()

  if (pagination) {
    params.set('limit', String(pagination.limit))
    params.set('offset', String(pagination.offset))
  }

  const search = filters.searchTerm.trim()
  if (search) params.set('search', search)
  if (filters.mode !== 'all') params.set('mode', filters.mode)
  if (filters.trigger !== 'all') params.set('trigger', filters.trigger)
  if (filters.rootCause !== 'all') params.set('root_cause', filters.rootCause)
  if (filters.actionType !== 'all') params.set('action_type', filters.actionType)
  if (filters.signalFamily !== 'all') params.set('signal_family', filters.signalFamily)
  if (filters.overturned !== 'all') params.set('overturned', filters.overturned)
  if (filters.decision !== 'all') params.set('decision', filters.decision)
  if (filters.model !== 'all') params.set('model', filters.model)
  if (filters.responseStatus !== 'all') params.set('response_status', filters.responseStatus)

  const query = params.toString()
  return query ? `?${query}` : ''
}

export async function fetchMetaRoutingJSON<T>(url: string, label: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch ${label}: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as T
}

export function formatPercent(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }
  return `${(value * 100).toFixed(1)}%`
}

export function formatSignedDecimal(value?: number, digits = 3) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }
  const prefix = value > 0 ? '+' : ''
  return `${prefix}${value.toFixed(digits)}`
}

export function formatLatencyDelta(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'N/A'
  }
  const prefix = value > 0 ? '+' : ''
  return `${prefix}${value} ms`
}

export function buildStatusTone(status?: number) {
  if (!status || status < 100) return styles.statusWarn
  if (status < 400) return styles.statusSuccess
  return styles.statusError
}

export function createMetaRoutingTableColumns(): Column<MetaRoutingFeedbackSummary>[] {
  return [
    {
      key: 'request_id',
      header: 'Request ID',
      width: '220px',
      sortable: true,
      render: (row) => (
        <span className={styles.requestId} title={row.request_id || ''}>
          {row.request_id || '-'}
        </span>
      ),
    },
    {
      key: 'timestamp',
      header: 'Created',
      width: '132px',
      sortable: true,
      render: (row) => <span className={styles.timestamp}>{formatDate(row.timestamp)}</span>,
    },
    {
      key: 'mode',
      header: 'Mode',
      width: '110px',
      sortable: true,
      render: (row) => (
        <span className={`${styles.pill} ${row.mode === 'active' ? styles.pillAccent : ''}`.trim()}>
          {row.mode || '-'}
        </span>
      ),
    },
    {
      key: 'pass_count',
      header: 'Passes',
      width: '88px',
      align: 'center',
      sortable: true,
      render: (row) => <span className={styles.mono}>{row.pass_count ?? 0}</span>,
    },
    {
      key: 'trigger_names',
      header: 'Triggers',
      width: '220px',
      render: (row) => renderPillRow(row.trigger_names, true),
    },
    {
      key: 'action_types',
      header: 'Actions',
      width: '200px',
      render: (row) => renderPillRow(row.action_types),
    },
    {
      key: 'final_decision_name',
      header: 'Final Decision',
      width: '220px',
      sortable: true,
      render: (row) => (
        <div>
          <div className={styles.decision}>{row.final_decision_name || '-'}</div>
          <div className={styles.mono}>{row.final_model || '-'}</div>
        </div>
      ),
    },
    {
      key: 'latency_delta_ms',
      header: 'Latency Δ',
      width: '110px',
      sortable: true,
      render: (row) => <span className={styles.mono}>{formatLatencyDelta(row.latency_delta_ms)}</span>,
    },
    {
      key: 'overturned_decision',
      header: 'Outcome',
      width: '120px',
      render: (row) => (
        <span className={`${styles.flag} ${row.overturned_decision ? styles.flagActive : ''}`.trim()}>
          {row.overturned_decision ? 'Overturned' : 'Stable'}
        </span>
      ),
    },
    {
      key: 'response_status',
      header: 'Status',
      width: '90px',
      align: 'center',
      render: (row) => (
        <span className={`${styles.statusBadge} ${buildStatusTone(row.response_status)}`}>
          {row.response_status || '-'}
        </span>
      ),
    },
  ]
}

function renderMetric(label: string, value: string | number | undefined) {
  return (
    <div className={styles.metricBlock}>
      <span className={styles.metricLabel}>{label}</span>
      <span className={styles.metricValue}>{value === undefined || value === '' ? '-' : value}</span>
    </div>
  )
}

function renderPassTrace(pass: MetaRoutingPassTrace) {
  const traceQuality = pass.trace_quality
  return (
    <article key={`${pass.kind || 'pass'}-${pass.index}`} className={styles.passCard}>
      <div className={styles.passHeader}>
        <span className={styles.passTitle}>
          Pass {pass.index + 1}
          {pass.kind ? ` · ${pass.kind}` : ''}
        </span>
        <div className={styles.pillRow}>
          <span className={`${styles.pill} ${pass.input_compressed ? styles.pillAccent : ''}`.trim()}>
            {pass.input_compressed ? 'compressed input' : 'full input'}
          </span>
          <span className={`${styles.pill} ${traceQuality?.fragile ? styles.pillAccent : ''}`.trim()}>
            {traceQuality?.fragile ? 'fragile' : 'stable'}
          </span>
        </div>
      </div>
      <div className={styles.passMetrics}>
        {renderMetric('Latency', formatLatencyDelta(pass.latency_ms))}
        {renderMetric('Decision', pass.decision_name || '-')}
        {renderMetric('Confidence', pass.decision_confidence?.toFixed(3))}
        {renderMetric('Margin', pass.decision_margin?.toFixed(3))}
        {renderMetric('Candidates', pass.decision_candidate_count)}
        {renderMetric('Model', pass.selected_model || '-')}
        {renderMetric('Runner-up', pass.runner_up_decision_name || '-')}
        {renderMetric('Winner basis', pass.decision_winner_basis || '-')}
        {renderMetric('Signal dominance', traceQuality?.signal_dominance?.toFixed(3))}
        {renderMetric('Avg signal confidence', traceQuality?.avg_signal_confidence?.toFixed(3))}
        {renderMetric('Boundary distance', traceQuality?.projection_boundary_min_distance?.toFixed(3))}
      </div>
      {pass.assessment ? (
        <div className={styles.recordStack}>
          {pass.assessment.triggers?.length ? (
            <div>
              <span className={styles.metricLabel}>Triggers</span>
              {renderPillRow(pass.assessment.triggers, true)}
            </div>
          ) : null}
          {pass.assessment.root_causes?.length ? (
            <div>
              <span className={styles.metricLabel}>Root Causes</span>
              {renderPillRow(pass.assessment.root_causes)}
            </div>
          ) : null}
          {pass.partition_conflicts?.length ? (
            <div>
              <span className={styles.metricLabel}>Partition Conflicts</span>
              {renderPillRow(pass.partition_conflicts)}
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  )
}

export function buildMetaRoutingRecordSections(
  detail: MetaRoutingFeedbackDetailResponse,
): ViewSection[] {
  const record = detail.record
  const trace = record.observation.trace
  const plan = record.action.plan
  const routerReplayID = record.outcome.router_replay_id

  const sections: ViewSection[] = [
    {
      title: 'Request',
      fields: [
        { label: 'Record ID', value: detail.id },
        { label: 'Created', value: formatDate(detail.timestamp) },
        { label: 'Mode', value: record.mode || '-' },
        { label: 'Request ID', value: record.observation.request_id || '-' },
        { label: 'Request model', value: record.observation.request_model || '-' },
        {
          label: 'Request query',
          value: record.observation.request_query || '-',
          fullWidth: true,
        },
      ],
    },
    {
      title: 'Final Outcome',
      fields: [
        { label: 'Final decision', value: record.outcome.final_decision_name || '-' },
        { label: 'Decision confidence', value: record.outcome.final_decision_confidence?.toFixed(3) || '-' },
        { label: 'Final model', value: record.outcome.final_model || '-' },
        { label: 'Response status', value: record.outcome.response_status || '-' },
        { label: 'Latency delta', value: formatLatencyDelta(trace?.latency_delta_ms) },
        { label: 'Decision margin delta', value: formatSignedDecimal(trace?.decision_margin_delta) },
        { label: 'Projection boundary delta', value: formatSignedDecimal(trace?.projection_boundary_delta) },
        { label: 'Overturned', value: trace?.overturned_decision ? 'Yes' : 'No' },
      ],
    },
    {
      title: 'Assessment and Actions',
      fields: [
        { label: 'Pass count', value: trace?.pass_count || 0 },
        { label: 'Planned', value: record.action.planned ? 'Yes' : 'No' },
        { label: 'Executed', value: record.action.executed ? 'Yes' : 'No' },
        { label: 'Executed passes', value: record.action.executed_pass_count || 0 },
        { label: 'Triggers', value: renderPillRow(trace?.trigger_names, true), fullWidth: true },
        { label: 'Root causes', value: renderPillRow(plan?.root_causes), fullWidth: true },
        { label: 'Executed actions', value: renderPillRow(record.action.executed_action_types), fullWidth: true },
        { label: 'Refined families', value: renderPillRow(trace?.refined_signal_families), fullWidth: true },
      ],
    },
    {
      title: 'Downstream Outcome',
      fields: [
        { label: 'Cache hit', value: record.outcome.cache_hit ? 'Yes' : 'No' },
        { label: 'Streaming', value: record.outcome.streaming ? 'Yes' : 'No' },
        { label: 'PII blocked', value: record.outcome.pii_blocked ? 'Yes' : 'No' },
        { label: 'Hallucination detected', value: record.outcome.hallucination_detected ? 'Yes' : 'No' },
        { label: 'Response jailbreak detected', value: record.outcome.response_jailbreak_detected ? 'Yes' : 'No' },
        { label: 'Unverified factual response', value: record.outcome.unverified_factual_response ? 'Yes' : 'No' },
        { label: 'RAG backend', value: record.outcome.rag_backend || '-' },
        { label: 'User feedback signals', value: renderPillRow(record.outcome.user_feedback_signals), fullWidth: true },
      ],
    },
  ]

  if (trace?.passes?.length) {
    sections.push({
      title: 'Pass Traces',
      fields: [
        {
          label: 'Recorded passes',
          value: <div className={styles.recordStack}>{trace.passes.map(renderPassTrace)}</div>,
          fullWidth: true,
        },
      ],
    })
  }

  sections.push({
    title: 'Cross-links',
    fields: [
      {
        label: 'Router replay',
        value: routerReplayID ? (
          <Link className={styles.linkButton} to={`/insights?replay_id=${encodeURIComponent(routerReplayID)}`}>
            Open in Insights
          </Link>
        ) : 'No replay record linked',
        fullWidth: true,
      },
    ],
  })

  return sections
}

export function distributionMax(data: MetaRoutingAggregateValue[]) {
  return data.reduce((max, entry) => Math.max(max, entry.value), 0)
}
