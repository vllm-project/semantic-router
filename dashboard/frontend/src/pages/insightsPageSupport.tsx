import type { Column } from '../components/DataTable'
import CollapsibleSection from '../components/CollapsibleSection'
import { formatRoutingMetadataValue } from '../components/routingMetadataDisplay'
import type { ViewField, ViewSection } from '../components/ViewPanel'
import { formatDateTime } from '../utils/dateTime'
import { Link } from 'react-router-dom'
import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'

import type {
  InsightsCostSummary,
  InsightsCurrencyCostSummary,
  InsightsRecord,
  Signal,
} from './insightsPageTypes'
import { buildProjectionTraceFields } from './insightsPageProjectionTrace'
import { buildRoutingExplanationSections } from './insightsPageRouting'
import { renderToolNamesCell } from './insightsPageToolTrace'
import styles from './InsightsPage.module.css'

export { filterInsightsRecords } from './insightsPageFilters'

export const formatInsightsDecisionName = (decision: string): string =>
  formatRoutingMetadataValue('x-vsr-selected-decision', decision)

export function getInsightsLifecyclePresentation(record: InsightsRecord) {
  const state = record.lifecycle_state || 'unknown'
  const successful =
    state === 'completed' && Boolean(record.response_status && record.response_status < 400)
  const errored =
    state === 'failed' ||
    state === 'aborted' ||
    (state === 'completed' && Boolean(record.response_status && record.response_status >= 400))
  const pending = state === 'in_progress'
  const label = record.response_status
    ? `${record.response_status} · ${state.replace('_', ' ')}`
    : state.replace('_', ' ')

  return { state, successful, errored, pending, label }
}

export function getUniqueDecisions(records: InsightsRecord[]) {
  const decisions = new Set<string>()
  records.forEach((record) => {
    if (record.decision) {
      decisions.add(record.decision)
    }
  })
  return Array.from(decisions).sort()
}

export function getUniqueModels(records: InsightsRecord[]) {
  const models = new Set<string>()
  records.forEach((record) => {
    if (record.selected_model) {
      models.add(record.selected_model)
    }
    if (record.original_model) {
      models.add(record.original_model)
    }
  })
  return Array.from(models).sort()
}

export function buildInsightsSummary(records: InsightsRecord[]): InsightsCostSummary {
  const groups = new Map<string, InsightsCurrencyCostSummary>()
  let costRecordCount = 0
  records.forEach((record) => {
    if (!hasCompleteCostData(record)) return
    const currency = record.currency!.trim().toUpperCase()
    const group = groups.get(currency) ?? {
      totalSaved: 0,
      baselineSpend: 0,
      actualSpend: 0,
      currency,
      costRecordCount: 0,
    }
    group.totalSaved += record.cost_savings!
    group.baselineSpend += record.baseline_cost!
    group.actualSpend += record.actual_cost!
    group.costRecordCount += 1
    groups.set(currency, group)
    costRecordCount += 1
  })
  const byCurrency = [...groups.values()].sort((a, b) => a.currency.localeCompare(b.currency))
  const single = byCurrency.length === 1 ? byCurrency[0] : undefined
  return {
    totalSaved: single?.totalSaved ?? 0,
    baselineSpend: single?.baselineSpend ?? 0,
    actualSpend: single?.actualSpend ?? 0,
    currency: single?.currency,
    costRecordCount,
    excludedRecordCount: records.length - costRecordCount,
    byCurrency,
  }
}

export function getInsightsRecordPath(recordId: string) {
  return `/insights/${encodeURIComponent(recordId)}`
}

export function buildInsightsRecordTitle(record: InsightsRecord | null | undefined) {
  if (!record) {
    return 'Record'
  }

  if (record.request_id) {
    return `Record: ${record.request_id.substring(0, 8)}...`
  }

  return `Record: ${record.decision ? formatInsightsDecisionName(record.decision) : record.id}`
}

function formatCompactIdentifier(value: string) {
  if (value.length <= 20) {
    return value
  }

  return `${value.slice(0, 8)}…${value.slice(-8)}`
}

function buildCompactListSummary(items: string[], maxVisibleItems: number) {
  if (items.length === 0) {
    return '-'
  }

  const visibleItems = items.slice(0, maxVisibleItems)
  const hiddenCount = items.length - visibleItems.length
  return hiddenCount > 0
    ? `${visibleItems.join(' · ')} · +${hiddenCount}`
    : visibleItems.join(' · ')
}

export function createInsightsTableColumns(): Column<InsightsRecord>[] {
  return [
    {
      key: 'request_id',
      header: 'Request ID',
      width: '190px',
      render: (row) => (
        <Link
          className={styles.requestIdLink}
          title={row.request_id || row.id}
          to={getInsightsRecordPath(row.id)}
        >
          <span className={styles.requestId} title={row.request_id || row.id}>
            {formatCompactIdentifier(row.request_id || row.id)}
          </span>
        </Link>
      ),
    },
    {
      key: 'timestamp',
      header: 'Created',
      width: '160px',
      sortable: true,
      render: (row) => <span className={styles.timestamp}>{formatDateTime(row.timestamp)}</span>,
    },
    {
      key: 'recipe',
      header: 'Recipe',
      width: '140px',
      sortable: true,
      render: (row) => <span className={styles.decision}>{row.recipe || 'default'}</span>,
    },
    {
      key: 'decision',
      header: 'Decision',
      width: '180px',
      sortable: true,
      render: (row) => (
        <span className={styles.decision}>
          {row.decision ? formatInsightsDecisionName(row.decision) : '-'}
        </span>
      ),
    },
    {
      key: 'signals',
      header: 'Signals',
      width: '180px',
      render: (row) => {
        const allSignals = collectSignals(row.signals)
        if (allSignals.length === 0) {
          return <span>-</span>
        }

        return (
          <span className={styles.tableSummaryText} title={allSignals.join(', ')}>
            {buildCompactListSummary(allSignals, 2)}
          </span>
        )
      },
    },
    {
      key: 'tools',
      header: 'Tools',
      width: '120px',
      render: (row) => renderToolNamesCell(row),
    },
    {
      key: 'reasoning_mode',
      header: 'Reasoning',
      width: '96px',
      align: 'center',
      render: (row) => (
        <span
          className={`${styles.reasoningBadge} ${
            row.reasoning_mode === 'on' ? styles.reasoningOn : styles.reasoningOff
          }`}
        >
          {row.reasoning_mode === 'on' ? 'On' : 'Off'}
        </span>
      ),
    },
    {
      key: 'selected_model',
      header: 'Model Change',
      width: '320px',
      render: (row) => (
        <div className={styles.modelChange}>
          <span className={styles.modelName}>{row.original_model || '-'}</span>
          <span className={styles.modelArrow}>→</span>
          <span className={styles.modelName}>{row.selected_model || '-'}</span>
        </div>
      ),
    },
    {
      key: 'actual_cost',
      header: 'Estimated Cost',
      width: '160px',
      sortable: true,
      render: (row) =>
        hasCompleteCostData(row) ? (
          renderCostValue(row.actual_cost, row.currency)
        ) : (
          <span className={styles.costValueMuted} title={getInsightsCostUnavailableReason(row)}>
            N/A
          </span>
        ),
    },
    {
      key: 'cost_savings',
      header: 'Estimated Savings',
      width: '180px',
      sortable: true,
      render: (row) => {
        if (!hasCompleteCostData(row)) {
          return (
            <span className={styles.costValueMuted} title={getInsightsCostUnavailableReason(row)}>
              N/A
            </span>
          )
        }

        return (
          <div className={styles.costCell}>
            <strong className={styles.costValuePositive}>
              {formatCurrency(row.cost_savings ?? 0, row.currency)}
            </strong>
            <span className={styles.costSubtle}>Baseline: {row.baseline_model}</span>
          </div>
        )
      },
    },
    {
      key: 'response_status',
      header: 'Status',
      width: '150px',
      align: 'center',
      render: (row) => {
        const lifecycle = getInsightsLifecyclePresentation(row)
        return (
          <span
            className={`${styles.statusBadge} ${
              lifecycle.successful
                ? styles.statusSuccess
                : lifecycle.errored
                  ? styles.statusError
                  : lifecycle.pending
                    ? styles.statusPending
                    : styles.statusUnknown
            }`}
          >
            {lifecycle.label}
          </span>
        )
      },
    },
    {
      key: 'flags',
      header: 'Flags',
      width: '160px',
      render: (row) => (
        <div className={styles.indicators}>
          <span className={`${styles.indicator} ${row.from_cache ? styles.indicatorActive : ''}`}>
            Cache
          </span>
          <span className={`${styles.indicator} ${row.streaming ? styles.indicatorActive : ''}`}>
            Stream
          </span>
        </div>
      ),
    },
  ]
}

export function buildInsightsRecordSections(
  record: InsightsRecord,
  options: { isReadonly: boolean },
): ViewSection[] {
  const sections: ViewSection[] = []

  sections.push({
    title: 'Lifecycle',
    fields: [
      { label: 'State', value: record.lifecycle_state || 'unknown' },
      { label: 'HTTP status', value: record.response_status || '-' },
      { label: 'Ended at', value: record.ended_at ? formatDateTime(record.ended_at) : '-' },
      {
        label: 'Duration',
        value: typeof record.duration_ms === 'number' ? `${record.duration_ms} ms` : '-',
      },
      { label: 'Terminal reason', value: record.terminal_reason || '-' },
    ],
  })

  sections.push({
    title: 'Decision Information',
    fields: [
      { label: 'Recipe', value: record.recipe || 'default' },
      {
        label: 'Decision name',
        value: record.decision ? formatInsightsDecisionName(record.decision) : '-',
      },
      { label: 'Decision tier', value: formatDecisionNumber(record.decision_tier) },
      { label: 'Decision priority', value: formatDecisionNumber(record.decision_priority) },
      {
        label: 'Confidence score',
        value:
          record.confidence_score_available === true && typeof record.confidence_score === 'number'
            ? `${(record.confidence_score * 100).toFixed(1)}%`
            : 'Score unavailable',
      },
      { label: 'Reasoning mode', value: record.reasoning_mode || '-' },
    ],
  })

  sections.push({
    title: 'Model Selection',
    fields: [
      { label: 'Original model', value: record.original_model || '-' },
      { label: 'Selected model', value: record.selected_model || '-' },
      { label: 'Selection method', value: record.selection_method || '-' },
      {
        label: 'Selection rationale',
        value: record.route_diagnostics?.selection_reasoning || 'Not recorded',
      },
    ],
  })

  sections.push(...buildRoutingExplanationSections(record))

  sections.push({
    title: 'Usage & Cost',
    fields: [
      { label: 'Context tokens', value: formatTokenValue(record.context_token_count) },
      { label: 'Prompt tokens', value: formatTokenValue(record.prompt_tokens) },
      { label: 'Completion tokens', value: formatTokenValue(record.completion_tokens) },
      { label: 'Total tokens', value: formatTokenValue(record.total_tokens) },
      { label: 'Baseline model', value: record.baseline_model || '-' },
      {
        label: 'Cost basis',
        value:
          getInsightsCostUnavailableReason(record) ||
          'Recorded tokens × configured model rates; excludes infrastructure charges and invoice adjustments.',
      },
      {
        label: 'Baseline basis',
        value:
          'New records compare the decision’s configured candidates in the same currency using the same recorded tokens. Direct requests compare against the selected model. Older records retain their captured baseline.',
      },
      {
        label: 'Estimated model cost',
        value: formatCurrencyOrNA(record.actual_cost, record.currency),
      },
      {
        label: 'Estimated baseline cost',
        value: formatCurrencyOrNA(record.baseline_cost, record.currency),
      },
      {
        label: 'Estimated savings',
        value: formatCurrencyOrNA(record.cost_savings, record.currency),
      },
    ],
  })

  const signalFields = buildSignalFields(record.signals)
  if (signalFields.length > 0) {
    sections.push({
      title: 'Signals',
      fields: signalFields,
    })
  }

  sections.push({
    title: 'Plugin Status',
    fields: [
      { label: 'Cache', value: record.from_cache ? 'Hit' : 'Miss' },
      { label: 'Cache similarity', value: formatSimilarityValue(record.cache_similarity) },
      { label: 'Streaming', value: record.streaming ? 'On' : 'Off' },
      { label: 'Guardrails', value: buildGuardrailsValue(record) },
      { label: 'RAG', value: buildRagValue(record) },
      { label: 'Hallucination Detection', value: buildHallucinationValue(record) },
    ],
  })

  const routingMetadataFields = buildRoutingMetadataFields(record)
  if (routingMetadataFields.length > 0) {
    sections.push({
      title: 'Routing Metadata',
      fields: routingMetadataFields,
    })
  }

  const projectionTraceFields = buildProjectionTraceFields(record)
  if (projectionTraceFields.length > 0) {
    sections.push({
      title: 'Projection Trace',
      fields: projectionTraceFields,
    })
  }

  const requestResponseFields = buildRequestResponseFields(record, options.isReadonly)
  if (requestResponseFields.length > 0) {
    sections.push({
      title: 'Request / Response',
      fields: requestResponseFields,
    })
  }

  return sections
}

export function collectSignals(signals: Signal): string[] {
  return ROUTER_CONFIG_EXTENSION.signals.flatMap(({ type }) =>
    (signals[type] ?? []).map((value) =>
      formatRoutingMetadataValue(`x-vsr-matched-${type.replace(/_/g, '-')}`, value),
    ),
  )
}

export function getInsightsCostUnavailableReason(record: InsightsRecord): string | undefined {
  if (record.lifecycle_state !== 'completed') return 'Request not completed'
  if (!Number.isFinite(record.total_tokens)) return 'Token usage unavailable'
  if (!Number.isFinite(record.actual_cost) || !record.currency?.trim()) {
    return 'Model pricing estimate unavailable'
  }
  if (
    !Number.isFinite(record.baseline_cost) ||
    !Number.isFinite(record.cost_savings) ||
    !record.baseline_model
  ) {
    return 'Baseline estimate unavailable'
  }
  return undefined
}

export function hasCompleteCostData(record: InsightsRecord) {
  return getInsightsCostUnavailableReason(record) === undefined
}

function buildSignalFields(signals: Signal): ViewField[] {
  return ROUTER_CONFIG_EXTENSION.signals.flatMap(({ type: key, display_name }) => {
    const values = signals[key]
    if (!values?.length) {
      return []
    }

    const label = `${display_name} signals`
    return [
      {
        label,
        value: (
          <div className={styles.modalSignalList}>
            {values.map((value) => (
              <span key={`${label}-${value}`} className={styles.modalSignalPill}>
                {formatRoutingMetadataValue(
                  `x-vsr-matched-${String(key).replace(/_/g, '-')}`,
                  value,
                )}
              </span>
            ))}
          </div>
        ),
        fullWidth: true,
      },
    ]
  })
}

function buildRoutingMetadataFields(record: InsightsRecord): ViewField[] {
  return [
    buildTagField('Projection outputs', record.projections),
    buildNumericMapField('Projection scores', record.projection_scores),
    buildNumericMapField('Signal confidences', record.signal_confidences),
    buildNumericMapField('Signal values', record.signal_values),
  ].filter((field): field is ViewField => field !== null)
}

function buildGuardrailsValue(record: InsightsRecord) {
  if (!(record.guardrails_enabled || record.jailbreak_enabled || record.pii_enabled)) {
    return 'Disabled'
  }

  if (record.jailbreak_detected || record.pii_detected) {
    return (
      <div className={styles.alertList}>
        {record.jailbreak_detected ? (
          <span className={styles.alertDanger}>
            Jailbreak: {record.jailbreak_type || 'detected'} (
            {record.jailbreak_score_available === true &&
            typeof record.jailbreak_confidence === 'number'
              ? `${(record.jailbreak_confidence * 100).toFixed(1)}%`
              : 'Score unavailable'}
            )
          </span>
        ) : null}
        {record.pii_detected ? (
          <span className={record.pii_blocked ? styles.alertDanger : styles.alertWarn}>
            {record.pii_blocked ? 'PII Blocked' : 'PII Found'}:{' '}
            {record.pii_entities?.join(', ') || 'detected'}
          </span>
        ) : null}
      </div>
    )
  }

  const enabledChecks = [
    record.jailbreak_enabled ? 'Jailbreak' : null,
    record.pii_enabled ? 'PII' : null,
  ]
    .filter(Boolean)
    .join(', ')

  return <span className={styles.alertSuccess}>Clean ({enabledChecks || 'enabled'})</span>
}

function buildRagValue(record: InsightsRecord) {
  if (!record.rag_enabled) {
    return 'Not used'
  }

  return (
    <div className={styles.pluginStack}>
      <span className={styles.alertInfo}>Context Retrieved</span>
      <span className={styles.costSubtle}>
        Backend: {record.rag_backend || 'unknown'} | Length: {record.rag_context_length || 0} chars
        | Score: {record.rag_similarity_score?.toFixed(3) || '-'}
      </span>
    </div>
  )
}

function buildHallucinationValue(record: InsightsRecord) {
  if (!record.hallucination_enabled) {
    return 'Disabled'
  }

  if (!record.hallucination_detected) {
    return <span className={styles.alertSuccess}>Not detected</span>
  }

  return (
    <div className={styles.pluginStack}>
      <span className={styles.alertDanger}>
        Detected ({((record.hallucination_confidence || 0) * 100).toFixed(1)}%)
      </span>
      {record.hallucination_spans?.length ? (
        <span className={styles.costSubtle}>
          Unsupported spans: {record.hallucination_spans.slice(0, 2).join(' | ')}
          {record.hallucination_spans.length > 2
            ? ` (+${record.hallucination_spans.length - 2})`
            : ''}
        </span>
      ) : null}
    </div>
  )
}

function buildRequestResponseFields(record: InsightsRecord, isReadonly: boolean): ViewField[] {
  if (isReadonly) {
    if (!record.request_body && !record.response_body) {
      return []
    }

    return [
      {
        label: 'Request body',
        value: renderReadonlyLock(),
      },
      {
        label: 'Response body',
        value: renderReadonlyLock(),
      },
    ]
  }

  const fields: ViewField[] = []
  if (record.request_body) {
    fields.push({
      label: 'Request body',
      value: renderBodyField(
        `request-${record.id}`,
        'request body',
        record.request_body,
        record.request_body_truncated || false,
      ),
      fullWidth: true,
    })
  }
  if (record.response_body) {
    fields.push({
      label: 'Response body',
      value: renderBodyField(
        `response-${record.id}`,
        'response body',
        record.response_body,
        record.response_body_truncated || false,
      ),
      fullWidth: true,
    })
  }

  return fields
}

function renderBodyField(id: string, title: string, body: string, truncated: boolean) {
  return (
    <CollapsibleSection
      id={id}
      title={title}
      isTruncated={truncated}
      defaultExpanded={false}
      content={<pre className={styles.bodyPreview}>{formatJson(body) || body}</pre>}
    />
  )
}

function renderReadonlyLock() {
  return (
    <div className={styles.readonlyLock}>
      <span>🔒</span>
      <span>Not available in read-only mode</span>
    </div>
  )
}

function buildTagField(label: string, values: string[] | undefined): ViewField | null {
  if (!values?.length) {
    return null
  }

  return {
    label,
    value: (
      <div className={styles.modalSignalList}>
        {values.map((value) => (
          <span key={`${label}-${value}`} className={styles.modalSignalPill}>
            {value}
          </span>
        ))}
      </div>
    ),
    fullWidth: true,
  }
}

function buildNumericMapField(
  label: string,
  values: Record<string, number> | undefined,
): ViewField | null {
  if (!values || Object.keys(values).length === 0) {
    return null
  }

  const entries = Object.entries(values).sort(([left], [right]) => left.localeCompare(right))
  return {
    label,
    value: (
      <div className={styles.pluginStack}>
        {entries.map(([key, value]) => (
          <span key={`${label}-${key}`} className={styles.costSubtle}>
            {key}: {formatNumericMetric(value)}
          </span>
        ))}
      </div>
    ),
    fullWidth: true,
  }
}

function formatDecisionNumber(value: number | undefined) {
  return typeof value === 'number' ? String(value) : '-'
}

function formatNumericMetric(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(3)
}

function renderCostValue(value?: number, currency?: string) {
  if (typeof value !== 'number' || !currency) {
    return <span className={styles.costValueMuted}>N/A</span>
  }

  return (
    <div className={styles.costCell}>
      <strong className={styles.costValue}>{formatCurrency(value, currency)}</strong>
    </div>
  )
}

function formatJson(jsonStr: string | undefined) {
  if (!jsonStr) {
    return null
  }

  try {
    return JSON.stringify(JSON.parse(jsonStr), null, 2)
  } catch {
    return jsonStr
  }
}

function formatCurrency(value: number, currency?: string) {
  if (!currency) {
    return 'N/A'
  }

  try {
    const minimumFractionDigits = Math.abs(value) >= 0.01 ? 2 : 4
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits,
      maximumFractionDigits: 4,
    }).format(value)
  } catch {
    return `${value.toFixed(4)} ${currency}`
  }
}

function formatCurrencyOrNA(value?: number, currency?: string) {
  return typeof value === 'number' && currency ? formatCurrency(value, currency) : 'N/A'
}

function formatTokenValue(value?: number) {
  return typeof value === 'number' ? value.toLocaleString('en-US') : '-'
}

function formatSimilarityValue(value?: number) {
  return typeof value === 'number' ? value.toFixed(3) : '-'
}
