import type { Column } from '../components/DataTable'
import CollapsibleSection from '../components/CollapsibleSection'
import { formatRoutingMetadataValue } from '../components/routingMetadataDisplay'
import type { ViewField, ViewSection } from '../components/ViewPanel'
import { formatDateTime } from '../utils/dateTime'
import { formatInsightsCost as formatCurrency } from '../utils/insightsCost'
import { Link } from 'react-router-dom'

import type {
  InsightsCostSummary,
  InsightsCurrencyCostSummary,
  InsightsRecord,
} from './insightsPageTypes'
import { buildProjectionTraceFields } from './insightsPageProjectionTrace'
import { buildRoutingMetadataFields } from './insightsRoutingMetadata'
import { buildRoutingExplanationSections } from './insightsPageRouting'
import { renderToolNamesCell } from './insightsPageToolTrace'
import { buildSignalFields, collectSignals } from './insightsRecordSignals'
import { buildInsightsPluginFields } from './insightsRecordPlugins'
import styles from './InsightsPage.module.css'

export { filterInsightsRecords } from './insightsPageFilters'
export { collectSignals } from './insightsRecordSignals'

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

export function getInsightsLifecycleStatusClass(
  lifecycle: ReturnType<typeof getInsightsLifecyclePresentation>,
) {
  if (lifecycle.successful) {
    return styles.statusSuccess
  }
  if (lifecycle.errored) {
    return styles.statusError
  }
  if (lifecycle.pending) {
    return styles.statusPending
  }
  return styles.statusUnknown
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
        hasCompleteCostData(row)
          ? renderCostValue(row.actual_cost, row.currency)
          : renderUnavailableCost(row),
    },
    {
      key: 'cost_savings',
      header: 'Estimated Savings',
      width: '180px',
      sortable: true,
      render: (row) => {
        if (!hasCompleteCostData(row)) {
          return renderUnavailableCost(row)
        }
        const zeroSavingsReason = getZeroSavingsReason(row)

        return (
          <div className={styles.costCell}>
            <strong className={zeroSavingsReason ? styles.costValue : styles.costValuePositive}>
              {zeroSavingsReason ? 'No savings' : formatCurrency(row.cost_savings, row.currency)}
            </strong>
            {zeroSavingsReason && <span className={styles.costSubtle}>{zeroSavingsReason}</span>}
            <span className={styles.costSubtle}>
              Baseline (configured rates): {row.baseline_model}
            </span>
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
          <span className={`${styles.statusBadge} ${getInsightsLifecycleStatusClass(lifecycle)}`}>
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
          {row.from_cache && (
            <span className={`${styles.indicator} ${styles.indicatorActive}`}>Cache hit</span>
          )}
          {row.streaming && (
            <span className={`${styles.indicator} ${styles.indicatorActive}`}>Streaming</span>
          )}
          {!row.from_cache && !row.streaming && <span>No recorded flags</span>}
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

  if (record.outcomes?.length) {
    sections.push({
      title: 'Outcomes',
      fields: record.outcomes.map((outcome, index) => ({
        label: `Outcome ${index + 1}`,
        value: [
          outcome.timestamp ? formatDateTime(outcome.timestamp) : 'Unknown time',
          `${outcome.source} → ${outcome.target}`,
          outcome.verdict,
        ].join(' · '),
      })),
    })
  }

  const projectionTraceFields = buildProjectionTraceFields(record)
  if (projectionTraceFields.length > 0) {
    sections.push({
      title: 'Projection Trace',
      fields: projectionTraceFields,
    })
  }

  const routingMetadataFields = buildRoutingMetadataFields(record)
  if (routingMetadataFields.length > 0) {
    sections.push({
      title: 'Routing Metadata',
      fields: routingMetadataFields,
    })
  }

  sections.push(...buildRoutingExplanationSections(record))

  sections.push({
    title: 'Usage & Cost',
    fields: [
      { label: 'Context tokens', value: formatTokenValue(record.context_token_count) },
      { label: 'Prompt tokens', value: formatTokenValue(record.prompt_tokens) },
      { label: 'Completion tokens', value: formatTokenValue(record.completion_tokens) },
      { label: 'Total tokens', value: formatTokenValue(record.total_tokens) },
      { label: 'Baseline model', value: record.baseline_model || 'Baseline not recorded' },
      {
        label: 'Cost basis',
        value:
          getInsightsCostUnavailableReason(record) ||
          'Recorded tokens × configured model rates at capture time; not a GPU bill or provider invoice.',
      },
      {
        label: 'Baseline basis',
        value:
          'New records use the highest estimate in the recipe’s complete model pool across all decisions at configured rates, in the same currency using the same recorded tokens. Direct requests compare against the selected model. Older records retain their captured baseline.',
      },
      {
        label: 'Current pricing',
        value: (
          <>
            <a href="/config/models">View current configured model rates</a>. Current rates may
            differ from this record; historical records are not repriced.
          </>
        ),
      },
      {
        label: 'Estimated model cost',
        value: formatRecordedCost(record, record.actual_cost),
      },
      {
        label: 'Estimated baseline cost',
        value: formatRecordedCost(record, record.baseline_cost),
      },
      {
        label: 'Estimated savings',
        value: getZeroSavingsReason(record)
          ? `No savings — ${getZeroSavingsReason(record)}`
          : formatRecordedCost(record, record.cost_savings),
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
    fields: buildInsightsPluginFields(record),
  })

  const requestResponseFields = buildRequestResponseFields(record, options.isReadonly)
  if (requestResponseFields.length > 0) {
    sections.push({
      title: 'Request / Response',
      fields: requestResponseFields,
    })
  }

  return sections
}

export function getInsightsCostUnavailableReason(record: InsightsRecord): string | undefined {
  return getUnavailableCost(record)?.reason
}

function getUnavailableCost(record: InsightsRecord) {
  if (record.lifecycle_state !== 'completed') {
    return { label: 'Not completed', reason: 'Request not completed' }
  }
  if (!Number.isFinite(record.total_tokens)) {
    return { label: 'Usage not recorded', reason: 'Token usage was not recorded for this request' }
  }
  if (!Number.isFinite(record.actual_cost) || !record.currency?.trim()) {
    return {
      label: 'Price not recorded',
      reason:
        'No pricing estimate was recorded with this request. Historical records are not repriced using current model rates.',
    }
  }
  if (
    !Number.isFinite(record.baseline_cost) ||
    !Number.isFinite(record.cost_savings) ||
    !record.baseline_model
  ) {
    return {
      label: 'Baseline not recorded',
      reason: 'Baseline estimate was not recorded for this request',
    }
  }
  return undefined
}

function renderUnavailableCost(record: InsightsRecord) {
  const unavailable = getUnavailableCost(record)
  return (
    <span className={styles.costValueMuted} title={unavailable?.reason}>
      {unavailable?.label || 'N/A'}
    </span>
  )
}

export function hasCompleteCostData(record: InsightsRecord) {
  return getInsightsCostUnavailableReason(record) === undefined
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

function formatDecisionNumber(value: number | undefined) {
  return typeof value === 'number' ? String(value) : '-'
}

function getZeroSavingsReason(record: InsightsRecord): string | null {
  if (
    !hasCompleteCostData(record) ||
    record.cost_savings !== 0 ||
    record.actual_cost !== record.baseline_cost
  ) {
    return null
  }
  return record.selected_model && record.selected_model === record.baseline_model
    ? 'Baseline model selected'
    : 'Equal estimated cost'
}

function renderCostValue(value?: number, currency?: string) {
  if (!Number.isFinite(value) || !currency?.trim()) {
    return <span className={styles.costValueMuted}>Price not recorded</span>
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

function formatRecordedCost(record: InsightsRecord, value?: number) {
  if (!Number.isFinite(value) || !record.currency?.trim()) {
    return getUnavailableCost(record)?.label || 'Price not recorded'
  }
  return formatCurrency(value, record.currency)
}

function formatTokenValue(value?: number) {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString('en-US')
    : 'Not recorded'
}
