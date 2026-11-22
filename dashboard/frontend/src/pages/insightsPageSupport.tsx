import { ChevronDown, Lock } from 'lucide-react'

import { formatRoutingMetadataValue } from '../components/routingMetadataDisplay'
import type { ViewField, ViewSection } from '../components/ViewPanel'
import { X_VSR_SELECTED_DECISION } from '../generated/vsrHeaders'
import type { StatusTone } from '../ui'
import { formatDateTime, formatSeconds } from '../utils/dateTime'
import { formatInsightsCost as formatCurrency } from '../utils/insightsCost'

import type {
  InsightsCostSummary,
  InsightsCurrencyCostSummary,
  InsightsRecord,
} from './insightsPageTypes'
import { CHEVRON, DETAIL, RAW } from './insightsEvidenceStyles'
import { buildProjectionTraceFields } from './insightsPageProjectionTrace'
import { buildRoutingMetadataFields } from './insightsRoutingMetadata'
import { buildRoutingExplanationSections } from './insightsPageRouting'
import { ModelName, type ModelLabels } from './insightsModelLabelSupport'
import { buildSignalFields } from './insightsRecordSignals'
import { buildInsightsPluginFields } from './insightsRecordPlugins'
import { buildProviderCacheSection } from './insightsProviderCache'
import { buildReviewSection } from './insightsReviewSupport'
import {
  CALLER_LOGIN_NOTE,
  describeBaselineBasis,
  describeMissingModelCost,
  describePricingSource,
  describeBudget,
  describeRouterCalls,
  isLooperRecord,
} from './insightsCostAccounting'

export { filterInsightsRecords } from './insightsPageFilters'
export { collectSignals } from './insightsRecordSignals'

export const formatInsightsDecisionName = (decision: string): string =>
  formatRoutingMetadataValue(X_VSR_SELECTED_DECISION, decision)

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

export function getInsightsLifecycleTone(
  lifecycle: ReturnType<typeof getInsightsLifecyclePresentation>,
): StatusTone {
  if (lifecycle.successful) {
    return 'success'
  }
  if (lifecycle.errored) {
    return 'danger'
  }
  if (lifecycle.pending) {
    return 'info'
  }
  return 'neutral'
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
  return `/observe/requests/${encodeURIComponent(recordId)}`
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

/** A record's estimated cost for a table cell, or why it has none. */
export function renderInsightsCostCell(record: InsightsRecord, labels: ModelLabels = {}) {
  return hasCompleteCostData(record)
    ? renderCostValue(record.actual_cost, record.currency)
    : renderUnavailableCost(record, labels)
}

/** A record's estimated savings for a table cell: the amount, why it is zero, or why it is unknown. */
export function renderInsightsSavingsCell(record: InsightsRecord, labels: ModelLabels = {}) {
  if (!hasCompleteCostData(record)) {
    return renderUnavailableCost(record, labels)
  }
  const zeroSavingsReason = getZeroSavingsReason(record)
  return (
    <span className="flex flex-col items-end gap-0.5">
      <strong
        className={
          zeroSavingsReason || (record.cost_savings ?? 0) < 0
            ? 'font-mono text-xs font-medium tabular'
            : 'font-mono text-xs font-medium text-success-foreground tabular'
        }
      >
        {zeroSavingsReason ? 'No savings' : formatCurrency(record.cost_savings, record.currency)}
      </strong>
      {zeroSavingsReason ? (
        <span className="text-[11px] text-subtle-foreground">{zeroSavingsReason}</span>
      ) : null}
      <span className="text-[11px] text-subtle-foreground">
        Baseline (configured rates): <ModelName name={record.baseline_model} compact />
      </span>
      {record.cost_accounting?.router_calls?.length ? (
        <span className="text-[11px] text-subtle-foreground">Before its Router calls</span>
      ) : null}
    </span>
  )
}

export function buildInsightsRecordSections(
  record: InsightsRecord,
  options: { isReadonly: boolean; modelLabels?: ModelLabels; canViewBodies?: boolean },
): ViewSection[] {
  const labels = options.modelLabels ?? {}
  const sections: ViewSection[] = []
  const reviewSection = buildReviewSection(record)
  if (reviewSection) {
    sections.push(reviewSection)
  }

  sections.push({
    title: 'Lifecycle',
    fields: [
      { label: 'State', value: record.lifecycle_state || 'unknown' },
      { label: 'HTTP status', value: record.response_status || '-' },
      { label: 'Ended at', value: record.ended_at ? formatDateTime(record.ended_at) : '-' },
      {
        label: 'Duration',
        value: typeof record.duration_ms === 'number' ? formatSeconds(record.duration_ms) : '-',
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
      { label: 'Original model', value: <ModelName name={record.original_model} /> },
      {
        label: 'Selected model',
        value: record.selected_model ? <ModelName name={record.selected_model} /> : '-',
      },
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
      {
        label: 'Baseline model',
        value: record.baseline_model ? (
          <ModelName name={record.baseline_model} />
        ) : (
          'Baseline not recorded'
        ),
      },
      ...buildRouterCallFields(record, labels),
      ...buildBudgetFields(record),
      { label: 'Cost basis', value: describeCostBasis(record, labels) },
      {
        label: 'Baseline basis',
        value: record.cost_accounting?.baseline_source
          ? `${describeBaselineBasis(record, labels)} Same currency and recorded tokens; a negative saving means the baseline was cheaper.`
          : 'New records use the highest estimate in the recipe’s complete model pool across all decisions at configured rates, in the same currency using the same recorded tokens, unless the route sets a cost_baseline_model. Direct requests compare against the selected model. Older records retain their captured baseline.',
      },
      {
        label: 'Current pricing',
        value: (
          <>
            <a className="text-primary hover:underline" href="/models/backends">
              View current configured model rates
            </a>
            . Current rates may differ from this record; historical records are not repriced.
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
        // The record's own savings are gross of its Router calls, as the
        // table's column says.
        label: record.cost_accounting?.router_calls?.length
          ? 'Estimated savings before its Router calls'
          : 'Estimated savings',
        value: getZeroSavingsReason(record)
          ? `No savings — ${getZeroSavingsReason(record)}`
          : formatRecordedCost(record, record.cost_savings),
      },
    ],
  })

  sections.push(buildProviderCacheSection(record))

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

  const requestResponseFields = buildRequestResponseFields(
    record,
    options.isReadonly,
    options.canViewBodies ?? true,
  )
  if (requestResponseFields.length > 0) {
    sections.push({
      title: 'Request / Response',
      fields: requestResponseFields,
    })
  }

  return sections
}

export function getInsightsCostUnavailableReason(
  record: InsightsRecord,
  labels: ModelLabels = {},
): string | undefined {
  return getUnavailableCost(record, labels)?.reason
}

function describeCostBasis(record: InsightsRecord, labels: ModelLabels): string {
  const rates = describePricingSource(record.cost_accounting)
  if (rates && Number.isFinite(record.actual_cost)) {
    const login = record.cost_accounting?.caller_login
      ? ` It ran on the client’s own login: this is the ${CALLER_LOGIN_NOTE}.`
      : ''
    return `Recorded tokens × ${rates} at capture time; not a GPU bill or provider invoice.${login}`
  }
  return (
    getInsightsCostUnavailableReason(record, labels) ||
    'Recorded tokens × configured model rates at capture time; not a GPU bill or provider invoice.'
  )
}

function buildRouterCallFields(record: InsightsRecord, labels: ModelLabels): ViewField[] {
  const value = describeRouterCalls(record.cost_accounting, labels)
  return value ? [{ label: 'Router calls', value }] : []
}

function buildBudgetFields(record: InsightsRecord): ViewField[] {
  const value = describeBudget(record.cost_accounting)
  return value ? [{ label: 'Spend budget', value }] : []
}

function getUnavailableCost(record: InsightsRecord, labels: ModelLabels = {}) {
  if (record.lifecycle_state !== 'completed') {
    return { label: 'Not completed', reason: 'Request not completed' }
  }
  if (!Number.isFinite(record.total_tokens)) {
    return { label: 'Usage not recorded', reason: 'Token usage was not recorded for this request' }
  }
  if (!Number.isFinite(record.actual_cost) || !record.currency?.trim()) {
    return getMissingModelCost(record, labels)
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

function getMissingModelCost(record: InsightsRecord, labels: ModelLabels) {
  if (isLooperRecord(record)) {
    return {
      label: 'In router calls',
      reason: `A Looper request: its cost is its calls, ${describeRouterCalls(record.cost_accounting, labels)}`,
    }
  }
  const missing = describeMissingModelCost(record, labels)
  if (missing) {
    return { label: 'No price', reason: missing }
  }
  return {
    label: 'Price not recorded',
    reason:
      'No pricing estimate was recorded with this request. Historical records are not repriced using current model rates.',
  }
}

function renderUnavailableCost(record: InsightsRecord, labels: ModelLabels) {
  const unavailable = getUnavailableCost(record, labels)
  return (
    <span className="text-xs text-subtle-foreground" title={unavailable?.reason}>
      {unavailable?.label || '—'}
    </span>
  )
}

export function hasCompleteCostData(record: InsightsRecord) {
  return getInsightsCostUnavailableReason(record) === undefined
}

function buildRequestResponseFields(
  record: InsightsRecord,
  isReadonly: boolean,
  canViewBodies: boolean,
): ViewField[] {
  if (isReadonly) {
    if (!record.request_body && !record.response_body) {
      return []
    }

    return [
      {
        label: 'Request body',
        value: renderLockedBody('Not available in read-only mode'),
      },
      {
        label: 'Response body',
        value: renderLockedBody('Not available in read-only mode'),
      },
    ]
  }

  const fields: ViewField[] = []
  if (record.request_body) {
    fields.push({
      label: 'Request body',
      value: renderBodyField(
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
        'response body',
        record.response_body,
        record.response_body_truncated || false,
      ),
      fullWidth: true,
    })
  }
  if (fields.length === 0 && !canViewBodies) {
    fields.push({
      label: 'Request and response bodies',
      value: renderLockedBody(
        'Hidden for your role: captured bodies and tool payloads need the config.write permission.',
      ),
      fullWidth: true,
    })
  }

  return fields
}

function renderBodyField(title: string, body: string, truncated: boolean) {
  return (
    <details className={DETAIL}>
      <summary>
        <span>
          Show the {title}
          {truncated ? ' · cut at the capture limit' : ''}
        </span>
        <ChevronDown className={CHEVRON} />
      </summary>
      <pre className={`${RAW} m-2 whitespace-pre-wrap break-all`}>{formatJson(body) || body}</pre>
    </details>
  )
}

function renderLockedBody(message: string) {
  return (
    <div className="inline-flex items-center gap-2 text-[13px] text-muted-foreground">
      <Lock className="size-3.5 shrink-0" />
      <span>{message}</span>
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
    return <span className="text-xs text-subtle-foreground">Price not recorded</span>
  }

  return <span className="font-mono text-xs tabular">{formatCurrency(value, currency)}</span>
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
