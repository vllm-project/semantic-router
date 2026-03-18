import type { Column } from '../components/DataTable'
import CollapsibleSection from '../components/CollapsibleSection'
import type { ViewField, ViewSection } from '../components/ViewModal'
import { formatDate } from '../types/evaluation'

import type {
  InsightsCostSummary,
  InsightsFilterType,
  InsightsRecord,
  Signal,
} from './insightsPageTypes'
import styles from './InsightsPage.module.css'

interface InsightsFilterState {
  searchTerm: string
  filter: InsightsFilterType
  decisionFilter: string
  modelFilter: string
}

const SIGNAL_TINTS: Record<string, string> = {
  keyword: 'rgba(118, 185, 0, 0.15)',
  embedding: 'rgba(0, 212, 255, 0.12)',
  domain: 'rgba(147, 51, 234, 0.14)',
  fact_check: 'rgba(34, 197, 94, 0.15)',
  user_feedback: 'rgba(236, 72, 153, 0.12)',
  preference: 'rgba(234, 179, 8, 0.15)',
  language: 'rgba(59, 130, 246, 0.15)',
  context: 'rgba(168, 85, 247, 0.14)',
  complexity: 'rgba(245, 158, 11, 0.15)',
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

export function filterInsightsRecords(records: InsightsRecord[], filters: InsightsFilterState) {
  const searchTerm = filters.searchTerm.trim().toLowerCase()

  return records.filter((record) => {
    if (filters.filter === 'cached' && !record.from_cache) {
      return false
    }
    if (filters.filter === 'streamed' && !record.streaming) {
      return false
    }
    if (filters.decisionFilter !== 'all' && record.decision !== filters.decisionFilter) {
      return false
    }
    if (
      filters.modelFilter !== 'all' &&
      record.selected_model !== filters.modelFilter &&
      record.original_model !== filters.modelFilter
    ) {
      return false
    }
    if (searchTerm && !record.request_id?.toLowerCase().includes(searchTerm)) {
      return false
    }

    return true
  })
}

export function buildInsightsSummary(records: InsightsRecord[]): InsightsCostSummary {
  let totalSaved = 0
  let baselineSpend = 0
  let actualSpend = 0
  let currency: string | undefined
  let costRecordCount = 0

  records.forEach((record) => {
    if (!hasCompleteCostData(record)) {
      return
    }

    totalSaved += record.cost_savings ?? 0
    baselineSpend += record.baseline_cost ?? 0
    actualSpend += record.actual_cost ?? 0
    currency = currency || record.currency
    costRecordCount += 1
  })

  return {
    totalSaved,
    baselineSpend,
    actualSpend,
    currency,
    costRecordCount,
    excludedRecordCount: records.length - costRecordCount,
  }
}

export function createInsightsTableColumns(): Column<InsightsRecord>[] {
  return [
    {
      key: 'request_id',
      header: 'Request ID',
      width: '260px',
      render: (row) => (
        <span className={styles.requestId} title={row.request_id || ''}>
          {row.request_id || '-'}
        </span>
      ),
    },
    {
      key: 'timestamp',
      header: 'Created',
      width: '130px',
      sortable: true,
      render: (row) => <span className={styles.timestamp}>{formatDate(row.timestamp)}</span>,
    },
    {
      key: 'decision',
      header: 'Decision',
      width: '180px',
      sortable: true,
      render: (row) => <span className={styles.decision}>{row.decision || '-'}</span>,
    },
    {
      key: 'signals',
      header: 'Signals',
      width: '220px',
      render: (row) => {
        const allSignals = collectSignals(row.signals)
        if (allSignals.length === 0) {
          return <span>-</span>
        }

        return (
          <div className={styles.signalList}>
            {allSignals.slice(0, 3).map((signal) => (
              <span key={`${row.id}-${signal}`} className={styles.signalPillCompact}>
                {signal}
              </span>
            ))}
            {allSignals.length > 3 ? (
              <span className={styles.signalOverflow}>+{allSignals.length - 3}</span>
            ) : null}
          </div>
        )
      },
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
      header: 'Actual Cost',
      width: '160px',
      sortable: true,
      render: (row) => renderCostValue(row.actual_cost, row.currency),
    },
    {
      key: 'cost_savings',
      header: 'Saved vs Baseline',
      width: '180px',
      sortable: true,
      render: (row) => {
        if (!hasCompleteCostData(row)) {
          return <span className={styles.costValueMuted}>N/A</span>
        }

        return (
          <div className={styles.costCell}>
            <strong className={styles.costValuePositive}>
              {formatCurrency(row.cost_savings ?? 0, row.currency)}
            </strong>
            <span className={styles.costSubtle}>
              Baseline: {row.baseline_model}
            </span>
          </div>
        )
      },
    },
    {
      key: 'response_status',
      header: 'Status',
      width: '80px',
      align: 'center',
      render: (row) => (
        <span
          className={`${styles.statusBadge} ${
            row.response_status && row.response_status < 400
              ? styles.statusSuccess
              : styles.statusError
          }`}
        >
          {row.response_status || '-'}
        </span>
      ),
    },
    {
      key: 'flags',
      header: 'Flags',
      width: '160px',
      render: (row) => (
        <div className={styles.indicators}>
          <span className={`${styles.indicator} ${row.from_cache ? styles.indicatorActive : ''}`}>Cache</span>
          <span className={`${styles.indicator} ${row.streaming ? styles.indicatorActive : ''}`}>Stream</span>
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
    title: 'Decision Information',
    fields: [
      { label: 'Decision name', value: record.decision || '-' },
      {
        label: 'Category',
        value: record.signals?.domain?.length ? record.signals.domain.join(', ') : record.category || '-',
      },
      {
        label: 'Confidence score',
        value:
          record.confidence_score !== undefined
            ? `${(record.confidence_score * 100).toFixed(1)}%`
            : '-',
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
    ],
  })

  sections.push({
    title: 'Usage & Cost',
    fields: [
      { label: 'Prompt tokens', value: formatTokenValue(record.prompt_tokens) },
      { label: 'Completion tokens', value: formatTokenValue(record.completion_tokens) },
      { label: 'Total tokens', value: formatTokenValue(record.total_tokens) },
      { label: 'Baseline model', value: record.baseline_model || '-' },
      { label: 'Actual cost', value: formatCurrencyOrNA(record.actual_cost, record.currency) },
      { label: 'Baseline cost', value: formatCurrencyOrNA(record.baseline_cost, record.currency) },
      { label: 'Saved vs baseline', value: formatCurrencyOrNA(record.cost_savings, record.currency) },
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
      { label: 'Streaming', value: record.streaming ? 'On' : 'Off' },
      { label: 'Guardrails', value: buildGuardrailsValue(record) },
      { label: 'RAG', value: buildRagValue(record) },
      { label: 'Hallucination Detection', value: buildHallucinationValue(record) },
    ],
  })

  const requestResponseFields = buildRequestResponseFields(record, options.isReadonly)
  if (requestResponseFields.length > 0) {
    sections.push({
      title: 'Request/Response',
      fields: requestResponseFields,
    })
  }

  return sections
}

export function collectSignals(signals: Signal): string[] {
  const allSignals: string[] = []
  if (signals.keyword?.length) allSignals.push(...signals.keyword)
  if (signals.embedding?.length) allSignals.push(...signals.embedding)
  if (signals.domain?.length) allSignals.push(...signals.domain)
  if (signals.fact_check?.length) allSignals.push(...signals.fact_check)
  if (signals.user_feedback?.length) allSignals.push(...signals.user_feedback)
  if (signals.preference?.length) allSignals.push(...signals.preference)
  if (signals.language?.length) allSignals.push(...signals.language)
  if (signals.context?.length) allSignals.push(...signals.context)
  if (signals.complexity?.length) allSignals.push(...signals.complexity)
  return allSignals
}

export function hasCompleteCostData(record: InsightsRecord) {
  return (
    typeof record.actual_cost === 'number' &&
    typeof record.baseline_cost === 'number' &&
    typeof record.cost_savings === 'number' &&
    typeof record.total_tokens === 'number' &&
    Boolean(record.currency) &&
    Boolean(record.baseline_model)
  )
}

function buildSignalFields(signals: Signal): ViewField[] {
  const signalEntries: Array<[keyof Signal, string]> = [
    ['keyword', 'Keyword matches'],
    ['embedding', 'Embedding matches'],
    ['domain', 'Domain matches'],
    ['fact_check', 'Fact check results'],
    ['user_feedback', 'User feedback'],
    ['preference', 'Preference signals'],
    ['language', 'Language signals'],
    ['context', 'Context signals'],
    ['complexity', 'Complexity signals'],
  ]

  return signalEntries.flatMap(([key, label]) => {
    const values = signals[key]
    if (!values?.length) {
      return []
    }

    return [
      {
        label,
        value: (
          <div className={styles.modalSignalList}>
            {values.map((value) => (
              <span
                key={`${label}-${value}`}
                className={styles.modalSignalPill}
                style={{ background: SIGNAL_TINTS[key] || 'rgba(255, 255, 255, 0.08)' }}
              >
                {value}
              </span>
            ))}
          </div>
        ),
        fullWidth: true,
      },
    ]
  })
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
            Jailbreak: {record.jailbreak_type || 'detected'} ({((record.jailbreak_confidence || 0) * 100).toFixed(1)}%)
          </span>
        ) : null}
        {record.pii_detected ? (
          <span className={record.pii_blocked ? styles.alertDanger : styles.alertWarn}>
            {record.pii_blocked ? 'PII Blocked' : 'PII Found'}: {record.pii_entities?.join(', ') || 'detected'}
          </span>
        ) : null}
      </div>
    )
  }

  const enabledChecks = [record.jailbreak_enabled ? 'Jailbreak' : null, record.pii_enabled ? 'PII' : null]
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
        Backend: {record.rag_backend || 'unknown'} | Length: {record.rag_context_length || 0} chars | Score:{' '}
        {record.rag_similarity_score?.toFixed(3) || '-'}
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
          {record.hallucination_spans.length > 2 ? ` (+${record.hallucination_spans.length - 2})` : ''}
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
      value: renderBodyField(`request-${record.id}`, 'request body', record.request_body, record.request_body_truncated || false),
      fullWidth: true,
    })
  }
  if (record.response_body) {
    fields.push({
      label: 'Response body',
      value: renderBodyField(`response-${record.id}`, 'response body', record.response_body, record.response_body_truncated || false),
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
