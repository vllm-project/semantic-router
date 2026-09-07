import type {
  EvidenceLevel,
  EvaluationRunStatus,
  EvaluationTrackStatus,
} from '../../types/evaluationPlane'
import type {
  EvaluationGate,
  EvaluationMetric,
  EvaluationReport,
} from '../../types/evaluationReport'

export const RUN_STATUS_LABELS: Record<EvaluationRunStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  sealing: 'Finalizing report',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export const TRACK_STATUS_LABELS: Record<EvaluationTrackStatus, string> = {
  ...RUN_STATUS_LABELS,
  unavailable: 'Not measured',
  skipped: 'Not selected',
}

interface EvaluationResultScopePresentation {
  label: string
  description: string
}

const EVALUATION_RESULT_SCOPE_PRESENTATION: Record<
  EvidenceLevel,
  EvaluationResultScopePresentation
> = {
  E0: {
    label: 'Diagnostic',
    description:
      'Checks the evaluation setup, data identity, and execution path without making a release recommendation.',
  },
  E1: {
    label: 'Signal validation',
    description: 'Measures signal availability, discrimination, latency, and failure behavior.',
  },
  E2: {
    label: 'Prediction validation',
    description: 'Measures prediction coverage, calibration, stability, and downstream usefulness.',
  },
  E3: {
    label: 'Routing validation',
    description:
      'Verifies routing decisions, fallback behavior, policy handling, and router latency.',
  },
  E4: {
    label: 'Model-pool validation',
    description:
      'Measures model alternatives, realized utility, regret, robustness, and pool usage.',
  },
  E5: {
    label: 'End-to-end validation',
    description:
      'Measures final task outcomes, live reliability, safety, complete cost, and capacity.',
  },
}

export function evaluationResultScopeLabel(level: EvidenceLevel): string {
  return EVALUATION_RESULT_SCOPE_PRESENTATION[level].label
}

export function evaluationResultScopeDescription(level: EvidenceLevel): string {
  return EVALUATION_RESULT_SCOPE_PRESENTATION[level].description
}

// Product guidance only. Gate identity, names, descriptions, dispositions, and
// thresholds are server-owned contract data and must never be inferred here.
const GATE_FOLLOW_UP_GUIDANCE: Record<string, string> = {
  G0: 'Repeat the run with the same pinned workload and configuration to confirm reproducibility.',
  G1: 'Validate the routing rules and signals on a labeled offline workload.',
  G2: 'Run live policy-enforcement cases and record whether every required policy was applied.',
  G3: 'Compare a baseline and candidate on the same assigned cohort.',
  G4: 'Run the declared workload shifts and measure how much quality and reliability change.',
  G5: 'Compare the saved candidate with a fresh live run of the unchanged system.',
  G6: 'Inject the expected failures and verify fallback, retry, and recovery behavior.',
  G7: 'Run repeated live load at the required service objective and measure available capacity.',
  G8: 'Run a guarded shadow or canary with exposure, stop, and rollback monitoring.',
  G9: 'Collect assigned online preference outcomes for the baseline and candidate.',
}

export function evaluationGateFollowUpGuidance(gateID: string): string {
  return (
    GATE_FOLLOW_UP_GUIDANCE[gateID] ||
    'Collect the missing results for this release check and repeat the evaluation.'
  )
}

export type EvaluationTone = 'neutral' | 'positive' | 'warning' | 'negative'

export function gateVerdictPresentation(gate: Pick<EvaluationGate, 'disposition' | 'verdict'>): {
  label: string
  tone: EvaluationTone
  explanation: string
} {
  switch (gate.verdict) {
    case 'pass':
      return {
        label: 'Passed',
        tone: 'positive',
        explanation: 'The measured result satisfied this check.',
      }
    case 'fail':
      return {
        label: 'Blocked',
        tone: 'negative',
        explanation: 'The measured result did not satisfy this check.',
      }
    case 'not_applicable':
      return {
        label: 'Not required',
        tone: 'neutral',
        explanation: 'This check does not apply to the selected change type.',
      }
    case 'unavailable':
      return gate.disposition === 'required'
        ? {
            label: 'Incomplete',
            tone: 'warning',
            explanation: 'This required check does not yet have enough results to complete.',
          }
        : {
            label: 'Not measured',
            tone: 'neutral',
            explanation: 'This recommended measurement was not produced by the run.',
          }
  }
}

export function clampFraction(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || typeof value === 'undefined' || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

function formatMetricNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(value)
}

function formatMetricCount(value: number, singular: string, plural = `${singular}s`): string {
  return `${new Intl.NumberFormat().format(value)} ${value === 1 ? singular : plural}`
}

const METRIC_THRESHOLD_OPERATORS: Readonly<Record<string, string>> = {
  '<': '<',
  '<=': '≤',
  lt: '<',
  lte: '≤',
  '=': '=',
  '==': '=',
  eq: '=',
  '>=': '≥',
  gte: '≥',
  '>': '>',
  gt: '>',
}

export function formatMetric(metric: Pick<EvaluationMetric, 'value' | 'unit'>): string {
  if (metric.value === null || !Number.isFinite(metric.value)) return '\u2014'
  const unit = metric.unit.trim().toLowerCase()
  switch (unit) {
    case 'ratio':
    case 'fraction':
      return formatPercent(metric.value)
    case 'percent':
    case '%':
      return `${metric.value.toFixed(1)}%`
    case 'ms':
      return `${metric.value.toFixed(Number.isInteger(metric.value) || metric.value >= 100 ? 0 : 1)} ms`
    case 's':
    case 'seconds':
      return `${metric.value.toFixed(2)} s`
    case 'usd':
      return new Intl.NumberFormat(undefined, {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: metric.value > 0 && metric.value < 0.01 ? 4 : 2,
        maximumFractionDigits: metric.value > 0 && metric.value < 0.01 ? 8 : 2,
      }).format(metric.value)
    case 'usd/request':
      return `${new Intl.NumberFormat(undefined, {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: metric.value > 0 && metric.value < 0.01 ? 4 : 2,
        maximumFractionDigits: metric.value > 0 && metric.value < 0.01 ? 8 : 2,
      }).format(metric.value)} / req`
    case 'count':
    case 'concurrency':
      return new Intl.NumberFormat().format(metric.value)
    case 'arms':
      return formatMetricCount(metric.value, 'model')
    case 'assignments':
      return formatMetricCount(metric.value, 'assignment')
    case 'attempts':
      return formatMetricCount(metric.value, 'attempt')
    case 'cases':
      return formatMetricCount(metric.value, 'case')
    case 'errors':
      return formatMetricCount(metric.value, 'error')
    case 'observations':
      return formatMetricCount(metric.value, 'observation')
    case 'pairs':
      return formatMetricCount(metric.value, 'pair')
    case 'requests':
      return formatMetricCount(metric.value, 'request')
    case 'seeds':
      return formatMetricCount(metric.value, 'trial')
    case 'segments':
      return formatMetricCount(metric.value, 'segment')
    case 'steps':
      return formatMetricCount(metric.value, 'step')
    case 'tasks':
      return formatMetricCount(metric.value, 'task')
    case 'effective samples':
      return formatMetricCount(metric.value, 'usable sample')
    case 'requests/s':
      return `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(metric.value)} req/s`
    case 'bits':
      return `${metric.value.toFixed(2)} bits`
    case 'boolean':
      return metric.value > 0 ? 'Yes' : 'No'
    case 'violations/case':
      return `${metric.value.toFixed(4)} / case`
    case 'usd/success':
      return `${new Intl.NumberFormat(undefined, {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 4,
      }).format(metric.value)} / success`
    case 'exposures/attempt':
      return `${formatMetricNumber(metric.value)} events / attempt`
    case 'exposures/trajectory':
      return `${formatMetricNumber(metric.value)} events / task`
    case 'non-inferiority-headroom':
    case 'p-value':
    case 'quality':
    case 'reward lift':
    case 'score':
      return formatMetricNumber(metric.value)
    default:
      return formatMetricNumber(metric.value)
  }
}

export function formatMetricThreshold(threshold: {
  operator: string
  value: number
  unit?: string
}): string {
  const operator = METRIC_THRESHOLD_OPERATORS[threshold.operator]
  const value = formatMetric({ value: threshold.value, unit: threshold.unit || '' })
  return operator ? `${operator} ${value}` : `Target ${value}`
}

export function formatDelta(metric: Pick<EvaluationMetric, 'delta' | 'unit'>): string | null {
  if (metric.delta === null || typeof metric.delta === 'undefined') return null
  const formatted = formatMetric({ value: Math.abs(metric.delta), unit: metric.unit })
  return `${metric.delta > 0 ? '+' : metric.delta < 0 ? '−' : ''}${formatted}`
}

export function formatConfidenceInterval(
  metric: Pick<EvaluationMetric, 'confidence_interval' | 'unit'>,
): string | null {
  if (!metric.confidence_interval || metric.confidence_interval.length !== 2) return null
  const [lower, upper] = metric.confidence_interval
  return `${formatMetric({ value: lower, unit: metric.unit })} \u2013 ${formatMetric({ value: upper, unit: metric.unit })}`
}

export function metricDeltaTone(
  metric: Pick<EvaluationMetric, 'delta' | 'direction'>,
): EvaluationTone {
  if (!metric.delta || !metric.direction || metric.direction === 'target') return 'neutral'
  const improved = metric.direction === 'higher_is_better' ? metric.delta > 0 : metric.delta < 0
  return improved ? 'positive' : 'negative'
}

export function evidenceRank(level: string): number {
  const parsed = Number(level.replace(/^E/, ''))
  return Number.isFinite(parsed) ? parsed : 0
}

export function effectiveGateVerdict(
  reported: EvaluationReport['summary']['verdict'],
  gates: EvaluationGate[],
): EvaluationReport['summary']['verdict'] {
  const requiredGates = gates.filter((gate) => gate.disposition === 'required')
  if (requiredGates.some((gate) => gate.verdict === 'fail')) return 'fail'
  if (requiredGates.some((gate) => gate.verdict === 'unavailable')) return 'unavailable'
  return reported
}

export function evaluationPromotionVerdict(
  report: EvaluationReport,
): EvaluationReport['summary']['verdict'] {
  return effectiveGateVerdict(report.summary.verdict, report.gates)
}

const HEADLINE_METRIC_PRIORITY = [
  'joint.realized_quality',
  'routing.accuracy',
  'model_pool.oracle_gain',
  'model_pool.oracle_quality',
  'agentic.success_rate',
  'multimodal.quality',
  'multimodal.support_rate',
  'preference.agreement',
  'joint.normalized_regret',
  'safety.violation_rate',
  'safety.block_accuracy',
  'capacity.throughput_rps',
  'capacity.latency_p95_ms',
  'capacity.success_rate',
  'capacity.cost_per_successful_request',
] as const

// The v2 control plane independently reduces only these values from sealed
// records, regardless of the report's claim level. Every other aggregate stays
// available in the explorer but cannot become a headline under this revision.
const SERVER_REDUCED_HEADLINES = new Set([
  'joint.normalized_regret',
  'safety.violation_rate',
  'safety.block_accuracy',
  'capacity.success_rate',
])

export function isServerReducedMetric(metricID: string): boolean {
  return SERVER_REDUCED_HEADLINES.has(metricID)
}

export function selectHeadlineMetrics(report: EvaluationReport, limit = 4): EvaluationMetric[] {
  const available = report.metrics.filter((metric) => {
    if (metric.value === null || !Number.isFinite(metric.value)) return false
    if (metric.track_id && !report.run.track_ids.includes(metric.track_id)) return false
    return isServerReducedMetric(metric.id)
  })
  const byID = new Map(available.map((metric) => [metric.id, metric]))
  const selected: EvaluationMetric[] = []
  for (const id of HEADLINE_METRIC_PRIORITY) {
    const metric = byID.get(id)
    if (!metric) continue
    selected.push(metric)
    byID.delete(id)
    if (selected.length === limit) return selected
  }
  for (const metric of available) {
    if (!byID.has(metric.id)) continue
    selected.push(metric)
    if (selected.length === limit) break
  }
  return selected
}
