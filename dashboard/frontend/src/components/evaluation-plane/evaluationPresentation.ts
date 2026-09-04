import type {
  EvaluationGate,
  EvaluationMetric,
  EvaluationRunStatus,
  GateVerdict,
} from '../../types/evaluationPlane'

export const RUN_STATUS_LABELS: Record<EvaluationRunStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export const GATE_VERDICT_LABELS: Record<GateVerdict, string> = {
  pass: 'Pass',
  fail: 'Fail',
  unavailable: 'Unavailable',
  waived: 'Waived',
  not_applicable: 'Not applicable',
}

export function clampFraction(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || typeof value === 'undefined' || !Number.isFinite(value)) return '-'
  return `${(value * 100).toFixed(1)}%`
}

export function formatMetric(metric: Pick<EvaluationMetric, 'value' | 'unit'>): string {
  if (metric.value === null || !Number.isFinite(metric.value)) return 'Unavailable'
  switch (metric.unit) {
    case 'ratio':
    case 'fraction':
      return formatPercent(metric.value)
    case 'percent':
    case '%':
      return `${metric.value.toFixed(1)}%`
    case 'ms':
      return `${metric.value.toFixed(metric.value >= 100 ? 0 : 1)} ms`
    case 's':
      return `${metric.value.toFixed(2)} s`
    case 'usd':
    case 'USD':
      return new Intl.NumberFormat(undefined, {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: metric.value < 1 ? 4 : 2,
      }).format(metric.value)
    case 'count':
      return new Intl.NumberFormat().format(metric.value)
    default:
      return `${metric.value.toFixed(3)}${metric.unit ? ` ${metric.unit}` : ''}`
  }
}

export function formatDelta(metric: Pick<EvaluationMetric, 'delta' | 'unit'>): string | null {
  if (metric.delta === null || typeof metric.delta === 'undefined') return null
  const formatted = formatMetric({ value: Math.abs(metric.delta), unit: metric.unit })
  return `${metric.delta > 0 ? '+' : metric.delta < 0 ? '−' : ''}${formatted}`
}

export function evidenceRank(level: string): number {
  const parsed = Number(level.replace(/^E/, ''))
  return Number.isFinite(parsed) ? parsed : 0
}

export function effectiveGateVerdict(reported: GateVerdict, gates: EvaluationGate[]): GateVerdict {
  const requiredGates = gates.filter((gate) => gate.disposition === 'required')
  if (requiredGates.some((gate) => gate.verdict === 'fail')) return 'fail'
  if (requiredGates.some((gate) => gate.verdict === 'unavailable')) return 'unavailable'
  return reported
}
