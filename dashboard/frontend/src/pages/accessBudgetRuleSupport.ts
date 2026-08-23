import type { RateLimitRule } from '../utils/routerManagementTypes'

const responseMetrics = new Set<RateLimitRule['metric']>([
  'input_tokens',
  'output_tokens',
  'total_tokens',
  'served_input_tokens',
  'served_output_tokens',
  'served_total_tokens',
  'cost',
])

export function allowedAlgorithms(metric: RateLimitRule['metric']): RateLimitRule['algorithm'][] {
  if (metric === 'concurrent_requests') return ['concurrency']
  if (responseMetrics.has(metric)) return ['sliding_log', 'calendar_window']
  return ['sliding_log', 'calendar_window', 'token_bucket', 'gcra']
}

function accounting(metric: RateLimitRule['metric']): RateLimitRule['accounting'] {
  return responseMetrics.has(metric) ? 'response_actual' : 'request'
}

export function durationInput(value?: string) {
  const minute = value?.match(/^PT(\d+)M$/)
  if (minute) return `${minute[1]}m`
  const hour = value?.match(/^PT(\d+)H$/)
  if (hour) return `${hour[1]}h`
  const second = value?.match(/^PT(\d+)S$/)
  if (second) return `${second[1]}s`
  const day = value?.match(/^P(\d+)D$/)
  if (day) return `${day[1]}d`
  return value || ''
}

export function isoDuration(value: string) {
  const normalized = value.trim().toLowerCase()
  const match = normalized.match(/^(\d+)(s|m|h|d)$/)
  if (!match) return value.trim().toUpperCase()
  const [, amount, unit] = match
  return unit === 'd' ? `P${amount}D` : `PT${amount}${unit.toUpperCase()}`
}

export function normalizeRule(rule: RateLimitRule): RateLimitRule {
  const next = { ...rule, accounting: accounting(rule.metric) }
  const allowed = allowedAlgorithms(next.metric)
  if (!allowed.includes(next.algorithm)) next.algorithm = allowed[0]
  if (next.metric === 'concurrent_requests') next.algorithm = 'concurrency'

  const base = {
    ruleId: next.ruleId,
    metric: next.metric,
    algorithm: next.algorithm,
    accounting: next.accounting,
    enforcement: next.enforcement,
    ordinal: next.ordinal,
  }
  if (next.algorithm === 'sliding_log') {
    return { ...base, limit: next.limit || '1', window: next.window || 'PT1M' }
  }
  if (next.algorithm === 'calendar_window') {
    return {
      ...base,
      limit: next.limit || '1',
      period: next.period || 'day',
      timezone: next.timezone || 'UTC',
    }
  }
  if (next.algorithm === 'token_bucket') {
    return {
      ...base,
      capacity: next.capacity || '1',
      refillAmount: next.refillAmount || '1',
      refillPeriod: next.refillPeriod || 'PT1M',
    }
  }
  if (next.algorithm === 'gcra') {
    return {
      ...base,
      emissionInterval: next.emissionInterval || 'PT1S',
      burstTolerance: next.burstTolerance ?? 0,
    }
  }
  return { ...base, limit: next.limit || '1' }
}
