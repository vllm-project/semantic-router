import type {
  AccessAPIKey,
  AccessGroup,
  AccessTeam,
  AccessUser,
  QuotaMeter,
  UsageSummary,
} from '../utils/inferenceAccessApi'
import type { ManagementCostSummary, RateLimitRule } from '../utils/routerManagementTypes'

export const EMPTY_USAGE: UsageSummary = {
  final: true,
  completeness: 'complete',
  granularity: 'hour',
  requests: 0,
  successful: 0,
  failed: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  activeKeys: 0,
  averageLatencyMs: 0,
  p95LatencyMs: 0,
  averageTtftMs: 0,
  p95TtftMs: 0,
  costs: [],
  series: [],
  byModel: [],
  byEntrypoint: [],
  byRecipe: [],
  byDecision: [],
  byUser: [],
  byTeam: [],
  byKey: [],
}

export const formatNumber = (value?: number) => new Intl.NumberFormat('en-US').format(value || 0)

const currencySymbols: Record<string, string> = {
  USD: '$',
  EUR: '€',
  GBP: '£',
  JPY: '¥',
}

export function formatExactDecimal(value?: string | null) {
  if (value === undefined || value === null || value === '') return '—'
  const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value)
  if (!match) return value
  const [, sign, integer, fraction] = match
  const grouped = integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
  return `${sign}${grouped}${fraction ? `.${fraction}` : ''}`
}

export function formatExactCost(amount: string, currency: string) {
  const code = currency.toUpperCase()
  const symbol = currencySymbols[code]
  const value = formatExactDecimal(amount)
  return symbol ? `${symbol}${value} ${code}` : `${value} ${code}`
}

export function formatCosts(costs: ManagementCostSummary[]) {
  const known = costs.filter((cost) => cost.completeness !== 'unknown')
  return known.length
    ? known.map((cost) => formatExactCost(cost.knownAmount, cost.currency)).join(' + ')
    : '—'
}

export function costCoverageLabel(costs: ManagementCostSummary[]) {
  if (!costs.length || costs.every((cost) => cost.completeness === 'unknown'))
    return 'No priced usage'
  return costs.every((cost) => cost.completeness === 'complete')
    ? 'Settled from actual usage'
    : 'Partial · incomplete requests excluded'
}

export function durationLabel(value?: string) {
  const match = value?.match(/^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$/)
  if (!match) return value || 'policy window'
  const units = [
    [match[1], 'day'],
    [match[2], 'hour'],
    [match[3], 'minute'],
    [match[4], 'second'],
  ] as const
  const parts = units
    .filter(([amount]) => amount && amount !== '0')
    .map(([amount, unit]) => `${amount} ${unit}${amount === '1' ? '' : 's'}`)
  return parts.join(' ') || 'policy window'
}

const rateMetricLabel = (metric: RateLimitRule['metric']) =>
  metric === 'requests'
    ? 'Requests'
    : metric === 'total_tokens'
      ? 'Tokens'
      : metric === 'input_tokens'
        ? 'Input tokens'
        : metric === 'output_tokens'
          ? 'Output tokens'
          : metric === 'served_total_tokens'
            ? 'Served tokens'
            : metric === 'served_input_tokens'
              ? 'Served input tokens'
              : metric === 'served_output_tokens'
                ? 'Served output tokens'
                : metric === 'concurrent_requests'
                  ? 'Concurrent requests'
                  : 'Spend'

export function rateLimitRuleLabel(rule: RateLimitRule) {
  const metric = rateMetricLabel(rule.metric)
  if (rule.algorithm === 'calendar_window') {
    return `${metric} ${formatExactDecimal(rule.limit)} / ${rule.period || 'calendar window'}`
  }
  if (rule.algorithm === 'token_bucket') {
    return `${metric} ${formatExactDecimal(rule.capacity)} capacity`
  }
  if (rule.algorithm === 'gcra') {
    return `${metric} / ${durationLabel(rule.emissionInterval)}`
  }
  if (rule.algorithm === 'concurrency') {
    return `${metric} ${formatExactDecimal(rule.limit)}`
  }
  return `${metric} ${formatExactDecimal(rule.limit)} / ${durationLabel(rule.window)}`
}

export function rateLimitPolicySummary(rules: RateLimitRule[], maximum = 2) {
  if (!rules.length) return 'No limits'
  const shown = rules.slice(0, maximum).map(rateLimitRuleLabel)
  const remaining = rules.length - shown.length
  return `${shown.join(' · ')}${remaining ? ` · +${remaining}` : ''}`
}

export function quotaMeterLabel(meter: QuotaMeter) {
  const metric =
    meter.metric === 'requests'
      ? 'Requests'
      : meter.metric === 'total_tokens'
        ? 'Tokens'
        : meter.metric === 'input_tokens'
          ? 'Input tokens'
          : meter.metric === 'output_tokens'
            ? 'Output tokens'
            : meter.metric === 'cost'
              ? 'Spend'
              : meter.metric.split('_').join(' ')
  return `${metric} / ${durationLabel(meter.window)}`
}

export function formatQuotaValue(meter: QuotaMeter, value: string | null) {
  if (value === null) return '—'
  return meter.metric === 'cost' && meter.currency
    ? formatExactCost(value, meter.currency)
    : formatExactDecimal(value)
}

export function quotaProgress(meter: QuotaMeter) {
  const limit = Number(meter.limit)
  const used = Number(meter.used)
  if (!Number.isFinite(limit) || !Number.isFinite(used) || limit <= 0) return 0
  return Math.min(100, Math.max(0, (used / limit) * 100))
}

export function quotaResetLabel(meter: QuotaMeter) {
  if (!meter.resetsAt) return ''
  const reset = new Date(meter.resetsAt)
  if (Number.isNaN(reset.getTime())) return ''
  return `Resets ${new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(reset)}`
}

export function quotaCapacityLabel(meter: QuotaMeter) {
  if (meter.remaining !== null) return `${formatQuotaValue(meter, meter.remaining)} left`
  if (meter.capacityState === 'fenced' || meter.completeness === 'partial') {
    return 'Finalizing usage'
  }
  return 'Syncing usage'
}

export function quotaCapacityNote(meter: QuotaMeter) {
  if (meter.remaining !== null && meter.completeness === 'complete') return ''
  if (meter.capacityState === 'fenced' || meter.completeness === 'partial') {
    return 'Recent requests are finalizing.'
  }
  return 'Usage is syncing.'
}
export const formatDate = (value?: string) =>
  value
    ? new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeStyle: 'short' }).format(
        new Date(value),
      )
    : 'Never'

export function effectiveResources(key: AccessAPIKey, groups: AccessGroup[]) {
  if (key.effectiveAccess?.length) return key.effectiveAccess
  const effective = groups.filter((group) => key.accessGroupIds.includes(group.id))
  const resources = effective.flatMap((group) => group.resources)
  return [
    ...new Map(
      resources.map((resource) => [`${resource.resourceType}:${resource.resourceId}`, resource]),
    ).values(),
  ]
}

export function ownerLabel(key: AccessAPIKey, users: AccessUser[], teams: AccessTeam[]) {
  if (key.ownerType === 'user')
    return users.find((user) => user.id === key.ownerId)?.name || key.ownerId
  return teams.find((team) => team.id === key.ownerId)?.name || key.ownerId || 'Unassigned'
}
