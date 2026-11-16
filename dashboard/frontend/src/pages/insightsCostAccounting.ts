import { formatInsightsCost } from '../utils/insightsCost'
import type {
  InsightsAggregateCostCoverage,
  InsightsCostAccounting,
  InsightsRecord,
} from './insightsPageTypes'

const UNPRICED_REASONS: Record<string, string> = {
  no_usage: 'no token usage reported',
  usage_breakdown_unavailable: 'usage too incomplete to price',
  pricing_not_configured: 'no price configured',
  currency_mismatch: 'priced in another currency',
  attempt_not_traced: 'Looper call without a per-call trace',
  price_not_recorded: 'recorded before prices were kept',
  not_completed: 'not completed',
}

const BASELINE_SOURCES: Record<string, string> = {
  configured_model: 'the configured baseline model',
  recipe_most_expensive: 'the recipe pool’s most expensive model',
  selected_model: 'the selected model (direct request)',
  unrecorded: 'an unrecorded rule (older records)',
}

const ROUTER_CALL_LABELS: Record<string, string> = {
  looper: 'Looper',
  systemone: 'System One',
  decision_model: 'Decision model',
}

export const CALLER_LOGIN_NOTE = 'API-price equivalent, not what the plan charges'

export const SET_PRICING_HINT =
  'Set providers.models[].pricing (rates per 1M tokens) for the model to see its cost.'

export function describeUnpricedReason(reason: string): string {
  return UNPRICED_REASONS[reason] ?? reason.replace(/_/g, ' ')
}

function describeBaselineRule(source: string): string {
  return BASELINE_SOURCES[source] ?? source.replace(/_/g, ' ')
}

/** Whether a record is a Looper request, whose cost is its calls rather than one model call. */
export function isLooperRecord(record: InsightsRecord): boolean {
  return Boolean(record.cost_accounting?.router_calls?.some((call) => call.kind === 'looper'))
}

/** Where a priced record's rates came from, marked as an estimate where it is one. */
export function describePricingSource(accounting?: InsightsCostAccounting): string | undefined {
  if (!accounting?.pricing_source) {
    return undefined
  }
  return accounting.pricing_source === 'catalog_list_price'
    ? 'the built-in catalog’s list price (an estimate: no pricing is configured)'
    : 'configured model rates'
}

/** Why a completed record shows no model cost, with how to fix it when the fix is pricing. */
export function describeMissingModelCost(record: InsightsRecord): string | undefined {
  const reason = record.cost_accounting?.unpriced_reason
  if (!reason) {
    return undefined
  }
  if (reason !== 'pricing_not_configured') {
    return `No model cost: ${describeUnpricedReason(reason)}.`
  }
  const login = record.cost_accounting?.caller_login
    ? ' It ran on the client’s own login, which its plan bills; a price shows the API-price equivalent, not what the plan charges.'
    : ''
  return `No price is configured for ${record.selected_model || 'this model'}.${login} ${SET_PRICING_HINT}`
}

export function describeBaselineBasis(record: InsightsRecord): string {
  const accounting = record.cost_accounting
  if (!accounting?.baseline_source) {
    return 'Older records used the highest estimate in the recipe’s complete model pool, or the selected model for a direct request, and keep that baseline.'
  }
  const rule = describeBaselineRule(accounting.baseline_source)
  if (accounting.baseline_unavailable) {
    return `No baseline: ${rule} is ${describeUnpricedReason(accounting.baseline_unavailable)}.`
  }
  const model = record.baseline_model ? ` (${record.baseline_model})` : ''
  return `The same recorded tokens at the rates of ${rule}${model}.`
}

/** The calls the Router made itself, with what they cost and what is left unpriced. */
export function describeRouterCalls(accounting?: InsightsCostAccounting): string | undefined {
  const calls = accounting?.router_calls ?? []
  if (calls.length === 0) {
    return undefined
  }
  return calls
    .map((call) => {
      const label = ROUTER_CALL_LABELS[call.kind] ?? call.kind
      const priced = call.calls - (call.unpriced_calls ?? 0)
      const parts = [
        `${call.calls} ${label} call${call.calls === 1 ? '' : 's'}${call.model ? ` (${call.model})` : ''}`,
      ]
      parts.push(`${call.prompt_tokens ?? 0} input / ${call.completion_tokens ?? 0} output tokens`)
      if (priced > 0) {
        parts.push(`${priced} priced at ${formatInsightsCost(call.cost, call.currency)}`)
      }
      if (call.unpriced_calls) {
        parts.push(`${call.unpriced_calls} unpriced (${formatReasonCounts(call.unpriced_reasons)})`)
      }
      return parts.join(', ')
    })
    .join('; ')
}

function sortedCounts(counts?: Record<string, number>): [string, number][] {
  return Object.entries(counts ?? {}).sort(
    ([a, left], [b, right]) => right - left || a.localeCompare(b),
  )
}

function formatReasonCounts(counts?: Record<string, number>): string {
  return sortedCounts(counts)
    .map(([reason, count]) => `${count} ${describeUnpricedReason(reason)}`)
    .join(', ')
}

/** Which Router calls the savings are net of, and the requests a missing price leaves out. */
function buildNetSavingsLines(coverage: InsightsAggregateCostCoverage): string[] {
  const lines: string[] = []
  const netted = sortedCounts(coverage.netted_router_calls)
  if (netted.length > 0) {
    lines.push(
      `Savings are net of the Router’s own calls for these requests: ${netted
        .map(([kind, count]) => plural(count, `${ROUTER_CALL_LABELS[kind] ?? kind} call`))
        .join(', ')}.`,
    )
  }
  const left = sortedCounts(coverage.net_unavailable)
  if (left.length > 0) {
    const count = left.reduce((sum, [, value]) => sum + value, 0)
    const hint = coverage.net_unavailable?.pricing_not_configured
      ? ' Set global.model_catalog.deployments.<name>.pricing; an empty block declares a free deployment.'
      : ''
    lines.push(
      `${plural(count, 'request')} left out of savings for a Router call without a price (${formatReasonCounts(coverage.net_unavailable)}).${hint}`,
    )
  }
  return lines
}

function plural(count: number, noun: string): string {
  return `${count.toLocaleString('en-US')} ${noun}${count === 1 ? '' : 's'}`
}

/** Sentences that state how much of the filtered traffic the estimates cover. */
export function buildCostCoverageLines(
  coverage: InsightsAggregateCostCoverage,
  costRecordCount: number,
): string[] {
  const requests =
    coverage.priced_records + coverage.partly_priced_records + coverage.unpriced_records
  const lines = [
    `Priced: ${coverage.priced_records} of ${plural(requests, 'request')}` +
      (coverage.partly_priced_records ? `, ${coverage.partly_priced_records} partly` : '') +
      (coverage.unpriced_records ? `, ${coverage.unpriced_records} without a price` : '') +
      '.',
  ]
  if (coverage.unpriced_records + coverage.partly_priced_records > 0) {
    lines.push(`Left out: ${formatReasonCounts(coverage.unpriced_reasons)}.`)
  }
  if (coverage.router_calls > 0) {
    lines.push(
      `Router calls: ${plural(coverage.router_calls, 'call')}, ${coverage.unpriced_router_calls} unpriced; their cost is shown apart from the model cost.`,
    )
  }
  const baselines = Object.entries(coverage.baseline_sources ?? {})
  if (baselines.length > 0) {
    lines.push(
      `Savings compare ${plural(costRecordCount, 'request')} with ${baselines
        .map(([source, count]) => `${describeBaselineRule(source)} (${count})`)
        .join(', ')}.`,
    )
  }
  lines.push(...buildNetSavingsLines(coverage))
  if (coverage.list_price_records > 0) {
    lines.push(
      `${plural(coverage.list_price_records, 'request')} priced at a catalog list price: an estimate, not a configured rate.`,
    )
  }
  if (coverage.caller_login_records > 0) {
    lines.push(
      `${plural(coverage.caller_login_records, 'request')} ran on a client’s own login: any cost shown is the ${CALLER_LOGIN_NOTE}.`,
    )
  }
  if (coverage.looper_hop_records > 0) {
    lines.push(
      `${plural(coverage.looper_hop_records, 'Looper hop record')} not counted: each request counts its own calls.`,
    )
  }
  return lines
}

/** Why no cost shows at all, when every request is unpriced, and how to set it. */
export function describeMissingAggregateCost(
  coverage: InsightsAggregateCostCoverage,
): string | undefined {
  if (coverage.priced_records + coverage.partly_priced_records > 0) {
    return undefined
  }
  if (!coverage.unpriced_reasons?.pricing_not_configured) {
    return undefined
  }
  const login =
    coverage.caller_login_records > 0
      ? ' Requests on a client’s own login (auth.credential: caller) are billed by the client’s plan, so the Router has no price for them; the built-in catalog lists no price for these models either.'
      : ''
  return `No cost is shown because the models these requests used have no price.${login} ${SET_PRICING_HINT} With pricing set, the cost of a login request is its ${CALLER_LOGIN_NOTE}.`
}
