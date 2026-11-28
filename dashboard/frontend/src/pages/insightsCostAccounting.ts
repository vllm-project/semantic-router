import { formatInsightsCost } from '../utils/insightsCost'
import { modelLabel, modelText, type ModelLabels } from './insightsModelLabelSupport'
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
  recipe_most_expensive: 'the priciest model in its pool',
  selected_model: 'the selected model (direct request)',
  unrecorded: 'an unrecorded rule (older records)',
}

const PRICE_TABLE =
  'at this deployment’s own price table (catalog list prices when the pricing source says so); not a quality comparison.'

/** What a range's saving is measured against, read from the rules its compared requests used. */
export interface SavingsReference {
  /** The figure's label: "Saved vs priciest model", "Saved vs <model>", or "Saved vs reference" when they differ. */
  label: string
  /** The reference in a word or two, beside the actual cost. */
  short: string
  /** The reference in a sentence: "the priciest model", a model's name, or "the reference". */
  noun: string
  /** How each request is priced for the comparison. */
  hint: string
  /** How many requests each rule compared, when they differ. */
  mixed?: string
}

/**
 * The reference of a range's saving. Direct requests, compared with their own
 * model, never decide it; a range that compared nothing reads by the
 * configuration.
 */
export function savingsReference(
  coverage: Pick<InsightsAggregateCostCoverage, 'baseline_sources'> | undefined,
  configuredModel?: string,
  labels: ModelLabels = {},
): SavingsReference {
  const counts = { ...(coverage?.baseline_sources ?? {}) }
  delete counts.selected_model
  const pool = counts.recipe_most_expensive ?? 0
  const configured = counts.configured_model ?? 0
  const other = Object.entries(counts)
    .filter(([source]) => source !== 'recipe_most_expensive' && source !== 'configured_model')
    .reduce((sum, [, count]) => sum + count, 0)
  const model = configuredModel ? modelLabel(labels, configuredModel) : undefined
  const configuredRule = configured > 0 || (pool + other === 0 && model !== undefined)
  if (pool === 0 && other === 0 && configuredRule) {
    return {
      label: `Saved vs ${model ?? 'configured model'}`,
      short: model ?? 'Configured model',
      noun: model ?? 'the configured model',
      hint: `Cost only: each request priced at ${model ?? 'the configured model'} for the same tokens, ${PRICE_TABLE}`,
    }
  }
  if (configured === 0 && other === 0) {
    return {
      label: 'Saved vs priciest model',
      short: 'Priciest model',
      noun: 'the priciest model',
      hint: `Cost only: each request priced at the most expensive model in its pool for the same tokens, ${PRICE_TABLE}`,
    }
  }
  const parts = [
    pool ? `${plural(pool, 'request')} vs the priciest model in its pool` : '',
    configured ? `${plural(configured, 'request')} vs ${model ?? 'the configured model'}` : '',
    other ? `${plural(other, 'request')} vs an unrecorded rule (older records)` : '',
  ].filter(Boolean)
  return {
    label: 'Saved vs reference',
    short: 'Reference',
    noun: 'the reference',
    hint: `Cost only: each request priced at its reference model for the same tokens, ${PRICE_TABLE}`,
    mixed: parts.length > 1 ? `References differ in this range: ${parts.join(', ')}.` : undefined,
  }
}

/** A range's saving as a figure: plain where it saved, with a minus sign where the requests cost more. */
export function formatSaving(value: number, currency?: string): string {
  const amount = formatInsightsCost(Math.abs(value), currency, { total: true })
  return value < 0 && amount !== '—' ? `−${amount}` : amount
}

/** What a record's saving compares with, in a few words under the figure. */
export function savingVersus(record: InsightsRecord, labels: ModelLabels = {}): string {
  if (!record.baseline_model) return 'No baseline recorded'
  const model = modelLabel(labels, record.baseline_model)
  switch (record.cost_accounting?.baseline_source) {
    case 'recipe_most_expensive':
      return `vs. ${model}, the priciest in its pool`
    case 'selected_model':
      return 'vs. its own model (a direct request)'
    default:
      return `vs. ${model}`
  }
}

/** A record's saving in words, against the model it was compared with. */
export function describeRecordSaving(record: InsightsRecord, labels: ModelLabels = {}): string {
  const value = record.cost_savings ?? 0
  const amount = formatInsightsCost(Math.abs(value), record.currency)
  const compared = value < 0 ? `Cost ${amount} more than` : `Saved ${amount} vs`
  const model = modelText(labels, record.baseline_model ?? '')
  switch (record.cost_accounting?.baseline_source) {
    case 'recipe_most_expensive':
      return `${compared} ${model}, the priciest model in its pool for these tokens (cost only)`
    case 'selected_model':
      return `A direct request, compared with its own model, ${model} (cost only)`
    default:
      return `${compared} ${model} (cost only)`
  }
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
export function describeMissingModelCost(
  record: InsightsRecord,
  labels: ModelLabels = {},
): string | undefined {
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
  const model = record.selected_model ? modelText(labels, record.selected_model) : 'this model'
  return `No price is configured for ${model}.${login} ${SET_PRICING_HINT}`
}

export function describeBaselineBasis(record: InsightsRecord, labels: ModelLabels = {}): string {
  const accounting = record.cost_accounting
  if (!accounting?.baseline_source) {
    return 'Older records used the highest estimate in the recipe’s complete model pool, or the selected model for a direct request, and keep that baseline.'
  }
  const rule = describeBaselineRule(accounting.baseline_source)
  if (accounting.baseline_unavailable) {
    return `No baseline: ${rule} is ${describeUnpricedReason(accounting.baseline_unavailable)}.`
  }
  const model = record.baseline_model ? `, ${modelText(labels, record.baseline_model)}` : ''
  return `The same recorded tokens at the rates of ${rule}${model}.`
}

const BUDGET_SETTLEMENTS: Record<string, string> = {
  usage: 'from its usage',
  estimate: 'its estimate, as it reported no usage that could be priced',
  released: 'nothing, as no provider answered it',
}

/** What the request's tightest spend budget had left, and what the request reserved and spent. */
export function describeBudget(accounting?: InsightsCostAccounting): string | undefined {
  const budget = accounting?.budget
  if (!budget) {
    return undefined
  }
  const cost = (value?: number) => formatInsightsCost(value, budget.currency)
  const left = `${budget.scope}: ${cost(budget.remaining)} of ${cost(budget.limit)} left`
  const unpriced = budget.unpriced_calls
    ? `; ${plural(budget.unpriced_calls, 'unpriced call')}`
    : ''
  if (!budget.settlement) {
    const refused =
      budget.remaining > 0
        ? `the call was estimated at ${cost(budget.estimate)}`
        : 'nothing was left'
    return `${left}; refused: ${refused}${unpriced}`
  }
  const settled = BUDGET_SETTLEMENTS[budget.settlement] ?? budget.settlement
  return `${left}; reserved ${cost(budget.estimate)}, charged ${cost(budget.charged)} ${settled}${unpriced}`
}

/** The calls the Router made itself, with what they cost and what is left unpriced. */
export function describeRouterCalls(
  accounting?: InsightsCostAccounting,
  labels: ModelLabels = {},
): string | undefined {
  const calls = accounting?.router_calls ?? []
  if (calls.length === 0) {
    return undefined
  }
  return calls
    .map((call) => {
      const label = ROUTER_CALL_LABELS[call.kind] ?? call.kind
      const priced = call.calls - (call.unpriced_calls ?? 0)
      const parts = [
        `${call.calls} ${label} call${call.calls === 1 ? '' : 's'}${call.model ? ` on ${modelText(labels, call.model)}` : ''}`,
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

/** The Router's own calls of the compared requests, counted apart, and the requests a missing price leaves out. */
function buildNetSavingsLines(coverage: InsightsAggregateCostCoverage): string[] {
  const lines: string[] = []
  const netted = sortedCounts(coverage.netted_router_calls)
  if (netted.length > 0) {
    lines.push(
      `The Router’s own calls for these requests are counted apart from the saving: ${netted
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

/** The requests that cost nothing because no model answered them, as sentences. */
export function describeFreeRecords(coverage: InsightsAggregateCostCoverage): string {
  return (
    (coverage.refused_records
      ? ` ${plural(coverage.refused_records, 'request')} refused before any model answered cost nothing.`
      : '') +
    (coverage.unanswered_records
      ? ` ${plural(coverage.unanswered_records, 'request')} no provider answered cost nothing.`
      : '')
  )
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
      `Differences compare ${plural(costRecordCount, 'request')} with ${baselines
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

/** How the cost-only estimate is made and what it leaves out, for one tooltip. */
export function describeEstimateMethod(
  summary: { coverage?: InsightsAggregateCostCoverage; excluded_record_count: number },
  recordCount: number,
  reference: SavingsReference = savingsReference(summary.coverage),
): string {
  const coverage = summary.coverage
  const parts = [
    reference.hint,
    'The saving is that reference cost less the cost of the model that answered; the Router’s own calls are shown on their own line and not taken off. Nothing here is a GPU bill or an invoice.',
  ]
  if (reference.mixed) parts.push(reference.mixed)
  if (coverage) {
    parts.push(
      `Priced: ${coverage.priced_records.toLocaleString('en-US')} of ${recordCount.toLocaleString('en-US')} requests` +
        (coverage.partly_priced_records
          ? `, ${coverage.partly_priced_records.toLocaleString('en-US')} partly`
          : '') +
        '.' +
        describeFreeRecords(coverage),
    )
  }
  if (summary.excluded_record_count > 0) {
    parts.push(`${plural(summary.excluded_record_count, 'request')} left out of the comparison.`)
  }
  return parts.join(' ')
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
