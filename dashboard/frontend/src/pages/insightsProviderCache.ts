import type { ViewSection } from '../components/ViewPanel'
import { formatInsightsCost } from '../utils/insightsCost'
import type { InsightsRecord, TokenCostComponent } from './insightsPageTypes'

export function buildProviderCacheSection(record: InsightsRecord): ViewSection {
  if (record.from_cache) {
    return {
      title: 'Provider Prompt Cache',
      fields: [
        { label: 'Accounting', value: 'Router response-cache hit; no provider request was made.' },
      ],
    }
  }
  const accounting = record.provider_cache
  if (!accounting) {
    return {
      title: 'Provider Prompt Cache',
      fields: [{ label: 'Accounting', value: 'Provider cache breakdown was not recorded.' }],
    }
  }
  const fields: ViewSection['fields'] = [{ label: 'Usage state', value: accounting.state }]
  for (const [label, component] of [
    ['Uncached input', accounting.uncached_input],
    ['Cache read', accounting.cache_read],
    ['Cache write', accounting.cache_write],
    ['Output', accounting.output],
  ] as const) {
    fields.push(
      { label: `${label} tokens`, value: formatComponentTokens(component) },
      { label: `${label} cost`, value: formatInsightsCost(component.cost, record.currency) },
    )
  }
  for (const [label, component] of [
    ['5m cache write', accounting.cache_write_5m],
    ['1h cache write', accounting.cache_write_1h],
  ] as const) {
    if (component.tokens !== undefined) {
      fields.push({ label: `${label} tokens`, value: formatComponentTokens(component) })
    }
  }
  const settled = accounting.state === 'complete' && record.lifecycle_state === 'completed'
  fields.push(
    {
      label: 'Same-model no-cache cost',
      value: formatInsightsCost(settled ? accounting.no_cache_cost : undefined, record.currency),
    },
    {
      label: 'Provider cache savings',
      value: formatInsightsCost(settled ? accounting.cache_savings : undefined, record.currency),
    },
    {
      label: 'Model-routing savings',
      value: formatInsightsCost(
        record.lifecycle_state === 'completed' ? record.cost_savings : undefined,
        record.currency,
      ),
    },
    {
      label: 'Savings basis',
      value:
        'Provider cache savings compare this model with all input priced at its standard input rate. Negative savings include cache-write premiums. Model-routing savings compare model choices using the observed cache usage; the two values are separate comparisons.',
    },
  )
  return { title: 'Provider Prompt Cache', fields }
}

function formatComponentTokens(component: TokenCostComponent): string {
  return component.tokens === undefined
    ? 'Not recorded'
    : `${component.tokens.toLocaleString('en-US')} (${component.provenance})`
}
