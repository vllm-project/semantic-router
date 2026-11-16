import { describe, expect, it } from 'vitest'
import { buildProviderCacheSection } from './insightsProviderCache'
import type { InsightsRecord, ProviderCacheAccounting } from './insightsPageTypes'

const unknown = { provenance: 'unknown' as const }
const accounting: ProviderCacheAccounting = {
  state: 'complete',
  uncached_input: { tokens: 10, provenance: 'derived', cost: 0.00002 },
  cache_read: { tokens: 10, provenance: 'authoritative', cost: 0.00001 },
  cache_write: { tokens: 80, provenance: 'authoritative', cost: 0.00052 },
  output: { tokens: 2, provenance: 'authoritative', cost: 0.00002 },
  cache_write_5m: { tokens: 30, provenance: 'authoritative' },
  cache_write_1h: { tokens: 50, provenance: 'authoritative' },
  no_cache_cost: 0.00022,
  cache_savings: -0.00035,
}
const record: InsightsRecord = {
  id: 'cache-cost',
  timestamp: '2026-10-15T00:00:00Z',
  signals: {},
  decision_tier: 0,
  decision_priority: 0,
  turn_index: 0,
  currency: 'USD',
  lifecycle_state: 'completed',
  cost_savings: 0.002,
  provider_cache: accounting,
}

describe('provider prompt-cache accounting', () => {
  it('shows component costs and separates cache savings from routing savings', () => {
    expect(buildProviderCacheSection(record).fields).toEqual(
      expect.arrayContaining([
        { label: 'Uncached input tokens', value: '10 (derived)' },
        { label: 'Cache write cost', value: '$0.0005' },
        { label: '1h cache write tokens', value: '50 (authoritative)' },
        { label: 'Same-model no-cache cost', value: '$0.0002' },
        { label: 'Provider cache savings', value: '-$0.0004' },
        { label: 'Model-routing savings', value: '$0.0020' },
      ]),
    )
  })

  it('distinguishes missing usage from authoritative zero', () => {
    const fields = buildProviderCacheSection({
      ...record,
      provider_cache: {
        ...accounting,
        state: 'partial',
        cache_read: unknown,
        cache_write: { tokens: 0, provenance: 'authoritative', cost: 0 },
      },
    }).fields
    expect(fields).toContainEqual({ label: 'Cache read tokens', value: 'Not recorded' })
    expect(fields).toContainEqual({ label: 'Cache read cost', value: 'N/A' })
    expect(fields).toContainEqual({ label: 'Cache write tokens', value: '0 (authoritative)' })
    expect(fields).toContainEqual({ label: 'Cache write cost', value: '$0.0000' })
    expect(fields).toContainEqual({ label: 'Provider cache savings', value: 'N/A' })
  })

  it('does not reinterpret Router response-cache hits as provider savings', () => {
    expect(buildProviderCacheSection({ ...record, from_cache: true }).fields).toEqual([
      { label: 'Accounting', value: 'Router response-cache hit; no provider request was made.' },
    ])
  })

  it('does not reprice historical records or show savings for incomplete requests', () => {
    expect(buildProviderCacheSection({ ...record, provider_cache: undefined }).fields).toEqual([
      { label: 'Accounting', value: 'Provider cache breakdown was not recorded.' },
    ])
    expect(
      buildProviderCacheSection({ ...record, lifecycle_state: 'aborted' }).fields,
    ).toContainEqual({
      label: 'Provider cache savings',
      value: 'N/A',
    })
  })
})
