import { describe, expect, it } from 'vitest'

import type { AccessUsageEvent, UsageSummary } from '../utils/inferenceAccessApi'
import {
  accessUsageEventToInsightsRecord,
  usageSummaryToInsightsAggregate,
} from './insightsPageApi'

const usageEvent: AccessUsageEvent = {
  id: 'namespace:admission-1',
  requestId: 'request-1',
  admissionId: 'admission-1',
  keyId: 'key-1',
  model: 'entrypoint-balanced',
  models: [{ id: 'model-fast', name: 'local/fast', revision: 7 }],
  entrypointId: 'entrypoint-balanced',
  recipeId: 'recipe-balance',
  decisionId: 'decision-simple',
  decisionName: 'Simple',
  decisionTier: 1,
  completedAt: '2026-08-25T00:00:01Z',
  routing: { entrypointName: 'vllm-sr/balance', recipeName: 'Balance' },
  statusCode: 200,
  promptTokens: 8,
  completionTokens: 3,
  totalTokens: 11,
  latencyMs: 1000,
  createdAt: '2026-08-25T00:00:00Z',
}

const slice = (id: string, requests: number) => ({
  id,
  requests,
  successful: requests,
  failed: 0,
  promptTokens: requests * 8,
  completionTokens: requests * 3,
  totalTokens: requests * 11,
  averageLatencyMs: 1000,
  p95LatencyMs: 1000,
  costs: [],
})

const usage: UsageSummary = {
  granularity: 'hour',
  requests: 40,
  successful: 39,
  failed: 1,
  promptTokens: 320,
  completionTokens: 120,
  totalTokens: 440,
  activeKeys: 1,
  averageLatencyMs: 1000,
  p95LatencyMs: 1500,
  averageTtftMs: 80,
  p95TtftMs: 120,
  costs: [
    {
      currency: 'USD',
      knownAmount: '12.5',
      completeness: 'partial',
      knownDispatches: '39',
      incompleteDispatches: '1',
    },
  ],
  series: [],
  byModel: [slice('model-fast', 40)],
  byEntrypoint: [slice('entrypoint-balanced', 40)],
  byRecipe: [slice('recipe-balance', 40)],
  byDecision: [slice('decision-simple', 40)],
  byUser: [],
  byTeam: [],
  byKey: [],
}

describe('Insights Management projection', () => {
  it('keeps durable route and Model names from request-log detail', () => {
    expect(accessUsageEventToInsightsRecord(usageEvent)).toMatchObject({
      recipe: 'Balance',
      decision: 'Simple',
      decision_tier: 1,
      original_model: 'vllm-sr/balance',
      selected_model: 'local/fast',
      ended_at: '2026-08-25T00:00:01Z',
    })
  })

  it('uses complete usage rollups instead of the visible log page', () => {
    const aggregate = usageSummaryToInsightsAggregate(usage, [
      accessUsageEventToInsightsRecord(usageEvent),
    ])

    expect(aggregate.record_count).toBe(40)
    expect(aggregate.summary).toMatchObject({
      actual_spend: 12.5,
      currency: 'USD',
      cost_record_count: 39,
      excluded_record_count: 1,
    })
    expect(aggregate.decision_distribution).toEqual([{ name: 'decision-simple', value: 40 }])
    expect(aggregate.token_breakdown.by_decision).toEqual([
      { name: 'decision-simple', input_tokens: 320, output_tokens: 120, total_tokens: 440 },
    ])
    expect(aggregate.available_recipes).toEqual(['recipe-balance'])
  })
})
