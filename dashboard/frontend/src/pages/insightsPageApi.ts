import {
  inferenceAccessApi,
  type AccessUsageEvent,
  type UsageSummary,
} from '../utils/inferenceAccessApi'
import { ManagementApiError } from '../utils/managementApiContract'

import type { InsightsAggregateResponse, InsightsRecord } from './insightsPageTypes'

const metadataString = (metadata: Record<string, unknown> | undefined, ...keys: string[]) => {
  for (const key of keys) {
    const value = metadata?.[key]
    if (typeof value === 'string' && value.trim()) return value
  }
  return undefined
}

const knownCost = (event: AccessUsageEvent): { amount: number; currency?: string } => {
  const cost = event.costs?.find((candidate) => candidate.completeness !== 'unknown')
  const amount = Number(cost?.knownAmount ?? 0)
  return { amount: Number.isFinite(amount) ? amount : 0, currency: cost?.currency }
}

export function accessUsageEventToInsightsRecord(event: AccessUsageEvent): InsightsRecord {
  const selectedModel = metadataString(
    event.metadata,
    'selectedModel',
    'selected_model',
    'logicalModelId',
  )
  const decision = metadataString(
    event.metadata,
    'decision',
    'selectedDecision',
    'selected_decision',
  )
  const cost = knownCost(event)
  const successful = event.statusCode >= 200 && event.statusCode < 400 && !event.errorCode

  return {
    id: event.id,
    timestamp: event.createdAt,
    request_id: event.requestId,
    recipe: event.recipeId,
    decision,
    decision_tier: 0,
    decision_priority: 0,
    original_model: event.entrypointId,
    selected_model: selectedModel ?? event.model,
    signals: {},
    response_status: event.statusCode,
    lifecycle_state: successful ? 'completed' : 'failed',
    duration_ms: event.latencyMs,
    terminal_reason: event.errorCode,
    from_cache: metadataString(event.metadata, 'cacheStatus', 'cache_status') === 'cached',
    streaming: event.streaming,
    prompt_tokens: event.promptTokens,
    completion_tokens: event.completionTokens,
    total_tokens: event.totalTokens,
    actual_cost: cost.amount,
    currency: cost.currency,
  }
}

const aggregateEntries = (
  values: Array<{
    id: string
    requests: number
    promptTokens: number
    completionTokens: number
    totalTokens: number
  }>,
) => ({
  distribution: values.map((item) => ({ name: item.id, value: item.requests })),
  tokens: values.map((item) => ({
    name: item.id,
    input_tokens: item.promptTokens,
    output_tokens: item.completionTokens,
    total_tokens: item.totalTokens,
  })),
})

export function usageSummaryToInsightsAggregate(
  usage: UsageSummary,
  visibleRecords: InsightsRecord[] = [],
): InsightsAggregateResponse {
  const models = aggregateEntries(usage.byModel)
  const decisions = new Map<string, number>()
  const recipes = new Set<string>()
  let actualSpend = 0
  let costRecordCount = 0

  for (const record of visibleRecords) {
    if (record.decision) decisions.set(record.decision, (decisions.get(record.decision) ?? 0) + 1)
    if (record.recipe) recipes.add(record.recipe)
    if (typeof record.actual_cost === 'number' && record.actual_cost > 0) {
      actualSpend += record.actual_cost
      costRecordCount += 1
    }
  }

  return {
    object: 'management.usage.aggregate',
    record_count: usage.requests,
    lifecycle: {
      completed: usage.successful,
      failed: usage.failed,
      aborted: 0,
      in_progress: 0,
      unknown: 0,
    },
    summary: {
      total_saved: 0,
      baseline_spend: 0,
      actual_spend: actualSpend,
      currency: visibleRecords.find((record) => record.currency)?.currency,
      cost_record_count: costRecordCount,
      excluded_record_count: Math.max(0, usage.requests - costRecordCount),
    },
    model_selection: models.distribution,
    decision_distribution: [...decisions].map(([name, value]) => ({ name, value })),
    signal_distribution: [],
    token_volume: {
      input_tokens: usage.promptTokens,
      output_tokens: usage.completionTokens,
      total_tokens: usage.totalTokens,
      excluded_record_count: 0,
    },
    token_breakdown: { by_decision: [], by_selected_model: models.tokens },
    available_recipes: [...recipes].sort(),
    available_decisions: [...decisions.keys()].sort(),
    available_models: usage.byModel.map((item) => item.id).sort(),
  }
}

export function isInsightsDataUnavailableError(error: unknown): boolean {
  return error instanceof ManagementApiError && [404, 501, 503].includes(error.status)
}

export async function fetchInsightsRecord(id: string): Promise<InsightsRecord> {
  return accessUsageEventToInsightsRecord(await inferenceAccessApi.requestLog(id))
}
