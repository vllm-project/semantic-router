import { describe, expect, it } from 'vitest'

import { formatRoutingMetadataValue } from './routingMetadataDisplay'

describe('formatRoutingMetadataValue', () => {
  it.each([
    ['unified_balance_recovery', 'Balance Recovery'],
    ['unified_speed_first_route', 'Speed First'],
    ['unified_cost_reasoning', 'Cost Reasoning'],
    ['unified_frontier_verified_answer', 'Frontier Verified Answer'],
    ['unified_privacy_sensitive_route', 'Privacy Sensitive'],
  ])('humanizes decision %s', (identifier, expected) => {
    expect(formatRoutingMetadataValue('x-vsr-selected-decision', identifier)).toBe(expected)
  })

  it.each([
    ['unified_balance_difficulty:hard', 'Balance Difficulty: Hard'],
    ['unified_speed_interactive', 'Speed Interactive'],
    ['unified_cost_budget_request', 'Cost Budget Request'],
    ['unified_frontier_workflow_intent', 'Frontier Workflow Intent'],
    ['unified_privacy_pii_strict', 'Privacy PII Strict'],
    ['needs_fact_check', 'Needs Fact Check'],
  ])('humanizes signal %s', (identifier, expected) => {
    expect(formatRoutingMetadataValue('x-vsr-matched-embeddings', identifier)).toBe(expected)
  })

  it('expands language codes and preserves model identifiers', () => {
    expect(formatRoutingMetadataValue('x-vsr-matched-language', 'zh')).toBe('Chinese')
    expect(formatRoutingMetadataValue('x-vsr-matched-Language', 'de')).toBe('German')
    expect(formatRoutingMetadataValue('x-vsr-selected-algorithm', 'multi_factor')).toBe(
      'Multi-Factor',
    )
    expect(formatRoutingMetadataValue('x-vsr-selected-model', 'local/qwen3.5-122b-frontier')).toBe(
      'local/qwen3.5-122b-frontier',
    )
  })
})
