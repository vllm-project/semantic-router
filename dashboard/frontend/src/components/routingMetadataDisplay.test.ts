import { readFileSync } from 'node:fs'
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

  it('humanizes every decision, signal, and projection in the maintained recipe', () => {
    const dsl = readFileSync(
      new URL('../../../../config/recipes/multi-objective/recipe.dsl', import.meta.url),
      'utf8',
    )
    const decisions = Array.from(dsl.matchAll(/^\s*ROUTE\s+([^\s(]+)/gm), (match) => match[1])
    const signals = Array.from(dsl.matchAll(/^\s*SIGNAL\s+(\S+)\s+([^\s{]+)/gm), (match) => ({
      type: match[1],
      name: match[2],
    }))
    const projections = Array.from(
      dsl.matchAll(/^\s*PROJECTION\s+(?:score|mapping)\s+([^\s{]+)/gm),
      (match) => match[1],
    )

    expect(decisions).toHaveLength(21)
    expect(signals.length).toBeGreaterThan(40)
    expect(projections).toHaveLength(14)

    for (const decision of decisions) {
      expect(formatRoutingMetadataValue('x-vsr-selected-decision', decision)).not.toMatch(
        /unified_|_/,
      )
    }
    for (const signal of signals) {
      expect(formatRoutingMetadataValue(`x-vsr-matched-${signal.type}`, signal.name)).not.toMatch(
        /unified_|_/,
      )
    }
    for (const projection of projections) {
      expect(formatRoutingMetadataValue('x-vsr-matched-projections', projection)).not.toMatch(
        /unified_|_/,
      )
    }
  })
})
