import { describe, expect, it } from 'vitest'

import {
  DECISION_FALLBACK_SCHEMA,
  DECISION_RELIABILITY_SCHEMA,
} from './configPageDecisionAdvancedSchemas'
import { mergeDecisionForSave, type DecisionConfig } from './configPageSupport'

describe('decision fallback', () => {
  it("edits the recipe policy's fields except the circuit breaker", () => {
    const types = Object.fromEntries(
      DECISION_FALLBACK_SCHEMA.map((field) => [field.key, field.type]),
    )
    expect(types).toEqual({
      enabled: 'boolean',
      max_attempts: 'number',
      total_timeout: 'string',
      per_attempt_timeout: 'string',
      retryable_status_codes: 'number[]',
    })
  })

  it('survives a save that does not touch it', () => {
    const existing = {
      name: 'escalate',
      description: '',
      priority: 1,
      rules: { operator: 'AND', conditions: [] },
      modelRefs: [],
      fallback: { enabled: true, max_attempts: 2 },
    } as unknown as DecisionConfig
    const untouched = mergeDecisionForSave(existing, {
      name: 'escalate',
      description: 'edited',
      priority: 1,
      rules: existing.rules,
      modelRefs: [],
    })
    expect(untouched.fallback).toEqual({ enabled: true, max_attempts: 2 })
  })
})

describe('decision reliability', () => {
  it('edits every field of the router schema with its type', () => {
    const types = Object.fromEntries(
      DECISION_RELIABILITY_SCHEMA.map((field) => [field.key, field.type]),
    )
    expect(types).toEqual({
      total_timeout: 'string',
      per_try_timeout: 'string',
      idle_timeout: 'string',
      first_byte_timeout: 'string',
      retry_count: 'number',
      retry_on: 'string',
      retriable_status_codes: 'number[]',
      retry_back_off_base: 'string',
      retry_back_off_max: 'string',
      retry_after_max: 'string',
    })
  })

  it('marks the fields only standalone mode honors', () => {
    const nativeOnly = DECISION_RELIABILITY_SCHEMA.filter((field) =>
      field.description?.includes('standalone mode only'),
    ).map((field) => field.key)
    expect(nativeOnly).toEqual([
      'idle_timeout',
      'first_byte_timeout',
      'retry_back_off_base',
      'retry_back_off_max',
      'retry_after_max',
    ])
  })

  it('survives a save that does not touch it', () => {
    const existing = {
      name: 'slow_route',
      description: '',
      priority: 1,
      rules: { operator: 'AND', conditions: [] },
      modelRefs: [],
      reliability: { total_timeout: '600s', retry_count: 1 },
    } as unknown as DecisionConfig
    const saved = mergeDecisionForSave(existing, {
      ...existing,
      reliability: undefined,
      description: 'edited',
    })
    expect(saved.reliability).toBeUndefined()
    const untouched = mergeDecisionForSave(existing, {
      name: 'slow_route',
      description: 'edited',
      priority: 1,
      rules: existing.rules,
      modelRefs: [],
    })
    expect(untouched.reliability).toEqual({ total_timeout: '600s', retry_count: 1 })
  })
})
