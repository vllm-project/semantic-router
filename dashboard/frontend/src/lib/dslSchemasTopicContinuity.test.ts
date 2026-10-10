import { describe, expect, it } from 'vitest'
import { getSignalFieldSchema, SIGNAL_TYPES } from './dslSchemas'
import { DECISION_SIGNAL_TYPES } from '../generated/routerConfigContract'

describe('topic_continuity signal authoring', () => {
  const fields = getSignalFieldSchema('topic_continuity')

  it('exposes the optional evidence policy fields', () => {
    expect(fields.map((field) => field.key)).toEqual([
      'description',
      'include_assistant',
      'thresholds',
      'limits',
    ])
    expect(fields.every((field) => !field.required)).toBe(true)
    const limits = fields.find((field) => field.key === 'limits')
    expect(limits?.fields?.map((field) => field.key)).toEqual([
      'max_prior_turns',
      'max_turn_bytes',
      'max_input_bytes',
    ])
  })

  it('is a declarable signal but never a decision condition', () => {
    expect(SIGNAL_TYPES).toContain('topic_continuity')
    expect(DECISION_SIGNAL_TYPES).not.toContain('topic_continuity')
  })
})
