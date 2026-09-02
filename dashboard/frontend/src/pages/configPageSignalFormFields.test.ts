import { describe, expect, it } from 'vitest'

import { buildSignalFormFields } from './configPageSignalFormFields'
import type { AddSignalFormState } from './configPageSupport'

function baseForm(overrides: Partial<AddSignalFormState> = {}): AddSignalFormState {
  return {
    type: 'Conversation',
    name: '',
    description: '',
    operator: 'AND',
    keywords: [],
    case_sensitive: false,
    threshold: 0.8,
    candidates: [],
    aggregation_method: 'mean',
    mmlu_categories: [],
    ...overrides,
  }
}

describe('signal form fields', () => {
  it('offers Conversation as a signal type', () => {
    const typeField = buildSignalFormFields().find((field) => field.name === 'type')
    expect((typeField as { options?: string[] }).options).toContain('Conversation')
    expect((typeField as { options?: string[] }).options).toHaveLength(19)
  })

  it('hides the conversation predicate unless the feature is a count', () => {
    const predicateField = buildSignalFormFields().find(
      (field) => field.name === 'conversation_predicate',
    )
    const shouldHide = predicateField?.shouldHide as (data: AddSignalFormState) => boolean

    expect(
      shouldHide(baseForm({ conversation_feature: { type: 'exists', source: { type: 'message' } } })),
    ).toBe(true)
    expect(
      shouldHide(baseForm({ conversation_feature: { type: 'count', source: { type: 'message' } } })),
    ).toBe(false)
    expect(
      shouldHide(baseForm({ type: 'Keywords', conversation_feature: { type: 'count', source: { type: 'message' } } })),
    ).toBe(true)
  })
})
