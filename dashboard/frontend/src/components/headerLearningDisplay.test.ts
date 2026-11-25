import { describe, expect, it } from 'vitest'

import {
  formatLearningHeaderValue,
  LEARNING_ACTIONS,
  LEARNING_REASONS,
  LEARNING_SCOPES,
} from './headerLearningDisplay'

describe('formatLearningHeaderValue', () => {
  it('renders the trace learning fields as readable method fields', () => {
    expect(
      formatLearningHeaderValue(
        LEARNING_ACTIONS,
        'adaptation=propose_switch,protection=allow_switch',
      ),
    ).toBe('adaptation: proposed switch · protection: switch allowed')

    expect(formatLearningHeaderValue(LEARNING_SCOPES, 'protection=conversation')).toBe(
      'protection: conversation',
    )
  })

  it('renders baseline reasons without legacy route-status language', () => {
    expect(formatLearningHeaderValue(LEARNING_REASONS, 'protection=new_conversation')).toBe(
      'protection: conversation baseline',
    )

    expect(formatLearningHeaderValue(LEARNING_REASONS, 'protection=new_session')).toBe(
      'protection: session baseline',
    )
  })
})
