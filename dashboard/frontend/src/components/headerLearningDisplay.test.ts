import { describe, expect, it } from 'vitest'

import { formatLearningHeaderValue } from './headerLearningDisplay'

describe('formatLearningHeaderValue', () => {
  it('renders split Router Learning headers as readable method fields', () => {
    expect(
      formatLearningHeaderValue(
        'x-vsr-learning-actions',
        'adaptation=propose_switch,protection=allow_switch',
      ),
    ).toBe('adaptation: proposed switch · protection: switch allowed')

    expect(
      formatLearningHeaderValue('x-vsr-learning-scopes', 'protection=conversation'),
    ).toBe('protection: conversation')
  })

  it('renders baseline reasons without legacy route-status language', () => {
    expect(
      formatLearningHeaderValue('x-vsr-learning-reasons', 'protection=new_conversation'),
    ).toBe('protection: conversation baseline')

    expect(
      formatLearningHeaderValue('x-vsr-learning-reasons', 'protection=new_session'),
    ).toBe('protection: session baseline')
  })
})
