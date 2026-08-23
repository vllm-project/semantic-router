import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { RateLimitRule } from '../utils/routerManagementTypes'
import AccessBudgetRuleEditor from './AccessBudgetRuleEditor'
import { durationInput, isoDuration, normalizeRule } from './accessBudgetRuleSupport'

const costRule: RateLimitRule = {
  metric: 'cost',
  algorithm: 'sliding_log',
  limit: '20.000000000000001',
  window: 'PT8H',
  accounting: 'response_actual',
  enforcement: 'enforce',
}

describe('AccessBudgetRuleEditor', () => {
  it('renders an actual-cost limit with a human-readable eight-hour window', () => {
    const markup = renderToStaticMarkup(
      createElement(AccessBudgetRuleEditor, { rules: [costRule], onChange: vi.fn() }),
    )

    expect(markup).toContain('Spend')
    expect(markup).toContain('20.000000000000001')
    expect(markup).toContain('value="8h"')
    expect(markup).toContain('actual settled model usage')
  })

  it('round-trips common product durations without exposing implementation units', () => {
    expect(isoDuration('8h')).toBe('PT8H')
    expect(isoDuration('1d')).toBe('P1D')
    expect(durationInput('PT8H')).toBe('8h')
    expect(durationInput('P1D')).toBe('1d')
  })

  it('closes incompatible algorithm fields when a rule becomes an actual metric', () => {
    const normalized = normalizeRule({
      metric: 'cost',
      algorithm: 'token_bucket',
      capacity: '100',
      refillAmount: '10',
      refillPeriod: 'PT1M',
      accounting: 'request',
      enforcement: 'enforce',
    })

    expect(normalized).toEqual({
      ruleId: undefined,
      metric: 'cost',
      algorithm: 'sliding_log',
      accounting: 'response_actual',
      enforcement: 'enforce',
      ordinal: undefined,
      limit: '1',
      window: 'PT1M',
    })
  })
})
