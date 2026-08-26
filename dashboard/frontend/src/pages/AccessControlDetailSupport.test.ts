import { describe, expect, it } from 'vitest'

import type { QuotaMeter } from '../utils/inferenceAccessApi'
import {
  durationLabel,
  formatCosts,
  formatQuotaValue,
  quotaMeterLabel,
  quotaCapacityLabel,
  quotaCapacityNote,
  quotaProgress,
  quotaResetLabel,
} from './AccessControlDetailSupport'

const meter = (overrides: Partial<QuotaMeter> = {}): QuotaMeter => ({
  policyId: 'policy-1',
  ruleId: 'rule-1',
  bindingId: 'binding-1',
  metric: 'requests',
  algorithm: 'fixed_window',
  accounting: 'actual',
  enforcement: 'hard',
  limit: '12',
  used: '2',
  remaining: '10',
  window: 'PT8H',
  resetsAt: '2026-08-23T08:00:00Z',
  completeness: 'complete',
  capacityState: 'available',
  ...overrides,
})

describe('API-key actual usage and quota presentation', () => {
  it('renders request, token, and cost meters from server snapshots including PT8H', () => {
    expect(durationLabel('PT8H')).toBe('8 hours')
    expect(quotaMeterLabel(meter())).toBe('Requests / 8 hours')
    expect(quotaMeterLabel(meter({ metric: 'total_tokens' }))).toBe('Tokens / 8 hours')
    expect(
      quotaMeterLabel(meter({ metric: 'cost', currency: 'USD', limit: '3.5', used: '1.25' })),
    ).toBe('Spend / 8 hours')
  })

  it('keeps exact server decimals for used, remaining, progress, and spend', () => {
    const cost = meter({
      metric: 'cost',
      currency: 'USD',
      limit: '3.500000',
      used: '1.250000',
      remaining: '2.250000',
    })
    expect(formatQuotaValue(cost, cost.used)).toBe('$1.250000 USD')
    expect(formatQuotaValue(cost, cost.remaining)).toBe('$2.250000 USD')
    expect(quotaProgress(cost)).toBeCloseTo(35.7142857)
    expect(
      formatCosts([
        {
          currency: 'USD',
          knownAmount: '1.250000',
          completeness: 'complete',
          knownDispatches: '2',
          incompleteDispatches: '0',
        },
      ]),
    ).toBe('$1.250000 USD')
    expect(quotaResetLabel(cost)).toMatch(/^Resets /)
  })

  it('distinguishes settled zero capacity from syncing and finalizing snapshots', () => {
    expect(quotaCapacityLabel(meter({ used: '0', remaining: '12' }))).toBe('12 left')
    expect(quotaCapacityNote(meter({ used: '0', remaining: '12' }))).toBe('')
    expect(
      quotaCapacityLabel(
        meter({ remaining: null, completeness: 'unknown', capacityState: 'unknown' }),
      ),
    ).toBe('Syncing usage')
    expect(
      quotaCapacityLabel(
        meter({ remaining: null, completeness: 'partial', capacityState: 'fenced' }),
      ),
    ).toBe('Finalizing usage')
    expect(
      quotaCapacityNote(
        meter({ remaining: null, completeness: 'partial', capacityState: 'fenced' }),
      ),
    ).toBe('Recent requests are finalizing.')
  })
})
