import { describe, expect, it } from 'vitest'

import { usageRangeBounds, usageRangeDays, type UsageScope } from './accessControlUsageRange'

const scope = (range: UsageScope['range']): UsageScope => ({
  type: 'global',
  id: '',
  model: '',
  range,
  granularity: 'auto',
  customFrom: '',
  customTo: '',
})

describe('usageRangeBounds', () => {
  const now = new Date(2026, 7, 21, 14, 30)

  it('aligns presets to local calendar boundaries', () => {
    expect(new Date(usageRangeBounds(scope('today'), now).from).getHours()).toBe(0)
    expect(usageRangeDays(scope('7d'), now)).toBeGreaterThan(6)
    expect(new Date(usageRangeBounds(scope('mtd'), now).from).getDate()).toBe(1)
    expect(new Date(usageRangeBounds(scope('ytd'), now).from).getMonth()).toBe(0)
  })

  it('uses inclusive custom dates and caps future end dates', () => {
    const custom = { ...scope('custom'), customFrom: '2026-08-10', customTo: '2026-08-30' }
    const bounds = usageRangeBounds(custom, now)
    expect(new Date(bounds.from).getDate()).toBe(10)
    expect(bounds.to).toBe(now.toISOString())
  })
})
