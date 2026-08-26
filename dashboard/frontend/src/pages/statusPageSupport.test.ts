import { describe, expect, it } from 'vitest'

import { formatStatusLabel } from './statusPageSupport'

describe('status page support', () => {
  it('formats public status labels', () => {
    expect(formatStatusLabel('not_running')).toBe('Not Running')
    expect(formatStatusLabel('operational')).toBe('Operational')
  })
})
