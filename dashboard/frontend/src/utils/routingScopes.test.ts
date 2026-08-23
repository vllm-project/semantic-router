import { describe, expect, it } from 'vitest'

import {
  countProjectionsInProfile,
  countSignalsInProfile,
  hasRoutingProfileContent,
} from './routingScopes'

describe('routing profile metrics', () => {
  const document = {
    signals: {
      keywords: [{ name: 'intent' }],
      context: [{ name: 'short' }, { name: 'long' }],
    },
    projections: {
      scores: [{ name: 'score' }],
      mappings: [{ name: 'band' }],
    },
    decisions: [{ name: 'route' }],
  }

  it('counts native Recipe document signals and projections', () => {
    expect(countSignalsInProfile(document)).toEqual({
      total: 3,
      byType: { keywords: 1, context: 2 },
    })
    expect(countProjectionsInProfile(document)).toBe(2)
  })

  it('distinguishes an empty Recipe document', () => {
    expect(hasRoutingProfileContent(document)).toBe(true)
    expect(hasRoutingProfileContent({ decisions: [] })).toBe(false)
  })
})
