import { describe, expect, it } from 'vitest'

import { projectionInputSource } from './configPageProjectionTableSupport'

describe('projection table support', () => {
  it('labels an input by its signal type and name', () => {
    expect(projectionInputSource({ type: 'keyword', name: 'urgent', weight: 0.5 })).toBe(
      'keyword:urgent',
    )
  })
})
