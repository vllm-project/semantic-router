import { describe, expect, it } from 'vitest'

import { projectionInputSource } from './configPageProjectionTableSupport'

describe('projection table support', () => {
  it('labels kb_metric inputs by knowledge base and metric', () => {
    expect(
      projectionInputSource({
        type: 'kb_metric',
        kb: 'privacy_kb',
        metric: 'private_vs_public',
        weight: 1,
        value_source: 'score',
      }),
    ).toBe('kb_metric:privacy_kb:private_vs_public')
    expect(projectionInputSource({ type: 'keyword', name: 'urgent', weight: 0.5 })).toBe(
      'keyword:urgent',
    )
  })
})
