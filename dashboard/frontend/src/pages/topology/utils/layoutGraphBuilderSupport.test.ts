import { describe, expect, it } from 'vitest'

import type { DecisionConfig } from '../types'
import { getDecisionReachability } from './layoutGraphBuilderSupport'

const decision = (conditions: DecisionConfig['rules']['conditions']): DecisionConfig => ({
  name: 'route',
  priority: 100,
  rules: { operator: 'AND', conditions },
  modelRefs: [],
})

describe('getDecisionReachability', () => {
  it('treats an empty-condition decision as a valid fallback', () => {
    expect(getDecisionReachability(decision([]), new Set())).toEqual({
      isFallback: true,
      isUnreachable: false,
    })
  })

  it('marks only missing signal references as unreachable', () => {
    const configured = new Set(['keyword:known'])

    expect(
      getDecisionReachability(
        decision([{ type: 'keyword', name: 'known' }]),
        configured,
      ),
    ).toMatchObject({ isFallback: false, isUnreachable: false })
    expect(
      getDecisionReachability(
        decision([{ type: 'keyword', name: 'missing' }]),
        configured,
      ),
    ).toEqual({
      isFallback: false,
      isUnreachable: true,
      unreachableReason: 'Referenced signals not configured',
    })
  })
})
