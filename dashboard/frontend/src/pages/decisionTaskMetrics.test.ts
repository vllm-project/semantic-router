import { describe, expect, it } from 'vitest'
import { decisionTaskQueries, readTaskResultDistribution } from './decisionTaskMetrics'

describe('task monitoring contracts', () => {
  it('separates unknown and error results, and escapes exact deployment/task selectors', () => {
    const query = decisionTaskQueries('judge"\\one', 'pii_spans', 3600)
    expect(query.calls).toContain(`deployment=${JSON.stringify('judge"\\one')}`)
    expect(query.calls).toContain('task="pii_spans"')
    expect(query.unknown).toContain('outcome="unknown"')
    expect(query.errors).toContain('outcome="error"')
    expect(query.errors).not.toContain('clamp_min')
    expect(query.results).toContain('sum by (result) (increase(')
    expect(query.results).toContain('[3600s]')
  })
  it('omits non-finite and absent counts rather than presenting them as successful predictions', () => {
    const result = readTaskResultDistribution({
      status: 'success',
      data: {
        resultType: 'vector',
        result: [
          { metric: { result: 'positive' }, value: [1, '12.4'] },
          { metric: { result: 'negative' }, value: [1, '3'] },
          { metric: { result: 'absent' }, value: [1, 'NaN'] },
          { metric: { result: 'zero' }, value: [1, '0'] },
          { metric: {}, value: [1, '5'] },
        ],
      },
    })
    expect(result).toEqual([
      { result: 'positive', count: 12.4 },
      { result: 'negative', count: 3 },
    ])
    expect(() => readTaskResultDistribution({ status: 'error' })).toThrow('unavailable')
  })
})
