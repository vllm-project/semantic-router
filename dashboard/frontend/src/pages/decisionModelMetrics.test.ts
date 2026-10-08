import { describe, expect, it } from 'vitest'
import {
  decisionModelMetricQueries,
  formatDecisionModelMetric,
  readDecisionModelMetricVector,
} from './decisionModelMetrics'

describe('decision model runtime metrics', () => {
  it('uses exact escaped deployment labels for router runtime counters and histograms', () => {
    const names = ['@Vela-2.0-0.3B/auto', 'custom"\\\n.*|other']
    const queries = decisionModelMetricQueries(names)
    const match = queries.calls.match(/deployment=~("(?:\\.|[^"\\])*")/)
    expect(match).not.toBeNull()
    const pattern = JSON.parse(match![1]) as string
    const matches = new RegExp(`^(?:${pattern})$`)
    expect(names.every((name) => matches.test(name))).toBe(true)
    expect(matches.test('@Vela-2x0-0x3B/auto')).toBe(false)
    expect(matches.test('other')).toBe(false)
    expect(queries.calls).toContain('vsr_model_runtime_requests_total')
    expect(queries.errors).toContain('outcome!="ok"')
    expect(queries.cache).toContain('result="hit"')
    expect(queries.forward).toContain('phase="forward"')
    expect(queries.p95).toContain('sum by (deployment, le)')
  })

  it('preserves a measured zero but treats missing, idle ratios and invalid observations as unknown', () => {
    const result = readDecisionModelMetricVector(
      {
        status: 'success',
        data: {
          resultType: 'vector',
          result: [
            { metric: { deployment: 'zero' }, value: [1, '0'] },
            { metric: { deployment: 'active' }, value: [1, '0.125'] },
            { metric: { deployment: 'idle' }, value: [1, 'NaN'] },
            { metric: { deployment: 'infinite' }, value: [1, '+Inf'] },
            { metric: { deployment: 'negative' }, value: [1, '-1'] },
            { metric: { deployment: 'empty' }, value: [1, ''] },
            { metric: { deployment: 'unrelated-backend-llm' }, value: [1, '999'] },
          ],
        },
      },
      ['zero', 'active', 'idle', 'infinite', 'negative', 'empty', 'missing'],
    )
    expect(result).toEqual({ zero: 0, active: 0.125 })
    expect(formatDecisionModelMetric(result.zero, 'percent')).toBe('0.0%')
    expect(formatDecisionModelMetric(result.active, 'seconds')).toBe('125.0 ms')
    expect(formatDecisionModelMetric(result.idle, 'percent')).toBe('Not reported')
    expect(formatDecisionModelMetric(result.missing, 'rate')).toBe('Not reported')
  })

  it('rejects failed or malformed Prometheus responses instead of reporting zero', () => {
    expect(() =>
      readDecisionModelMetricVector({ status: 'error', error: 'unavailable' }, []),
    ).toThrow()
    expect(() => readDecisionModelMetricVector(null, [])).toThrow()
    expect(() =>
      readDecisionModelMetricVector(
        { status: 'success', data: { resultType: 'matrix', result: [] } },
        [],
      ),
    ).toThrow()
    expect(
      readDecisionModelMetricVector(
        { status: 'success', data: { resultType: 'vector', result: [] } },
        ['model'],
      ),
    ).toEqual({})
  })
})
