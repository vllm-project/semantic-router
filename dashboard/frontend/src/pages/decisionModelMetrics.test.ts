import { describe, expect, it } from 'vitest'
import {
  DECISION_MODEL_TIME_WINDOWS,
  decisionModelChartPoints,
  decisionModelMetricQueries,
  decisionModelMetricRange,
  formatDecisionModelMetric,
  formatDecisionModelRateAxis,
  readDecisionModelMetricMatrix,
} from './decisionModelMetrics'

const matrix = (result: unknown[]) => ({
  status: 'success',
  data: { resultType: 'matrix', result },
})
const range = { start: 100, end: 160, step: 15 }

describe('decision model runtime metrics', () => {
  it('keeps quiet-runtime traffic ticks distinct from zero and from one another', () => {
    const labels = [0, 0.005, 0.01, 0.015, 0.02].map(formatDecisionModelRateAxis)
    expect(new Set(labels).size).toBe(5)
    expect(Number(formatDecisionModelRateAxis(0.005))).toBe(0.005)
    expect(Number(formatDecisionModelRateAxis(0.02))).toBe(0.02)
    expect(formatDecisionModelRateAxis(123_000).length).toBeLessThan(7)
  })
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

  it('bounds history requests and includes the most recent observation for every window', () => {
    for (const window of DECISION_MODEL_TIME_WINDOWS) {
      const result = decisionModelMetricRange(window.seconds, 1_800_000_123_456)
      expect(result.end).toBe(1_800_000_123)
      expect(result.end - result.start).toBe(window.seconds)
      expect(result.step).toBeGreaterThanOrEqual(15)
      expect((result.end - result.start) / result.step + 1).toBeLessThanOrEqual(181)
      expect((result.end - result.start) % result.step).toBe(0)
    }
  })

  it('retains measured zero and missing intervals while filtering unrelated or invalid samples', () => {
    const result = readDecisionModelMetricMatrix(
      matrix([
        {
          metric: { deployment: 'model' },
          values: [
            [100, '0'],
            [115, '0.125'],
            [145, 'NaN'],
            [160, '+Inf'],
            [85, '999'],
            [175, '999'],
            [101, '999'],
            ['130', '999'],
            [NaN, '999'],
          ],
        },
        { metric: { deployment: 'backend-llm' }, values: [[160, '999']] },
      ]),
      ['model'],
      range,
    )
    expect(result).toEqual({
      model: [
        { time: 100_000, value: 0 },
        { time: 115_000, value: 0.125 },
        { time: 130_000, value: null },
        { time: 145_000, value: null },
        { time: 160_000, value: null },
      ],
    })
    expect(formatDecisionModelMetric(result.model[0].value, 'percent')).toBe('0.0%')
    expect(formatDecisionModelMetric(result.model[1].value, 'seconds')).toBe('125.0 ms')
    expect(formatDecisionModelMetric(result.model[result.model.length - 1]?.value, 'percent')).toBe(
      'Not reported',
    )
  })

  it('does not reuse an earlier healthy sample after a series stops reporting', () => {
    const result = readDecisionModelMetricMatrix(
      matrix([
        {
          metric: { deployment: 'model' },
          values: [
            [100, '0.5'],
            [115, '0.6'],
          ],
        },
      ]),
      ['model'],
      range,
    )
    const points = decisionModelChartPoints({ calls: result }, 'model')
    expect(points[1]).toEqual({ time: 115_000, calls: 0.6 })
    expect(points[points.length - 1]).toEqual({ time: 160_000, calls: null })
  })

  it('treats negative, blank and malformed observations as gaps, not successful calls', () => {
    const result = readDecisionModelMetricMatrix(
      matrix([
        {
          metric: { deployment: 'model' },
          values: [[100, '-1'], [115, ''], [130, null], [145, 'NaN'], null],
        },
        { metric: { deployment: 'missing' } },
      ]),
      ['model', 'missing'],
      range,
    )
    expect(result.model.every(({ value }) => value === null)).toBe(true)
    expect(result.missing).toBeUndefined()
  })

  it('merges independent metrics by timestamp without inventing values for failed queries', () => {
    const points = decisionModelChartPoints(
      {
        calls: {
          model: [
            { time: 1_000, value: 0 },
            { time: 2_000, value: 1 },
          ],
        },
        p95: {
          model: [
            { time: 1_000, value: 0.12 },
            { time: 2_000, value: null },
          ],
        },
        cache: { other: [{ time: 2_000, value: 99 }] },
      },
      'model',
    )
    expect(points).toEqual([
      { time: 1_000, calls: 0, p95: 0.12 },
      { time: 2_000, calls: 1, p95: null },
    ])
    expect(formatDecisionModelMetric(points[points.length - 1]?.errors, 'percent')).toBe(
      'Not reported',
    )
  })

  it('rejects failed or malformed Prometheus responses instead of reporting zero', () => {
    expect(() => readDecisionModelMetricMatrix({ status: 'error' }, [], range)).toThrow()
    expect(() => readDecisionModelMetricMatrix(null, [], range)).toThrow()
    expect(() =>
      readDecisionModelMetricMatrix(
        { status: 'success', data: { resultType: 'vector', result: [] } },
        [],
        range,
      ),
    ).toThrow()
    expect(readDecisionModelMetricMatrix(matrix([]), ['model'], range)).toEqual({})
  })
})
