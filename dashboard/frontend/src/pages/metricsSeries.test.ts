import { describe, expect, it } from 'vitest'

import {
  OTHER_COLOR,
  RANKED_COLORS,
  chartRows,
  rankedColor,
  rankedSeries,
  ratioValues,
  readMatrix,
  readVector,
  sampleCount,
  sampleTimes,
  sumValues,
  timeTicks,
  totalOf,
  valuesWhere,
} from './metricsSeries'

const grid = { start: 100, end: 160, step: 15 }
const matrix = (result: unknown[]) => ({
  status: 'success',
  data: { resultType: 'matrix', result },
})
const vector = (result: unknown[]) => ({
  status: 'success',
  data: { resultType: 'vector', result },
})

describe('Prometheus range results', () => {
  it('places samples on the window grid and keeps missing samples as gaps', () => {
    expect(sampleCount(grid)).toBe(5)
    expect(sampleTimes(grid)).toEqual([100_000, 115_000, 130_000, 145_000, 160_000])
    const series = readMatrix(
      matrix([
        {
          metric: { outcome: 'success', job: 'semantic-router' },
          values: [
            [100, '2'],
            [130, '0'],
            [145, 'NaN'],
            [160, '+Inf'],
          ],
        },
      ]),
      grid,
    )
    expect(series).toEqual([
      { labels: { outcome: 'success', job: 'semantic-router' }, values: [2, null, 0, null, null] },
    ])
  })

  it('ignores samples off the grid or outside the window and malformed entries', () => {
    const [series] = readMatrix(
      matrix([
        {
          metric: { model: 'qwen2.5' },
          values: [
            [107, '1'],
            [85, '1'],
            [175, '1'],
            ['115', '1'],
            [115, 3],
            [115.0004, '4'],
          ],
        },
        { metric: { model: 'broken' } },
      ]),
      grid,
    )
    expect(series.values).toEqual([null, 4, null, null, null])
  })

  it('rejects an answer that is not a successful matrix', () => {
    expect(() => readMatrix({ status: 'error', error: 'bad_data' }, grid)).toThrow(
      'Prometheus did not return a time series.',
    )
    expect(() => readMatrix(vector([]), grid)).toThrow()
    expect(() => readMatrix(null, grid)).toThrow()
  })

  it('reads instant vectors with NaN as unknown', () => {
    expect(
      readVector(
        vector([
          { metric: { quantile: 'p95' }, value: [160, '0.42'] },
          { metric: { quantile: 'p99' }, value: [160, 'NaN'] },
          { metric: {} },
        ]),
      ),
    ).toEqual([
      { labels: { quantile: 'p95' }, value: 0.42 },
      { labels: { quantile: 'p99' }, value: null },
    ])
    expect(() => readVector(matrix([]))).toThrow('Prometheus did not return a result.')
  })
})

describe('series arithmetic', () => {
  it('adds series and leaves a point unknown only when every series lacks it', () => {
    expect(
      sumValues(
        [
          [1, null, null, 2],
          [3, 4, null, null],
        ],
        4,
      ),
    ).toEqual([4, 4, null, 2])
    expect(sumValues([], 2)).toEqual([null, null])
  })

  it('divides only where the whole is known and positive', () => {
    expect(ratioValues([1, null, 2, 1], [4, 2, 0, null])).toEqual([0.25, 0, null, null])
  })

  it('selects series by label and totals instant samples', () => {
    const series = [
      { labels: { outcome: 'success' }, values: [1, null] },
      { labels: { outcome: 'timeout' }, values: [null, 2] },
      { labels: { outcome: 'success' }, values: [1, 1] },
    ]
    expect(valuesWhere(series, 'outcome', 'success', 2)).toEqual([2, 1])
    expect(valuesWhere(series, 'outcome', 'canceled', 2)).toEqual([null, null])
    expect(
      totalOf([
        { labels: {}, value: 2 },
        { labels: {}, value: null },
      ]),
    ).toBe(2)
    expect(totalOf([])).toBeNull()
  })

  it('keys chart rows by position so a label with a dot cannot become a path', () => {
    expect(chartRows([1_000, 2_000], [{ key: 's0', values: [1, null] }])).toEqual([
      { time: 1_000, s0: 1 },
      { time: 2_000, s0: null },
    ])
  })
})

describe('ranked series', () => {
  const series = (name: string, values: (number | null)[]) => ({ name, values })

  it('colors names by their place in the order and adds the rest together', () => {
    const order = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    const ranked = rankedSeries(
      [
        series('h', [1, null]),
        series('b', [2, 2]),
        series('g', [null, null]),
        series('a', [5, 5]),
        series('unlisted', [1, 1]),
      ],
      order,
      2,
      true,
    )
    expect(ranked.map((entry) => entry.label)).toEqual(['a', 'b', '3 others'])
    expect(ranked[0].color).toBe(RANKED_COLORS[0])
    expect(ranked[1].color).toBe(RANKED_COLORS[1])
    expect(ranked[2]).toMatchObject({ color: OTHER_COLOR, values: [2, 1] })
    expect(ranked[2].source).toBeUndefined()
    expect(rankedColor(order, 'b')).toBe(RANKED_COLORS[1])
    expect(rankedColor(order, 'h')).toBe(OTHER_COLOR)
  })

  it('leaves the rest out of a metric that does not add up, such as a percentile', () => {
    const order = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    const ranked = rankedSeries(
      order.map((name) => series(name, [1])),
      order,
      1,
      false,
    )
    expect(ranked).toHaveLength(RANKED_COLORS.length)
    expect(ranked.some((entry) => entry.label === 'g')).toBe(false)
  })

  it('ranks by size when there is no order to follow', () => {
    const ranked = rankedSeries([series('small', [1]), series('large', [9])], [], 1, true)
    expect(ranked.map((entry) => entry.label)).toEqual(['large', 'small'])
  })
})

describe('time ticks', () => {
  it('places round ticks inside the window', () => {
    const start = new Date(2026, 10, 18, 9, 7).getTime()
    const ticks = timeTicks(start, start + 3_600_000)
    expect(ticks.length).toBeGreaterThanOrEqual(5)
    expect(ticks.length).toBeLessThanOrEqual(7)
    for (const tick of ticks) {
      expect(tick).toBeGreaterThanOrEqual(start)
      expect(new Date(tick).getMinutes() % 10).toBe(0)
    }
    const days = timeTicks(start, start + 7 * 86_400_000)
    expect(days.every((tick) => new Date(tick).getHours() === 0)).toBe(true)
  })
})
