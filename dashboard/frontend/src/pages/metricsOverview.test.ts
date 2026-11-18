import { describe, expect, it } from 'vitest'

import {
  amountsByCurrency,
  availabilityBuckets,
  cacheRateSeries,
  cacheSummary,
  decisionId,
  decisionRows,
  modelRows,
  parseDecisionKey,
  quantileSeries,
  quantileSummary,
  requestSeries,
  requestSummary,
  scrapeTargets,
  tokenSummary,
} from './metricsOverview'

const sample = (labels: Record<string, string>, value: number | null) => ({ labels, value })
const series = (labels: Record<string, string>, values: (number | null)[]) => ({ labels, values })

describe('request figures', () => {
  it('splits outcomes into succeeded and failed, with no error rate without traffic', () => {
    const result = requestSeries(
      [
        series({ outcome: 'success' }, [3, 0, null]),
        series({ outcome: 'server_error' }, [1, 0, null]),
        series({ outcome: 'timeout' }, [null, 0, null]),
      ],
      3,
    )
    expect(result.succeeded).toEqual([3, 0, null])
    expect(result.failed).toEqual([1, 0, null])
    expect(result.total).toEqual([4, 0, null])
    expect(result.errorRate).toEqual([0.25, null, null])
  })

  it('totals the range and leaves the error rate unknown without requests', () => {
    expect(
      requestSummary([
        sample({ outcome: 'success' }, 95),
        sample({ outcome: 'client_error' }, 3),
        sample({ outcome: 'canceled' }, 2),
      ]),
    ).toEqual({ total: 100, failed: 5, errorRate: 0.05 })
    expect(requestSummary([sample({ outcome: 'success' }, 0)])).toEqual({
      total: 0,
      failed: 0,
      errorRate: null,
    })
    expect(requestSummary([])).toEqual({ total: null, failed: null, errorRate: null })
  })

  it('reads latency quantiles and token totals by their tags', () => {
    expect(
      quantileSummary([sample({ quantile: 'p50' }, 0.2), sample({ quantile: 'p99' }, null)]),
    ).toEqual({ p50: 0.2, p95: null, p99: null })
    expect(quantileSeries([series({ quantile: 'p95' }, [1, null])], 2)).toEqual({
      p50: [null, null],
      p95: [1, null],
      p99: [null, null],
    })
    expect(
      tokenSummary([sample({ type: 'prompt' }, 30), sample({ type: 'completion' }, 70)]),
    ).toEqual({ prompt: 30, completion: 70, total: 100 })
  })
})

describe('spend and cache', () => {
  it('adds spend per currency, largest first', () => {
    expect(
      amountsByCurrency([
        sample({ currency: 'EUR' }, 1),
        sample({ currency: 'USD' }, 2),
        sample({}, 0.5),
        sample({ currency: 'CNY' }, null),
      ]),
    ).toEqual([
      { currency: 'USD', amount: 2.5 },
      { currency: 'EUR', amount: 1 },
    ])
  })

  it('reports no cache until the Router records a lookup, and the hit rate after', () => {
    expect(cacheSummary([])).toBeNull()
    expect(cacheSummary([sample({ result: 'miss' }, 4)])).toEqual({
      hits: 0,
      misses: 4,
      lookups: 4,
      hitRate: 0,
    })
    expect(
      cacheSummary([sample({ result: 'hit' }, 0), sample({ result: 'miss' }, 0)])?.hitRate,
    ).toBeNull()
    expect(
      cacheRateSeries(
        [series({ result: 'hit' }, [1, null, 0]), series({ result: 'miss' }, [3, 2, 0])],
        3,
      ),
    ).toEqual([0.25, 0, null])
  })
})

describe('per-model rows', () => {
  it('gathers each statistic of a model and orders models by requests', () => {
    const rows = modelRows([
      sample({ model: 'small', stat: 'requests' }, 10),
      sample({ model: 'large', stat: 'requests' }, 90),
      sample({ model: 'large', stat: 'errors' }, 9),
      sample({ model: 'large', stat: 'latency' }, 1.5),
      sample({ model: 'large', stat: 'tokens' }, 4_000),
      sample({ model: 'large', stat: 'cost', currency: 'USD' }, 0.4),
      sample({ model: 'large', stat: 'cost', currency: 'EUR' }, 0.1),
      sample({ model: 'auto', stat: 'errors' }, 3),
      sample({ stat: 'requests' }, 1),
    ])
    expect(rows.map((row) => row.model)).toEqual(['large', 'small', 'auto'])
    expect(rows[0]).toEqual({
      model: 'large',
      requests: 90,
      errors: 9,
      errorRate: 0.1,
      latency: 1.5,
      tokens: 4_000,
      cost: [
        { currency: 'USD', amount: 0.4 },
        { currency: 'EUR', amount: 0.1 },
      ],
    })
    expect(rows[1].errorRate).toBe(0)
    // Errors recorded before routing name the requested model, which has no requests.
    expect(rows[2]).toMatchObject({ requests: null, errors: 3, errorRate: null })
  })

  it('caps a model error rate at every request', () => {
    const [row] = modelRows([
      sample({ model: 'flaky', stat: 'requests' }, 2),
      sample({ model: 'flaky', stat: 'errors' }, 3),
    ])
    expect(row.errorRate).toBe(1)
  })
})

describe('per-decision rows', () => {
  it('splits the decision keys of the Router cache metrics', () => {
    expect(parseDecisionKey('coding')).toEqual({ recipe: 'default', decision: 'coding' })
    expect(parseDecisionKey('support%3A+tier+1::math+%2B+logic')).toEqual({
      recipe: 'support: tier 1',
      decision: 'math + logic',
    })
    expect(parseDecisionKey('bad%::x')).toEqual({ recipe: 'bad%', decision: 'x' })
  })

  it('adds selections per decision, ranks its models and joins its cache lookups', () => {
    const rows = decisionRows(
      [
        sample({ recipe: 'default', decision: 'coding', model: 'qwen' }, 30),
        sample({ recipe: 'default', decision: 'coding', model: 'llama' }, 50),
        sample({ recipe: 'support', decision: 'math', model: 'qwen' }, 5),
        sample({ recipe: 'default', decision: 'unknown', model: 'llama' }, 10),
        sample({ recipe: 'default', decision: 'empty', model: 'llama' }, null),
      ],
      [
        sample({ decision_name: 'coding', result: 'hit' }, 20),
        sample({ decision_name: 'coding', result: 'miss' }, 60),
        sample({ decision_name: 'support::math', result: 'miss' }, 5),
        sample({ decision_name: 'support::faq', result: 'hit' }, 2),
      ],
    )
    expect(rows.map((row) => [row.recipe, row.decision, row.requests])).toEqual([
      ['default', 'coding', 80],
      ['default', 'unknown', 10],
      ['support', 'math', 5],
      ['support', 'faq', 0],
    ])
    expect(rows[0].id).toBe(decisionId('default', 'coding'))
    expect(rows[0].models).toEqual([
      { model: 'llama', requests: 50 },
      { model: 'qwen', requests: 30 },
    ])
    expect(rows[0].cache).toEqual({ hits: 20, misses: 60, lookups: 80, hitRate: 0.25 })
    expect(rows[1].cache).toBeNull()
    expect(rows[2].cache?.hitRate).toBe(0)
    expect(rows[3].cache?.hitRate).toBe(1)
  })
})

describe('router health', () => {
  it('reads each scrape target and its share of successful scrapes', () => {
    expect(
      scrapeTargets([
        series({ instance: 'router-b:9190' }, [1, 0, null, 1]),
        series({ instance: 'router-a:9190' }, [1, 1, 1, 0]),
        series({ instance: 'router-c:9190' }, [1, null]),
      ]),
    ).toEqual([
      {
        instance: 'router-a:9190',
        values: [1, 1, 1, 0],
        up: false,
        availability: 0.75,
      },
      {
        instance: 'router-b:9190',
        values: [1, 0, null, 1],
        up: true,
        availability: 2 / 3,
      },
      { instance: 'router-c:9190', values: [1, null], up: null, availability: 1 },
    ])
  })

  it('groups scrapes into spans for the status strip', () => {
    const times = [0, 1, 2, 3, 4, 5]
    expect(availabilityBuckets([1, 1, 1, 0, null, null], times, 3)).toEqual([
      { start: 0, end: 1, state: 'up' },
      { start: 2, end: 3, state: 'partial' },
      { start: 4, end: 5, state: 'none' },
    ])
    expect(availabilityBuckets([0, 0], [0, 1], 60).map((bucket) => bucket.state)).toEqual([
      'down',
      'down',
    ])
  })
})
