import { afterEach, describe, expect, it, vi } from 'vitest'
import { testQueryDryRun } from './api'

const mockResponse = (body: unknown, status = 200) => {
  const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  }))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('live topology preview', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('preserves unavailable results and tolerates null arrays from older dashboards', async () => {
    mockResponse({
      query: 'hello', mode: 'dry-run', isAccurate: false,
      matchedSignals: null, matchedModels: null, highlightedPath: null,
      warning: 'Router API unavailable',
    })

    const result = await testQueryDryRun('hello')
    expect(result).toMatchObject({
      mode: 'dry-run', isAccurate: false,
      matchedSignals: [], matchedModels: [], highlightedPath: [],
      matchedDecision: null, warning: 'Router API unavailable',
    })
  })

  it('retains diagnostic traces and signal errors on HTTP 503', async () => {
    const trace = [{ decision_name: 'formal_math_proof', state: 'unknown', root_trace: {
      node_type: 'NOT', state: 'unknown', children: [
        { node_type: 'leaf', signal_type: 'fact_check', signal_name: 'verify', signal_error: 'unavailable' },
      ],
    } }]
    mockResponse({
      query: 'prove this theorem', mode: 'dry-run', isAccurate: true,
      decisionError: 'decision unresolved', warning: 'decision unresolved',
      signalErrors: { 'fact_check:verify': 'unavailable' },
      appliedUnknownPolicies: { formal_math_proof: 'fail_request' },
      evalTrace: trace,
      evaluatedRules: [{
        decisionName: 'formal_math_proof', expression: 'NOT(fact_check:verify)',
        ruleOperator: 'NOT', state: 'unknown', isMatch: false, priority: 100,
        matchedCount: 0, totalCount: 1,
      }],
    }, 503)

    const result = await testQueryDryRun('prove this theorem')
    expect(result.isAccurate).toBe(false)
    expect(result.mode).toBe('dry-run')
    expect(result.evalTrace).toEqual(trace)
    expect(result.signalErrors).toEqual({ 'fact_check:verify': 'unavailable' })
    expect(result.appliedUnknownPolicies).toEqual({ formal_math_proof: 'fail_request' })
    expect(result.decisionError).toBe('decision unresolved')
    expect(result.warning).toBe('decision unresolved')
    expect(result.evaluatedRules?.[0]).toMatchObject({
      condition: 'NOT(fact_check:verify)', state: 'unknown', result: false,
    })
  })

  it('preserves concrete selection separately from candidates and routes the selected entrypoint', async () => {
    const fetchMock = mockResponse({
      query: 'hello', mode: 'dry-run', isAccurate: true,
      matchedModels: ['backend-b'], selectedModel: 'backend-b',
      recommendedModels: ['backend-a', 'backend-b'],
      selectionStatus: 'selected', selectionMethod: 'static',
      evaluatedRules: [{
        decisionName: 'balance', expression: 'AND(domain:math, NOT(OR(keyword:a, keyword:b)))',
        ruleOperator: 'AND', state: 'true', isMatch: true, priority: 100,
        matchedCount: 2, totalCount: 2,
      }],
    })

    const result = await testQueryDryRun('hello', 'vllm-sr/balance')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ query: 'hello', mode: 'dry-run', model: 'vllm-sr/balance' })
    expect(result.selectedModel).toBe('backend-b')
    expect(result.matchedModels).toEqual(['backend-b'])
    expect(result.recommendedModels).toEqual(['backend-a', 'backend-b'])
    expect(result.evaluatedRules?.[0].condition).toBe('AND(domain:math, NOT(OR(keyword:a, keyword:b)))')
  })

  it('reports non-JSON gateway errors without fabricating routing results', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('gateway timeout', { status: 504 })))
    await expect(testQueryDryRun('hello')).rejects.toThrow('Router preview failed (HTTP 504)')
  })

  it('rejects unrelated JSON error payloads instead of treating them as preview results', async () => {
    mockResponse({ error: 'not authenticated' }, 401)
    await expect(testQueryDryRun('hello')).rejects.toThrow('Router preview failed (HTTP 401)')
  })
})
