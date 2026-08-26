import { afterEach, describe, expect, it, vi } from 'vitest'
import { MANAGEMENT_API_MEDIA_TYPE } from '../generated/managementApiContract'

import { setManagementNamespace } from './managementApiContract'
import { inferenceAccessApi } from './inferenceAccessApi'

const response = (payload: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  headers: new Headers({ 'Content-Type': MANAGEMENT_API_MEDIA_TYPE }),
  json: async () => payload,
})

afterEach(() => {
  setManagementNamespace(null)
  vi.unstubAllGlobals()
})

describe('Router Management access client', () => {
  it('resolves an Agent response to its scoped request log by exact request ID', async () => {
    const requestId = '11111111-1111-4111-8111-111111111111'
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: [
          {
            admissionId: 'admission-1',
            eventId: '22222222-2222-4222-8222-222222222222',
            externalRequestId: requestId,
            occurredAt: '2026-08-25T00:00:00Z',
            completedAt: '2026-08-25T00:00:01Z',
            protocol: 'openai_chat_v1',
            path: '/v1/chat/completions',
            statusCode: 200,
            usageState: 'known_actual',
            inputTokens: '8',
            outputTokens: '3',
            latencyMilliseconds: 1000,
            stream: true,
            toolCall: false,
            costs: [],
            models: [],
          },
        ],
        page: { hasMore: false, pageSize: 10 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      inferenceAccessApi.requestLogs({ q: ` ${requestId} `, limit: 10 }),
    ).resolves.toMatchObject({
      items: [expect.objectContaining({ requestId })],
    })
    const requestedURL = new URL(String(fetchMock.mock.calls[0][0]), 'http://dashboard.local')
    expect(requestedURL.pathname).toBe('/api/router/management/v1/request-logs')
    expect(requestedURL.searchParams.get('requestId')).toBe(requestId)
    expect(requestedURL.searchParams.get('pageSize')).toBe('10')
    expect(requestedURL.searchParams.has('grain')).toBe(false)
  })

  it('preserves typed route, quota, and dispatch evidence on request detail', async () => {
    const namespaceID = '11111111-1111-4111-8111-111111111111'
    setManagementNamespace(namespaceID)
    const fetchMock = vi.fn().mockResolvedValue(
      response({
        data: {
          request: {
            admissionId: 'admission-1',
            eventId: '22222222-2222-4222-8222-222222222222',
            occurredAt: '2026-08-25T00:00:00Z',
            completedAt: '2026-08-25T00:00:01Z',
            protocol: 'openai_chat_v1',
            path: '/v1/chat/completions',
            statusCode: 200,
            usageState: 'known_actual',
            inputTokens: '8',
            outputTokens: '3',
            latencyMilliseconds: 1000,
            stream: true,
            toolCall: false,
            decisionId: 'decision-simple',
            decisionName: 'Simple',
            decisionTier: 1,
            models: [{ id: 'model-fast', name: 'local/fast', revision: 7 }],
            costs: [],
          },
          routing: { entrypointName: 'vllm-sr/balance', recipeName: 'Balance' },
          quotaReceipts: [{ ruleId: 'rpm', metric: 'requests', amount: '1' }],
          dispatches: [
            {
              dispatchId: 'dispatch-1',
              ordinal: 0,
              dispatchType: 'primary',
              modelId: 'model-fast',
              inputTokens: '8',
              cacheReadTokens: '0',
              cacheWriteTokens: '0',
              outputTokens: '3',
              usageState: 'known_actual',
              cost: {
                currency: 'USD',
                knownAmount: '0.1',
                completeness: 'complete',
                knownDispatches: '1',
                incompleteDispatches: '0',
              },
              evidenceDigest: 'a'.repeat(64),
              startedAt: '2026-08-25T00:00:00Z',
              completedAt: '2026-08-25T00:00:01Z',
              attempts: [],
            },
          ],
        },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      inferenceAccessApi.requestLog(`${namespaceID}:admission-1`),
    ).resolves.toMatchObject({
      decisionName: 'Simple',
      model: 'local/fast',
      models: [{ name: 'local/fast' }],
      routing: { recipeName: 'Balance' },
      quotaReceipts: [{ metric: 'requests' }],
      dispatches: [{ dispatchId: 'dispatch-1' }],
    })
  })

  it('uses the exact key usage summary and scopes every supporting series query', async () => {
    const totals = {
      requests: '12',
      successfulRequests: '12',
      inputTokens: '100',
      outputTokens: '40',
      totalTokens: '140',
      incompleteDispatches: '0',
      completeness: 'complete',
      latency: {
        sampleCount: '12',
        totalMilliseconds: '4920',
        averageMilliseconds: 410,
        p50Milliseconds: 250,
        p95Milliseconds: 750,
        p99Milliseconds: 1000,
        percentilesAreEstimated: true,
      },
      ttft: {
        sampleCount: '12',
        totalMilliseconds: '1104',
        averageMilliseconds: 92,
        p50Milliseconds: 64,
        p95Milliseconds: 125,
        p99Milliseconds: 250,
        percentilesAreEstimated: true,
      },
      costs: [
        {
          currency: 'USD',
          knownAmount: '0.0014',
          completeness: 'complete',
          knownDispatches: '12',
          incompleteDispatches: '0',
        },
      ],
    }
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api-keys/key-1/usage')) {
        return Promise.resolve(response({ totals, grain: 'minute', final: true }))
      }
      if (url.includes('/usage/series')) {
        return Promise.resolve(response({ points: [], grain: 'minute', final: true }))
      }
      return Promise.resolve(
        response({ dimension: 'api_key', rows: [], grain: 'minute', final: true }),
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    const usage = await inferenceAccessApi.keyUsage('key-1', {
      from: '2026-08-23T00:00:00Z',
      granularity: 'minute',
      model: 'internal-model-id',
    })

    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    const exact = urls.find((url) => url.includes('/api-keys/key-1/usage'))
    expect(exact).not.toContain('apiKeyId=')
    expect(
      urls.filter((url) => url.includes('/usage/series') || url.includes('/usage/breakdowns')),
    ).toHaveLength(7)
    expect(
      urls
        .filter((url) => url.includes('/usage/series') || url.includes('/usage/breakdowns'))
        .every((url) => url.includes('apiKeyId=key-1')),
    ).toBe(true)
    expect(urls.some((url) => url.includes('dimension=logical_model'))).toBe(false)
    expect(urls.some((url) => url.includes('logicalModelId='))).toBe(false)
    expect(usage.costs[0]).toMatchObject({ currency: 'USD', knownAmount: '0.0014' })
  })

  it('keeps required usage available when an optional internal breakdown is forbidden', async () => {
    const timing = {
      sampleCount: '0',
      totalMilliseconds: '0',
      averageMilliseconds: 0,
      p50Milliseconds: 0,
      p95Milliseconds: 0,
      p99Milliseconds: 0,
      percentilesAreEstimated: false,
    }
    const totals = {
      requests: '0',
      successfulRequests: '0',
      inputTokens: '0',
      outputTokens: '0',
      totalTokens: '0',
      incompleteDispatches: '0',
      completeness: 'complete',
      costs: [],
      latency: timing,
      ttft: timing,
    }
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('dimension=logical_model')) {
        return Promise.resolve(
          response(
            { error: { code: 'forbidden', message: 'Internal dimensions are hidden.' } },
            403,
          ),
        )
      }
      if (url.includes('/usage/series')) {
        return Promise.resolve(response({ points: [], grain: 'hour', final: true }))
      }
      if (url.includes('/usage/breakdowns')) {
        return Promise.resolve(
          response({ dimension: 'api_key', rows: [], grain: 'hour', final: true }),
        )
      }
      return Promise.resolve(response({ totals, grain: 'hour', final: true }))
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      inferenceAccessApi.keyUsage('key-1', {}, { internalDimensions: true }),
    ).resolves.toMatchObject({ requests: 0, final: true, byModel: [] })
    expect(
      fetchMock.mock.calls
        .map(([input]) => String(input))
        .some((url) => url.includes('dimension=logical_model')),
    ).toBe(true)
  })

  it('builds overview from bounded statistics plus the existing usage ledger queries', async () => {
    const timing = {
      sampleCount: '0',
      totalMilliseconds: '0',
      averageMilliseconds: 0,
      p50Milliseconds: 0,
      p95Milliseconds: 0,
      p99Milliseconds: 0,
      percentilesAreEstimated: false,
    }
    const totals = {
      requests: '12',
      successfulRequests: '11',
      inputTokens: '100',
      outputTokens: '40',
      totalTokens: '140',
      incompleteDispatches: '0',
      completeness: 'complete',
      costs: [],
      latency: timing,
      ttft: timing,
    }
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/statistics')) {
        return Promise.resolve(
          response({
            asOf: '2026-08-23T00:00:00Z',
            expiringBefore: '2026-09-22T00:00:00Z',
            teams: '3',
            activeApiKeys: '10000',
            expiringApiKeys: '8',
            accessPolicies: '4',
            activeRatePolicies: '5',
          }),
        )
      }
      if (url.includes('/usage/series')) {
        return Promise.resolve(response({ points: [], grain: 'hour', final: true }))
      }
      if (url.includes('/usage/breakdowns')) {
        return Promise.resolve(
          response({ dimension: 'api_key', rows: [], grain: 'hour', final: true }),
        )
      }
      return Promise.resolve(response({ totals, grain: 'hour', final: true }))
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(inferenceAccessApi.overview()).resolves.toMatchObject({
      users: null,
      teams: '3',
      activeKeys: '10000',
      requestsToday: 12,
      tokensToday: 140,
    })
    const urls = fetchMock.mock.calls.map(([input]) => String(input))
    expect(urls).toHaveLength(9)
    expect(
      urls.some((url) =>
        /\/(users|teams|api-keys|access-policies|rate-limit-policies)(\?|$)/.test(url),
      ),
    ).toBe(false)
  })
})
