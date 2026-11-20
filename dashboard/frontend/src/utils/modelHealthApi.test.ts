import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  fetchModelHealth,
  ModelHealthApiError,
  modelHealthIssue,
  parseModelHealthSnapshot,
} from './modelHealthApi'

afterEach(() => {
  vi.unstubAllGlobals()
})

/** A fetch that never answers, and rejects like the browser once its signal aborts. */
const hangingFetch = () =>
  vi.fn(
    (_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () =>
          reject(new DOMException('The operation was aborted.', 'AbortError')),
        )
      }),
  )

describe('model health API', () => {
  it('keeps valid entries and drops malformed ones', () => {
    const snapshot = parseModelHealthSnapshot({
      checkedAt: '2026-11-18T10:00:00Z',
      intervalSeconds: 30,
      models: [
        {
          model: 'qwen3-8b',
          status: 'healthy',
          latencyMs: 412,
          p95Ms: 640,
          recentMs: [400, 'slow', 640, -1],
          endpoints: [
            { name: 'primary', status: 'healthy', latencyMs: 412, httpStatus: 200 },
            { name: 'broken' },
          ],
        },
        { model: 'mystery', status: 'flaky' },
        { status: 'down' },
      ],
    })
    expect(snapshot.models).toHaveLength(1)
    expect(snapshot.models[0].recentMs).toEqual([400, 640])
    expect(snapshot.models[0].endpoints).toEqual([
      { name: 'primary', status: 'healthy', latencyMs: 412, httpStatus: 200, reason: undefined },
    ])
  })

  it('rejects an answer without models and falls back to a 30 s interval', () => {
    expect(() => parseModelHealthSnapshot({ checkedAt: 'now' })).toThrow(ModelHealthApiError)
    expect(parseModelHealthSnapshot({ models: [], intervalSeconds: 1 }).intervalSeconds).toBe(30)
  })

  it('asks for the named models and a refresh, and reports the server message', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ intervalSeconds: 30, models: [] })))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ error: 'model_not_found', message: 'Model "x" is gone.' }), {
          status: 404,
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await fetchModelHealth({ models: ['a b', 'c'], refresh: true })
    expect(fetchMock.mock.calls[0][0]).toBe('/api/models/health?model=a+b&model=c&refresh=1')

    const failure = await fetchModelHealth().catch((error: unknown) => error)
    expect(fetchMock.mock.calls[1][0]).toBe('/api/models/health')
    expect(failure).toBeInstanceOf(ModelHealthApiError)
    expect(failure).toMatchObject({
      status: 404,
      code: 'model_not_found',
      message: 'Model "x" is gone.',
    })
  })

  it('reports a network failure, a timeout and an unreadable answer as answers that never came', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(fetchModelHealth()).rejects.toMatchObject({
      status: 0,
      code: 'network',
      message: 'The Dashboard could not be reached.',
    })

    vi.stubGlobal('fetch', hangingFetch())
    await expect(fetchModelHealth({ timeoutMs: 5 })).rejects.toMatchObject({
      status: 0,
      code: 'timeout',
    })

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>')))
    await expect(fetchModelHealth()).rejects.toMatchObject({
      status: 502,
      code: 'invalid_health_contract',
    })
  })

  it('rejects with the abort itself when the caller cancels', async () => {
    vi.stubGlobal('fetch', hangingFetch())
    const controller = new AbortController()
    const request = fetchModelHealth({ signal: controller.signal })
    controller.abort()
    const failure = await request.catch((error: unknown) => error)
    expect(failure).not.toBeInstanceOf(ModelHealthApiError)
    expect(failure).toMatchObject({ name: 'AbortError' })
  })

  it('tells failures that repeat from failures that may pass', () => {
    expect(modelHealthIssue(new ModelHealthApiError('signed out', 401))).toBe('forbidden')
    expect(modelHealthIssue(new ModelHealthApiError('forbidden', 403))).toBe('forbidden')
    for (const status of [404, 405, 501]) {
      expect(modelHealthIssue(new ModelHealthApiError('missing', status))).toBe('unsupported')
    }
    for (const status of [0, 500, 502, 503, 504]) {
      expect(modelHealthIssue(new ModelHealthApiError('passing', status))).toBe('unreachable')
    }
    expect(modelHealthIssue(new TypeError('Failed to fetch'))).toBe('unreachable')
  })
})
