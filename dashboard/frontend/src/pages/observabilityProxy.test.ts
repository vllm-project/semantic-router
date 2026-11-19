import { describe, expect, it, vi } from 'vitest'

import {
  ObservabilityError,
  classifyObservabilityFailure,
  fetchObservabilityJson,
  isAbortError,
  observabilityErrorMessage,
} from './observabilityProxy'

const respond = (status: number, body: string, contentType: string) =>
  vi.fn(async () => new Response(body, { status, headers: { 'Content-Type': contentType } }))

describe('observability proxy failures', () => {
  it('reads the error message of Prometheus and Jaeger error bodies', () => {
    expect(
      observabilityErrorMessage({ status: 'error', errorType: 'bad_data', error: 'parse error' }),
    ).toBe('parse error')
    expect(
      observabilityErrorMessage({
        data: null,
        errors: [
          { code: 404, msg: 'trace not found' },
          { code: 500, msg: 'storage' },
        ],
      }),
    ).toBe('trace not found; storage')
    expect(observabilityErrorMessage({ status: 'success', data: {} })).toBeNull()
    expect(observabilityErrorMessage('<html>')).toBeNull()
  })

  it('tells an unconfigured service from an unreachable one and a rejected query', () => {
    const classify = (status: number, contentType: string, body: unknown = null) =>
      classifyObservabilityFailure('Prometheus', status, contentType, body).failure
    expect(classify(503, 'text/html; charset=utf-8')).toBe('unconfigured')
    expect(classify(502, 'text/plain')).toBe('unreachable')
    expect(classify(504, 'text/plain')).toBe('unreachable')
    expect(classify(503, 'text/plain')).toBe('unreachable')
    // Prometheus itself answers 503 with a JSON body when a query times out.
    expect(classify(503, 'application/json', { status: 'error', error: 'query timed out' })).toBe(
      'query',
    )
    expect(classify(400, 'application/json', { status: 'error', error: 'bad_data' })).toBe('query')
    expect(classify(401, 'application/json')).toBe('denied')
    expect(classify(403, 'text/plain')).toBe('denied')
    expect(classify(404, 'text/plain')).toBe('query')
    expect(classifyObservabilityFailure('Jaeger', 403, '', null).message).toContain(
      'Access to Jaeger was denied',
    )
  })

  it("keeps the message the Dashboard's own error body gives, and its kind by status", () => {
    const classify = (status: number) =>
      classifyObservabilityFailure('Jaeger', status, 'application/json', {
        error: 'jaeger_timeout',
        message: `Explained ${status}`,
      })
    const timeout = classify(504)
    expect([timeout.failure, timeout.message, timeout.status]).toEqual([
      'timeout',
      'Explained 504',
      504,
    ])
    expect(classify(503).failure).toBe('unconfigured')
    expect(classify(502).failure).toBe('unreachable')
    expect(classify(400).failure).toBe('query')
    expect(classify(400).message).toBe('Explained 400')
  })
})

describe('fetchObservabilityJson', () => {
  it('returns the JSON body of a successful read and asks for JSON', async () => {
    const fetcher = respond(200, '{"status":"success","data":{}}', 'application/json')
    await expect(
      fetchObservabilityJson('Prometheus', '/embedded/prometheus/api/v1/query?query=up', {
        fetcher,
      }),
    ).resolves.toEqual({ status: 'success', data: {} })
    const [, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit]
    expect(new Headers(init.headers).get('Accept')).toBe('application/json')
  })

  it('classifies the Dashboard page for an unconfigured service', async () => {
    const error = await fetchObservabilityJson('Jaeger', '/embedded/jaeger/api/services', {
      fetcher: respond(503, '<!DOCTYPE html><title>Jaeger Not Configured</title>', 'text/html'),
    }).catch((caught: unknown) => caught)
    expect(error).toBeInstanceOf(ObservabilityError)
    expect((error as ObservabilityError).failure).toBe('unconfigured')
    expect((error as ObservabilityError).status).toBe(503)
  })

  it('rejects a successful answer that is a page rather than data', async () => {
    const error = await fetchObservabilityJson('Grafana', '/embedded/grafana/api/health', {
      fetcher: respond(200, '<html></html>', 'text/html'),
    }).catch((caught: unknown) => caught)
    expect((error as ObservabilityError).failure).toBe('query')
    expect((error as ObservabilityError).message).toContain('instead of data')
  })

  it('rejects a successful JSON answer that does not parse as a page rather than data', async () => {
    const error = await fetchObservabilityJson('Jaeger', '/embedded/jaeger/api/services', {
      fetcher: respond(200, '{"data": [', 'application/json'),
    }).catch((caught: unknown) => caught)
    expect((error as ObservabilityError).failure).toBe('query')
    expect((error as ObservabilityError).message).toContain('instead of data')
  })

  it('classifies a failed answer by its status when its JSON does not parse', async () => {
    const error = await fetchObservabilityJson('Jaeger', '/embedded/jaeger/api/services', {
      fetcher: respond(502, '<html>Bad Gateway</html>', 'application/json'),
    }).catch((caught: unknown) => caught)
    expect((error as ObservabilityError).failure).toBe('unreachable')
    expect((error as ObservabilityError).status).toBe(502)
  })

  it('reports a network failure as unreachable', async () => {
    const error = await fetchObservabilityJson('Prometheus', '/embedded/prometheus/api/v1/query', {
      fetcher: vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    }).catch((caught: unknown) => caught)
    expect((error as ObservabilityError).failure).toBe('unreachable')
  })

  it('times out a stalled read, and lets a caller abort without an error of its own', async () => {
    const stalled = vi.fn(
      (_url: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () =>
            reject(new DOMException('The operation was aborted.', 'AbortError')),
          )
        }),
    )
    const timedOut = await fetchObservabilityJson('Prometheus', '/slow', {
      fetcher: stalled,
      timeoutMs: 5,
    }).catch((caught: unknown) => caught)
    expect((timedOut as ObservabilityError).failure).toBe('timeout')

    const controller = new AbortController()
    const pending = fetchObservabilityJson('Prometheus', '/slow', {
      fetcher: stalled,
      signal: controller.signal,
    }).catch((caught: unknown) => caught)
    controller.abort()
    expect(isAbortError(await pending)).toBe(true)
  })

  describe('a body that does not finish arriving', () => {
    // Headers and the start of a large answer arrive, then the stream either
    // stalls until the read is aborted, as fetch does, or drops.
    const partial = (ending: 'stall' | 'drop') =>
      vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
        const body = new ReadableStream<Uint8Array>({
          start(stream) {
            stream.enqueue(new TextEncoder().encode('{"data":[{"traceID":"4bf92f35","spans":['))
            if (ending === 'drop') stream.error(new TypeError('network error'))
            init?.signal?.addEventListener('abort', () =>
              stream.error(new DOMException('The operation was aborted.', 'AbortError')),
            )
          },
        })
        return new Response(body, { status: 200, headers: { 'Content-Type': 'application/json' } })
      })

    it('times out instead of calling the answer a page', async () => {
      const error = await fetchObservabilityJson('Jaeger', '/api/observability/traces/summaries', {
        fetcher: partial('stall'),
        timeoutMs: 5,
      }).catch((caught: unknown) => caught)
      expect((error as ObservabilityError).failure).toBe('timeout')
      expect((error as ObservabilityError).message).toBe(
        'Jaeger took too long to respond. Try again.',
      )
    })

    it('reports a dropped connection as unreachable', async () => {
      const error = await fetchObservabilityJson('Jaeger', '/api/observability/traces/summaries', {
        fetcher: partial('drop'),
      }).catch((caught: unknown) => caught)
      expect((error as ObservabilityError).failure).toBe('unreachable')
      expect((error as ObservabilityError).message).not.toContain('instead of data')
    })

    it('lets a caller abort it without an error of its own', async () => {
      const controller = new AbortController()
      const pending = fetchObservabilityJson('Jaeger', '/embedded/jaeger/api/traces/4bf92f35', {
        fetcher: partial('stall'),
        signal: controller.signal,
      }).catch((caught: unknown) => caught)
      await new Promise((resolve) => setTimeout(resolve, 0))
      controller.abort()
      expect(isAbortError(await pending)).toBe(true)
    })
  })
})
