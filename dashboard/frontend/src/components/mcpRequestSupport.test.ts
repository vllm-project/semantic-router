import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  createLatestMCPRequestRunner,
  fetchMCPToolsDatabase,
  getMCPRequestErrorMessage,
  isMCPAbortError,
} from './mcpRequestSupport'

const failing = (status: number, statusText: string, body: string) =>
  new Response(body, { status, statusText })

const jsonError = (status: number, statusText: string, body: unknown) =>
  new Response(JSON.stringify(body), {
    status,
    statusText,
    headers: { 'Content-Type': 'application/json' },
  })

describe('MCP request support', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('aborts the previous request and drops a stale response even when the request ignores abort', async () => {
    const runner = createLatestMCPRequestRunner()
    const firstSuccess = vi.fn()
    const secondSuccess = vi.fn()
    let firstSignal: AbortSignal | undefined
    let resolveFirst: ((value: string) => void) | undefined

    const first = runner.run(
      (signal) => {
        firstSignal = signal
        return new Promise<string>((resolve) => {
          resolveFirst = resolve
        })
      },
      { onSuccess: firstSuccess },
    )
    const second = runner.run(async () => 'fresh', { onSuccess: secondSuccess })

    await second
    resolveFirst?.('stale')
    await first

    expect(firstSignal?.aborted).toBe(true)
    expect(firstSuccess).not.toHaveBeenCalled()
    expect(secondSuccess).toHaveBeenCalledWith('fresh')
    expect(runner.isInFlight()).toBe(false)
  })

  it('suppresses abort failures while surfacing actionable request errors', async () => {
    const runner = createLatestMCPRequestRunner()
    const onError = vi.fn()
    const controller = new AbortController()
    controller.abort()

    await runner.run(
      async (signal) => {
        if (signal.aborted) throw new DOMException('Aborted', 'AbortError')
        return 'unexpected'
      },
      { onError },
      controller.signal,
    )
    expect(onError).not.toHaveBeenCalled()

    await runner.run(
      async () => {
        throw new Error('gateway unavailable')
      },
      { onError },
    )
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'gateway unavailable' }),
    )
    expect(isMCPAbortError({ name: 'AbortError' })).toBe(true)
    expect(getMCPRequestErrorMessage(null, 'fallback')).toBe('fallback')
  })

  it('loads tools-db with an abort signal and keeps successful loads unchanged', async () => {
    const signal = new AbortController().signal
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => [],
    }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchMCPToolsDatabase(signal)).resolves.toEqual([])
    expect(fetchMock).toHaveBeenCalledWith('/api/tools-db', { signal })
  })

  it('surfaces the sanitized tools database reason instead of only the status', async () => {
    const signal = new AbortController().signal
    const cases = [
      {
        response: failing(404, 'Not Found', 'Tools database not found\n'),
        reason: 'Tools database not found',
      },
      {
        response: failing(500, 'Internal Server Error', 'Failed to read tools database\n'),
        reason: 'Failed to read tools database',
      },
      {
        response: failing(500, 'Internal Server Error', 'Failed to parse tools database\n'),
        reason: 'Failed to parse tools database',
      },
      {
        response: jsonError(500, 'Internal Server Error', {
          message: 'Failed to parse tools database',
        }),
        reason: 'Failed to parse tools database',
      },
    ]

    for (const { response, reason } of cases) {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => response),
      )
      await expect(fetchMCPToolsDatabase(signal)).rejects.toThrow(new Error(reason))
    }
  })

  it('falls back to the status-based message when the failure body is unusable', async () => {
    const signal = new AbortController().signal
    const fallback = 'Failed to load the tools database (500 Internal Server Error)'
    const cases = [
      failing(500, 'Internal Server Error', ''),
      failing(500, 'Internal Server Error', '  \n\t'),
      jsonError(500, 'Internal Server Error', { error: { code: 500 } }),
      failing(500, 'Internal Server Error', '<html><body>Bad gateway</body></html>'),
    ]

    for (const response of cases) {
      vi.stubGlobal(
        'fetch',
        vi.fn(async () => response),
      )
      await expect(fetchMCPToolsDatabase(signal)).rejects.toThrow(new Error(fallback))
    }
  })

  it('falls back to the status-based message when the failure body cannot be read', async () => {
    const signal = new AbortController().signal
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 502,
        statusText: 'Bad Gateway',
        text: async () => {
          throw new Error('stream closed')
        },
      })),
    )

    await expect(fetchMCPToolsDatabase(signal)).rejects.toThrow(
      new Error('Failed to load the tools database (502 Bad Gateway)'),
    )
  })
})
