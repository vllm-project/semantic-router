import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  createLatestMCPRequestRunner,
  fetchMCPToolsDatabase,
  getMCPRequestErrorMessage,
  isMCPAbortError,
} from './mcpRequestSupport'

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

  it('loads tools-db with an abort signal and rejects non-success responses', async () => {
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

    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
      })),
    )
    await expect(fetchMCPToolsDatabase(signal)).rejects.toThrow('503 Service Unavailable')
  })
})
