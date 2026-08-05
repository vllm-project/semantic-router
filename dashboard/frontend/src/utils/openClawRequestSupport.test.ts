import { describe, expect, it, vi } from 'vitest'

import { createLatestOpenClawRequest, fetchOpenClawJSON } from './openClawRequestSupport'

describe('OpenClaw request support', () => {
  it('aborts the previous request and only publishes the latest response', async () => {
    const request = createLatestOpenClawRequest()
    const firstSuccess = vi.fn()
    const secondSuccess = vi.fn()
    const first = request.run(
      (signal) =>
        new Promise<string>((_resolve, reject) => {
          signal.addEventListener(
            'abort',
            () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
            { once: true },
          )
        }),
      { onSuccess: firstSuccess },
    )
    const second = request.run(async () => 'current', { onSuccess: secondSuccess })

    await expect(first).resolves.toEqual({ status: 'aborted' })
    await expect(second).resolves.toEqual({ status: 'success', value: 'current' })
    expect(firstSuccess).not.toHaveBeenCalled()
    expect(secondSuccess).toHaveBeenCalledWith('current')
  })

  it('drops a stale response even when the underlying task ignores abort', async () => {
    const request = createLatestOpenClawRequest()
    let resolveRequest: (value: string) => void = () => undefined
    const onSuccess = vi.fn()
    const pending = request.run(
      () =>
        new Promise<string>((resolve) => {
          resolveRequest = resolve
        }),
      { onSuccess },
    )

    request.cancel()
    resolveRequest('stale')

    await expect(pending).resolves.toEqual({ status: 'aborted' })
    expect(onSuccess).not.toHaveBeenCalled()
    expect(request.isInFlight()).toBe(false)
  })

  it('forwards AbortSignal and preserves server error detail', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      expect(init?.signal).toBe(controller.signal)
      return new Response(JSON.stringify({ message: 'OpenClaw permission denied' }), {
        status: 403,
        statusText: 'Forbidden',
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchOpenClawJSON('/api/openclaw/teams', {}, controller.signal)).rejects.toThrow(
      'OpenClaw permission denied',
    )
    expect(fetchMock).toHaveBeenCalledOnce()
    vi.unstubAllGlobals()
  })
})
