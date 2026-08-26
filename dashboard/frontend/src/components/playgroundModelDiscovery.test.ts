import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  fetchPlaygroundModelPayload,
  PlaygroundModelDiscoveryTimeoutError,
} from './playgroundModelDiscovery'

describe('Playground model discovery', () => {
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('bounds a model request that never responds', async () => {
    vi.useFakeTimers()
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise<Response>(() => undefined)),
    )

    const request = fetchPlaygroundModelPayload(
      '/v1/models',
      new AbortController().signal,
      async () => 'delegated-secret',
      250,
    )
    const rejection = expect(request).rejects.toBeInstanceOf(PlaygroundModelDiscoveryTimeoutError)

    await vi.advanceTimersByTimeAsync(250)
    await rejection
  })

  it('preserves caller cancellation instead of presenting a timeout', async () => {
    const controller = new AbortController()
    vi.stubGlobal(
      'fetch',
      vi.fn(() => new Promise<Response>(() => undefined)),
    )
    const request = fetchPlaygroundModelPayload(
      '/v1/models',
      controller.signal,
      async () => 'delegated-secret',
      10_000,
    )

    controller.abort()

    await expect(request).rejects.toMatchObject({ name: 'AbortError' })
  })
})
