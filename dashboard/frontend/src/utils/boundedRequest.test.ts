import { afterEach, describe, expect, it, vi } from 'vitest'
import { withRequestTimeout } from './boundedRequest'
afterEach(() => vi.useRealTimers())
describe('bounded page requests', () => {
  it('bounds body decoding and releases the deadline before a later retry', async () => {
    vi.useFakeTimers()
    const readBody = vi.fn(
      (signal: AbortSignal) =>
        new Promise<string>((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(signal.reason))
        }),
    )
    const pending = expect(withRequestTimeout(readBody)).rejects.toThrow('timed out')
    await vi.advanceTimersByTimeAsync(15_000)
    await pending
    expect(vi.getTimerCount()).toBe(0)
    await expect(withRequestTimeout(async () => 'recovered')).resolves.toBe('recovered')
    expect(vi.getTimerCount()).toBe(0)
  })
  it('preserves caller cancellation instead of reporting a network timeout', async () => {
    const caller = new AbortController()
    let child: AbortSignal | undefined
    const pending = expect(
      withRequestTimeout((signal) => {
        child = signal
        return new Promise((_resolve, reject) =>
          signal.addEventListener('abort', () => reject(signal.reason)),
        )
      }, caller.signal),
    ).rejects.toMatchObject({ name: 'AbortError' })
    caller.abort()
    await pending
    expect(child?.aborted).toBe(true)
  })
  it('does not begin work for an already cancelled route', async () => {
    const caller = new AbortController()
    caller.abort()
    const load = vi.fn()
    await expect(withRequestTimeout(load, caller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    })
    expect(load).not.toHaveBeenCalled()
  })
})
