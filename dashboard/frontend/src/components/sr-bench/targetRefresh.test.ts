import { afterEach, describe, expect, it, vi } from 'vitest'
import { EMPTY_TARGETS_POLL_MS, watchTargets } from './targetRefresh'

afterEach(() => {
  vi.useRealTimers()
})

describe('watchTargets', () => {
  it('refetches on focus and polls while no target is registered', () => {
    vi.useFakeTimers()
    const source = new EventTarget()
    const refetch = vi.fn()
    const stop = watchTargets(source, refetch, EMPTY_TARGETS_POLL_MS)

    source.dispatchEvent(new Event('focus'))
    expect(refetch).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(EMPTY_TARGETS_POLL_MS * 2)
    expect(refetch).toHaveBeenCalledTimes(3)

    stop()
    source.dispatchEvent(new Event('focus'))
    vi.advanceTimersByTime(EMPTY_TARGETS_POLL_MS * 2)
    expect(refetch).toHaveBeenCalledTimes(3)
  })

  it('only refetches on focus once targets exist', () => {
    vi.useFakeTimers()
    const source = new EventTarget()
    const refetch = vi.fn()
    const stop = watchTargets(source, refetch, null)

    vi.advanceTimersByTime(EMPTY_TARGETS_POLL_MS * 5)
    expect(refetch).not.toHaveBeenCalled()
    source.dispatchEvent(new Event('focus'))
    expect(refetch).toHaveBeenCalledTimes(1)
    stop()
  })
})
