import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createDashboardObservation } from './dashboardObservation'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: Error) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
const settle = () => vi.advanceTimersByTimeAsync(0)

describe('shared dashboard observations', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(10_000)
  })
  afterEach(() => vi.useRealTimers())

  it('shares a slow read across consumers and waits for completion before polling', async () => {
    const pending = deferred<string>()
    const load = vi.fn(() => pending.promise)
    const resource = createDashboardObservation(load, { pollMs: 1000 })
    const stopA = resource.subscribe(vi.fn())
    const stopB = resource.subscribe(vi.fn())
    const first = resource.refresh()
    expect(resource.refresh()).toBe(first)
    await vi.advanceTimersByTimeAsync(25_000)
    expect(load).toHaveBeenCalledTimes(1)
    pending.resolve('ready')
    await first
    await vi.advanceTimersByTimeAsync(999)
    expect(load).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(load).toHaveBeenCalledTimes(2)
    stopA()
    stopB()
  })

  it('aborts only after the last consumer leaves and discards the late response', async () => {
    const pending = deferred<string>()
    let signal!: AbortSignal
    const resource = createDashboardObservation((next) => {
      signal = next
      return pending.promise
    })
    const stopA = resource.subscribe(vi.fn())
    const stopB = resource.subscribe(vi.fn())
    await settle()
    stopA()
    expect(signal.aborted).toBe(false)
    stopB()
    expect(signal.aborted).toBe(true)
    pending.resolve('late')
    await settle()
    expect(resource.getSnapshot().data).toBeNull()
  })

  it('survives StrictMode unsubscribe/resubscribe without starting the abandoned read', async () => {
    const load = vi.fn(async () => 'mounted')
    const resource = createDashboardObservation(load)
    const stop = resource.subscribe(vi.fn())
    stop()
    const stopAgain = resource.subscribe(vi.fn())
    await settle()
    expect(load).toHaveBeenCalledTimes(1)
    expect(resource.getSnapshot()).toMatchObject({ data: 'mounted', refreshing: false })
    stopAgain()
  })

  it('publishes independent runtime facts before a slow configuration read finishes', async () => {
    const configRead = deferred<string>()
    const config = createDashboardObservation(() => configRead.promise)
    const inventory = createDashboardObservation(async () => ({ ready_replicas: 2 }))
    const stopConfig = config.subscribe(vi.fn())
    const stopInventory = inventory.subscribe(vi.fn())
    await settle()
    expect(config.getSnapshot().loading).toBe(true)
    expect(inventory.getSnapshot()).toMatchObject({ loading: false, data: { ready_replicas: 2 } })
    stopConfig()
    stopInventory()
    configRead.resolve('late')
  })

  it('reuses fresh data when navigating back and marks expired observations during refresh', async () => {
    const load = vi.fn(async () => 'first')
    const resource = createDashboardObservation(load, { freshMs: 5000, pollMs: 0 })
    const stop = resource.subscribe(vi.fn())
    await settle()
    stop()
    const stopAgain = resource.subscribe(vi.fn())
    await settle()
    expect(load).toHaveBeenCalledTimes(1)
    stopAgain()
    await vi.advanceTimersByTimeAsync(6000)
    const pending = deferred<string>()
    load.mockImplementationOnce(() => pending.promise)
    const stopExpired = resource.subscribe(vi.fn())
    await settle()
    expect(resource.getSnapshot()).toMatchObject({ data: 'first', stale: true, refreshing: true })
    pending.resolve('current')
    await settle()
    expect(resource.getSnapshot()).toMatchObject({ data: 'current', stale: false })
    stopExpired()
  })

  it('retains the last observation with an explicit error instead of erasing the page', async () => {
    const load = vi.fn(async () => ({ ready: true }))
    const resource = createDashboardObservation(load, { pollMs: 0 })
    const stop = resource.subscribe(vi.fn())
    await settle()
    load.mockRejectedValueOnce(new Error('slow network'))
    await resource.refresh()
    expect(resource.getSnapshot()).toMatchObject({
      data: { ready: true },
      stale: true,
      loading: false,
      error: 'slow network',
    })
    await resource.refresh()
    expect(resource.getSnapshot()).toMatchObject({ stale: false, error: null })
    stop()
  })

  it('invalidates in-flight configuration reads after a write without accepting an old response', async () => {
    const old = deferred<string>()
    const next = deferred<string>()
    const load = vi.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise)
    const resource = createDashboardObservation<string>(load)
    const stop = resource.subscribe(vi.fn())
    await settle()
    resource.invalidate()
    await settle()
    next.resolve('new generation')
    await settle()
    old.resolve('old generation')
    await settle()
    expect(resource.getSnapshot().data).toBe('new generation')
    stop()
  })

  it('clears cached private data and cancels work when the authenticated session changes', async () => {
    const pending = deferred<string>()
    const load = vi.fn().mockResolvedValueOnce('private').mockReturnValueOnce(pending.promise)
    const resource = createDashboardObservation<string>(load)
    const stop = resource.subscribe(vi.fn())
    await settle()
    void resource.refresh()
    await settle()
    resource.clear()
    pending.resolve('old session')
    await settle()
    expect(resource.getSnapshot()).toMatchObject({ data: null, error: null, updatedAt: null })
    stop()
  })

  it('backs off repeated failures and recovers automatically after the third failure', async () => {
    const load = vi.fn().mockRejectedValue(new Error('offline'))
    const resource = createDashboardObservation<string>(load, { pollMs: 0 })
    const stop = resource.subscribe(vi.fn())
    await vi.advanceTimersByTimeAsync(14_000)
    expect(load).toHaveBeenCalledTimes(4)
    load.mockResolvedValue('recovered')
    await vi.advanceTimersByTimeAsync(16_000)
    expect(resource.getSnapshot()).toMatchObject({ data: 'recovered', error: null })
    stop()
  })

  it('caps the retry delay so a visible page can still recover after a long outage', async () => {
    const load = vi.fn().mockRejectedValue(new Error('offline'))
    const resource = createDashboardObservation<string>(load, { pollMs: 0 })
    const stop = resource.subscribe(vi.fn())
    await vi.advanceTimersByTimeAsync(122_000)
    expect(load).toHaveBeenCalledTimes(7)
    load.mockResolvedValue('online')
    await vi.advanceTimersByTimeAsync(60_000)
    expect(resource.getSnapshot().data).toBe('online')
    stop()
  })

  it('skips polling hidden tabs without aborting work already in progress', async () => {
    let hidden = false
    const load = vi.fn(async () => 'ready')
    const resource = createDashboardObservation(load, { pollMs: 1000, isHidden: () => hidden })
    const stop = resource.subscribe(vi.fn())
    await settle()
    hidden = true
    await vi.advanceTimersByTimeAsync(10_000)
    expect(load).toHaveBeenCalledTimes(1)
    hidden = false
    await vi.advanceTimersByTimeAsync(1000)
    expect(load).toHaveBeenCalledTimes(2)
    stop()
  })
})
