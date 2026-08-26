import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  AUTH_SESSION_REVALIDATION_INTERVAL_MS,
  subscribeToAuthSessionRevalidation,
} from './authSessionRevalidation'

function createBrowserHarness() {
  let focus: EventListener | undefined
  let visibilityChange: EventListener | undefined
  let interval: TimerHandler | undefined

  const browserWindow = {
    addEventListener: vi.fn((type: string, listener: EventListener) => {
      if (type === 'focus') focus = listener
    }),
    removeEventListener: vi.fn(),
    setInterval: vi.fn((handler: TimerHandler) => {
      interval = handler
      return 17
    }),
    clearInterval: vi.fn(),
  } as unknown as Window
  const browserDocument = {
    hidden: false,
    addEventListener: vi.fn((type: string, listener: EventListener) => {
      if (type === 'visibilitychange') visibilityChange = listener
    }),
    removeEventListener: vi.fn(),
  } as unknown as Document

  return {
    browserDocument,
    browserWindow,
    focus: () => focus?.(new Event('focus')),
    interval: () => {
      if (typeof interval === 'function') interval()
    },
    visibilityChange: () => visibilityChange?.(new Event('visibilitychange')),
  }
}

describe('Dashboard session revalidation', () => {
  afterEach(() => vi.useRealTimers())

  it('revalidates visible sessions on focus and a bounded interval', async () => {
    const revalidate = vi.fn(async () => undefined)
    const browser = createBrowserHarness()
    const unsubscribe = subscribeToAuthSessionRevalidation(
      revalidate,
      browser.browserWindow,
      browser.browserDocument,
    )

    browser.focus()
    await Promise.resolve()
    expect(revalidate).toHaveBeenCalledTimes(1)

    browser.interval()
    await Promise.resolve()
    expect(revalidate).toHaveBeenCalledTimes(2)

    unsubscribe()
    expect(browser.browserWindow.clearInterval).toHaveBeenCalledWith(17)
    expect(browser.browserWindow.setInterval).toHaveBeenCalledWith(
      expect.any(Function),
      AUTH_SESSION_REVALIDATION_INTERVAL_MS,
    )
  })

  it('does not overlap authorization snapshots', async () => {
    let release: (() => void) | undefined
    const revalidate = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          release = resolve
        }),
    )
    const browser = createBrowserHarness()
    const unsubscribe = subscribeToAuthSessionRevalidation(
      revalidate,
      browser.browserWindow,
      browser.browserDocument,
    )

    browser.focus()
    browser.focus()
    expect(revalidate).toHaveBeenCalledTimes(1)
    release?.()
    await vi.waitFor(() => expect(revalidate).toHaveBeenCalledTimes(1))

    unsubscribe()
  })
})
