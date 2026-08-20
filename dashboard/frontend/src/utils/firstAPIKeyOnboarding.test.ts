import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  isFirstAPIKeyOnboardingPending,
  markFirstAPIKeyOnboardingHandled,
  markFirstAPIKeyOnboardingPending,
} from './firstAPIKeyOnboarding'

afterEach(() => vi.unstubAllGlobals())

describe('invitation API key onboarding', () => {
  it('persists a per-user first-login prompt until it is handled', () => {
    const values = new Map<string, string>()
    vi.stubGlobal('window', {
      localStorage: {
        getItem: (key: string) => values.get(key) ?? null,
        setItem: (key: string, value: string) => values.set(key, value),
      },
    })

    markFirstAPIKeyOnboardingPending('user-a')

    expect(isFirstAPIKeyOnboardingPending('user-a')).toBe(true)
    expect(isFirstAPIKeyOnboardingPending('user-b')).toBe(false)

    markFirstAPIKeyOnboardingHandled('user-a')
    expect(isFirstAPIKeyOnboardingPending('user-a')).toBe(false)
  })
})
