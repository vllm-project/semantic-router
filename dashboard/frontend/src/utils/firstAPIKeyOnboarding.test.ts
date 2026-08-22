import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ensureFirstAPIKey,
  handoffFirstAPIKey,
  isFirstAPIKeyOnboardingPending,
  markFirstAPIKeyOnboardingHandled,
  markFirstAPIKeyOnboardingPending,
  takeFirstAPIKeyHandoff,
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

  it('creates exactly one key named after the invited user', async () => {
    const create = vi.fn().mockResolvedValue({ id: 'key-a', name: 'Andy Luo' })

    const key = await ensureFirstAPIKey('  Andy Luo  ', {
      list: vi.fn().mockResolvedValue({ items: [], total: 0 }),
      create,
    })

    expect(create).toHaveBeenCalledWith('Andy Luo')
    expect(key).toMatchObject({ id: 'key-a', name: 'Andy Luo' })
  })

  it('reuses an existing key without creating another one', async () => {
    const create = vi.fn()
    const existing = { id: 'key-a', name: 'Andy Luo' }

    const key = await ensureFirstAPIKey('Andy Luo', {
      list: vi.fn().mockResolvedValue({ items: [existing], total: 1 }),
      create,
    })

    expect(create).not.toHaveBeenCalled()
    expect(key).toBe(existing)
  })

  it('turns a concurrent create conflict into the winning key', async () => {
    const winningKey = { id: 'key-b', name: 'Andy Luo' }
    const list = vi
      .fn()
      .mockResolvedValueOnce({ items: [], total: 0 })
      .mockResolvedValueOnce({ items: [winningKey], total: 1 })

    const key = await ensureFirstAPIKey('Andy Luo', {
      list,
      create: vi.fn().mockRejectedValue(new Error('you already have an API key')),
    })

    expect(key).toBe(winningKey)
    expect(list).toHaveBeenCalledTimes(2)
  })

  it('hands the created secret across routes exactly once', () => {
    const created = {
      id: 'key-a',
      name: 'Andy Luo',
      secret: 'vsr_secret',
    } as Parameters<typeof handoffFirstAPIKey>[0]

    handoffFirstAPIKey(created)

    expect(takeFirstAPIKeyHandoff()).toBe(created)
    expect(takeFirstAPIKeyHandoff()).toBeNull()
  })
})
