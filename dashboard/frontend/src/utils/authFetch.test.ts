import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  getStoredAuthToken,
  normalizeAuthToken,
  storeAuthToken,
  STORAGE_KEY,
} from './authFetch'

class MemoryStorage {
  private readonly values = new Map<string, string>()

  getItem(key: string): string | null {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value)
  }

  removeItem(key: string): void {
    this.values.delete(key)
  }
}

describe('auth token storage', () => {
  let storage: MemoryStorage

  beforeEach(() => {
    storage = new MemoryStorage()
    vi.stubGlobal('window', {
      location: {
        origin: 'https://dashboard.example.test',
        protocol: 'https:',
      },
      localStorage: storage,
    })
    vi.stubGlobal('document', { cookie: '' })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('normalizes bounded cookie-safe tokens', () => {
    expect(normalizeAuthToken(' header.payload.signature ')).toBe('header.payload.signature')
    expect(normalizeAuthToken('')).toBeNull()
    expect(normalizeAuthToken('header payload')).toBeNull()
    expect(normalizeAuthToken('header;payload')).toBeNull()
    expect(normalizeAuthToken(`header\npayload`)).toBeNull()
    expect(normalizeAuthToken('x'.repeat(8193))).toBeNull()
  })

  it('stores trimmed tokens in localStorage and session cookie', () => {
    const token = storeAuthToken(' header.payload.signature ')

    expect(token).toBe('header.payload.signature')
    expect(storage.getItem(STORAGE_KEY)).toBe('header.payload.signature')
    expect(document.cookie).toBe(
      'vsr_session=header.payload.signature; Path=/; SameSite=Lax; Secure',
    )
  })

  it('clears malformed stored tokens before reuse', () => {
    storage.setItem(STORAGE_KEY, 'bad token')

    expect(getStoredAuthToken()).toBeNull()
    expect(storage.getItem(STORAGE_KEY)).toBeNull()
    expect(document.cookie).toBe('vsr_session=; Path=/; SameSite=Lax; Max-Age=0; Secure')
  })
})
