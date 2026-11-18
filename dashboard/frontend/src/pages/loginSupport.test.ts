import { describe, expect, it, vi } from 'vitest'

import {
  formatServerMessage,
  resolvePostAuthTarget,
  sanitizePostAuthTarget,
  settleWithin,
  targetPathname,
} from './loginSupport'

describe('post-authentication target', () => {
  it('returns to the protected route that started sign-in', () => {
    expect(resolvePostAuthTarget(false, '/observe/requests?status=error#row-4')).toBe(
      '/observe/requests?status=error#row-4',
    )
    expect(resolvePostAuthTarget(false, null)).toBe('/')
    expect(resolvePostAuthTarget(false, undefined)).toBe('/')
  })

  it('keeps setup mode on the setup wizard regardless of the requested route', () => {
    expect(resolvePostAuthTarget(true, '/routing/decisions')).toBe('/setup')
    expect(resolvePostAuthTarget(true, null)).toBe('/setup')
  })

  it('rejects off-site, protocol-relative and auth-surface targets', () => {
    for (const candidate of [
      'https://evil.example/',
      '//evil.example/path',
      'javascript:alert(1)',
      '/login?next=users',
      '/invite/abc',
      '/auth/transition?to=/settings/access',
      '',
    ]) {
      expect(sanitizePostAuthTarget(candidate, '/')).toBe('/')
    }
    expect(sanitizePostAuthTarget('/settings/access', '/')).toBe('/settings/access')
  })

  it('turns server error bodies into readable form messages', () => {
    expect(formatServerMessage('invitation is no longer available\n', 'Failed')).toBe(
      'Invitation is no longer available',
    )
    expect(formatServerMessage('{"error":"too many attempts, retry later"}', 'Failed')).toBe(
      'Too many attempts, retry later',
    )
    expect(formatServerMessage('{"message":"Rate limited"}', 'Failed')).toBe('Rate limited')
    expect(formatServerMessage('Session service unavailable', 'Failed')).toBe(
      'Session service unavailable',
    )
    expect(formatServerMessage('   ', 'Request failed (503)')).toBe('Request failed (503)')
  })

  it('extracts the route pathname used to preload its chunk', () => {
    expect(targetPathname('/observe/requests?status=error#row')).toBe('/observe/requests')
  })

  it('stops waiting for a slow route preload after the grace period', async () => {
    vi.useFakeTimers()
    let settled = false
    const pending = settleWithin(new Promise(() => undefined), 1500).then(() => {
      settled = true
    })
    await vi.advanceTimersByTimeAsync(1499)
    expect(settled).toBe(false)
    await vi.advanceTimersByTimeAsync(1)
    await pending
    expect(settled).toBe(true)
    vi.useRealTimers()
    await expect(settleWithin(undefined, 10)).resolves.toBeUndefined()
  })
})
