import { readFileSync } from 'node:fs'
import { describe, expect, it, vi } from 'vitest'

import {
  fetchBootstrapOpen,
  onPageReturn,
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

describe('first-admin form freshness', () => {
  const jsonResponse = (body: unknown, ok = true) =>
    ({ ok, json: async () => body }) as unknown as Response

  it('reports whether the first-admin form is still open', async () => {
    const open = vi.fn(async () => jsonResponse({ canRegister: true }))
    const closed = vi.fn(async () => jsonResponse({ canRegister: false }))

    expect(await fetchBootstrapOpen(open as unknown as typeof fetch)).toBe(true)
    expect(open).toHaveBeenCalledWith('/api/auth/bootstrap/can-register', { method: 'GET' })
    expect(await fetchBootstrapOpen(closed as unknown as typeof fetch)).toBe(false)
  })

  it('returns unknown on server or network errors so a stale check cannot hide the form', async () => {
    const serverError = vi.fn(async () => jsonResponse({}, false))
    const networkError = vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    })

    expect(await fetchBootstrapOpen(serverError as unknown as typeof fetch)).toBeNull()
    expect(await fetchBootstrapOpen(networkError as unknown as typeof fetch)).toBeNull()
  })

  it('rechecks when the window regains focus or the tab becomes visible again', () => {
    const win = new EventTarget()
    const doc = Object.assign(new EventTarget(), { hidden: true })
    const onReturn = vi.fn()

    const stop = onPageReturn(onReturn, win, doc)

    doc.dispatchEvent(new Event('visibilitychange'))
    expect(onReturn).not.toHaveBeenCalled()

    doc.hidden = false
    doc.dispatchEvent(new Event('visibilitychange'))
    win.dispatchEvent(new Event('focus'))
    expect(onReturn).toHaveBeenCalledTimes(2)

    stop()
    win.dispatchEvent(new Event('focus'))
    doc.dispatchEvent(new Event('visibilitychange'))
    expect(onReturn).toHaveBeenCalledTimes(2)
  })

  it('moves the login page to sign-in when a recheck or submit finds setup already done', () => {
    const source = readFileSync(new URL('./LoginPage.tsx', import.meta.url), 'utf8')

    expect(source).toContain('onPageReturn(recheck)')
    expect(source).toContain('if (!active || open !== false || pendingRef.current) return')
    expect(source).toContain('response.status === 409 || (await fetchBootstrapOpen()) === false')
    expect(source.match(/BOOTSTRAP_CLOSED_MESSAGE/g)?.length).toBeGreaterThanOrEqual(3)
  })
})
