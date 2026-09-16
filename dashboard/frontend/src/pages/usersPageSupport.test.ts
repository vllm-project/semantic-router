import { describe, expect, it } from 'vitest'

import {
  createLatestUsersRequest,
  describeUserUpdateFailure,
  isUsersRequestAbortError,
  MAX_PASSWORD_BYTES,
  passwordByteLength,
  validateUserPassword,
} from './usersPageSupport'

describe('users page request support', () => {
  it('aborts and marks an older user-list request stale', () => {
    const requests = createLatestUsersRequest()
    const first = requests.start()
    const second = requests.start()

    expect(first.signal.aborted).toBe(true)
    expect(first.isCurrent()).toBe(false)
    expect(second.signal.aborted).toBe(false)
    expect(second.isCurrent()).toBe(true)

    requests.abort()
    expect(second.signal.aborted).toBe(true)
    expect(second.isCurrent()).toBe(false)
  })

  it('recognizes fetch abort errors without hiding real failures', () => {
    expect(isUsersRequestAbortError(new DOMException('stale', 'AbortError'))).toBe(true)
    expect(isUsersRequestAbortError(new Error('network failed'))).toBe(false)
  })
})

describe('user password validation', () => {
  it('measures passwords in bytes rather than characters', () => {
    expect(passwordByteLength('a'.repeat(72))).toBe(72)
    // 25 CJK characters are 75 bytes, so a character count would wrongly pass.
    expect('好'.repeat(25)).toHaveLength(25)
    expect(passwordByteLength('好'.repeat(25))).toBe(75)
  })

  it('accepts a password at the bcrypt limit and rejects one past it', () => {
    expect(validateUserPassword('a'.repeat(MAX_PASSWORD_BYTES), 'edit')).toBeNull()
    expect(validateUserPassword('a'.repeat(MAX_PASSWORD_BYTES + 1), 'edit')).toContain(
      'at most 72 bytes',
    )
  })

  it('rejects a multibyte password that is short in characters but too long in bytes', () => {
    const password = '好'.repeat(25)

    expect(password.length).toBeLessThan(MAX_PASSWORD_BYTES)
    expect(validateUserPassword(password, 'edit')).toContain('at most 72 bytes')
  })

  it('treats a blank password as keep-current when editing and as missing when creating', () => {
    expect(validateUserPassword('', 'edit')).toBeNull()
    expect(validateUserPassword('', 'create')).toContain('required')
  })
})

describe('user update failure reporting', () => {
  it('reports a failure before the role write as a plain failure', () => {
    expect(describeUserUpdateFailure(new Error('Forbidden'), false)).toBe('Forbidden')
  })

  it('says what was saved when the role write committed and a later step failed', () => {
    const message = describeUserUpdateFailure(new Error('Unauthorized'), true)

    expect(message).toContain('Role and status were saved')
    expect(message).toContain('Unauthorized')
  })

  it('describes non-Error rejections without losing the reason', () => {
    expect(describeUserUpdateFailure('connection reset', true)).toContain('connection reset')
  })
})
