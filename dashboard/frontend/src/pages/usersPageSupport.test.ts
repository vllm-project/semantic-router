import { describe, expect, it } from 'vitest'

import { createLatestUsersRequest, isUsersRequestAbortError } from './usersPageSupport'

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
