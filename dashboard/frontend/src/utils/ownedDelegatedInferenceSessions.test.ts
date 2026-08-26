import { describe, expect, it } from 'vitest'

import { OwnedDelegatedInferenceSessions } from './ownedDelegatedInferenceSessions'

describe('owned delegated inference sessions', () => {
  it('retires only sessions claimed by this owner', () => {
    const revoked: string[] = []
    const sessions = new OwnedDelegatedInferenceSessions((resourceId) => revoked.push(resourceId))
    sessions.activate('key-a')
    const claim = sessions.begin('key-a')
    expect(claim).not.toBeNull()
    expect(sessions.claim(claim!, { keyId: 'key-a', resourceId: 'session-a' })).toBe(true)

    sessions.retire('not-owned')
    sessions.deactivate()
    sessions.deactivate()

    expect(revoked).toEqual(['session-a'])
  })

  it('revokes an issuance that resolves after a key switch', () => {
    const revoked: string[] = []
    const sessions = new OwnedDelegatedInferenceSessions((resourceId) => revoked.push(resourceId))
    sessions.activate('key-a')
    const staleClaim = sessions.begin('key-a')

    sessions.deactivate()
    sessions.activate('key-b')
    expect(sessions.claim(staleClaim!, { keyId: 'key-a', resourceId: 'stale-session' })).toBe(false)

    const currentClaim = sessions.begin('key-b')
    expect(sessions.claim(currentClaim!, { keyId: 'key-b', resourceId: 'current-session' })).toBe(
      true,
    )
    expect(revoked).toEqual(['stale-session'])

    sessions.deactivate()
    expect(revoked).toEqual(['stale-session', 'current-session'])
  })

  it('invalidates StrictMode-era work without double-revoking an owned session', () => {
    const revoked: string[] = []
    const sessions = new OwnedDelegatedInferenceSessions((resourceId) => revoked.push(resourceId))
    sessions.activate('key-a')
    const firstMountClaim = sessions.begin('key-a')
    expect(sessions.claim(firstMountClaim!, { keyId: 'key-a', resourceId: 'first-mount' })).toBe(
      true,
    )

    sessions.deactivate()
    sessions.activate('key-a')
    expect(
      sessions.claim(firstMountClaim!, { keyId: 'key-a', resourceId: 'late-first-mount' }),
    ).toBe(false)
    sessions.deactivate()

    expect(revoked).toEqual(['first-mount', 'late-first-mount'])
  })
})
