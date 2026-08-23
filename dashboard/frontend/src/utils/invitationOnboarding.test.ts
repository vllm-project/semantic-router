import { afterEach, describe, expect, it } from 'vitest'

import {
  claimInvitationOnboarding,
  clearInvitationOnboarding,
  peekInvitationOnboarding,
  stageInvitationOnboarding,
} from './invitationOnboarding'

const key = {
  id: 'key-1',
  name: "Test User's API key",
  prefix: 'vsr_test_user',
  ownerType: 'user' as const,
  ownerId: 'user-1',
  status: 'active' as const,
  accessGroupIds: [],
  secret: 'vsr_secret',
}

describe('invitation onboarding navigation', () => {
  afterEach(clearInvitationOnboarding)

  it('keeps a valid secret only in memory until the intended user claims it', () => {
    stageInvitationOnboarding({ displayName: 'Test User', onboardingKey: key })
    expect(peekInvitationOnboarding('user-1')).toMatchObject({
      displayName: 'Test User',
      onboardingKey: { id: 'key-1' },
    })
    expect(claimInvitationOnboarding('user-1')).toMatchObject({ onboardingKey: { id: 'key-1' } })
    expect(peekInvitationOnboarding('user-1')).toBeNull()
  })

  it('rejects incomplete or cross-user handoffs', () => {
    expect(() => stageInvitationOnboarding({ displayName: '', onboardingKey: key })).toThrowError(
      'incomplete or expired',
    )
    stageInvitationOnboarding({ displayName: 'Test User', onboardingKey: key })
    expect(peekInvitationOnboarding('another-user')).toBeNull()
    expect(() =>
      stageInvitationOnboarding({
        displayName: 'Test User',
        onboardingKey: { ...key, secret: '' },
      }),
    ).toThrowError('incomplete or expired')
  })
})
