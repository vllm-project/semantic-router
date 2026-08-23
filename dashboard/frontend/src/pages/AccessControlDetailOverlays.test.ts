import { describe, expect, it } from 'vitest'

import type { AccessAPIKey, AccessTeam } from '../utils/inferenceAccessApi'
import { canManageSelfServiceKey } from './AccessControlDetailOverlaysSupport'

const userKey = {
  id: 'user-key',
  ownerType: 'user',
  ownerId: 'user-1',
} as AccessAPIKey

const teamKey = {
  id: 'team-key',
  ownerType: 'team',
  ownerId: 'team-1',
} as AccessAPIKey

const team = {
  id: 'team-1',
  members: [
    { teamId: 'team-1', userId: 'user-1', role: 'admin' },
    { teamId: 'team-1', userId: 'user-2', role: 'member' },
  ],
} as AccessTeam

describe('self-service API key actions', () => {
  it('never turns delegated inference authority into personal-key mutation', () => {
    expect(canManageSelfServiceKey(userKey.id, [userKey, teamKey], [team], 'user-1')).toBe(false)
  })

  it('limits Team-key actions to that Team admin', () => {
    expect(canManageSelfServiceKey(teamKey.id, [userKey, teamKey], [team], 'user-1')).toBe(true)
    expect(canManageSelfServiceKey(teamKey.id, [userKey, teamKey], [team], 'user-2')).toBe(false)
    expect(canManageSelfServiceKey(teamKey.id, [userKey, teamKey], [team], 'user-3')).toBe(false)
  })
})
