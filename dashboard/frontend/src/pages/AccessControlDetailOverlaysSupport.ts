import type { AccessAPIKey, AccessTeam } from '../utils/inferenceAccessApi'

export function canManageSelfServiceKey(
  keyId: string,
  keys: readonly AccessAPIKey[],
  teams: readonly AccessTeam[],
  userId: string,
): boolean {
  const key = keys.find((candidate) => candidate.id === keyId)
  if (!key || key.ownerType !== 'team') return false
  return teams.some(
    (team) =>
      team.id === key.ownerId &&
      team.members.some(
        (membership) => membership.userId === userId && membership.role === 'admin',
      ),
  )
}
