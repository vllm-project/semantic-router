import type { DashboardMemberInvitation } from '../utils/dashboardMemberInvitations'
import type { AccessPage, AccessUser } from '../utils/inferenceAccessApi'
import type { DashboardMember } from './AccessControlViewTypes'

export interface AccessIdentityRow {
  key: string
  access?: AccessUser
  member?: DashboardMember
  invitation?: DashboardMemberInvitation
  name: string
  email: string
}

type AccessUserPageLoader = (params: {
  cursor?: string
  limit: number
  q?: string
}) => Promise<AccessPage<AccessUser>>

export const normalizeIdentityEmail = (email: string) => email.trim().toLowerCase()

export function mergeAccessIdentityRows(
  users: readonly AccessUser[],
  members: readonly DashboardMember[],
  invitations: readonly DashboardMemberInvitation[],
): AccessIdentityRow[] {
  const memberByEmail = new Map(
    members
      .map((member) => [normalizeIdentityEmail(member.email), member] as const)
      .filter(([email]) => email),
  )
  const invitationByEmail = new Map(
    invitations
      .filter((item) => item.status === 'pending')
      .map((invitation) => [normalizeIdentityEmail(invitation.email), invitation] as const)
      .filter(([email]) => email),
  )
  const matchedMemberIds = new Set<string>()
  const rows = users.map<AccessIdentityRow>((user) => {
    const email = normalizeIdentityEmail(user.email)
    const member = memberByEmail.get(email)
    if (member) matchedMemberIds.add(member.id)
    return {
      key: member ? `member:${member.id}` : `access:${user.id}`,
      access: user,
      member,
      invitation: invitationByEmail.get(email),
      name: member?.name || user.name,
      email: member?.email || user.email,
    }
  })

  members.forEach((member) => {
    if (matchedMemberIds.has(member.id)) return
    rows.push({
      key: `member:${member.id}`,
      access: undefined,
      member,
      name: member.name,
      email: member.email,
    })
  })
  return rows
}

export async function loadAllAccessUsers(loadPage: AccessUserPageLoader): Promise<AccessUser[]> {
  const users = new Map<string, AccessUser>()
  const seenCursors = new Set<string>()
  let cursor: string | undefined
  do {
    const page = await loadPage({ cursor, limit: 200 })
    page.items.forEach((user) => users.set(user.id, user))
    const nextCursor = page.hasMore ? page.nextCursor : undefined
    if (nextCursor && seenCursors.has(nextCursor)) {
      throw new Error('Router returned a repeated user cursor.')
    }
    if (nextCursor) seenCursors.add(nextCursor)
    cursor = nextCursor
  } while (cursor)
  return Array.from(users.values())
}

export async function findAccessUserByEmail(
  email: string,
  loadPage: AccessUserPageLoader,
): Promise<AccessUser | null> {
  const normalized = normalizeIdentityEmail(email)
  if (!normalized) return null
  const seenCursors = new Set<string>()
  let cursor: string | undefined
  do {
    const page = await loadPage({ cursor, limit: 200, q: normalized })
    const match = page.items.find((user) => normalizeIdentityEmail(user.email) === normalized)
    if (match) return match
    const nextCursor = page.hasMore ? page.nextCursor : undefined
    if (nextCursor && seenCursors.has(nextCursor)) {
      throw new Error('Router returned a repeated user cursor.')
    }
    if (nextCursor) seenCursors.add(nextCursor)
    cursor = nextCursor
  } while (cursor)
  return null
}
