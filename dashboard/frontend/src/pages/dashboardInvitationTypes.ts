export type DashboardRole = 'admin' | 'write' | 'read'
export type InvitationKind = 'personal' | 'shared'
export type InvitationStatus = 'pending' | 'accepted' | 'revoked' | 'expired'

export interface DashboardInvitation {
  id: string
  email?: string
  name?: string
  role: DashboardRole
  kind: InvitationKind
  maxUses: number
  usedCount: number
  remainingUses: number
  status: InvitationStatus
  expiresAt: number
  acceptedAt?: number
  revokedAt?: number
  createdAt: number
  createdBy?: string
}
