export interface DashboardMemberInvitationDraft {
  email: string
  name: string
  role: string
  teamId: string
  teamRole: 'member' | 'admin'
  expiresInHours: number
  sendEmail: boolean
}

export function createDashboardMemberInvitationDraft(): DashboardMemberInvitationDraft {
  return {
    email: '',
    name: '',
    role: 'read',
    teamId: '',
    teamRole: 'member',
    expiresInHours: 168,
    sendEmail: true,
  }
}
