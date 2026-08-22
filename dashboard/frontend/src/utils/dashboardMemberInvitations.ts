export interface DashboardMemberInvitation {
  id: string
  email: string
  name: string
  role: 'admin' | 'write' | 'read'
  teamId?: string
  teamName?: string
  teamRole?: 'admin' | 'member'
  status: 'pending' | 'accepted' | 'revoked' | 'expired'
  expiresAt: number
  acceptedAt?: number
  revokedAt?: number
  createdAt: number
  createdBy: string
  updatedAt: number
  lastSentAt?: number
  deliveryStatus: string
  deliveryError?: string
  invitationToken?: string
  invitationPath?: string
}

export interface DashboardMemberInvitationInput {
  email: string
  name: string
  role: string
  teamId?: string
  teamRole?: 'admin' | 'member'
  expiresInHours: number
  sendEmail: boolean
}

const errorMessage = async (response: Response) =>
  (await response.text()) || `Request failed (${response.status})`

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return response.json() as Promise<T>
}

export const dashboardMemberInvitationApi = {
  list: () => request<{ items: DashboardMemberInvitation[] }>('/api/admin/invitations'),
  create: (input: DashboardMemberInvitationInput) =>
    request<DashboardMemberInvitation>('/api/admin/invitations', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
  resend: (id: string, sendEmail: boolean) =>
    request<DashboardMemberInvitation>(`/api/admin/invitations/${id}/resend`, {
      method: 'POST',
      body: JSON.stringify({ sendEmail }),
    }),
  revoke: (id: string) =>
    request<DashboardMemberInvitation>(`/api/admin/invitations/${id}`, { method: 'DELETE' }),
}

export const absoluteInvitationURL = (invitation: DashboardMemberInvitation) =>
  invitation.invitationPath
    ? new URL(invitation.invitationPath, window.location.origin).toString()
    : ''
