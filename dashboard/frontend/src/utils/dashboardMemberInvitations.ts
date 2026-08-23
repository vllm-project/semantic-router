import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'

import { getManagementNamespace } from './managementApiContract'

export interface DashboardMemberInvitation {
  id: string
  namespaceId: string
  revision: number
  email: string
  name: string
  role: 'admin' | 'write' | 'read'
  teamId?: string
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
  const namespace = getManagementNamespace()
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...(namespace ? { [MANAGEMENT_API_HEADERS.namespace]: namespace } : {}),
      ...init?.headers,
    },
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return response.json() as Promise<T>
}

const mutationHeaders = (revision?: number) => ({
  [MANAGEMENT_API_HEADERS.idempotencyKey]: crypto.randomUUID(),
  ...(revision ? { [MANAGEMENT_API_HEADERS.ifMatch]: `"invitation:${revision}"` } : {}),
})

export const dashboardMemberInvitationApi = {
  list: () => request<{ items: DashboardMemberInvitation[] }>('/api/admin/invitations'),
  create: (input: DashboardMemberInvitationInput) =>
    request<DashboardMemberInvitation>('/api/admin/invitations', {
      method: 'POST',
      headers: mutationHeaders(),
      body: JSON.stringify(input),
    }),
  resend: (id: string, revision: number, sendEmail: boolean) =>
    request<DashboardMemberInvitation>(`/api/admin/invitations/${id}/resend`, {
      method: 'POST',
      headers: mutationHeaders(revision),
      body: JSON.stringify({ sendEmail }),
    }),
  revoke: (id: string, revision: number) =>
    request<DashboardMemberInvitation>(`/api/admin/invitations/${id}`, {
      method: 'DELETE',
      headers: mutationHeaders(revision),
    }),
}

export const absoluteInvitationURL = (invitation: DashboardMemberInvitation) =>
  invitation.invitationPath
    ? new URL(invitation.invitationPath, window.location.origin).toString()
    : ''
