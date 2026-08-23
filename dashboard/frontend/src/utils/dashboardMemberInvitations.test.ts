import { afterEach, describe, expect, it, vi } from 'vitest'
import { dashboardMemberInvitationApi } from './dashboardMemberInvitations'
import { setManagementNamespace } from './managementApiContract'

afterEach(() => {
  setManagementNamespace(null)
  vi.unstubAllGlobals()
})

describe('dashboard member invitation API', () => {
  it('creates a Router-owned invitation with namespace, Team, and idempotency', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'invite-a' }),
    })
    vi.stubGlobal('fetch', fetchMock)
    setManagementNamespace('20000000-0000-4000-8000-000000000001')

    await dashboardMemberInvitationApi.create({
      email: 'member@example.com',
      name: 'Member',
      role: 'read',
      teamId: '20000000-0000-4000-8000-000000000002',
      teamRole: 'member',
      expiresInHours: 168,
      sendEmail: false,
    })

    const [, init] = fetchMock.mock.calls[0]
    expect(fetchMock.mock.calls[0][0]).toBe('/api/admin/invitations')
    expect(init.headers['VLLM-SR-Namespace']).toBe('20000000-0000-4000-8000-000000000001')
    expect(init.headers['Idempotency-Key']).toMatch(/^[0-9a-f-]{36}$/)
    expect(JSON.parse(init.body)).toEqual({
      email: 'member@example.com',
      name: 'Member',
      role: 'read',
      teamId: '20000000-0000-4000-8000-000000000002',
      teamRole: 'member',
      expiresInHours: 168,
      sendEmail: false,
    })
  })
})
