import { afterEach, describe, expect, it, vi } from 'vitest'
import { dashboardMemberInvitationApi } from './dashboardMemberInvitations'

afterEach(() => vi.unstubAllGlobals())

describe('dashboard member invitation API', () => {
  it('keeps the optional Team assignment explicit', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'invite-a' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await dashboardMemberInvitationApi.create({
      email: 'member@example.com',
      name: 'Member',
      role: 'read',
      teamId: 'team-a',
      expiresInHours: 168,
      sendEmail: false,
    })

    const [, init] = fetchMock.mock.calls[0]
    expect(fetchMock.mock.calls[0][0]).toBe('/api/admin/invitations')
    expect(JSON.parse(init.body)).toMatchObject({ teamId: 'team-a' })
  })
})
