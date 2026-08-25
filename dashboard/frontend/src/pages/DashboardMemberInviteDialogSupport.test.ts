import { describe, expect, it } from 'vitest'

import { createDashboardMemberInvitationDraft } from './DashboardMemberInviteDialogSupport'

describe('Dashboard invitation draft', () => {
  it('starts every invitation from the complete product defaults', () => {
    expect(createDashboardMemberInvitationDraft()).toEqual({
      email: '',
      name: '',
      role: 'read',
      teamId: '',
      teamRole: 'member',
      expiresInHours: 168,
      sendEmail: true,
    })
  })

  it('returns a fresh draft for every dialog opening', () => {
    const first = createDashboardMemberInvitationDraft()
    first.role = 'admin'
    first.sendEmail = false

    expect(createDashboardMemberInvitationDraft()).toMatchObject({ role: 'read', sendEmail: true })
  })
})
