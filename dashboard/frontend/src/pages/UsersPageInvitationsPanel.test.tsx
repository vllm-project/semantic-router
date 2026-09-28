import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { DashboardInvitation } from './dashboardInvitationTypes'
import { InvitationProgress } from './UsersPageInvitationsPanel'

const invitation = (overrides: Partial<DashboardInvitation>): DashboardInvitation => ({
  id: 'invite-1',
  role: 'read',
  kind: 'shared',
  maxUses: 10,
  usedCount: 3,
  remainingUses: 7,
  status: 'pending',
  expiresAt: 1_800_000_000,
  createdAt: 1_700_000_000,
  ...overrides,
})

describe('UsersPageInvitationsPanel', () => {
  it('presents shared invitation capacity as tickets', () => {
    const markup = renderToStaticMarkup(
      createElement(InvitationProgress, { invitation: invitation({}) }),
    )

    expect(markup).toContain('3 / 10 used')
    expect(markup).toContain('7 tickets left')
    expect(markup).toContain('width:30%')
  })

  it('makes an accepted personal invitation explicit', () => {
    const markup = renderToStaticMarkup(
      createElement(InvitationProgress, {
        invitation: invitation({
          kind: 'personal',
          maxUses: 1,
          usedCount: 1,
          remainingUses: 0,
          status: 'accepted',
          acceptedAt: 1_750_000_000,
        }),
      }),
    )

    expect(markup).toContain('Accepted and signed in')
    expect(markup).not.toContain('ticket available')
  })
})
