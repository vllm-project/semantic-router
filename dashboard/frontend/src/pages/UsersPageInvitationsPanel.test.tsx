import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'
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

const render = (value: DashboardInvitation) =>
  renderToStaticMarkup(
    createElement(TooltipProvider, null, createElement(InvitationProgress, { invitation: value })),
  )

describe('UsersPageInvitationsPanel', () => {
  it('shows a shared link as one usage value', () => {
    const markup = render(invitation({}))

    expect(markup).toContain('3<span class="text-subtle-foreground"> of 10 used</span>')
    expect(markup).not.toContain('tickets left')
  })

  it('says whether a personal invitation was accepted', () => {
    const accepted = render(
      invitation({
        kind: 'personal',
        maxUses: 1,
        usedCount: 1,
        remainingUses: 0,
        status: 'accepted',
        acceptedAt: 1_750_000_000,
      }),
    )
    expect(accepted).toContain('Joined')
    expect(render(invitation({ kind: 'personal', maxUses: 1, usedCount: 0 }))).toContain('Not yet')
  })
})
