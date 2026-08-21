import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readSource = (name: string) => readFileSync(new URL(name, import.meta.url), 'utf8')

describe('dashboard product surfaces', () => {
  it('puts a calm system overview in the first dashboard surface', () => {
    const page = readSource('./DashboardPage.tsx')
    const hero = readSource('./DashboardRoutingHero.tsx')

    expect(page.indexOf('<DashboardRoutingHero')).toBeLessThan(page.indexOf('mainGrid'))
    expect(hero).toContain('Your model system, at a glance.')
    expect(hero).toContain('Every capability path visible, governed, and ready.')
    expect(hero).not.toContain('activeRoute')
    expect(hero).not.toContain('routeDots')
  })

  it('uses one shared authentication composition for login and invitations', () => {
    const login = readSource('./LoginPage.tsx')
    const invite = readSource('./InviteAcceptPage.tsx')

    expect(login).toContain('<AuthExperienceShell')
    expect(invite).toContain('<AuthExperienceShell')
    expect(invite).toContain('Your invitation is here')
    expect(invite).toContain('Build what one model can’t.')
    expect(invite).toContain('Valid until')
    expect(invite).toContain('markFirstAPIKeyOnboardingPending')
  })

  it('opens API key creation directly from the invitation welcome', () => {
    const invite = readSource('./InviteAcceptPage.tsx')
    const shell = readSource('../app/AuthenticatedShell.tsx')
    const access = readSource('./AccessControlPage.tsx')

    expect(invite).toContain('markFirstAPIKeyOnboardingPending')
    expect(shell).toContain('<InviteCompletionDialog')
    expect(shell).toContain('/access/api-keys?create=key&from=invitation')
    expect(access).toContain("detailParams.get('create')")
    expect(access).toContain("openCreate('key')")
  })

  it('uses invitation as the only dashboard user creation path', () => {
    const access = readSource('./AccessControlPage.tsx')
    const invitation = readSource('./DashboardMemberInviteDialog.tsx')

    expect(access).toContain('<span aria-hidden="true">+</span> Invite user')
    expect(access).not.toContain("target === 'user'")
    expect(access).not.toContain("? 'New user'")
    expect(invitation).toContain('Team (optional)')
    expect(invitation).toContain('Dashboard role')
  })

  it('keeps every Build manager on the shared banner composition', () => {
    const manager = readSource('./ConfigPageManagerLayout.tsx')

    expect(manager).toContain('className={styles.headerGrid}')
    expect(manager).toContain('className={styles.surfacePulse}')
    expect(manager).toContain('Semantic Router')
  })
})
