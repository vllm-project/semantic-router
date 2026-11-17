import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'
import UsersPageAuditPanel from './UsersPageAuditPanel'

describe('UsersPageAuditPanel', () => {
  it('renders search, response and date filters, refresh, paging and a loading state', () => {
    const markup = renderToStaticMarkup(
      createElement(TooltipProvider, null, createElement(UsersPageAuditPanel)),
    )

    expect(markup).toContain('Search activity')
    expect(markup).toContain('aria-label="Response status"')
    expect(markup).toContain('aria-label="From date"')
    expect(markup).toContain('aria-label="To date"')
    expect(markup).toContain('More filters')
    expect(markup).toContain('Refresh audit logs')
    expect(markup).toContain('Page 1 / 1')
    expect(markup).toContain('0 records')
    expect(markup).toContain('data-slot="skeleton"')
  })

  it('keeps the audit and invitation surfaces behind users.manage in their parent page', () => {
    const source = readFileSync(new URL('./UsersPage.tsx', import.meta.url), 'utf8')

    expect(source).toContain("activeView === 'audit' && canManageUsers")
    expect(source).toContain("activeView === 'invitations' && canManageUsers")
    expect(source).toContain('<UsersPageAuditPanel actors={actors} />')
    expect(source).toContain('<UsersPageInvitationsPanel')
  })
})
