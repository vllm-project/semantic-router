import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'
import UsersPageAuditPanel from './UsersPageAuditPanel'

describe('UsersPageAuditPanel', () => {
  it('filters with chips, exports, pages and shows a loading state without a refresh button', () => {
    const markup = renderToStaticMarkup(
      createElement(TooltipProvider, null, createElement(UsersPageAuditPanel)),
    )

    expect(markup).toContain('Search activity')
    for (const chip of ['Person', 'Action', 'Result', 'Time']) expect(markup).toContain(chip)
    expect(markup).toContain('Newest first')
    expect(markup).toContain('Export CSV')
    expect(markup).not.toMatch(/Refresh|More filters|aria-label="Response status"/)
    expect(markup).toContain('Page 1 / 1')
    expect(markup).toContain('0 events')
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
