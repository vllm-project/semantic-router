import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import UsersPageAuditPanel from './UsersPageAuditPanel'

describe('UsersPageAuditPanel', () => {
  it('renders enterprise filters, count, paging, refresh, and an explicit empty state', () => {
    const markup = renderToStaticMarkup(createElement(UsersPageAuditPanel))

    expect(markup).toContain('Audit logs')
    expect(markup).toContain('0 records')
    expect(markup).toContain('Action, path, IP, user agent, or status code')
    expect(markup).toContain('Exact user ID')
    expect(markup).toContain('user.update')
    expect(markup).toContain('Exact resource')
    expect(markup).toContain('All responses')
    expect(markup).toContain('Reset filters')
    expect(markup).toContain('Refresh')
    expect(markup).toContain('No audit log entries have been recorded.')
    expect(markup).toContain('Page 1 / 1')
  })

  it('keeps the audit surface behind users.manage in its parent page', () => {
    const source = readFileSync(new URL('./UsersPage.tsx', import.meta.url), 'utf8')

    expect(source).toContain('showAudit && canManageUsers')
    expect(source).toContain('canManageUsers')
    expect(source).toContain('<UsersPageAuditPanel />')
  })
})
