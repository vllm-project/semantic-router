import { describe, expect, it } from 'vitest'

import {
  auditLogsCsv,
  auditTimeBounds,
  auditTimeLabel,
  buildAuditLogQuery,
  createLatestAuditRequest,
  normalizeAuditLogPage,
  type AuditLog,
} from './usersAuditLogSupport'

const auditLog: AuditLog = {
  id: 7,
  userId: 'user-a',
  action: 'user.update',
  resource: 'users/user-b',
  method: 'PATCH',
  path: '/api/admin/users/user-b',
  ip: '192.0.2.1',
  userAgent: 'console',
  statusCode: 204,
  createdAt: 1_704_153_600,
}

describe('users audit log support', () => {
  it('builds the server query from trimmed filters and paging controls', () => {
    const query = new URLSearchParams(
      buildAuditLogQuery({
        query: '  update  ',
        user: ' user-a ',
        action: 'user.update',
        resource: ' users/user-b ',
        status: 'success',
        from: '2024-01-01',
        to: '2024-01-31',
        sort: 'createdAt',
        order: 'desc',
        page: 3,
        limit: 50,
      }),
    )

    expect(Object.fromEntries(query.entries())).toEqual({
      q: 'update',
      user: 'user-a',
      action: 'user.update',
      resource: 'users/user-b',
      status: 'success',
      from: '2024-01-01',
      to: '2024-01-31',
      sort: 'createdAt',
      order: 'desc',
      page: '3',
      limit: '50',
    })
  })

  it('normalizes paged and legacy array responses', () => {
    expect(
      normalizeAuditLogPage({ logs: [auditLog], total: 42, page: 2, limit: 20 }, 1, 10),
    ).toEqual({ logs: [auditLog], total: 42, page: 2, limit: 20 })
    expect(normalizeAuditLogPage([auditLog], 4, 25)).toEqual({
      logs: [auditLog],
      total: 1,
      page: 4,
      limit: 25,
    })
  })

  it('aborts and marks an older request stale when a new request starts', () => {
    const requests = createLatestAuditRequest()
    const first = requests.start()
    const second = requests.start()

    expect(first.signal.aborted).toBe(true)
    expect(first.isCurrent()).toBe(false)
    expect(second.signal.aborted).toBe(false)
    expect(second.isCurrent()).toBe(true)

    requests.abort()
    expect(second.signal.aborted).toBe(true)
    expect(second.isCurrent()).toBe(false)
  })

  it('keeps the facets the server adds and drops malformed ones', () => {
    const page = normalizeAuditLogPage(
      {
        logs: [],
        total: 0,
        facets: {
          users: [{ value: 'user-a', count: 2 }],
          actions: [{ value: 'user.update', count: 1 }, { value: 7 } as never],
        },
      },
      1,
      20,
    )
    expect(page.facets).toEqual({
      users: [{ value: 'user-a', count: 2 }],
      actions: [{ value: 'user.update', count: 1 }],
    })
    expect(normalizeAuditLogPage({ logs: [] }, 1, 20).facets).toBeUndefined()
  })

  it('turns time ranges into query bounds and chip labels', () => {
    expect(auditTimeBounds({ preset: '24h', from: '', to: '' }, 100_000)).toEqual({
      from: String(100_000 - 86_400),
      to: '',
    })
    expect(auditTimeBounds({ preset: 'custom', from: '2026-11-01', to: '2026-11-18' })).toEqual({
      from: '2026-11-01',
      to: '2026-11-18',
    })
    expect(auditTimeBounds({ preset: 'any', from: '2026-11-01', to: '' })).toEqual({
      from: '',
      to: '',
    })
    expect(auditTimeLabel({ preset: '7d', from: '', to: '' })).toBe('Last 7 days')
    expect(auditTimeLabel({ preset: 'custom', from: '2026-11-01', to: '' })).toBe(
      'Since 2026-11-01',
    )
    expect(auditTimeLabel({ preset: 'any', from: '', to: '' })).toBeNull()
  })

  it('exports entries as CSV that spreadsheets will not run as formulas', () => {
    const csv = auditLogsCsv(
      [{ ...auditLog, resource: '=HYPERLINK("x")', userAgent: 'agent, "quoted"' }],
      (userId) => (userId === 'user-a' ? 'Ada Lin' : ''),
    )
    const [header, row] = csv.trimEnd().split('\n')
    expect(header).toBe('time,user_id,user,action,resource,method,path,status_code,ip,user_agent')
    expect(row).toBe(
      [
        '2024-01-02T00:00:00.000Z',
        'user-a',
        'Ada Lin',
        'user.update',
        `"'=HYPERLINK(""x"")"`,
        'PATCH',
        '/api/admin/users/user-b',
        '204',
        '192.0.2.1',
        '"agent, ""quoted"""',
      ].join(','),
    )
  })
})
