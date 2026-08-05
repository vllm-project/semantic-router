import { describe, expect, it } from 'vitest'

import {
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
  ip: '10.0.0.1',
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
})
