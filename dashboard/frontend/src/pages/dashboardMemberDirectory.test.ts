import { describe, expect, it, vi } from 'vitest'
import { loadAllDashboardMembers } from './dashboardMemberDirectory'

describe('Dashboard member directory', () => {
  it('reads every bounded server page instead of truncating the directory', async () => {
    const loadPage = vi.fn(async (page: number, _limit: number) => ({
      users:
        page === 1
          ? [
              {
                id: 'member-1',
                email: 'one@example.com',
                name: 'One',
                role: 'read',
                status: 'active',
              },
            ]
          : [
              {
                id: 'member-2',
                email: 'two@example.com',
                name: 'Two',
                role: 'read',
                status: 'active',
              },
            ],
      total: 2,
      page,
      limit: 1,
    }))

    const members = await loadAllDashboardMembers(loadPage)

    expect(members.map((member) => member.id)).toEqual(['member-1', 'member-2'])
    expect(loadPage).toHaveBeenNthCalledWith(1, 1, 200, undefined)
    expect(loadPage).toHaveBeenNthCalledWith(2, 2, 200, undefined)
  })

  it('deduplicates a member repeated across a changing paginated snapshot', async () => {
    const loadPage = vi
      .fn()
      .mockResolvedValueOnce({
        users: [
          {
            id: 'member-1',
            email: 'one@example.com',
            name: 'One',
            role: 'read',
            status: 'active',
          },
        ],
        total: 2,
        page: 1,
        limit: 1,
      })
      .mockResolvedValueOnce({
        users: [
          {
            id: 'member-1',
            email: 'one@example.com',
            name: 'One',
            role: 'read',
            status: 'active',
          },
          {
            id: 'member-2',
            email: 'two@example.com',
            name: 'Two',
            role: 'read',
            status: 'active',
          },
        ],
        total: 2,
        page: 2,
        limit: 1,
      })

    await expect(loadAllDashboardMembers(loadPage)).resolves.toHaveLength(2)
  })
})
