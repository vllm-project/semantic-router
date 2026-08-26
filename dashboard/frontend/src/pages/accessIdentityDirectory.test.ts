import { describe, expect, it, vi } from 'vitest'
import type { AccessUser } from '../utils/inferenceAccessApi'
import {
  findAccessUserByEmail,
  loadAllAccessUsers,
  mergeAccessIdentityRows,
} from './accessIdentityDirectory'

const accessUser = (id: string, email: string): AccessUser => ({
  id,
  email,
  name: id,
  status: 'active',
  memberships: [],
  accessGroupIds: [],
})

describe('Access identity directory', () => {
  it('unifies identities by normalized email and includes Dashboard-only members', () => {
    const rows = mergeAccessIdentityRows(
      [accessUser('router-1', 'ONE@example.com')],
      [
        {
          id: 'login-1',
          email: 'one@example.com',
          name: 'One',
          role: 'read',
          status: 'active',
        },
        {
          id: 'login-2',
          email: 'two@example.com',
          name: 'Two',
          role: 'write',
          status: 'active',
        },
      ],
      [],
    )

    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({ access: { id: 'router-1' }, member: { id: 'login-1' } })
    expect(rows[1]).toMatchObject({ access: undefined, member: { id: 'login-2' } })
  })

  it('reads every Router cursor page exactly once', async () => {
    const loadPage = vi
      .fn()
      .mockResolvedValueOnce({
        items: [accessUser('router-1', 'one@example.com')],
        total: 2,
        limit: 200,
        hasMore: true,
        nextCursor: 'page-2',
      })
      .mockResolvedValueOnce({
        items: [accessUser('router-2', 'two@example.com')],
        total: 2,
        limit: 200,
        hasMore: false,
      })

    await expect(loadAllAccessUsers(loadPage)).resolves.toHaveLength(2)
    expect(loadPage).toHaveBeenNthCalledWith(1, { cursor: undefined, limit: 200 })
    expect(loadPage).toHaveBeenNthCalledWith(2, { cursor: 'page-2', limit: 200 })
  })

  it('fails instead of looping when Router repeats a cursor', async () => {
    const loadPage = vi.fn(async () => ({
      items: [],
      total: 1,
      limit: 200,
      hasMore: true,
      nextCursor: 'same-page',
    }))

    await expect(loadAllAccessUsers(loadPage)).rejects.toThrow('repeated user cursor')
    expect(loadPage).toHaveBeenCalledTimes(2)
  })

  it('resolves direct links by exact normalized email and never by another identity id', async () => {
    const loadPage = vi.fn(async () => ({
      items: [
        accessUser('same-dashboard-id', 'other@example.com'),
        accessUser('router-1', 'person@example.com'),
      ],
      total: 2,
      limit: 200,
      hasMore: false,
    }))

    await expect(findAccessUserByEmail(' Person@Example.com ', loadPage)).resolves.toMatchObject({
      id: 'router-1',
    })
    await expect(findAccessUserByEmail('missing@example.com', loadPage)).resolves.toBeNull()
  })
})
