import { describe, expect, it, vi } from 'vitest'
import {
  deleteUnifiedUser,
  findLinkedModelUser,
  UnifiedUserDeletionError,
  type UnifiedUserDeletionProgress,
} from './unifiedUserDeletion'

const initialProgress = (): UnifiedUserDeletionProgress => ({
  dashboardLoginRemoved: false,
  modelIdentityDeleted: false,
})

describe('unified user deletion', () => {
  it('links Dashboard and Router identities only by normalized email', () => {
    const users = [
      { id: 'model-id', email: ' Person@Example.com ' },
      { id: 'dashboard-id', email: 'other@example.com' },
    ]

    expect(findLinkedModelUser({ email: 'person@example.com' }, users)).toEqual(users[0])
    expect(findLinkedModelUser({ email: 'missing@example.com' }, users)).toBeNull()
  })

  it('deletes the Router identity before removing the Dashboard login', async () => {
    const calls: string[] = []
    const progress = await deleteUnifiedUser(initialProgress(), {
      removeDashboardLogin: vi.fn(async () => {
        calls.push('login')
      }),
      deleteModelIdentity: vi.fn(async () => {
        calls.push('identity')
      }),
    })

    expect(calls).toEqual(['identity', 'login'])
    expect(progress).toEqual({ dashboardLoginRemoved: true, modelIdentityDeleted: true })
  })

  it('preserves completed progress so a retry only repeats the failed step', async () => {
    const deleteModelIdentity = vi.fn(async () => undefined)
    const firstRemove = vi.fn(async () => {
      throw new Error('Login removal failed')
    })

    let progress = initialProgress()
    try {
      await deleteUnifiedUser(progress, {
        removeDashboardLogin: firstRemove,
        deleteModelIdentity,
      })
      throw new Error('deletion unexpectedly succeeded')
    } catch (error) {
      expect(error).toBeInstanceOf(UnifiedUserDeletionError)
      progress = (error as UnifiedUserDeletionError).progress
      expect(progress).toEqual({ dashboardLoginRemoved: false, modelIdentityDeleted: true })
    }

    const removeDashboardLogin = vi.fn(async () => undefined)
    const completed = await deleteUnifiedUser(progress, {
      removeDashboardLogin,
      deleteModelIdentity,
    })

    expect(removeDashboardLogin).toHaveBeenCalledTimes(1)
    expect(firstRemove).toHaveBeenCalledTimes(1)
    expect(deleteModelIdentity).toHaveBeenCalledTimes(1)
    expect(completed).toEqual({ dashboardLoginRemoved: true, modelIdentityDeleted: true })
  })

  it('does not remove the login when Router deletion fails', async () => {
    const removeDashboardLogin = vi.fn(async () => undefined)

    await expect(
      deleteUnifiedUser(initialProgress(), {
        removeDashboardLogin,
        deleteModelIdentity: vi.fn(async () => {
          throw new Error('Router is unavailable')
        }),
      }),
    ).rejects.toMatchObject({
      message: 'Router is unavailable',
      progress: { dashboardLoginRemoved: false, modelIdentityDeleted: false },
    })
    expect(removeDashboardLogin).not.toHaveBeenCalled()
  })
})
