export interface UnifiedUserDeletionProgress {
  dashboardLoginRemoved: boolean
  modelIdentityDeleted: boolean
}

export interface UnifiedUserDeletionDependencies {
  removeDashboardLogin: () => Promise<void>
  deleteModelIdentity: () => Promise<void>
}

export class UnifiedUserDeletionError extends Error {
  readonly progress: UnifiedUserDeletionProgress

  constructor(message: string, progress: UnifiedUserDeletionProgress) {
    super(message)
    this.name = 'UnifiedUserDeletionError'
    this.progress = progress
  }
}

const errorMessage = (error: unknown, fallback: string) =>
  error instanceof Error && error.message.trim() ? error.message : fallback

const normalizedEmail = (email: string) => email.trim().toLowerCase()

export function findLinkedModelUser<T extends { id: string; email: string }>(
  member: { email: string },
  users: readonly T[],
): T | null {
  const email = normalizedEmail(member.email)
  return users.find((user) => email && normalizedEmail(user.email) === email) ?? null
}

/**
 * Remove the Dashboard login first, then tombstone the Router identity. If the
 * second step fails, the remaining Router user stays visible and the returned
 * progress lets a retry resume without repeating step one.
 */
export async function deleteUnifiedUser(
  current: UnifiedUserDeletionProgress,
  dependencies: UnifiedUserDeletionDependencies,
): Promise<UnifiedUserDeletionProgress> {
  let progress = { ...current }

  if (!progress.dashboardLoginRemoved) {
    try {
      await dependencies.removeDashboardLogin()
      progress = { ...progress, dashboardLoginRemoved: true }
    } catch (error) {
      throw new UnifiedUserDeletionError(
        errorMessage(error, 'Could not remove Dashboard login'),
        progress,
      )
    }
  }

  if (!progress.modelIdentityDeleted) {
    try {
      await dependencies.deleteModelIdentity()
      progress = { ...progress, modelIdentityDeleted: true }
    } catch (error) {
      throw new UnifiedUserDeletionError(
        errorMessage(error, 'Could not delete model identity'),
        progress,
      )
    }
  }

  return progress
}
