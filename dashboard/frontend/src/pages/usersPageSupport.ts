export type UsersPageRolePermissions = Record<string, string[]>

export type UsersPageRolePermissionsPayload = {
  rolePermissions?: UsersPageRolePermissions
}

export const EMPTY_ROLE_PERMISSIONS = Object.freeze({}) as UsersPageRolePermissions

/**
 * Mirrors MaxPasswordBytes in dashboard/backend/auth/password.go. The server is
 * the enforcing side; this exists so the dialog can reject an oversized
 * password before it sends anything, rather than after a partial write.
 */
export const MAX_PASSWORD_BYTES = 72

/**
 * bcrypt measures passwords in bytes, so a character count is not a substitute:
 * 25 CJK characters are 75 bytes. This is also why the password input carries no
 * maxLength attribute, which counts UTF-16 units and would silently truncate a
 * pasted password into one the user cannot sign in with.
 */
export const passwordByteLength = (password: string) => new TextEncoder().encode(password).length

/**
 * Returns a message when the password cannot be submitted, or null when it can.
 * A blank password means "leave the current one alone" while editing, but a new
 * account has nothing to fall back to.
 */
export const validateUserPassword = (password: string, mode: 'create' | 'edit') => {
  if (!password) {
    return mode === 'create' ? 'A password is required to create a user.' : null
  }

  const byteLength = passwordByteLength(password)
  if (byteLength > MAX_PASSWORD_BYTES) {
    return `Password must be at most ${MAX_PASSWORD_BYTES} bytes (this one is ${byteLength}). Passwords are measured in bytes, so accented and non-Latin characters count as more than one.`
  }

  return null
}

/**
 * Builds the dialog error for a failed user edit.
 *
 * The role and status update and the password rotation are two requests, so the
 * first can commit and the second still fail, whether from a rejected password,
 * an expired session, or a dropped connection. Reporting that as a flat failure
 * is what made the dialog lie about what had been saved.
 */
export const describeUserUpdateFailure = (error: unknown, roleAndStatusSaved: boolean) => {
  const reason = error instanceof Error ? error.message : String(error)
  if (!roleAndStatusSaved) {
    return reason
  }

  return `Role and status were saved. The password was not changed: ${reason}`
}

export const isUsersRequestAbortError = (error: unknown) =>
  error instanceof DOMException
    ? error.name === 'AbortError'
    : error instanceof Error && error.name === 'AbortError'

export const createLatestUsersRequest = () => {
  let sequence = 0
  let activeController: AbortController | null = null

  return {
    start() {
      activeController?.abort()
      const controller = new AbortController()
      const requestSequence = ++sequence
      activeController = controller

      return {
        signal: controller.signal,
        isCurrent: () => requestSequence === sequence,
        finish: () => {
          if (requestSequence === sequence) {
            activeController = null
          }
        },
      }
    },
    abort() {
      sequence += 1
      activeController?.abort()
      activeController = null
    },
  }
}
