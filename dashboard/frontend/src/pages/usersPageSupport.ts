export type UsersPageRolePermissions = Record<string, string[]>

export type UsersPageRolePermissionsPayload = {
  rolePermissions?: UsersPageRolePermissions
}

export const EMPTY_ROLE_PERMISSIONS = Object.freeze({}) as UsersPageRolePermissions

// Mirrors MaxPasswordBytes in dashboard/backend/auth/password.go.
export const MAX_PASSWORD_BYTES = 72

// bcrypt counts bytes, not characters: 25 CJK characters are 75 bytes. A
// maxLength attribute is not a substitute, it counts UTF-16 units and would
// silently truncate a pasted password.
export const passwordByteLength = (password: string) => new TextEncoder().encode(password).length

// Blank means "keep the current password" when editing; a new account has none.
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

// The role write can commit and the password write still fail, so the message
// has to say which of the two happened.
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
