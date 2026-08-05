export type UsersPageRolePermissions = Record<string, string[]>

export type UsersPageRolePermissionsPayload = {
  rolePermissions?: UsersPageRolePermissions
}

export const EMPTY_ROLE_PERMISSIONS = Object.freeze({}) as UsersPageRolePermissions

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
