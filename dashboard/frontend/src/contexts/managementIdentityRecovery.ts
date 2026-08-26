import { ManagementApiError } from '../utils/managementApiContract'

export type ManagementIdentityStatus = 'ready' | 'unavailable' | 'error'

export interface ManagementIdentityFailure {
  detail: string
  status: Exclude<ManagementIdentityStatus, 'ready'>
}

const RECOVERY_BASE_DELAY_MS = 2_000
const RECOVERY_MAX_DELAY_MS = 30_000

export function classifyManagementIdentityFailure(cause: unknown): ManagementIdentityFailure {
  const detail =
    cause instanceof Error && cause.message.trim()
      ? cause.message.trim()
      : 'Router Management identity is unavailable.'
  return {
    detail,
    status: cause instanceof ManagementApiError && cause.status === 503 ? 'unavailable' : 'error',
  }
}

export function managementIdentityRecoveryDelay(attempt: number): number {
  const exponent = Math.max(0, Math.min(Math.trunc(attempt), 30))
  return Math.min(RECOVERY_BASE_DELAY_MS * 2 ** exponent, RECOVERY_MAX_DELAY_MS)
}
