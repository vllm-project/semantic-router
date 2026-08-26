import { describe, expect, it } from 'vitest'

import { ManagementApiError } from '../utils/managementApiContract'
import {
  classifyManagementIdentityFailure,
  managementIdentityRecoveryDelay,
} from './managementIdentityRecovery'

describe('Router Management identity recovery', () => {
  it('reserves outage mode for an explicit HTTP 503', () => {
    expect(
      classifyManagementIdentityFailure(new ManagementApiError('Temporarily unavailable', 503)),
    ).toEqual({ detail: 'Temporarily unavailable', status: 'unavailable' })
    expect(classifyManagementIdentityFailure(new ManagementApiError('Unauthorized', 401))).toEqual({
      detail: 'Unauthorized',
      status: 'error',
    })
    expect(classifyManagementIdentityFailure(new TypeError('Malformed response'))).toEqual({
      detail: 'Malformed response',
      status: 'error',
    })
  })

  it('backs off retries without ever exceeding the recovery ceiling', () => {
    expect([0, 1, 2, 3].map(managementIdentityRecoveryDelay)).toEqual([
      2_000, 4_000, 8_000, 16_000,
    ])
    expect(managementIdentityRecoveryDelay(4)).toBe(30_000)
    expect(managementIdentityRecoveryDelay(100)).toBe(30_000)
  })
})
