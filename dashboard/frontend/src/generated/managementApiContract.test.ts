import { describe, expect, it } from 'vitest'

import {
  MANAGEMENT_API_BASE_PATH,
  MANAGEMENT_API_HEADERS,
  MANAGEMENT_API_MEDIA_TYPE,
  MANAGEMENT_API_OPERATIONS,
  managementApiPath,
  type ManagementApiOperationId,
} from './managementApiContract'

describe('generated Management API contract', () => {
  it('builds encoded paths from operation ids', () => {
    expect(
      managementApiPath('getNamespacesByNamespaceIdRequestLogsByAdmissionId', {
        namespaceId: 'namespace/one',
        admissionId: 'request two',
      }),
    ).toBe('/management/v1/namespaces/namespace%2Fone/request-logs/request%20two')
  })

  it('rejects a missing path parameter at runtime', () => {
    const uncheckedPath = managementApiPath as (
      operationId: ManagementApiOperationId,
      parameters?: Record<string, string>,
    ) => string
    expect(() => uncheckedPath('getApiKeysByKeyId')).toThrow(
      'Missing Management API path parameter: keyId.',
    )
  })

  it('carries the canonical versioned transport metadata', () => {
    expect(MANAGEMENT_API_BASE_PATH).toBe('/management/v1')
    expect(MANAGEMENT_API_MEDIA_TYPE).toContain('.management.v1+json')
    expect(MANAGEMENT_API_HEADERS.namespace).toBe('VLLM-SR-Namespace')
    expect(MANAGEMENT_API_OPERATIONS.postApiKeys.method).toBe('POST')
  })
})
