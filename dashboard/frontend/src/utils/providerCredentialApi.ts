import {
  ManagementApiError,
  hasOnlyKeys,
  isNonEmptyString,
  isRecord,
  managementOperationRequest,
} from './managementApiContract'
import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import type { ProviderConnectionValue } from './providerCatalogApi'

export interface CreateProviderCredentialInput {
  providerId: string
  catalogRevision: string
  name: string
  secret: string
  baseUrl?: string
  connectionFields?: Record<string, ProviderConnectionValue>
}

export interface ProviderCredentialSummary {
  id: string
  providerId: string
  name: string
  revision: number
  status: 'active'
}

export interface CreateProviderCredentialResult {
  data: ProviderCredentialSummary
}

// ProviderCredential transport is intentionally isolated here. The Router
// endpoint still owns origin normalization, adapter selection, encryption,
// persistence, and authorization; the Dashboard never derives those values.
export async function createProviderCredential(
  input: CreateProviderCredentialInput,
  signal?: AbortSignal,
): Promise<CreateProviderCredentialResult> {
  const payload = await managementOperationRequest('postProviderCredentials', {
    body: input,
    headers: { [MANAGEMENT_API_HEADERS.idempotencyKey]: crypto.randomUUID() },
    signal,
  })
  if (
    !isRecord(payload) ||
    !hasOnlyKeys(payload, ['data']) ||
    !isRecord(payload.data) ||
    !hasOnlyKeys(payload.data, ['id', 'providerId', 'name', 'revision', 'status']) ||
    !isNonEmptyString(payload.data.id) ||
    !isNonEmptyString(payload.data.providerId) ||
    !isNonEmptyString(payload.data.name) ||
    !Number.isSafeInteger(payload.data.revision) ||
    Number(payload.data.revision) < 1 ||
    payload.data.status !== 'active'
  ) {
    throw new ManagementApiError('Provider credential returned an invalid contract.', 502)
  }
  return payload as unknown as CreateProviderCredentialResult
}
