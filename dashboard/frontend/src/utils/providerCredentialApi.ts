import {
  MANAGEMENT_API_HEADERS,
  type ProviderCredentialCreateRequest,
  type ProviderCredentialDetail,
} from '../generated/managementApiContract'

import { ManagementApiError, managementApiClient } from './managementApiContract'

export type CreateProviderCredentialInput = ProviderCredentialCreateRequest
export type CreateProviderCredentialResult = ProviderCredentialDetail

// Provider credential transport is intentionally isolated here. The Router
// owns origin normalization, adapter selection, encryption, persistence, and
// authorization; the Dashboard only maps the product form to the generated
// Management contract.
export async function createProviderCredential(
  input: CreateProviderCredentialInput,
  signal?: AbortSignal,
): Promise<CreateProviderCredentialResult> {
  const mutation = await managementApiClient.postProviderCredentials({
    body: input,
    headers: { [MANAGEMENT_API_HEADERS.idempotencyKey]: crypto.randomUUID() },
    signal,
  })
  if (!('resource' in mutation.data) || mutation.data.resource.kind !== 'provider_credential') {
    throw new ManagementApiError(
      'Provider credential creation did not return a resource reference.',
      502,
    )
  }

  const detail = await managementApiClient.getProviderCredentialsByCredentialId({
    pathParameters: { credentialId: mutation.data.resource.id },
    signal,
  })
  return detail.data
}
