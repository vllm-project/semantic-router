import type { ModelPricing, ProviderModelConfig, ProviderReliability } from './configPageSupport'

export interface ConnectModelAdvancedValues {
  namePrefix: string
  reasoningFamily: string
  description: string
  modality: string
  parameterSize: string
  contextWindow: string
  capabilities: string
  tags: string
  inputCost: string
  outputCost: string
  cacheReadCost: string
  cacheWriteCost: string
  maxRetries: string
  retryOn: string
  loadBalancing: string
  healthCheckPath: string
  healthCheckInterval: string
  healthCheckTimeout: string
}

export const emptyConnectModelAdvancedValues = (): ConnectModelAdvancedValues => ({
  namePrefix: '',
  reasoningFamily: '',
  description: '',
  modality: '',
  parameterSize: '',
  contextWindow: '',
  capabilities: '',
  tags: '',
  inputCost: '',
  outputCost: '',
  cacheReadCost: '',
  cacheWriteCost: '',
  maxRetries: '',
  retryOn: '',
  loadBalancing: '',
  healthCheckPath: '',
  healthCheckInterval: '',
  healthCheckTimeout: '',
})

export const requestedConnectedModelName = (prefix: string, model: string) => {
  const normalizedPrefix = prefix.trim().replace(/^\/+|\/+$/g, '')
  return normalizedPrefix ? `${normalizedPrefix}/${model}` : model
}

export function resolveConnectedModelName(
  prefix: string,
  providerId: string,
  model: string,
  reservedNames: ReadonlySet<string>,
): string {
  const requested = requestedConnectedModelName(prefix, model)
  if (!reservedNames.has(requested)) return requested

  const providerScoped = requestedConnectedModelName(providerId, model)
  if (!reservedNames.has(providerScoped)) return providerScoped

  let suffix = 2
  while (reservedNames.has(`${providerScoped}-${suffix}`)) suffix += 1
  return `${providerScoped}-${suffix}`
}

interface ConnectedProviderModelOptions {
  name: string
  catalog?: string
  providerModelID: string
  catalogProviderModelID?: string
  providerID: string
  providerAPIFormat: string
  baseURL: string
  apiKey: string
  reasoningFamily?: string
  pricing?: ModelPricing
  reliability?: ProviderReliability
}

/**
 * Builds the user-authored provider model entry for Quick Connect.
 *
 * Catalog-backed entries retain the catalog identity, backend provider, and
 * any operator-defined deployment name required by the binding. The canonical
 * materializer owns stable upstream model IDs and protocols, including models
 * that support Responses but not Chat Completions. Custom models have no such
 * binding, so they keep the provider-wide defaults selected by the user.
 */
export function buildConnectedProviderModel({
  name,
  catalog,
  providerModelID,
  catalogProviderModelID,
  providerID,
  providerAPIFormat,
  baseURL,
  apiKey,
  reasoningFamily,
  pricing,
  reliability,
}: ConnectedProviderModelOptions): ProviderModelConfig {
  return {
    name,
    ...(catalog
      ? {
          catalog,
          ...(catalogProviderModelID?.trim()
            ? { provider_model_id: catalogProviderModelID.trim() }
            : {}),
        }
      : {
          ...(reasoningFamily ? { reasoning: { family: reasoningFamily } } : {}),
          provider_model_id: providerModelID,
          api_format: providerAPIFormat,
        }),
    pricing,
    reliability,
    backend_refs: [
      {
        name: `${providerID}-primary`,
        base_url: baseURL,
        provider: providerID,
        ...(apiKey ? { api_key: apiKey } : {}),
      },
    ],
  }
}
