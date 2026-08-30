export interface ConnectModelAdvancedValues {
  namePrefix: string
  reasoningFamily: string
  description: string
  modality: string
  parameterSize: string
  contextWindow: string
  capabilities: string
  tags: string
  qualityScore: string
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
  qualityScore: '',
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
