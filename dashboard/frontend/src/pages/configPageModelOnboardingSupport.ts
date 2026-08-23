import type {
  DiscoveredProviderModel,
  ProviderCatalogItem,
  ProviderConnectionField,
  ProviderConnectionValue,
} from '../utils/providerCatalogApi'
import type {
  RoutingBulkImportRequest,
  RoutingExecution,
  RoutingPricing,
} from '../utils/routingManagementApi'
import { normalizedPrefix } from './configPageModelImportSupport'

export type EditableConnectionValue = string | boolean

const durationPattern = /^(?<amount>[1-9]\d*)(?<unit>ms|s|m|h)$/
const durationUnitMilliseconds = { ms: 1, s: 1_000, m: 60_000, h: 3_600_000 } as const
const minimumModelTimeoutMilliseconds = 1_000
const maximumModelTimeoutMilliseconds = 24 * 60 * 60 * 1_000

export function initialProviderFieldValue(field: ProviderConnectionField): EditableConnectionValue {
  if (field.kind === 'boolean') return field.default === 'true'
  if (field.kind === 'select' && field.required && !field.default) {
    return field.options?.[0]?.value ?? ''
  }
  return field.default ?? ''
}

export function initialProviderConnectionFields(
  provider: ProviderCatalogItem,
): Record<string, EditableConnectionValue> {
  return Object.fromEntries(
    provider.connectionFields.map((field) => [field.name, initialProviderFieldValue(field)]),
  )
}

export function validatedProviderConnectionFields(
  provider: ProviderCatalogItem,
  values: Record<string, EditableConnectionValue>,
): Record<string, ProviderConnectionValue> {
  const result: Record<string, ProviderConnectionValue> = {}
  for (const field of provider.connectionFields) {
    const value = values[field.name]
    if (field.kind === 'boolean') {
      result[field.name] = value === true
      continue
    }
    const text = typeof value === 'string' ? value.trim() : ''
    if (!text) {
      if (field.required) throw new Error(`${field.label} is required.`)
      continue
    }
    if (field.kind === 'integer') {
      const number = Number(text)
      if (!Number.isSafeInteger(number)) throw new Error(`${field.label} must be a whole number.`)
      result[field.name] = number
      continue
    }
    result[field.name] = text
  }
  return result
}

export interface ExecutionFormValues {
  maxRetries: string
  requestTimeout: string
  streamTimeout: string
}

export type ModelExecutionOverrides = Partial<RoutingExecution>

export function buildModelExecutionOverrides(
  values: ExecutionFormValues,
): ModelExecutionOverrides | undefined {
  const execution: ModelExecutionOverrides = {}
  if (values.maxRetries.trim()) {
    const retries = Number(values.maxRetries)
    if (!Number.isSafeInteger(retries) || retries < 0 || retries > 5) {
      throw new Error('Max retries must be a whole number from 0 to 5.')
    }
    execution.maxRetries = retries
  }
  for (const [label, value, key] of [
    ['Request timeout', values.requestTimeout, 'requestTimeout'],
    ['Stream timeout', values.streamTimeout, 'streamTimeout'],
  ] as const) {
    const normalized = value.trim()
    if (normalized && !validModelTimeout(normalized)) {
      throw new Error(`${label} must be a duration from 1s to 24h, such as 30s or 5m.`)
    }
    if (normalized) execution[key] = normalized
  }
  return Object.keys(execution).length > 0 ? execution : undefined
}

function validModelTimeout(value: string): boolean {
  const match = durationPattern.exec(value)
  if (!match?.groups) return false
  const amount = Number(match.groups.amount)
  const unit = match.groups.unit as keyof typeof durationUnitMilliseconds
  if (!Number.isSafeInteger(amount)) return false
  const multiplier = durationUnitMilliseconds[unit]
  if (!multiplier) return false
  const milliseconds = amount * multiplier
  return (
    Number.isSafeInteger(milliseconds) &&
    milliseconds >= minimumModelTimeoutMilliseconds &&
    milliseconds <= maximumModelTimeoutMilliseconds
  )
}

export interface PricingFormValues {
  inputCost: string
  outputCost: string
  cacheReadCost: string
  cacheWriteCost: string
}

export type ModelPricingOverrides = Partial<RoutingPricing>

const decimalPattern = /^(?:0|[1-9]\d*)(?:\.\d+)?$/

export function isDecimalString(value: string): boolean {
  return decimalPattern.test(value)
}

export function buildModelPricingOverrides(
  values: PricingFormValues,
): ModelPricingOverrides | undefined {
  const inputs: Array<[keyof ModelPricingOverrides, string, string]> = [
    ['inputCostPerMillionTokens', values.inputCost, 'Input cost'],
    ['outputCostPerMillionTokens', values.outputCost, 'Output cost'],
    ['cacheReadCostPerMillionTokens', values.cacheReadCost, 'Cache read cost'],
    ['cacheWriteCostPerMillionTokens', values.cacheWriteCost, 'Cache write cost'],
  ]
  const pricing: ModelPricingOverrides = {}
  for (const [key, value, label] of inputs) {
    const normalized = value.trim()
    if (normalized && !isDecimalString(normalized)) {
      throw new Error(`${label} must be a non-negative decimal.`)
    }
    if (normalized) pricing[key] = normalized
  }
  return Object.keys(pricing).length > 0 ? pricing : undefined
}

interface BuildBulkImportRequestInput {
  provider: ProviderCatalogItem
  interfaceId?: string
  catalogRevision: string
  discoveryClaim: string
  credentialId?: string
  baseUrl?: string
  connectionFields: Record<string, ProviderConnectionValue>
  models: DiscoveredProviderModel[]
  selectedCatalogItemIds: ReadonlySet<string>
  namePrefix: string
  execution?: ModelExecutionOverrides
  pricing?: ModelPricingOverrides
}

const defaultExecution: RoutingExecution = {
  maxRetries: 0,
  requestTimeout: '300s',
  streamTimeout: '300s',
}

const defaultPricing: RoutingPricing = {
  inputCostPerMillionTokens: null,
  outputCostPerMillionTokens: null,
  cacheReadCostPerMillionTokens: null,
  cacheWriteCostPerMillionTokens: null,
}

/**
 * Builds the Router-owned bulk-import command from one signed discovery page.
 * Selection order intentionally follows the discovery result, because the
 * Router verifies that order against the signed claim.
 */
export function buildRoutingBulkImportRequest({
  provider,
  interfaceId,
  catalogRevision,
  discoveryClaim,
  credentialId,
  baseUrl,
  connectionFields,
  models,
  selectedCatalogItemIds,
  namePrefix,
  execution,
  pricing,
}: BuildBulkImportRequestInput): RoutingBulkImportRequest {
  const prefix = normalizedPrefix(namePrefix)
  const selections = models
    .filter((model) => selectedCatalogItemIds.has(model.catalogItemId))
    .map((model) => ({
      catalogItemId: model.catalogItemId,
      name: `${prefix}${model.providerModelId}`,
      aliases: [],
      capabilities: [...model.capabilities],
      loras: [],
      execution: { ...defaultExecution, ...execution },
      pricing: { ...defaultPricing, ...pricing },
    }))

  if (selections.length === 0) throw new Error('Select at least one model.')

  const normalizedCredentialId = credentialId?.trim()
  const normalizedBaseUrl = baseUrl?.trim()
  const normalizedInterfaceId = interfaceId?.trim()
  return {
    providerId: provider.providerId,
    ...(normalizedInterfaceId ? { interfaceId: normalizedInterfaceId } : {}),
    catalogRevision,
    discoveryClaim,
    ...(normalizedCredentialId ? { credentialId: normalizedCredentialId } : {}),
    ...(provider.origin.mode === 'user_supplied' && normalizedBaseUrl
      ? { baseUrl: normalizedBaseUrl }
      : {}),
    ...(Object.keys(connectionFields).length > 0 ? { connectionFields } : {}),
    weight: '1',
    selections,
  }
}
