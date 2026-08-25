import type {
  DiscoveredProviderModel,
  ProviderCatalogItem,
  ProviderConnectionField,
  ProviderConnectionValue,
} from '../utils/providerCatalogApi'
import type {
  FallbackTrigger,
  RoutingBulkImportRequest,
  RoutingModelControl,
  RoutingModelControlWrite,
  RoutingPricing,
} from '../utils/routingManagementApi'
import { normalizedPrefix } from './configPageModelImportSupport'

export type EditableConnectionValue = string | boolean

const durationComponentPattern = /([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(ns|us|µs|μs|ms|s|m|h)/gy
const durationUnitMilliseconds = {
  ns: 0.000_001,
  us: 0.001,
  'µs': 0.001,
  'μs': 0.001,
  ms: 1,
  s: 1_000,
  m: 60_000,
  h: 3_600_000,
} as const
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

export interface ControlFormValues {
  maxRetries: string
  retryOn: FallbackTrigger[]
  requestTimeout: string
  streamTimeout: string
}

export const MODEL_RETRY_TRIGGERS: readonly FallbackTrigger[] = [
  'unavailable',
  'timeout',
]

export type ModelControlOverrides = RoutingModelControlWrite

export function buildModelControlOverrides(
  values: ControlFormValues,
): ModelControlOverrides | undefined {
  const control: ModelControlOverrides = {}
  if (values.maxRetries.trim()) {
    const retries = Number(values.maxRetries)
    if (!Number.isSafeInteger(retries) || retries < 0 || retries > 5) {
      throw new Error('Max retries must be a whole number from 0 to 5.')
    }
    const triggers = [...new Set(values.retryOn)]
    if (triggers.some((value) => !MODEL_RETRY_TRIGGERS.includes(value))) {
      throw new Error('Retry on contains an unsupported condition.')
    }
    if (retries === 0 && triggers.length) {
      throw new Error('Retry conditions require at least one retry.')
    }
    control.retry = {
      count: retries,
      on: retries > 0 ? (triggers.length ? triggers : ['unavailable']) : [],
    }
  } else if (values.retryOn.length > 0) {
    throw new Error('Set Max retries before choosing retry conditions.')
  }
  const timeout: Partial<RoutingModelControl['timeout']> = {}
  for (const [label, value, key] of [
    ['Request timeout', values.requestTimeout, 'request'],
    ['Stream timeout', values.streamTimeout, 'stream'],
  ] as const) {
    const normalized = value.trim()
    if (normalized && !validModelTimeout(normalized)) {
      throw new Error(`${label} must be a duration from 1s to 24h, such as 30s or 5m.`)
    }
    if (normalized) timeout[key] = normalized
  }
  if (Object.keys(timeout).length) control.timeout = timeout
  return Object.keys(control).length > 0 ? control : undefined
}

function validModelTimeout(value: string): boolean {
  const source = value.startsWith('+') ? value.slice(1) : value
  if (!source) return false
  let cursor = 0
  let milliseconds = 0
  durationComponentPattern.lastIndex = 0
  while (cursor < source.length) {
    durationComponentPattern.lastIndex = cursor
    const match = durationComponentPattern.exec(source)
    if (!match || match.index !== cursor) return false
    const amount = Number(match[1])
    const unit = match[2] as keyof typeof durationUnitMilliseconds
    if (!Number.isFinite(amount) || amount < 0) return false
    milliseconds += amount * durationUnitMilliseconds[unit]
    cursor = durationComponentPattern.lastIndex
  }
  return (
    Number.isFinite(milliseconds) &&
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
const maximumModelPrice = 1_000_000n

function isModelPrice(value: string): boolean {
  if (!decimalPattern.test(value)) return false
  const [whole, fraction = ''] = value.split('.')
  if (fraction.length > 9) return false
  const wholeValue = BigInt(whole)
  return (
    wholeValue < maximumModelPrice || (wholeValue === maximumModelPrice && !/[1-9]/.test(fraction))
  )
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
    if (normalized && !isModelPrice(normalized)) {
      throw new Error(`${label} must be a decimal from 0 to 1,000,000 with up to 9 decimals.`)
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
  control?: ModelControlOverrides
  pricing?: ModelPricingOverrides
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
  control,
  pricing,
}: BuildBulkImportRequestInput): RoutingBulkImportRequest {
  const prefix = normalizedPrefix(namePrefix)
  const selections = models
    .filter((model) => selectedCatalogItemIds.has(model.catalogItemId))
    .map((model) => ({
      catalogItemId: model.catalogItemId,
      name: `${prefix}${model.providerModelId}`,
      aliases: [],
      ...(model.capabilities?.length ? { capabilities: [...model.capabilities] } : {}),
      loras: [],
      ...(control ? { control } : {}),
      ...(pricing ? { pricing } : {}),
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
