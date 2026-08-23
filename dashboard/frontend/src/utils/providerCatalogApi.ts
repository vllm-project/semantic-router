import {
  ManagementApiError,
  hasOnlyKeys,
  isNonEmptyString,
  isOptionalString,
  isRecord,
  isStringArray,
  managementOperationRequest,
} from './managementApiContract'

export type ProviderCredentialMode = 'none' | 'optional' | 'required'
export type ProviderOriginMode = 'fixed' | 'user_supplied'
export type ProviderConnectionFieldKind = 'text' | 'boolean' | 'integer' | 'select'
export type ProviderConnectionValue = string | boolean | number

export interface ProviderCatalogIcon {
  source: 'lobe' | 'asset' | 'url'
  value: string
  color: boolean
}

export interface ProviderCatalogDisplay {
  name: string
  description: string
  category: string
  icon: ProviderCatalogIcon
  monogram?: string
  accent?: string
}

export interface ProviderCredentialPrompt {
  mode: ProviderCredentialMode
  label?: string
  hint?: string
}

export interface ProviderOriginPrompt {
  mode: ProviderOriginMode
  defaultUrl?: string
  baseUrlRequired: boolean
  label?: string
  hint?: string
}

export interface ProviderFieldOption {
  value: string
  label: string
}

export interface ProviderConnectionField {
  name: string
  label: string
  kind: ProviderConnectionFieldKind
  required: boolean
  advanced: boolean
  default?: string
  hint?: string
  placeholder?: string
  options?: ProviderFieldOption[]
}

export interface ProviderInterface {
  id: string
  label: string
  default: boolean
  capabilities: string[]
}

export interface ProviderCatalogItem {
  providerId: string
  revision: string
  display: ProviderCatalogDisplay
  credential: ProviderCredentialPrompt
  origin: ProviderOriginPrompt
  discoverySupported: boolean
  capabilities: string[]
  connectionFields: ProviderConnectionField[]
  interfaces: ProviderInterface[]
}

export interface ManagementPageInfo {
  nextCursor?: string
  hasMore: boolean
  pageSize: number
}

export interface ProviderCatalogPage {
  data: ProviderCatalogItem[]
  page: ManagementPageInfo
  catalogRevision: string
  categories: string[]
}

export interface ProviderCatalogDetail {
  data: ProviderCatalogItem
  catalogRevision: string
}

export interface ProviderCatalogFilters {
  cursor?: string
  pageSize?: number
  search?: string
  category?: string
  capability?: string
}

export interface DiscoverProviderModelsInput {
  credentialId?: string
  baseUrl?: string
  connectionFields?: Record<string, ProviderConnectionValue>
  search?: string
  pageSize?: number
  cursor?: string
}

export interface DiscoveredProviderModel {
  catalogItemId: string
  providerModelId: string
  displayName: string
  capabilities: string[]
}

export interface DiscoveredProviderModelPage {
  data: DiscoveredProviderModel[]
  page: ManagementPageInfo
  catalogRevision: string
  discoveryRevision: string
  expiresAt: string
}

const providerIdPattern = /^[a-z][a-z0-9._-]{0,127}$/
const revisionPattern = /^sha256:[a-f0-9]{64}$/

const exact = (
  value: unknown,
  required: readonly string[],
  optional: readonly string[] = [],
): value is Record<string, unknown> => isRecord(value) && hasOnlyKeys(value, required, optional)

function isDisplay(value: unknown): value is ProviderCatalogDisplay {
  return (
    exact(value, ['name', 'description', 'category', 'icon'], ['monogram', 'accent']) &&
    isNonEmptyString(value.name) &&
    isNonEmptyString(value.description) &&
    isNonEmptyString(value.category) &&
    isIcon(value.icon) &&
    isOptionalString(value.monogram) &&
    isOptionalString(value.accent)
  )
}

function isIcon(value: unknown): value is ProviderCatalogIcon {
  if (
    !exact(value, ['source', 'value', 'color']) ||
    !['lobe', 'asset', 'url'].includes(String(value.source)) ||
    !isNonEmptyString(value.value) ||
    typeof value.color !== 'boolean'
  ) {
    return false
  }
  if (value.source === 'lobe') return /^[a-z0-9][a-z0-9-]{0,127}$/.test(value.value)
  if (value.source === 'asset') {
    return (
      value.value.startsWith('/') && !value.value.startsWith('//') && !value.value.includes('..')
    )
  }
  try {
    const parsed = new URL(value.value)
    return (
      parsed.protocol === 'https:' &&
      parsed.username === '' &&
      parsed.password === '' &&
      parsed.hash === ''
    )
  } catch {
    return false
  }
}

function isCredential(value: unknown): value is ProviderCredentialPrompt {
  return (
    exact(value, ['mode'], ['label', 'hint']) &&
    (value.mode === 'none' || value.mode === 'optional' || value.mode === 'required') &&
    isOptionalString(value.label) &&
    isOptionalString(value.hint)
  )
}

function isOrigin(value: unknown): value is ProviderOriginPrompt {
  if (
    !exact(value, ['mode', 'baseUrlRequired'], ['defaultUrl', 'label', 'hint']) ||
    (value.mode !== 'fixed' && value.mode !== 'user_supplied') ||
    typeof value.baseUrlRequired !== 'boolean' ||
    !isOptionalString(value.defaultUrl) ||
    !isOptionalString(value.label) ||
    !isOptionalString(value.hint)
  ) {
    return false
  }
  return value.mode === 'fixed'
    ? value.baseUrlRequired === false && isNonEmptyString(value.defaultUrl)
    : value.baseUrlRequired === true && value.defaultUrl === undefined
}

function isFieldOption(value: unknown): value is ProviderFieldOption {
  return (
    exact(value, ['value', 'label']) &&
    isNonEmptyString(value.value) &&
    isNonEmptyString(value.label)
  )
}

function isConnectionField(value: unknown): value is ProviderConnectionField {
  if (
    !exact(
      value,
      ['name', 'label', 'kind', 'required', 'advanced'],
      ['default', 'hint', 'placeholder', 'options'],
    ) ||
    !providerIdPattern.test(String(value.name)) ||
    !isNonEmptyString(value.label) ||
    !['text', 'boolean', 'integer', 'select'].includes(String(value.kind)) ||
    typeof value.required !== 'boolean' ||
    typeof value.advanced !== 'boolean' ||
    !isOptionalString(value.default) ||
    !isOptionalString(value.hint) ||
    !isOptionalString(value.placeholder) ||
    (value.options !== undefined &&
      (!Array.isArray(value.options) || !value.options.every(isFieldOption)))
  ) {
    return false
  }
  return value.kind !== 'select' || (Array.isArray(value.options) && value.options.length > 0)
}

function isProviderInterface(value: unknown): value is ProviderInterface {
  return (
    exact(value, ['id', 'label', 'default', 'capabilities']) &&
    providerIdPattern.test(String(value.id)) &&
    isNonEmptyString(value.label) &&
    typeof value.default === 'boolean' &&
    isStringArray(value.capabilities)
  )
}

function isProvider(value: unknown): value is ProviderCatalogItem {
  return (
    exact(value, [
      'providerId',
      'revision',
      'display',
      'credential',
      'origin',
      'discoverySupported',
      'capabilities',
      'connectionFields',
      'interfaces',
    ]) &&
    isNonEmptyString(value.providerId) &&
    providerIdPattern.test(value.providerId) &&
    validRevision(value.revision) &&
    isDisplay(value.display) &&
    isCredential(value.credential) &&
    isOrigin(value.origin) &&
    typeof value.discoverySupported === 'boolean' &&
    isStringArray(value.capabilities) &&
    Array.isArray(value.connectionFields) &&
    value.connectionFields.every(isConnectionField) &&
    Array.isArray(value.interfaces) &&
    value.interfaces.length > 0 &&
    value.interfaces.every(isProviderInterface) &&
    value.interfaces.filter((providerInterface) => providerInterface.default).length === 1
  )
}

function isPage(value: unknown): value is ManagementPageInfo {
  return (
    exact(value, ['hasMore', 'pageSize'], ['nextCursor']) &&
    typeof value.hasMore === 'boolean' &&
    Number.isSafeInteger(value.pageSize) &&
    Number(value.pageSize) >= 1 &&
    Number(value.pageSize) <= 200 &&
    (value.nextCursor === undefined || isNonEmptyString(value.nextCursor)) &&
    (!value.hasMore || isNonEmptyString(value.nextCursor))
  )
}

function validRevision(value: unknown): value is string {
  return typeof value === 'string' && revisionPattern.test(value)
}

function parseProviderPage(payload: unknown): ProviderCatalogPage {
  if (
    !exact(payload, ['data', 'page', 'catalogRevision', 'categories']) ||
    !Array.isArray(payload.data) ||
    !payload.data.every(isProvider) ||
    !isPage(payload.page) ||
    !validRevision(payload.catalogRevision) ||
    !isStringArray(payload.categories)
  ) {
    throw new ManagementApiError('Provider catalog returned an invalid contract.', 502)
  }
  return payload as unknown as ProviderCatalogPage
}

function parseProviderDetail(payload: unknown): ProviderCatalogDetail {
  if (
    !exact(payload, ['data', 'catalogRevision']) ||
    !isProvider(payload.data) ||
    !validRevision(payload.catalogRevision)
  ) {
    throw new ManagementApiError('Provider detail returned an invalid contract.', 502)
  }
  return payload as unknown as ProviderCatalogDetail
}

function isDiscoveredModel(value: unknown): value is DiscoveredProviderModel {
  return (
    exact(value, ['catalogItemId', 'providerModelId', 'displayName', 'capabilities']) &&
    isNonEmptyString(value.catalogItemId) &&
    isNonEmptyString(value.providerModelId) &&
    isNonEmptyString(value.displayName) &&
    isStringArray(value.capabilities)
  )
}

function parseDiscoveredPage(payload: unknown): DiscoveredProviderModelPage {
  if (
    !exact(payload, ['data', 'page', 'catalogRevision', 'discoveryRevision', 'expiresAt']) ||
    !Array.isArray(payload.data) ||
    !payload.data.every(isDiscoveredModel) ||
    !isPage(payload.page) ||
    !validRevision(payload.catalogRevision) ||
    !isNonEmptyString(payload.discoveryRevision) ||
    !isNonEmptyString(payload.expiresAt) ||
    Number.isNaN(Date.parse(payload.expiresAt))
  ) {
    throw new ManagementApiError('Model discovery returned an invalid contract.', 502)
  }
  return payload as unknown as DiscoveredProviderModelPage
}

function appendOptionalQuery(query: URLSearchParams, name: string, value?: string): void {
  const normalized = value?.trim()
  if (normalized) query.set(name, normalized)
}

function validatePageSize(pageSize?: number): void {
  if (pageSize !== undefined && (!Number.isInteger(pageSize) || pageSize < 1 || pageSize > 200)) {
    throw new RangeError('pageSize must be between 1 and 200.')
  }
}

export async function listProviderCatalog(
  filters: ProviderCatalogFilters = {},
  signal?: AbortSignal,
): Promise<ProviderCatalogPage> {
  validatePageSize(filters.pageSize)
  const query = new URLSearchParams()
  appendOptionalQuery(query, 'cursor', filters.cursor)
  if (filters.pageSize !== undefined) query.set('pageSize', String(filters.pageSize))
  appendOptionalQuery(query, 'search', filters.search)
  appendOptionalQuery(query, 'category', filters.category)
  appendOptionalQuery(query, 'capability', filters.capability)
  return parseProviderPage(await managementOperationRequest('getProviders', { query, signal }))
}

export async function getProviderCatalogDetail(
  providerId: string,
  signal?: AbortSignal,
): Promise<ProviderCatalogDetail> {
  if (!providerIdPattern.test(providerId)) throw new TypeError('providerId is invalid.')
  return parseProviderDetail(
    await managementOperationRequest('getProvidersByProviderId', {
      pathParameters: { providerId },
      signal,
    }),
  )
}

export async function discoverProviderModels(
  providerId: string,
  input: DiscoverProviderModelsInput,
  signal?: AbortSignal,
): Promise<DiscoveredProviderModelPage> {
  if (!providerIdPattern.test(providerId)) throw new TypeError('providerId is invalid.')
  validatePageSize(input.pageSize)
  return parseDiscoveredPage(
    await managementOperationRequest('postProvidersByProviderIdDiscoverModels', {
      pathParameters: { providerId },
      body: input,
      signal,
    }),
  )
}
