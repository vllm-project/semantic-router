import {
  type DiscoverModelsPage,
  type DiscoverModelsRequest,
  type DiscoveredModel,
  type PageInfo,
  type ProviderCatalogDetail as WireProviderCatalogDetail,
  type ProviderCatalogDisplay as WireProviderCatalogDisplay,
  type ProviderCatalogIcon as WireProviderCatalogIcon,
  type ProviderCatalogItem as WireProviderCatalogItem,
  type ProviderCatalogPage as WireProviderCatalogPage,
  type ProviderConnectionField as WireProviderConnectionField,
  type ProviderCredentialPrompt as WireProviderCredentialPrompt,
  type ProviderFieldOption as WireProviderFieldOption,
  type ProviderInterface as WireProviderInterface,
  type ProviderOriginPrompt as WireProviderOriginPrompt,
} from '../generated/managementApiContract'
import { managementOperationRequest } from './managementApiContract'

export type ProviderCredentialMode = WireProviderCredentialPrompt['mode']
export type ProviderOriginMode = WireProviderOriginPrompt['mode']
export type ProviderConnectionFieldKind = WireProviderConnectionField['kind']
export type ProviderConnectionValue = string | boolean | number

export type ProviderCatalogIcon = WireProviderCatalogIcon
export type ProviderCatalogDisplay = WireProviderCatalogDisplay
export type ProviderCredentialPrompt = WireProviderCredentialPrompt
export type ProviderOriginPrompt = WireProviderOriginPrompt
export type ProviderFieldOption = WireProviderFieldOption
export type ProviderConnectionField = WireProviderConnectionField
export type ProviderInterface = WireProviderInterface
export type ProviderCatalogItem = WireProviderCatalogItem
export type ManagementPageInfo = PageInfo
export type ProviderCatalogPage = WireProviderCatalogPage
export type ProviderCatalogDetail = WireProviderCatalogDetail

export interface ProviderCatalogFilters {
  cursor?: string
  pageSize?: number
  search?: string
  category?: string
  /** Filter Provider transport support; this is not a Model capability filter. */
  capability?: string
}

export type DiscoverProviderModelsInput = DiscoverModelsRequest
export type DiscoveredProviderModel = DiscoveredModel
export type DiscoveredProviderModelPage = DiscoverModelsPage

const providerIdPattern = /^[a-z][a-z0-9._-]{0,127}$/

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
  return managementOperationRequest('getProviders', { query, signal })
}

export async function getProviderCatalogDetail(
  providerId: string,
  signal?: AbortSignal,
): Promise<ProviderCatalogDetail> {
  if (!providerIdPattern.test(providerId)) throw new TypeError('providerId is invalid.')
  return managementOperationRequest('getProvidersByProviderId', {
    pathParameters: { providerId },
    signal,
  })
}

export async function discoverProviderModels(
  providerId: string,
  input: DiscoverProviderModelsInput,
  signal?: AbortSignal,
): Promise<DiscoveredProviderModelPage> {
  if (!providerIdPattern.test(providerId)) throw new TypeError('providerId is invalid.')
  validatePageSize(input.pageSize)
  return managementOperationRequest('postProvidersByProviderIdDiscoverModels', {
    pathParameters: { providerId },
    body: input,
    signal,
  })
}
