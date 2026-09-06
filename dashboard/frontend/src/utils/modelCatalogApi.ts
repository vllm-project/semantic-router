import type {
  BuiltInModelCatalog,
  BuiltInModelCatalogVersion,
  BuiltInModelMetadata,
  BuiltInModelRole,
  BuiltInModelVerification,
  ModelCatalogChannel,
} from '../types/modelCatalog'

export class ModelCatalogApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ModelCatalogApiError'
    this.status = status
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0
}

function isNonEmptyStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.length > 0 && value.every(isNonEmptyString)
}

function isCatalogChannel(value: unknown): value is ModelCatalogChannel {
  return value === 'latest' || value === 'release'
}

function isCatalogVersion(value: unknown): value is BuiltInModelCatalogVersion {
  return (
    isRecord(value) &&
    isNonEmptyString(value.catalog_version) &&
    isCatalogChannel(value.channel) &&
    isNonEmptyString(value.default_model) &&
    isNonEmptyStringArray(value.enabled_models)
  )
}

function isCatalogRole(value: unknown): value is BuiltInModelRole {
  return (
    isRecord(value) &&
    isNonEmptyString(value.name) &&
    typeof value.required === 'boolean' &&
    Number.isInteger(value.minimum_candidates) &&
    Number(value.minimum_candidates) >= 1 &&
    isNonEmptyStringArray(value.traits) &&
    isNonEmptyStringArray(value.recommended_pool)
  )
}

function isVerification(value: unknown): value is BuiltInModelVerification {
  return (
    isRecord(value) &&
    value.status === 'verified' &&
    isNonEmptyString(value.authority) &&
    typeof value.asset_sha256 === 'string' &&
    /^sha256:[0-9a-f]{64}$/.test(value.asset_sha256)
  )
}

function isCatalogModel(value: unknown): value is BuiltInModelMetadata {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    isNonEmptyString(value.description) &&
    value.kind === 'virtual' &&
    isNonEmptyString(value.family) &&
    Number.isInteger(value.generation) &&
    Number(value.generation) >= 1 &&
    isNonEmptyString(value.policy_version) &&
    isNonEmptyString(value.entrypoint) &&
    isNonEmptyString(value.recipe) &&
    isNonEmptyStringArray(value.protocols) &&
    isNonEmptyStringArray(value.traits) &&
    Array.isArray(value.roles) &&
    value.roles.length > 0 &&
    value.roles.every(isCatalogRole) &&
    isNonEmptyString(value.catalog_version) &&
    isCatalogChannel(value.channel) &&
    typeof value.compatible === 'boolean' &&
    isNonEmptyString(value.compatibility_reason) &&
    typeof value.enabled_by_default === 'boolean' &&
    typeof value.default === 'boolean' &&
    isVerification(value.verification)
  )
}

function isBuiltInModelCatalog(value: unknown): value is BuiltInModelCatalog {
  if (!isRecord(value)) return false
  return (
    Array.isArray(value.catalogs) &&
    value.catalogs.length > 0 &&
    value.catalogs.every(isCatalogVersion) &&
    Array.isArray(value.models) &&
    value.models.length > 0 &&
    value.models.every(isCatalogModel)
  )
}

export async function getBuiltInModelCatalog(signal?: AbortSignal): Promise<BuiltInModelCatalog> {
  const response = await fetch('/api/models/catalog', { signal })
  if (!response.ok) {
    throw new ModelCatalogApiError(
      `Built-in model catalog is unavailable (HTTP ${response.status}).`,
      response.status,
    )
  }
  const payload: unknown = await response.json()
  if (!isBuiltInModelCatalog(payload)) {
    throw new ModelCatalogApiError('Built-in model catalog returned an invalid contract.', 502)
  }
  return payload
}
