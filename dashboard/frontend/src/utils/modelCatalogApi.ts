import type {
  BuiltInModelCatalog,
  BuiltInModelCatalogVersion,
  BuiltInModelMetadata,
  BuiltInModelRole,
  CatalogBenchmark,
  CatalogEvaluation,
  CatalogIndex,
  CatalogIndexResult,
  CatalogOffering,
  CatalogProtocol,
  CatalogProvider,
  CatalogReasoningFamily,
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

function isStringArray(value: unknown, allowEmpty = false): value is string[] {
  return Array.isArray(value) && (allowEmpty || value.length > 0) && value.every(isNonEmptyString)
}

function isNumberRecord(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.values(value).every((item) => typeof item === 'number')
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
    isStringArray(value.enabled_models) &&
    isNonEmptyString(value.default_intelligence_index)
  )
}

function isCatalogRole(value: unknown): value is BuiltInModelRole {
  return (
    isRecord(value) &&
    isNonEmptyString(value.name) &&
    typeof value.required === 'boolean' &&
    Number.isInteger(value.minimum_candidates) &&
    Number(value.minimum_candidates) >= 1 &&
    isStringArray(value.traits) &&
    isStringArray(value.recommended_pool)
  )
}

function isVerification(value: unknown, virtual: boolean): boolean {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.authority) ||
    !['claimed', 'imported', 'reproduced'].includes(String(value.status)) ||
    !isNonEmptyString(value.verified_at)
  ) {
    return false
  }
  return (
    !virtual ||
    (typeof value.asset_sha256 === 'string' && /^sha256:[0-9a-f]{64}$/.test(value.asset_sha256))
  )
}

function isCatalogModel(value: unknown): value is BuiltInModelMetadata {
  if (!isRecord(value)) return false
  const virtual = value.kind === 'virtual'
  return (
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    isNonEmptyString(value.description) &&
    (virtual || value.kind === 'physical') &&
    isNonEmptyString(value.family) &&
    ['experimental', 'active', 'deprecated', 'removed'].includes(String(value.lifecycle)) &&
    isStringArray(value.capabilities) &&
    isRecord(value.modalities) &&
    isStringArray(value.modalities.input) &&
    isStringArray(value.modalities.output) &&
    isStringArray(value.protocols) &&
    isVerification(value.verification, virtual) &&
    (!virtual ||
      (Number.isInteger(value.generation) &&
        Number(value.generation) >= 1 &&
        isNonEmptyString(value.policy_version) &&
        isNonEmptyString(value.entrypoint) &&
        isNonEmptyString(value.recipe) &&
        isStringArray(value.traits) &&
        Array.isArray(value.roles) &&
        value.roles.length > 0 &&
        value.roles.every(isCatalogRole)))
  )
}

function isCatalogProtocol(value: unknown): value is CatalogProtocol {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    isNonEmptyString(value.wire_format) &&
    Array.isArray(value.operations) &&
    value.operations.length > 0 &&
    value.operations.every(
      (operation) =>
        isRecord(operation) &&
        isNonEmptyString(operation.id) &&
        ['GET', 'POST', 'DELETE'].includes(String(operation.method)) &&
        typeof operation.path === 'string' &&
        operation.path.startsWith('/'),
    ) &&
    isStringArray(value.capabilities)
  )
}

function isCatalogProvider(value: unknown): value is CatalogProvider {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    isNonEmptyString(value.description) &&
    ['start_here', 'model_api', 'private_runtime'].includes(String(value.category)) &&
    ['native', 'compatible', 'runtime'].includes(String(value.support_tier)) &&
    isStringArray(value.protocols) &&
    isNonEmptyString(value.default_protocol) &&
    isStringArray(value.supported_operations) &&
    value.supported_operations.length > 0 &&
    (value.reasoning_transport === undefined ||
      ['chat_template_kwargs', 'top_level_effort', 'deepseek_thinking'].includes(
        String(value.reasoning_transport),
      )) &&
    isRecord(value.auth) &&
    ['none', 'bearer', 'api_key_header'].includes(String(value.auth.strategy)) &&
    typeof value.auth.header === 'string' &&
    typeof value.auth.prefix === 'string' &&
    isRecord(value.presentation) &&
    isNonEmptyString(value.presentation.logo) &&
    isNonEmptyString(value.presentation.monogram) &&
    typeof value.presentation.monochrome === 'boolean' &&
    isRecord(value.conformance) &&
    ['unverified', 'fixture_verified', 'live_verified'].includes(String(value.conformance.status))
  )
}

function isReasoningFamily(value: unknown): value is CatalogReasoningFamily {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    ['chat_template_kwargs', 'reasoning_effort', 'top_level_reasoning_effort'].includes(
      String(value.type),
    ) &&
    isNonEmptyString(value.parameter) &&
    isStringArray(value.levels) &&
    isNonEmptyString(value.default) &&
    value.levels.includes(value.default)
  )
}

function isOffering(value: unknown): value is CatalogOffering {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.provider) &&
    isNonEmptyString(value.model) &&
    isNonEmptyString(value.provider_model_id) &&
    isStringArray(value.protocols) &&
    ['experimental', 'active', 'deprecated', 'removed'].includes(String(value.lifecycle)) &&
    isRecord(value.verification) &&
    ['claimed', 'imported', 'reproduced'].includes(String(value.verification.status))
  )
}

function isBenchmark(value: unknown): value is CatalogBenchmark {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    isNonEmptyString(value.domain) &&
    Array.isArray(value.metrics) &&
    value.metrics.length > 0 &&
    value.metrics.every(
      (metric) =>
        isRecord(metric) &&
        isNonEmptyString(metric.id) &&
        isNonEmptyString(metric.unit) &&
        ['higher_is_better', 'lower_is_better'].includes(String(metric.direction)) &&
        Array.isArray(metric.range) &&
        metric.range.length === 2 &&
        metric.range.every((bound) => typeof bound === 'number'),
    )
  )
}

function isEvaluation(value: unknown): value is CatalogEvaluation {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.model) &&
    isRecord(value.subject) &&
    isNumberRecord(value.metrics) &&
    Object.keys(value.metrics).length > 0 &&
    ['available', 'missing', 'failed', 'not_applicable', 'withheld'].includes(
      String(value.status),
    ) &&
    isRecord(value.evidence) &&
    ['vendor_claimed', 'third_party', 'vllm_sr_reproduced', 'operator'].includes(
      String(value.evidence.provenance),
    ) &&
    ['claimed', 'imported', 'reproduced'].includes(String(value.evidence.verification)) &&
    typeof value.evidence.redistributable === 'boolean'
  )
}

function isNormalization(value: unknown): boolean {
  if (!isRecord(value)) return false
  const type = String(value.type)
  if (type === 'identity' || type === 'one_minus') return true
  if (type === 'linear_clamp') {
    return typeof value.min === 'number' && typeof value.max === 'number' && value.min < value.max
  }
  if (type === 'piecewise_linear') {
    return (
      Array.isArray(value.points) &&
      value.points.length >= 2 &&
      value.points.every(
        (point) =>
          isRecord(point) && typeof point.input === 'number' && typeof point.output === 'number',
      )
    )
  }
  if (type === 'logistic') {
    return typeof value.k === 'number' && value.k !== 0 && typeof value.x0 === 'number'
  }
  return type === 'lookup' && isNumberRecord(value.values) && Object.keys(value.values).length > 0
}

function isIndex(value: unknown): value is CatalogIndex {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    value.aggregation === 'weighted_mean' &&
    Array.isArray(value.scale) &&
    value.scale.length === 2 &&
    value.scale.every((bound) => typeof bound === 'number') &&
    isRecord(value.missing) &&
    ['require_all', 'require_coverage', 'reported_only'].includes(String(value.missing.policy)) &&
    isNumberRecord(value.domains) &&
    Array.isArray(value.components) &&
    value.components.length > 0 &&
    value.components.every(
      (component) =>
        isRecord(component) &&
        (isNonEmptyString(component.metric) !== isNonEmptyString(component.index)) &&
        typeof component.weight === 'number' &&
        component.weight > 0 &&
        isNormalization(component.normalization),
    )
  )
}

function isIndexResult(value: unknown): value is CatalogIndexResult {
  return (
    isRecord(value) &&
    isNonEmptyString(value.model) &&
    isNonEmptyString(value.index) &&
    ['available', 'missing', 'failed', 'not_applicable', 'withheld'].includes(
      String(value.status),
    ) &&
    (value.score === null || typeof value.score === 'number') &&
    typeof value.coverage === 'number' &&
    value.coverage >= 0 &&
    value.coverage <= 1 &&
    Array.isArray(value.components) &&
    value.components.every(
      (component) =>
        isRecord(component) &&
        (isNonEmptyString(component.metric) !== isNonEmptyString(component.index)) &&
        typeof component.weight === 'number' &&
        ['available', 'missing', 'failed', 'not_applicable', 'withheld'].includes(
          String(component.status),
        ),
    ) &&
    isStringArray(value.provenance, true)
  )
}

function isBuiltInModelCatalog(value: unknown): value is BuiltInModelCatalog {
  if (!isRecord(value) || value.schema_version !== 'vllm-sr/model-catalog/v2') return false
  return (
    Array.isArray(value.catalogs) &&
    value.catalogs.length > 0 &&
    value.catalogs.every(isCatalogVersion) &&
    Array.isArray(value.protocols) &&
    value.protocols.length > 0 &&
    value.protocols.every(isCatalogProtocol) &&
    Array.isArray(value.providers) &&
    value.providers.length > 0 &&
    value.providers.every(isCatalogProvider) &&
    Array.isArray(value.reasoning_families) &&
    value.reasoning_families.every(isReasoningFamily) &&
    Array.isArray(value.models) &&
    value.models.length > 0 &&
    value.models.every(isCatalogModel) &&
    Array.isArray(value.offerings) &&
    value.offerings.every(isOffering) &&
    Array.isArray(value.benchmarks) &&
    value.benchmarks.length > 0 &&
    value.benchmarks.every(isBenchmark) &&
    Array.isArray(value.evaluations) &&
    value.evaluations.every(isEvaluation) &&
    Array.isArray(value.indices) &&
    value.indices.length > 0 &&
    value.indices.every(isIndex) &&
    Array.isArray(value.index_results) &&
    value.index_results.every(isIndexResult)
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
