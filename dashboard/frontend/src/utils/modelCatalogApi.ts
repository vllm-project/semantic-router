import type {
  BuiltInModelCatalog,
  BuiltInModelCatalogVersion,
  BuiltInModelMetadata,
  BuiltInModelRole,
  CatalogBenchmark,
  CatalogEvaluation,
  CatalogIndex,
  CatalogIndexResult,
  CatalogModelBinding,
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

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isNonEmptyStringRecord(value: unknown): value is Record<string, string> {
  return (
    isRecord(value) &&
    Object.keys(value).length > 0 &&
    Object.entries(value).every(([key, item]) => isNonEmptyString(key) && isNonEmptyString(item))
  )
}

function isISOCalendarDate(value: unknown): value is string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const parsed = new Date(`${value}T00:00:00Z`)
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value
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
  const hasValidAssetDigest =
    typeof value.asset_sha256 === 'string' && /^sha256:[0-9a-f]{64}$/.test(value.asset_sha256)
  return virtual ? hasValidAssetDigest : value.asset_sha256 === undefined || hasValidAssetDigest
}

function isCatalogModel(value: unknown): value is BuiltInModelMetadata {
  if (!isRecord(value)) return false
  const virtual = value.kind === 'virtual'
  return (
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    isNonEmptyString(value.description) &&
    (virtual || value.kind === 'physical') &&
    isNonEmptyString(value.publisher) &&
    isRecord(value.presentation) &&
    isNonEmptyString(value.presentation.logo) &&
    isNonEmptyString(value.presentation.monogram) &&
    typeof value.presentation.monochrome === 'boolean' &&
    isRecord(value.distribution) &&
    ['proprietary_api', 'open_weights', 'router_recipe'].includes(
      String(value.distribution.type),
    ) &&
    isNonEmptyString(value.distribution.source) &&
    (value.distribution.type !== 'open_weights' || isNonEmptyString(value.distribution.license)) &&
    isNonEmptyString(value.family) &&
    ['experimental', 'active', 'deprecated', 'removed'].includes(String(value.lifecycle)) &&
    isStringArray(value.capabilities) &&
    isRecord(value.modalities) &&
    isStringArray(value.modalities.input) &&
    isStringArray(value.modalities.output) &&
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
    isNonEmptyString(value.default_base_path) &&
    value.default_base_path.startsWith('/') &&
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
      [
        'chat_template_kwargs',
        'top_level_effort',
        'top_level_boolean',
        'top_level_effort_template_switch',
        'top_level_effort_boolean_switch',
        'reasoning_object',
        'thinking_object',
        'thinking_object_effort',
        'output_config_effort',
        'deepseek_thinking',
      ].includes(String(value.reasoning_transport))) &&
    isRecord(value.auth) &&
    ['none', 'bearer', 'api_key_header'].includes(String(value.auth.strategy)) &&
    typeof value.auth.header === 'string' &&
    typeof value.auth.prefix === 'string' &&
    isRecord(value.presentation) &&
    isNonEmptyString(value.presentation.logo) &&
    isNonEmptyString(value.presentation.monogram) &&
    typeof value.presentation.monochrome === 'boolean' &&
    (value.presentation.featured === undefined ||
      typeof value.presentation.featured === 'boolean') &&
    isRecord(value.conformance) &&
    ['unverified', 'fixture_verified', 'live_verified'].includes(
      String(value.conformance.status),
    ) &&
    (value.models === undefined ||
      (Array.isArray(value.models) && value.models.every(isCatalogModelBinding)))
  )
}

function isReasoningFamily(value: unknown): value is CatalogReasoningFamily {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    [
      'chat_template_kwargs',
      'reasoning_effort',
      'reasoning_mode',
      'top_level_reasoning_effort',
    ].includes(String(value.type)) &&
    isNonEmptyString(value.parameter) &&
    (value.activation_parameter === undefined ||
      (isNonEmptyString(value.activation_parameter) &&
        value.activation_parameter !== value.parameter)) &&
    (value.effort_flags === undefined || isValidReasoningEffortFlags(value)) &&
    ((['reasoning_effort', 'top_level_reasoning_effort'].includes(String(value.type)) &&
      isStringArray(value.levels)) ||
      (!['reasoning_effort', 'top_level_reasoning_effort'].includes(String(value.type)) &&
        (value.levels === undefined || isStringArray(value.levels, true)))) &&
    (value.default === undefined ||
      (isNonEmptyString(value.default) &&
        isStringArray(value.levels) &&
        value.levels.includes(value.default))) &&
    isStringArray(value.modes) &&
    value.modes.length > 0 &&
    value.modes.every((mode) => ['enabled', 'disabled', 'adaptive'].includes(mode)) &&
    isNonEmptyString(value.default_mode) &&
    value.modes.includes(value.default_mode)
  )
}

function isValidReasoningEffortFlags(value: Record<string, unknown>): boolean {
  if (
    String(value.type) !== 'reasoning_effort' ||
    !isNonEmptyString(value.parameter) ||
    !isNonEmptyString(value.activation_parameter) ||
    !isNonEmptyStringRecord(value.effort_flags) ||
    !isStringArray(value.levels)
  ) {
    return false
  }

  const flags = value.effort_flags
  const levels = value.levels
  const parameters = Object.values(flags)
  const activeLevels = levels.filter((level) => level !== value.disabled)
  return (
    Object.keys(flags).every((effort) => levels.includes(effort)) &&
    new Set(parameters).size === parameters.length &&
    !parameters.includes(value.parameter) &&
    !parameters.includes(value.activation_parameter) &&
    activeLevels.length - Object.keys(flags).length <= 1
  )
}

function isReasoningEffortsByProtocol(
  value: unknown,
  protocols: unknown,
  reasoningEfforts: unknown,
): value is Record<string, string[]> | undefined {
  if (value === undefined) return true
  if (
    !isRecord(value) ||
    Object.keys(value).length === 0 ||
    !isStringArray(protocols) ||
    !isStringArray(reasoningEfforts)
  ) {
    return false
  }
  const boundProtocols = new Set(protocols)
  const providerEfforts = new Set(reasoningEfforts)
  return Object.entries(value).every(
    ([protocol, efforts]) =>
      boundProtocols.has(protocol) &&
      isStringArray(efforts) &&
      new Set(efforts).size === efforts.length &&
      efforts.every((effort) => providerEfforts.has(effort)),
  )
}

function isCatalogModelBinding(value: unknown): value is CatalogModelBinding {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.catalog) &&
    ['first_party', 'managed_cloud', 'gateway', 'self_hosted'].includes(
      String(value.relationship),
    ) &&
    isStringArray(value.protocols) &&
    (value.reasoning_transport === undefined ||
      [
        'chat_template_kwargs',
        'top_level_effort',
        'top_level_boolean',
        'top_level_effort_template_switch',
        'top_level_effort_boolean_switch',
        'reasoning_object',
        'thinking_object',
        'thinking_object_effort',
        'output_config_effort',
        'deepseek_thinking',
      ].includes(String(value.reasoning_transport))) &&
    (value.reasoning_modes === undefined ||
      (isStringArray(value.reasoning_modes) &&
        value.reasoning_modes.every((mode) =>
          ['enabled', 'disabled', 'adaptive'].includes(mode),
        ))) &&
    (value.reasoning_efforts === undefined || isStringArray(value.reasoning_efforts)) &&
    isReasoningEffortsByProtocol(
      value.reasoning_efforts_by_protocol,
      value.protocols,
      value.reasoning_efforts,
    ) &&
    ['experimental', 'active', 'deprecated', 'removed'].includes(String(value.lifecycle)) &&
    isRecord(value.verification) &&
    ['claimed', 'imported', 'reproduced'].includes(String(value.verification.status))
  )
}

function isMetricNormalization(value: unknown): boolean {
  if (!isRecord(value)) return false
  if (value.type === 'identity' || value.type === 'one_minus') return true
  if (value.type === 'linear_clamp') {
    return isFiniteNumber(value.min) && isFiniteNumber(value.max) && value.min < value.max
  }
  if (value.type === 'piecewise_linear') {
    if (!Array.isArray(value.points) || value.points.length < 2) return false
    let previous = Number.NEGATIVE_INFINITY
    return value.points.every((point) => {
      if (
        !isRecord(point) ||
        !isFiniteNumber(point.input) ||
        !isFiniteNumber(point.output) ||
        point.output < 0 ||
        point.output > 1 ||
        point.input <= previous
      ) {
        return false
      }
      previous = point.input
      return true
    })
  }
  if (value.type === 'logistic') {
    return isFiniteNumber(value.k) && value.k !== 0 && isFiniteNumber(value.x0)
  }
  return (
    value.type === 'lookup' &&
    isRecord(value.values) &&
    Object.keys(value.values).length > 0 &&
    Object.entries(value.values).every(
      ([key, item]) => key.length > 0 && isFiniteNumber(item) && item >= 0 && item <= 1,
    )
  )
}

function isBenchmark(value: unknown): value is CatalogBenchmark {
  return (
    isRecord(value) &&
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.display_name) &&
    isNonEmptyString(value.domain) &&
    (value.tags === undefined ||
      (isStringArray(value.tags) &&
        value.tags.length > 0 &&
        new Set(value.tags).size === value.tags.length &&
        value.tags.every((tag) => /^[a-z0-9][a-z0-9._-]*$/.test(tag)))) &&
    isNonEmptyString(value.default_profile) &&
    Array.isArray(value.profiles) &&
    value.profiles.length > 0 &&
    value.profiles.every(
      (profile) =>
        isRecord(profile) &&
        isNonEmptyString(profile.id) &&
        isNonEmptyString(profile.display_name) &&
        isNonEmptyString(profile.description),
    ) &&
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
        metric.range.every(isFiniteNumber) &&
        metric.range[0] < metric.range[1] &&
        (metric.normalization === undefined || isMetricNormalization(metric.normalization)),
    )
  )
}

function isEvaluation(value: unknown): value is CatalogEvaluation {
  if (!isRecord(value)) return false
  const status = String(value.status)
  const metricsAreValid =
    isNumberRecord(value.metrics) &&
    (status !== 'available' || Object.keys(value.metrics).length > 0)
  const measuredAtIsValid = value.measured_at === undefined || isISOCalendarDate(value.measured_at)
  const observedAtIsValid = value.observed_at === undefined || isISOCalendarDate(value.observed_at)
  const availableHasCalendarAnchor =
    status !== 'available' ||
    isISOCalendarDate(value.measured_at) ||
    isISOCalendarDate(value.observed_at)
  return (
    isNonEmptyString(value.id) &&
    isNonEmptyString(value.model) &&
    isNonEmptyString(value.benchmark) &&
    isNonEmptyString(value.benchmark_profile) &&
    isNonEmptyString(value.reasoning_effort) &&
    isRecord(value.subject) &&
    metricsAreValid &&
    measuredAtIsValid &&
    observedAtIsValid &&
    availableHasCalendarAnchor &&
    ['available', 'missing', 'failed', 'not_applicable', 'withheld'].includes(status) &&
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
    value.components.every((component) => {
      if (!isRecord(component)) return false
      const metricReference =
        isNonEmptyString(component.benchmark) &&
        isNonEmptyString(component.metric) &&
        isNonEmptyString(component.benchmark_profile) &&
        !isNonEmptyString(component.index)
      const indexReference =
        isNonEmptyString(component.index) &&
        !isNonEmptyString(component.benchmark) &&
        !isNonEmptyString(component.metric) &&
        !isNonEmptyString(component.benchmark_profile)
      return (
        (metricReference || indexReference) &&
        typeof component.weight === 'number' &&
        component.weight > 0 &&
        isNormalization(component.normalization)
      )
    })
  )
}

function isIndexResult(value: unknown): value is CatalogIndexResult {
  return (
    isRecord(value) &&
    isNonEmptyString(value.model) &&
    isNonEmptyString(value.reasoning_effort) &&
    isNonEmptyString(value.index) &&
    value.status === 'available' &&
    typeof value.score === 'number' &&
    typeof value.coverage === 'number' &&
    value.coverage >= 0 &&
    value.coverage <= 1 &&
    Array.isArray(value.components) &&
    value.components.every((component) => {
      if (!isRecord(component)) return false
      const metricReference =
        isNonEmptyString(component.benchmark) &&
        isNonEmptyString(component.metric) &&
        isNonEmptyString(component.benchmark_profile) &&
        !isNonEmptyString(component.index)
      const indexReference =
        isNonEmptyString(component.index) &&
        !isNonEmptyString(component.benchmark) &&
        !isNonEmptyString(component.metric) &&
        !isNonEmptyString(component.benchmark_profile)
      return (
        (metricReference || indexReference) &&
        typeof component.weight === 'number' &&
        ['available', 'missing', 'failed', 'not_applicable', 'withheld'].includes(
          String(component.status),
        ) &&
        (component.value === null ||
          component.value === undefined ||
          typeof component.value === 'number') &&
        (component.normalized === null ||
          component.normalized === undefined ||
          typeof component.normalized === 'number') &&
        (component.evaluation === undefined || isNonEmptyString(component.evaluation))
      )
    }) &&
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
