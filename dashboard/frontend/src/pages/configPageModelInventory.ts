import type { ConfigData, NormalizedModel } from './configPageSupport'

const benchmarkIdentityPattern =
  /^[a-z0-9][a-z0-9._-]*(?:\/[a-z0-9][a-z0-9._-]*)+@[0-9]+(?:\.[0-9]+\.[0-9]+)?$/

const isISOCalendarDate = (value: string): boolean => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const [year, month, day] = value.split('-').map(Number)
  const parsed = new Date(Date.UTC(year, month - 1, day))
  return (
    parsed.getUTCFullYear() === year &&
    parsed.getUTCMonth() === month - 1 &&
    parsed.getUTCDate() === day
  )
}

const isNumericInput = (value: unknown): boolean =>
  (typeof value === 'number' && Number.isFinite(value)) ||
  (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value)))

export type ModelEndpointFilter = 'all' | 'configured' | 'missing'
export type ModelRoleFilter = 'all' | 'default' | 'standard'

export interface ModelInventoryFilters {
  search: string
  reasoningFamily: string
  endpointState: ModelEndpointFilter
  role: ModelRoleFilter
  defaultModel: string
}

function searchableModelValues(model: NormalizedModel): string[] {
  return [
    model.name,
    model.provider_model_id,
    model.api_format,
    model.reasoning_family,
    model.modality,
    model.param_size,
    model.description,
    ...(model.tags ?? []),
    ...(model.capabilities ?? []),
    ...(model.endpoints ?? []).flatMap((endpoint) => [endpoint.name, endpoint.protocol]),
    ...(model.backend_refs ?? []).flatMap((backend) => [backend.name, backend.provider]),
  ].filter((value): value is string => typeof value === 'string' && value.length > 0)
}

export function filterModelInventory(
  models: NormalizedModel[],
  filters: ModelInventoryFilters,
): NormalizedModel[] {
  const query = filters.search.trim().toLocaleLowerCase()

  return models.filter((model) => {
    if (
      query &&
      !searchableModelValues(model).some((value) => value.toLocaleLowerCase().includes(query))
    ) {
      return false
    }

    if (filters.reasoningFamily === '__unassigned__' && model.reasoning_family) {
      return false
    }
    if (
      filters.reasoningFamily !== 'all' &&
      filters.reasoningFamily !== '__unassigned__' &&
      model.reasoning_family !== filters.reasoningFamily
    ) {
      return false
    }

    const endpointCount = model.endpoints?.length ?? 0
    if (filters.endpointState === 'configured' && endpointCount === 0) {
      return false
    }
    if (filters.endpointState === 'missing' && endpointCount > 0) {
      return false
    }

    const isDefault = model.name === filters.defaultModel
    if (filters.role === 'default' && !isDefault) {
      return false
    }
    if (filters.role === 'standard' && isDefault) {
      return false
    }

    return true
  })
}

export function getReasoningFamilyFilterOptions(models: NormalizedModel[]): string[] {
  return [
    ...new Set(
      models
        .map((model) => model.reasoning_family?.trim())
        .filter((family): family is string => Boolean(family)),
    ),
  ].sort((left, right) => left.localeCompare(right, undefined, { sensitivity: 'base' }))
}

export function getModelReferenceCounts(config: ConfigData | null): Map<string, number> {
  const counts = new Map<string, number>()
  const decisions = [
    ...(config?.routing?.decisions ?? config?.decisions ?? []),
    ...(config?.recipes ?? []).flatMap((recipe) => recipe.routing.decisions ?? []),
  ]

  for (const decision of decisions) {
    const models = new Set(
      (decision.modelRefs ?? []).map((reference) => reference.model).filter(Boolean),
    )
    collectAlgorithmModelReferences(decision.algorithm, models)
    collectAlgorithmModelReferences(decision.candidateIterations, models)
    for (const model of models) {
      counts.set(model, (counts.get(model) ?? 0) + 1)
    }
  }

  return counts
}

function collectAlgorithmModelReferences(
  value: unknown,
  references: Set<string>,
  fieldName = '',
): void {
  if (typeof value === 'string') {
    if (fieldName === 'model' || fieldName.endsWith('_model')) {
      const modelName = value.trim()
      if (modelName) references.add(modelName)
    }
    return
  }

  if (Array.isArray(value)) {
    if (fieldName === 'models' || fieldName === 'model_names' || fieldName.endsWith('_models')) {
      for (const modelName of value) {
        if (typeof modelName === 'string' && modelName.trim()) references.add(modelName.trim())
        else collectAlgorithmModelReferences(modelName, references)
      }
      return
    }
    for (const item of value) collectAlgorithmModelReferences(item, references)
    return
  }

  if (!value || typeof value !== 'object') return
  for (const [key, nestedValue] of Object.entries(value)) {
    collectAlgorithmModelReferences(nestedValue, references, key)
  }
}

export function getModelDeleteBlocker(
  modelName: string,
  defaultModel: string,
  referenceCounts: ReadonlyMap<string, number>,
): string | null {
  if (modelName === defaultModel) {
    return 'Choose a different default model before deleting this model.'
  }

  const references = referenceCounts.get(modelName) ?? 0
  if (references > 0) {
    return `Remove this model from ${references} routing ${references === 1 ? 'decision' : 'decisions'} before deleting it.`
  }

  return null
}

export function validateNewModelName(rawName: unknown, existingModels: NormalizedModel[]): string {
  const modelName = typeof rawName === 'string' ? rawName.trim() : ''
  if (!modelName) {
    throw new Error('Model name is required.')
  }
  if (existingModels.some((model) => model.name === modelName)) {
    throw new Error(`Model "${modelName}" already exists.`)
  }
  return modelName
}

export function validateModelStructuredFields(data: Record<string, unknown>): void {
  validateModelReasoningFields(data)
  validateModelCollectionShapes(data)
  validateModelLoras(data.loras)
  validateModelBackends(data.backend_refs)
  validateModelEvaluations(data.evaluations)
  validateModelObjectShapes(data)
  validateExternalModelIDs(data.external_model_ids)
  validateModelPricing(data.pricing)
}

function validateModelReasoningFields(data: Record<string, unknown>): void {
  const catalog = typeof data.catalog === 'string' ? data.catalog.trim() : ''
  const reasoningFamily = textValue(data.reasoning_family)
  const reasoningType = textValue(data.reasoning_type)
  const reasoningParameter = textValue(data.reasoning_parameter)
  const customReasoning = [
    reasoningFamily,
    reasoningType,
    reasoningParameter,
    textValue(data.reasoning_default),
    textValue(data.reasoning_levels),
  ].some(Boolean)
  if (catalog && customReasoning) {
    throw new Error('Built-in catalog models inherit reasoning; clear the custom reasoning fields.')
  }
  if (reasoningFamily && (reasoningType || reasoningParameter)) {
    throw new Error('Choose a built-in reasoning family or inline reasoning fields, not both.')
  }
  if ((reasoningType || reasoningParameter) && (!reasoningType || !reasoningParameter)) {
    throw new Error('Inline reasoning requires both type and parameter.')
  }
}

const textValue = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

function validateModelCollectionShapes(data: Record<string, unknown>): void {
  const arrayFields = [
    ['backend_refs', 'Backend Refs'],
    ['loras', 'LoRAs'],
    ['evaluations', 'Evaluations'],
  ] as const
  for (const [field, label] of arrayFields) {
    const value = data[field]
    if (value !== undefined && !Array.isArray(value)) {
      throw new Error(`${label} must be a JSON array.`)
    }
  }

  const stringListFields = [
    ['tags', 'Tags'],
    ['capabilities', 'Capabilities'],
  ] as const
  for (const [field, label] of stringListFields) {
    const value = data[field]
    if (
      value !== undefined &&
      (!Array.isArray(value) || value.some((item) => typeof item !== 'string'))
    ) {
      throw new Error(`${label} must be a list of text values.`)
    }
  }
}

function validateModelLoras(value: unknown): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) {
        throw new Error(`LoRA adapter ${index + 1} must be a structured object.`)
      }
      const lora = item as Record<string, unknown>
      if (typeof lora.name !== 'string' || !lora.name.trim()) {
        throw new Error(`LoRA adapter ${index + 1} requires a name.`)
      }
      if (lora.description !== undefined && typeof lora.description !== 'string') {
        throw new Error(`LoRA adapter ${index + 1} description must be text.`)
      }
    })
  }
}

function validateModelBackends(value: unknown): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) {
        throw new Error(`Provider backend ${index + 1} must be a structured object.`)
      }

      const backend = item as Record<string, unknown>
      const endpoint = typeof backend.endpoint === 'string' ? backend.endpoint.trim() : ''
      const baseUrl = typeof backend.base_url === 'string' ? backend.base_url.trim() : ''
      if (!endpoint && !baseUrl) {
        throw new Error(`Provider backend ${index + 1} requires an endpoint or base URL.`)
      }
      if (typeof backend.provider !== 'string' || !backend.provider.trim()) {
        throw new Error(`Provider backend ${index + 1} requires a catalog Provider ID.`)
      }
      if (
        backend.protocol !== undefined &&
        backend.protocol !== 'http' &&
        backend.protocol !== 'https'
      ) {
        throw new Error(`Provider backend ${index + 1} protocol must be HTTP or HTTPS.`)
      }
      if (
        backend.weight !== undefined &&
        (typeof backend.weight !== 'number' ||
          !Number.isFinite(backend.weight) ||
          backend.weight < 0)
      ) {
        throw new Error(`Provider backend ${index + 1} weight must be zero or greater.`)
      }
      if (
        backend.extra_headers !== undefined &&
        (!backend.extra_headers ||
          typeof backend.extra_headers !== 'object' ||
          Array.isArray(backend.extra_headers) ||
          Object.values(backend.extra_headers).some(
            (headerValue) => typeof headerValue !== 'string',
          ))
      ) {
        throw new Error(
          `Provider backend ${index + 1} extra headers must contain text key/value pairs.`,
        )
      }
    })
  }
}

function validateModelEvaluations(value: unknown): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) {
        throw new Error(`Evaluation ${index + 1} must be a structured object.`)
      }
      validateModelEvaluation(item as Record<string, unknown>, index)
    })
  }
}

function validateModelEvaluation(evaluation: Record<string, unknown>, index: number): void {
  if (
    typeof evaluation.benchmark !== 'string' ||
    !benchmarkIdentityPattern.test(evaluation.benchmark.trim())
  ) {
    throw new Error(`Evaluation ${index + 1} requires a namespaced, versioned benchmark.`)
  }
  if (
    !evaluation.metrics ||
    typeof evaluation.metrics !== 'object' ||
    Array.isArray(evaluation.metrics) ||
    Object.keys(evaluation.metrics).length === 0
  ) {
    throw new Error(`Evaluation ${index + 1} requires at least one metric.`)
  }
  for (const [metric, rawValue] of Object.entries(evaluation.metrics as Record<string, unknown>)) {
    if (!metric.trim() || !isNumericInput(rawValue)) {
      throw new Error(`Evaluation ${index + 1} metrics must be non-empty numeric pairs.`)
    }
  }
  if (
    evaluation.measured_at !== undefined &&
    (typeof evaluation.measured_at !== 'string' || !isISOCalendarDate(evaluation.measured_at))
  ) {
    throw new Error(`Evaluation ${index + 1} measured_at must use YYYY-MM-DD.`)
  }
  if (evaluation.metadata !== undefined) {
    validateEvaluationMetadata(evaluation.metadata, index)
  }
}

function validateEvaluationMetadata(value: unknown, index: number): void {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Evaluation ${index + 1} metadata must be a JSON object.`)
  }
  for (const [key, item] of Object.entries(value)) {
    const scalar =
      item === null ||
      typeof item === 'string' ||
      typeof item === 'boolean' ||
      (typeof item === 'number' && Number.isFinite(item))
    if (!key.trim() || !scalar) {
      throw new Error(`Evaluation ${index + 1} metadata must contain scalar values.`)
    }
  }
}

function validateModelObjectShapes(data: Record<string, unknown>): void {
  const objectFields = [
    ['external_model_ids', 'External Model IDs'],
    ['pricing', 'Pricing'],
  ] as const
  for (const [field, label] of objectFields) {
    const value = data[field]
    if (value !== undefined && (!value || typeof value !== 'object' || Array.isArray(value))) {
      throw new Error(`${label} must be a JSON object.`)
    }
  }
}

function validateExternalModelIDs(value: unknown): void {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    for (const [provider, modelId] of Object.entries(value)) {
      if (!provider.trim() || typeof modelId !== 'string' || !modelId.trim()) {
        throw new Error('External Model IDs must contain non-empty provider/model ID pairs.')
      }
    }
  }
}

function validateModelPricing(value: unknown): void {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const pricing = value as Record<string, unknown>
    if (pricing.currency !== undefined && typeof pricing.currency !== 'string') {
      throw new Error('Pricing currency must be text.')
    }
    const rateFields = [
      'prompt_per_1m',
      'cached_input_per_1m',
      'cache_write_per_1m',
      'completion_per_1m',
    ]
    for (const field of rateFields) {
      const value = pricing[field]
      if (
        value !== undefined &&
        (typeof value !== 'number' || !Number.isFinite(value) || value < 0)
      ) {
        throw new Error(`Pricing ${field} must be zero or greater.`)
      }
    }
  }
}
