import type {
  ConfigData,
  ModelConfigEntry,
  NormalizedModel,
  ReasoningFamily,
  VLLMEndpoint,
} from './configPageSupport'

export type LegacyCategoriesConfig = Pick<
  ConfigData,
  'categories' | 'model_config' | 'default_model' | 'default_reasoning_effort'
>

export interface DashboardEditorConfigState {
  editorConfig: ConfigData
  legacyCategoriesConfig: LegacyCategoriesConfig | null
  normalizedLegacyKeys: string[]
}

type SignalSections = NonNullable<ConfigData['signals']>

type LegacySignalSection = Exclude<keyof ConfigData, keyof SignalSections | 'signals'>
type CanonicalSignalSection = keyof SignalSections

const LEGACY_SIGNAL_SECTION_MAP: Array<{
  legacyKey: LegacySignalSection
  signalKey: CanonicalSignalSection
  project?: (value: unknown) => SignalSections[CanonicalSignalSection]
}> = [
  { legacyKey: 'keyword_rules', signalKey: 'keywords' },
  { legacyKey: 'embedding_rules', signalKey: 'embeddings' },
  {
    legacyKey: 'categories',
    signalKey: 'domains',
    project: (value) =>
      Array.isArray(value)
        ? value.map((category) => {
            const domain = category as {
              name: string
              description?: string
              mmlu_categories?: string[]
            }
            return {
              name: domain.name,
              description: domain.description || '',
              mmlu_categories: domain.mmlu_categories,
            }
          })
        : undefined,
  },
  { legacyKey: 'fact_check_rules', signalKey: 'fact_check' },
  { legacyKey: 'user_feedback_rules', signalKey: 'user_feedbacks' },
  { legacyKey: 'preference_rules', signalKey: 'preferences' },
  { legacyKey: 'language_rules', signalKey: 'language' },
  { legacyKey: 'context_rules', signalKey: 'context' },
  { legacyKey: 'complexity_rules', signalKey: 'complexity' },
  { legacyKey: 'jailbreak', signalKey: 'jailbreak' },
  { legacyKey: 'pii', signalKey: 'pii' },
]

const cloneJson = <T,>(value: T): T => {
  if (value === undefined || value === null) {
    return value
  }

  return JSON.parse(JSON.stringify(value)) as T
}

const getProviderModelsFromLegacyConfig = (
  modelConfig: Record<string, ModelConfigEntry> | undefined,
  endpoints: VLLMEndpoint[] | undefined
): NonNullable<ConfigData['providers']>['models'] => {
  if (!modelConfig) {
    return []
  }

  const endpointByName = new Map((endpoints || []).map((endpoint) => [endpoint.name, endpoint]))

  return Object.entries(modelConfig).map(([name, configEntry]) => ({
    name,
    reasoning_family: configEntry.reasoning_family,
    endpoints: (configEntry.preferred_endpoints || [])
      .map((endpointName) => {
        const endpoint = endpointByName.get(endpointName)
        if (!endpoint) {
          return null
        }

        return {
          name: endpoint.name,
          weight: endpoint.weight || 1,
          endpoint: `${endpoint.address}:${endpoint.port}`,
          protocol: 'http' as const,
        }
      })
      .filter((endpoint): endpoint is NonNullable<typeof endpoint> => endpoint !== null),
    pricing: cloneJson(configEntry.pricing),
  }))
}

const normalizeSignals = (
  config: ConfigData,
  normalizedLegacyKeys: Set<string>
): ConfigData['signals'] => {
  const nextSignals = cloneJson(config.signals || {}) as SignalSections

  for (const { legacyKey, signalKey, project } of LEGACY_SIGNAL_SECTION_MAP) {
    if (nextSignals[signalKey]) {
      continue
    }

    const legacyValue = config[legacyKey]
    if (!Array.isArray(legacyValue) || legacyValue.length === 0) {
      continue
    }

    const projectedValue = project ? project(legacyValue) : legacyValue
    const signalMap = nextSignals as Record<string, unknown>
    signalMap[signalKey] = cloneJson(projectedValue)
    normalizedLegacyKeys.add(String(legacyKey))
  }

  return Object.keys(nextSignals).length > 0 ? nextSignals : undefined
}

export const normalizeDashboardConfigForEditor = (
  config: ConfigData
): DashboardEditorConfigState => {
  const normalizedLegacyKeys = new Set<string>()

  const providers = cloneJson(config.providers || {
    models: [],
    default_model: '',
  })

  providers.models = providers.models || []
  providers.default_model = providers.default_model || ''

  if (providers.models.length === 0 && config.model_config) {
    providers.models = getProviderModelsFromLegacyConfig(config.model_config, config.vllm_endpoints)
    normalizedLegacyKeys.add('model_config')
    if (config.vllm_endpoints?.length) {
      normalizedLegacyKeys.add('vllm_endpoints')
    }
  }

  if (!providers.default_model && config.default_model) {
    providers.default_model = config.default_model
    normalizedLegacyKeys.add('default_model')
  }

  if (!providers.reasoning_families && config.reasoning_families) {
    providers.reasoning_families = cloneJson(config.reasoning_families)
    normalizedLegacyKeys.add('reasoning_families')
  }

  if (providers.default_reasoning_effort === undefined && config.default_reasoning_effort !== undefined) {
    providers.default_reasoning_effort = config.default_reasoning_effort
    normalizedLegacyKeys.add('default_reasoning_effort')
  }

  const editorConfig: ConfigData = {
    ...cloneJson(config),
    listeners: cloneJson(config.listeners || []),
    decisions: cloneJson(config.decisions || []),
    providers,
    signals: normalizeSignals(config, normalizedLegacyKeys),
  }

  delete editorConfig.categories
  delete editorConfig.complexity_rules
  delete editorConfig.context_rules
  delete editorConfig.default_model
  delete editorConfig.default_reasoning_effort
  delete editorConfig.embedding_rules
  delete editorConfig.fact_check_rules
  delete editorConfig.jailbreak
  delete editorConfig.keyword_rules
  delete editorConfig.language_rules
  delete editorConfig.modality_rules
  delete editorConfig.model_config
  delete editorConfig.pii
  delete editorConfig.preference_rules
  delete editorConfig.reasoning_families
  delete editorConfig.role_bindings
  delete editorConfig.user_feedback_rules
  delete editorConfig.vllm_endpoints

  const legacyCategoriesConfig = Array.isArray(config.categories)
    ? cloneJson({
        categories: config.categories,
        model_config: config.model_config,
        default_model: config.default_model,
        default_reasoning_effort: config.default_reasoning_effort,
      })
    : null

  if (legacyCategoriesConfig?.categories?.length) {
    normalizedLegacyKeys.add('categories')
  }

  return {
    editorConfig,
    legacyCategoriesConfig,
    normalizedLegacyKeys: Array.from(normalizedLegacyKeys).sort(),
  }
}

export const getDashboardModels = (config: ConfigData | null): NormalizedModel[] =>
  (config?.providers?.models || []).map((model) => ({
    name: model.name,
    reasoning_family: model.reasoning_family,
    endpoints: model.endpoints || [],
    access_key: model.access_key,
    pricing: model.pricing,
  }))

export const getDashboardDefaultModel = (config: ConfigData | null): string =>
  config?.providers?.default_model || ''

export const getDashboardReasoningFamilies = (
  config: ConfigData | null
): Record<string, ReasoningFamily> => config?.providers?.reasoning_families || {}
