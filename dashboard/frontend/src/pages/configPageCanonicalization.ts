import type {
  BackendRefEntry,
  ConfigData,
  ModelConfigEntry,
  RoutingModelCard,
} from './configPageSupport'

type CanonicalSignalSections = NonNullable<NonNullable<ConfigData['routing']>['signals']>

const LEGACY_SIGNAL_SECTIONS = [
  ['keyword_rules', 'keywords'],
  ['embedding_rules', 'embeddings'],
  ['categories', 'domains'],
  ['fact_check_rules', 'fact_check'],
  ['user_feedback_rules', 'user_feedbacks'],
  ['preference_rules', 'preferences'],
  ['language_rules', 'language'],
  ['context_rules', 'context'],
  ['complexity_rules', 'complexity'],
  ['jailbreak', 'jailbreak'],
  ['pii', 'pii'],
] as const satisfies ReadonlyArray<
  readonly [keyof ConfigData, keyof CanonicalSignalSections]
>

const LEGACY_GLOBAL_ROOT_KEYS = [
  'semantic_cache',
  'memory',
  'response_api',
  'router_replay',
  'api',
  'observability',
  'tools',
  'looper',
  'clear_route_cache',
  'model_selection',
  'embedding_models',
  'bert_model',
  'external_models',
  'prompt_guard',
  'classifier',
  'hallucination_mitigation',
  'feedback_detector',
] as const

type MutableRecord = Record<string, unknown>

const asRecord = (value: unknown): MutableRecord | undefined => {
  if (!value || Array.isArray(value) || typeof value !== 'object') {
    return undefined
  }
  return value as MutableRecord
}

const assignIfMissing = <K extends keyof MutableRecord>(
  target: MutableRecord,
  key: K,
  value: unknown,
) => {
  if (value === undefined || value === null) {
    return
  }
  const current = target[key]
  if (current === undefined || current === null || current === '') {
    target[key] = value
  }
}

const ensureNestedRecord = (root: MutableRecord, path: string[]): MutableRecord => {
  let current = root
  for (const segment of path) {
    const existing = asRecord(current[segment])
    if (existing) {
      current = existing
      continue
    }
    const created: MutableRecord = {}
    current[segment] = created
    current = created
  }
  return current
}

const cloneUnknown = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

export const cloneConfigData = (value: ConfigData): ConfigData => cloneUnknown(value)

export const ensureRoutingConfig = (cfg: ConfigData) => {
  if (!cfg.routing) {
    cfg.routing = {}
  }
  if (!cfg.routing.modelCards) {
    cfg.routing.modelCards = []
  }
  return cfg.routing
}

export const ensureProvidersConfig = (cfg: ConfigData) => {
  if (!cfg.providers) {
    cfg.providers = { models: [] }
  }
  if (!cfg.providers.models) {
    cfg.providers.models = []
  }
  return cfg.providers
}

export const ensureProviderDefaultsConfig = (cfg: ConfigData) => {
  const providers = ensureProvidersConfig(cfg)
  if (!providers.defaults) {
    providers.defaults = {}
  }
  return providers.defaults
}

export const upsertRoutingModelCard = (
  cfg: ConfigData,
  name: string,
  patch: Partial<RoutingModelCard> = {},
) => {
  const routing = ensureRoutingConfig(cfg)
  const cards = [...(routing.modelCards || [])]
  const index = cards.findIndex((card) => card.name === name)
  if (index >= 0) {
    cards[index] = { ...cards[index], ...patch, name }
  } else {
    cards.push({ name, ...patch })
  }
  routing.modelCards = cards
}

export const removeRoutingModelCard = (cfg: ConfigData, name: string) => {
  const routing = ensureRoutingConfig(cfg)
  routing.modelCards = (routing.modelCards || []).filter((card) => card.name !== name)
}

const promoteLegacySignals = (cfg: ConfigData) => {
  const routing = ensureRoutingConfig(cfg)
  const signals = { ...(routing.signals || {}) }
  const mutableSignals = signals as MutableRecord
  let changed = false

  for (const [legacyKey, canonicalKey] of LEGACY_SIGNAL_SECTIONS) {
    const legacyValue = cfg[legacyKey]
    if (!Array.isArray(legacyValue) || legacyValue.length === 0) {
      continue
    }
    if (!signals[canonicalKey] || signals[canonicalKey]?.length === 0) {
      mutableSignals[canonicalKey] = cloneUnknown(legacyValue)
      changed = true
    }
  }

  if (changed) {
    routing.signals = signals
  }
}

const buildLegacyBackendCatalog = (cfg: ConfigData): Record<string, BackendRefEntry> => {
  const catalog: Record<string, BackendRefEntry> = {}
  const rawConfig = cfg as MutableRecord
  const providerProfiles = asRecord(rawConfig.provider_profiles) || {}

  for (const endpoint of cfg.vllm_endpoints || []) {
    const name = endpoint.name?.trim()
    if (!name) {
      continue
    }

    const backendRef: BackendRefEntry = { name }
    const profileName =
      typeof endpoint.provider_profile === 'string' ? endpoint.provider_profile.trim() : ''
    const profile = profileName ? asRecord(providerProfiles[profileName]) : undefined

    if (profile) {
      assignIfMissing(backendRef as MutableRecord, 'base_url', profile.base_url)
      assignIfMissing(backendRef as MutableRecord, 'provider', profile.type)
      assignIfMissing(backendRef as MutableRecord, 'auth_header', profile.auth_header)
      assignIfMissing(backendRef as MutableRecord, 'auth_prefix', profile.auth_prefix)
      assignIfMissing(backendRef as MutableRecord, 'extra_headers', profile.extra_headers)
      assignIfMissing(backendRef as MutableRecord, 'api_version', profile.api_version)
      assignIfMissing(backendRef as MutableRecord, 'chat_path', profile.chat_path)
    }

    const address = endpoint.address?.trim()
    if (address) {
      backendRef.endpoint = endpoint.port > 0 ? `${address}:${endpoint.port}` : address
    }
    if (endpoint.protocol) {
      backendRef.protocol = endpoint.protocol
    }
    if (typeof endpoint.weight === 'number') {
      backendRef.weight = endpoint.weight
    }
    if (endpoint.type) {
      backendRef.type = endpoint.type
    }
    if (endpoint.api_key) {
      backendRef.api_key = endpoint.api_key
    }
    if (endpoint.api_key_env) {
      backendRef.api_key_env = endpoint.api_key_env
    }

    catalog[name] = backendRef
  }

  return catalog
}

const legacyBackendRefsForModel = (
  modelConfig: ModelConfigEntry,
  backendCatalog: Record<string, BackendRefEntry>,
): BackendRefEntry[] => {
  const backendRefs: BackendRefEntry[] = []
  for (const endpointName of modelConfig.preferred_endpoints || []) {
    const backend = backendCatalog[endpointName]
    if (!backend) {
      continue
    }
    const nextBackend = cloneUnknown(backend)
    if (!nextBackend.api_key && modelConfig.access_key) {
      nextBackend.api_key = modelConfig.access_key
    }
    backendRefs.push(nextBackend)
  }
  return backendRefs
}

const mergeLegacyModelIntoProviderModel = (
  existing: NonNullable<ConfigData['providers']>['models'][number],
  modelConfig: ModelConfigEntry,
  backendCatalog: Record<string, BackendRefEntry>,
) => {
  if (!existing.reasoning_family && modelConfig.reasoning_family) {
    existing.reasoning_family = modelConfig.reasoning_family
  }
  if (!existing.provider_model_id && modelConfig.model_id) {
    existing.provider_model_id = modelConfig.model_id
  }
  if (!existing.pricing && modelConfig.pricing) {
    existing.pricing = cloneUnknown(modelConfig.pricing)
  }
  if (!existing.api_format && modelConfig.api_format) {
    existing.api_format = modelConfig.api_format
  }
  if (!existing.external_model_ids && modelConfig.external_model_ids) {
    existing.external_model_ids = cloneUnknown(modelConfig.external_model_ids)
  }
  if ((!existing.backend_refs || existing.backend_refs.length === 0) && modelConfig.preferred_endpoints) {
    existing.backend_refs = legacyBackendRefsForModel(modelConfig, backendCatalog)
  }
}

const promoteLegacyModelBindings = (cfg: ConfigData) => {
  if (!cfg.model_config) {
    return
  }

  const providers = ensureProvidersConfig(cfg)
  const providerModelsByName = new Map(providers.models.map((model) => [model.name, model]))
  const backendCatalog = buildLegacyBackendCatalog(cfg)

  for (const [modelName, modelConfig] of Object.entries(cfg.model_config)) {
    const cardPatch: Partial<RoutingModelCard> = {}
    if (modelConfig.param_size) {
      cardPatch.param_size = modelConfig.param_size
    }
    if (modelConfig.context_window_size) {
      cardPatch.context_window_size = modelConfig.context_window_size
    }
    if (modelConfig.description) {
      cardPatch.description = modelConfig.description
    }
    if (modelConfig.capabilities) {
      cardPatch.capabilities = cloneUnknown(modelConfig.capabilities)
    }
    if (modelConfig.loras) {
      cardPatch.loras = cloneUnknown(modelConfig.loras)
    }
    if (modelConfig.tags) {
      cardPatch.tags = cloneUnknown(modelConfig.tags)
    }
    if (typeof modelConfig.quality_score === 'number') {
      cardPatch.quality_score = modelConfig.quality_score
    }
    if (modelConfig.modality) {
      cardPatch.modality = modelConfig.modality
    }
    upsertRoutingModelCard(cfg, modelName, cardPatch)

    const existing = providerModelsByName.get(modelName)
    if (existing) {
      mergeLegacyModelIntoProviderModel(existing, modelConfig, backendCatalog)
      continue
    }

    const providerModel = {
      name: modelName,
      reasoning_family: modelConfig.reasoning_family,
      provider_model_id: modelConfig.model_id,
      backend_refs: legacyBackendRefsForModel(modelConfig, backendCatalog),
      pricing: modelConfig.pricing ? cloneUnknown(modelConfig.pricing) : undefined,
      api_format: modelConfig.api_format,
      external_model_ids: modelConfig.external_model_ids
        ? cloneUnknown(modelConfig.external_model_ids)
        : undefined,
    }
    providers.models.push(providerModel)
    providerModelsByName.set(modelName, providerModel)
  }
}

const promoteLegacyProviderDefaults = (cfg: ConfigData) => {
  const defaults = ensureProviderDefaultsConfig(cfg)
  if (!defaults.default_model && cfg.default_model) {
    defaults.default_model = cfg.default_model
  }
  if (!defaults.reasoning_families && cfg.reasoning_families) {
    defaults.reasoning_families = cloneUnknown(cfg.reasoning_families)
  }
  if (!defaults.default_reasoning_effort && cfg.default_reasoning_effort) {
    defaults.default_reasoning_effort = cfg.default_reasoning_effort
  }
}

const promoteLegacyGlobalBlocks = (cfg: ConfigData) => {
  if (!cfg.global) {
    cfg.global = {}
  }

  const globalRoot = cfg.global as MutableRecord
  const rawConfig = cfg as MutableRecord

  const placeBlockIfMissing = (path: string[], value: unknown) => {
    if (value === undefined || value === null) {
      return
    }
    const parent = ensureNestedRecord(globalRoot, path.slice(0, -1))
    assignIfMissing(parent, path[path.length - 1], cloneUnknown(value))
  }

  const placeLegacyBertEmbeddingIfMissing = (value: unknown) => {
    const legacy = asRecord(value)
    if (!legacy) {
      return
    }
    const semantic = ensureNestedRecord(globalRoot, ['model_catalog', 'embeddings', 'semantic'])
    assignIfMissing(semantic, 'bert_model_path', cloneUnknown(legacy.model_id))
    assignIfMissing(semantic, 'use_cpu', cloneUnknown(legacy.use_cpu))

    if (legacy.threshold !== undefined && legacy.threshold !== null && legacy.threshold !== '') {
      const embeddingConfig = ensureNestedRecord(semantic, ['embedding_config'])
      assignIfMissing(embeddingConfig, 'min_score_threshold', cloneUnknown(legacy.threshold))
    }
  }

  placeBlockIfMissing(['stores', 'semantic_cache'], cfg.semantic_cache)
  placeBlockIfMissing(['stores', 'memory'], cfg.memory)
  placeBlockIfMissing(['services', 'response_api'], cfg.response_api)
  placeBlockIfMissing(['services', 'router_replay'], cfg.router_replay)
  placeBlockIfMissing(['services', 'api'], cfg.api)
  placeBlockIfMissing(['services', 'observability'], cfg.observability)
  placeBlockIfMissing(['integrations', 'tools'], cfg.tools)
  placeBlockIfMissing(['integrations', 'looper'], cfg.looper)
  placeBlockIfMissing(['router', 'clear_route_cache'], cfg.clear_route_cache)
  placeBlockIfMissing(['router', 'model_selection'], cfg.model_selection)
  placeBlockIfMissing(['model_catalog', 'embeddings', 'semantic'], cfg.embedding_models)
  placeLegacyBertEmbeddingIfMissing((cfg as MutableRecord).bert_model)
  placeBlockIfMissing(['model_catalog', 'external'], cfg.external_models)
  placeBlockIfMissing(['model_catalog', 'modules', 'prompt_guard'], cfg.prompt_guard)
  placeBlockIfMissing(['model_catalog', 'modules', 'classifier'], cfg.classifier)
  placeBlockIfMissing(
    ['model_catalog', 'modules', 'hallucination_mitigation'],
    cfg.hallucination_mitigation,
  )
  placeBlockIfMissing(['model_catalog', 'modules', 'feedback_detector'], cfg.feedback_detector)

  const legacyGlobalModels = asRecord(globalRoot.models)
  if (legacyGlobalModels) {
    const embeddings = asRecord(legacyGlobalModels.embeddings)
    if (embeddings) {
      placeBlockIfMissing(['model_catalog', 'embeddings', 'semantic'], embeddings.semantic)
      placeLegacyBertEmbeddingIfMissing(embeddings.bert)
    }
    placeBlockIfMissing(['model_catalog', 'system'], legacyGlobalModels.system)
    placeBlockIfMissing(['model_catalog', 'external'], legacyGlobalModels.external)
    delete globalRoot.models
  }

  const legacyGlobalModules = asRecord(globalRoot.modules)
  if (legacyGlobalModules) {
    placeBlockIfMissing(['model_catalog', 'modules'], legacyGlobalModules)
    delete globalRoot.modules
  }

  const legacyRuntime = asRecord(globalRoot.runtime)
  if (legacyRuntime) {
    const legacyRuntimeRouter = asRecord(legacyRuntime.router)
    if (legacyRuntimeRouter) {
      for (const [key, value] of Object.entries(legacyRuntimeRouter)) {
        placeBlockIfMissing(['router', key], value)
      }
      delete legacyRuntime.router
    }
    for (const [key, value] of Object.entries(legacyRuntime)) {
      if (key === 'router') {
        continue
      }
      switch (key) {
        case 'semantic_cache':
        case 'memory':
        case 'vector_store':
          placeBlockIfMissing(['stores', key], value)
          break
        case 'tools':
        case 'looper':
          placeBlockIfMissing(['integrations', key], value)
          break
        case 'embedding_models':
          placeBlockIfMissing(['model_catalog', 'embeddings', 'semantic'], value)
          break
        case 'bert_model':
          placeLegacyBertEmbeddingIfMissing(value)
          break
        case 'external_models':
          placeBlockIfMissing(['model_catalog', 'external'], value)
          break
        case 'system_models':
          placeBlockIfMissing(['model_catalog', 'system'], value)
          break
        case 'prompt_guard':
        case 'classifier':
        case 'hallucination_mitigation':
        case 'feedback_detector':
        case 'modality_detector':
        case 'prompt_compression':
          placeBlockIfMissing(['model_catalog', 'modules', key], value)
          break
        default:
          placeBlockIfMissing(['services', key], value)
      }
    }
    delete globalRoot.runtime
  }

  const modelCatalog = ensureNestedRecord(globalRoot, ['model_catalog'])
  const legacyModules = asRecord(modelCatalog.modules)
  if (legacyModules) {
    modelCatalog.modules = legacyModules
  }

  const rawGlobal = asRecord(rawConfig.global)
  if (rawGlobal) {
    delete rawGlobal.modules
  }
}

const stripLegacyRootFields = (cfg: ConfigData) => {
  const rawConfig = cfg as MutableRecord
  for (const [legacyKey] of LEGACY_SIGNAL_SECTIONS) {
    delete rawConfig[legacyKey]
  }
  delete rawConfig.categories
  delete rawConfig.default_model
  delete rawConfig.reasoning_families
  delete rawConfig.default_reasoning_effort
  delete rawConfig.model_config
  delete rawConfig.vllm_endpoints
  for (const key of LEGACY_GLOBAL_ROOT_KEYS) {
    delete rawConfig[key]
  }
  delete rawConfig.provider_profiles
}

export const canonicalizeConfigForManagerSave = (updatedConfig: ConfigData): ConfigData => {
  const next = cloneConfigData(updatedConfig)
  const routing = ensureRoutingConfig(next)
  if (next.signals) {
    routing.signals = cloneUnknown(next.signals)
    delete next.signals
  }
  if (next.decisions) {
    routing.decisions = cloneUnknown(next.decisions)
    delete next.decisions
  }

  promoteLegacySignals(next)
  promoteLegacyProviderDefaults(next)
  promoteLegacyModelBindings(next)
  promoteLegacyGlobalBlocks(next)
  stripLegacyRootFields(next)

  if (!next.version) {
    next.version = 'v0.3'
  }
  return next
}

export const projectCanonicalConfigForManager = (data: ConfigData): ConfigData => {
  const next = canonicalizeConfigForManagerSave(data)
  if (next.routing?.signals) {
    next.signals = cloneUnknown(next.routing.signals)
  }
  if (next.routing?.decisions) {
    next.decisions = cloneUnknown(next.routing.decisions)
  }
  return next
}
