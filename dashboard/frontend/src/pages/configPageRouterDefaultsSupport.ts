import type { EditFormData, FieldConfig } from '../components/EditModal'
import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import { X_AUTHZ_USER_ID } from '../generated/vsrHeaders'
import {
  decisionBindingsBadges,
  decisionBindingsEditData,
  decisionBindingsFields,
  decisionBindingsPatch,
  decisionBindingsState,
  decisionBindingsSummary,
} from './configPageDecisionBindingsSupport'
import { DEFAULT_SECTIONS, SECTION_META } from './configPageRouterDefaultsCatalog'
import { routerStructuredField } from './configPageRouterStructuredFields'
import { normalizeRouterStructuredFields } from './configPageRouterStructuredSchema'
import {
  embeddingModelsBadges,
  embeddingModelsEditData,
  embeddingModelsFields,
  embeddingModelsPatch,
  embeddingModelsSummary,
  embeddingModelsView,
} from './configPageEmbeddingModelsSupport'
import type { CanonicalGlobalConfig, ConfigData } from './configPageSupport'
import {
  generatedRouterValueField,
  mergeGeneratedRouterFields,
} from './configPageGeneratedFieldSupport'
import {
  CURATED_ROUTER_SECTIONS,
  type RouterLayerKey,
  type RouterSystemKey,
} from './configPageRouterSectionCatalog'

export type { RouterLayerKey, RouterSystemKey } from './configPageRouterSectionCatalog'
export type RouterConfigSectionData = Partial<Record<RouterSystemKey, unknown>>

export interface RouterSectionBadge {
  label: string
  tone: 'active' | 'inactive' | 'info'
}

export interface RouterSectionSummaryItem {
  label: string
  value: string
}

export interface RouterSectionCard {
  key: string
  layer: string
  path: string[]
  title: string
  eyebrow: string
  description: string
  data: unknown
  sourceLabel: string
  sourceTone: 'active' | 'inactive' | 'info'
  status: RouterSectionBadge
  badges: RouterSectionBadge[]
  summary: RouterSectionSummaryItem[]
  editData: EditFormData
  editFields: FieldConfig[]
  save: (data: EditFormData) => Partial<ConfigData>
  /** A deprecated section is shown, but the Dashboard no longer writes it. */
  deprecation?: string
}

interface RouterSectionContext {
  config: ConfigData | null
  routerConfig: RouterConfigSectionData
  routerDefaults: CanonicalGlobalConfig | null
}

export const ROUTER_LAYER_META: Record<RouterLayerKey, { title: string; description: string }> = {
  router: {
    title: 'Router',
    description:
      'Router controls and shared routing defaults. Each recipe can override its own strategy and fallback.',
  },
  services: {
    title: 'Services',
    description: 'APIs, replay, observability, and other router-owned service surfaces.',
  },
  stores: {
    title: 'Stores',
    description: 'Shared storage-backed capabilities such as semantic cache and memory.',
  },
  integrations: {
    title: 'Integrations',
    description: 'Auxiliary runtime integrations used by routing.',
  },
  model_catalog: {
    title: 'Model Catalog',
    description: 'Router-owned embedding catalogs, deployments, and decision model bindings.',
  },
}

export function routerLayerMeta(layer: string): { title: string; description: string } {
  return (
    ROUTER_LAYER_META[layer as RouterLayerKey] ?? {
      title: layer
        .split('_')
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' '),
      description: 'Canonical router-wide configuration generated from the Router contract.',
    }
  )
}

function cloneDefaultSection(key: RouterSystemKey): unknown {
  return JSON.parse(JSON.stringify(DEFAULT_SECTIONS[key]))
}

function asObject(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function asArray(value: unknown): Array<Record<string, unknown>> | undefined {
  return Array.isArray(value) ? (value as Array<Record<string, unknown>>) : undefined
}

function stringOrFallback(value: unknown, fallback = 'Not set'): string {
  if (typeof value === 'string' && value.trim()) {
    return value
  }
  if (typeof value === 'number') {
    return String(value)
  }
  if (typeof value === 'boolean') {
    return value ? 'Enabled' : 'Disabled'
  }
  return fallback
}

function percentOrFallback(value: unknown, fallback = 'Not set'): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : fallback
}

// A store without embedding.deployment embeds with embeddings.default.
function embeddingDeploymentSummary(section: Record<string, unknown> | undefined): string {
  return stringOrFallback(asObject(section?.embedding)?.deployment, 'embeddings.default')
}

function enabledBadge(value: boolean | undefined): RouterSectionBadge {
  if (value === undefined) {
    return { label: 'Configured', tone: 'info' }
  }
  return value ? { label: 'Enabled', tone: 'active' } : { label: 'Disabled', tone: 'inactive' }
}

function sourceBadge(
  key: RouterSystemKey,
  routerDefaults: CanonicalGlobalConfig | null,
  data: unknown,
): { label: string; tone: 'active' | 'inactive' | 'info' } {
  if (getSectionValue(routerDefaults, key) !== undefined) {
    return { label: 'router effective defaults', tone: 'active' }
  }
  if (data !== undefined) {
    return { label: 'config.yaml override', tone: 'info' }
  }
  return { label: 'Router default available', tone: 'inactive' }
}

function getNestedValue(value: unknown, path: readonly string[]): unknown {
  let current: unknown = value
  for (const segment of path) {
    const objectValue = asObject(current)
    if (!objectValue) {
      return undefined
    }
    current = objectValue[segment]
  }
  return current
}

function buildNestedPatch(path: readonly string[], value: unknown): Record<string, unknown> {
  if (path.length === 0) {
    return {}
  }
  if (path.length === 1) {
    return { [path[0]]: value }
  }
  const [head, ...tail] = path
  return { [head]: buildNestedPatch(tail, value) }
}

function getSectionValue(
  config: ConfigData | CanonicalGlobalConfig | null,
  key: RouterSystemKey,
): unknown {
  if (!config) {
    return undefined
  }

  const sectionPath = CURATED_ROUTER_SECTIONS[key].path
  const canonicalValue = getNestedValue(config, sectionPath)
  if (canonicalValue !== undefined) {
    return canonicalValue
  }

  return getNestedValue((config as ConfigData).global, sectionPath)
}

function summaryForKey(key: RouterSystemKey, data: unknown): RouterSectionSummaryItem[] {
  const section = asObject(data)

  switch (key) {
    case 'router_core':
      return [
        { label: 'Config source', value: stringOrFallback(section?.config_source, 'file') },
        { label: 'Default strategy', value: stringOrFallback(section?.strategy, 'priority') },
        {
          label: 'List backend models',
          value: section?.list_backend_models ? 'Enabled' : 'Disabled',
        },
      ]
    case 'learning':
      return [
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
        {
          label: 'Candidate set',
          value: stringOrFallback(asObject(section?.adaptation)?.candidate_set, 'decision'),
        },
        {
          label: 'Protection scope',
          value: stringOrFallback(asObject(section?.protection)?.scope, 'conversation'),
        },
        {
          label: 'State backend',
          value: stringOrFallback(asObject(section?.state_store)?.backend, 'local'),
        },
      ]
    case 'response_api':
      return [
        { label: 'Store backend', value: stringOrFallback(section?.store_backend) },
        { label: 'TTL', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
        { label: 'Max responses', value: stringOrFallback(section?.max_responses) },
      ]
    case 'router_replay':
      return [
        { label: 'Store backend', value: stringOrFallback(section?.store_backend) },
        { label: 'Retention', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
        { label: 'Async writes', value: stringOrFallback(section?.async_writes, 'Disabled') },
      ]
    case 'authz':
      return [
        { label: 'Fail open', value: stringOrFallback(section?.fail_open, 'Disabled') },
        {
          label: 'User header',
          value: stringOrFallback(asObject(section?.identity)?.user_id_header, X_AUTHZ_USER_ID),
        },
        { label: 'Providers', value: `${(asArray(section?.providers) || []).length}` },
      ]
    case 'ratelimit':
      return [
        { label: 'Fail open', value: stringOrFallback(section?.fail_open, 'Disabled') },
        { label: 'Providers', value: `${(asArray(section?.providers) || []).length}` },
        {
          label: 'Rules',
          value: `${(asArray(section?.providers) || []).flatMap((provider) => asArray(provider.rules) || []).length}`,
        },
      ]
    case 'management_api':
      return [
        { label: 'Bind address', value: stringOrFallback(section?.bind_address) },
        { label: 'Port', value: stringOrFallback(section?.port) },
        { label: 'Remote exposure', value: stringOrFallback(section?.remote_exposure, 'Disabled') },
      ]
    case 'startup_status':
      return [
        { label: 'Store backend', value: stringOrFallback(section?.store_backend) },
        { label: 'Redis', value: asObject(section?.redis) ? 'Configured' : 'Not configured' },
      ]
    case 'memory':
      return [
        { label: 'Milvus address', value: stringOrFallback(asObject(section?.milvus)?.address) },
        { label: 'Embedding', value: embeddingDeploymentSummary(section) },
        {
          label: 'Similarity threshold',
          value: percentOrFallback(section?.default_similarity_threshold),
        },
      ]
    case 'response_cache':
      return [
        { label: 'Backend', value: stringOrFallback(section?.backend_type) },
        { label: 'Retention', value: section?.ttl_seconds ? `${section.ttl_seconds}s` : 'Not set' },
      ]
    case 'vector_store':
      return [
        { label: 'Backend', value: stringOrFallback(section?.backend_type) },
        { label: 'Embedding', value: embeddingDeploymentSummary(section) },
        { label: 'Enabled', value: stringOrFallback(section?.enabled, 'Disabled') },
      ]
    case 'admission':
      return [{ label: 'Policies', value: `${section ? Object.keys(section).length : 0}` }]
    case 'embedding_models':
      return embeddingModelsSummary(data)
    case 'observability':
      return [
        { label: 'Metrics', value: asObject(section?.metrics)?.enabled ? 'Enabled' : 'Disabled' },
        {
          label: 'Tracing provider',
          value: stringOrFallback(asObject(section?.tracing)?.provider),
        },
        {
          label: 'Trace endpoint',
          value: stringOrFallback(asObject(asObject(section?.tracing)?.exporter)?.endpoint),
        },
      ]
    case 'looper':
      return [
        {
          label: 'Timeout',
          value: section?.timeout_seconds ? `${section.timeout_seconds}s` : 'Not set',
        },
        { label: 'Headers', value: `${Object.keys(asObject(section?.headers) || {}).length}` },
      ]
    case 'clear_route_cache':
      return [
        { label: 'Startup behavior', value: data ? 'Clear route cache' : 'Retain route cache' },
      ]
    case 'model_selection':
      return [{ label: 'Model selection', value: section?.enabled ? 'Enabled' : 'Disabled' }]
    case 'api': {
      const batchClassification = asObject(section?.batch_classification)
      const metrics = asObject(batchClassification?.metrics)
      const batchRanges = Array.isArray(metrics?.batch_size_ranges)
        ? metrics.batch_size_ranges.length
        : 0
      return [
        { label: 'Metrics', value: metrics?.enabled ? 'Enabled' : 'Disabled' },
        { label: 'Batch size ranges', value: `${batchRanges}` },
        {
          label: 'Sample rate',
          value: typeof metrics?.sample_rate === 'number' ? `${metrics.sample_rate}` : 'Not set',
        },
      ]
    }
  }

  const keys = section ? Object.keys(section).length : 0
  return [{ label: 'Fields', value: `${keys}` }]
}

function badgesForKey(key: RouterSystemKey, data: unknown): RouterSectionBadge[] {
  const badges: RouterSectionBadge[] = []

  if (key === 'embedding_models') {
    badges.push(...embeddingModelsBadges(data))
  }

  return badges
}

function statusForKey(data: unknown): RouterSectionBadge {
  if (data === undefined) {
    return { label: 'Missing', tone: 'inactive' }
  }
  if (typeof data === 'boolean') {
    return enabledBadge(data)
  }
  if (Array.isArray(data)) {
    return data.length > 0
      ? { label: `${data.length} configured`, tone: 'active' }
      : { label: 'Not configured', tone: 'inactive' }
  }
  const section = asObject(data)
  return enabledBadge(typeof section?.enabled === 'boolean' ? section.enabled : undefined)
}

function curatedFieldsForKey(key: RouterSystemKey): FieldConfig[] {
  switch (key) {
    case 'router_core':
      return [
        {
          name: 'config_source',
          label: 'Config Source',
          type: 'select',
          options: ['file', 'kubernetes'],
          required: true,
        },
        {
          name: 'strategy',
          label: 'Default Decision Strategy',
          type: 'select',
          options: ['priority', 'confidence'],
          description: 'Used when a recipe omits strategy. Priority is the built-in default.',
        },
        {
          name: 'list_backend_models',
          label: 'List Backend Models',
          type: 'boolean',
        },
        routerStructuredField(key, 'streamed_body'),
        routerStructuredField(key, 'skip_processing'),
      ]
    case 'learning':
      return [
        { name: 'enabled', label: 'Enable Router Learning', type: 'boolean' },
        routerStructuredField(key, 'adaptation'),
        routerStructuredField(key, 'protection'),
        routerStructuredField(key, 'state_store'),
      ]
    case 'response_api':
      return [
        { name: 'enabled', label: 'Enable Response API', type: 'boolean' },
        {
          name: 'store_backend',
          label: 'Store Backend',
          type: 'select',
          options: ['memory', 'milvus', 'redis'],
          required: true,
        },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '86400' },
        { name: 'max_responses', label: 'Max Responses', type: 'number', placeholder: '1000' },
      ]
    // The switches and capture settings of Replay, memory and the response
    // cache are the router_replay, context and response_cache plugins of
    // global.plugins in v0.5; these cards edit only the stores.
    case 'router_replay':
      return [
        {
          name: 'store_backend',
          label: 'Store Backend',
          type: 'select',
          options: ['memory', 'redis', 'postgres', 'milvus'],
          required: true,
        },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '2592000' },
        { name: 'async_writes', label: 'Async Writes', type: 'boolean' },
      ]
    case 'authz':
      return [
        { name: 'fail_open', label: 'Fail Open', type: 'boolean' },
        routerStructuredField(key, 'identity'),
        routerStructuredField(key, 'providers'),
      ]
    case 'ratelimit':
      return [
        { name: 'fail_open', label: 'Fail Open', type: 'boolean' },
        routerStructuredField(key, 'providers'),
      ]
    case 'memory':
      return [
        { name: 'auto_store', label: 'Auto Store Facts', type: 'boolean' },
        routerStructuredField(key, 'milvus'),
        routerStructuredField(key, 'embedding'),
        {
          name: 'default_retrieval_limit',
          label: 'Default Retrieval Limit',
          type: 'number',
          placeholder: '5',
        },
        {
          name: 'default_similarity_threshold',
          label: 'Similarity Threshold',
          type: 'percentage',
          placeholder: '70',
        },
        { name: 'hybrid_search', label: 'Hybrid Search', type: 'boolean' },
        { name: 'hybrid_mode', label: 'Hybrid Mode', type: 'text', placeholder: 'weighted' },
        { name: 'adaptive_threshold', label: 'Adaptive Threshold', type: 'boolean' },
        routerStructuredField(key, 'reflection'),
      ]
    case 'response_cache':
      return [
        {
          name: 'backend_type',
          label: 'Backend Type',
          type: 'select',
          options: ['memory', 'milvus', 'redis'],
          required: true,
        },
        { name: 'max_entries', label: 'Max Entries', type: 'number', placeholder: '1000' },
        { name: 'ttl_seconds', label: 'TTL (seconds)', type: 'number', placeholder: '3600' },
        {
          name: 'eviction_policy',
          label: 'Eviction Policy',
          type: 'select',
          options: ['fifo', 'lru', 'lfu'],
        },
        routerStructuredField(key, 'embedding'),
        routerStructuredField(key, 'redis'),
        routerStructuredField(key, 'milvus'),
      ]
    case 'vector_store':
      return [
        { name: 'enabled', label: 'Enable Vector Store', type: 'boolean' },
        {
          name: 'backend_type',
          label: 'Backend Type',
          type: 'select',
          options: ['memory', 'milvus', 'llama_stack'],
          required: true,
        },
        {
          name: 'file_storage_dir',
          label: 'File Storage Dir',
          type: 'text',
          placeholder: '/var/lib/vsr/data',
        },
        {
          name: 'max_file_size_mb',
          label: 'Max File Size (MB)',
          type: 'number',
          placeholder: '50',
        },
        routerStructuredField(key, 'embedding'),
        { name: 'ingestion_workers', label: 'Ingestion Workers', type: 'number', placeholder: '2' },
        {
          name: 'ingestion_drain_timeout_seconds',
          label: 'Ingestion Drain Timeout (s)',
          type: 'number',
          placeholder: '25',
        },
        routerStructuredField(key, 'supported_formats'),
        routerStructuredField(key, 'memory'),
        routerStructuredField(key, 'milvus'),
        routerStructuredField(key, 'llama_stack'),
      ]
    case 'embedding_models':
      return embeddingModelsFields()
    case 'observability':
      return [routerStructuredField(key, 'metrics'), routerStructuredField(key, 'tracing')]
    case 'looper':
      return [
        {
          name: 'timeout_seconds',
          label: 'Timeout (seconds)',
          type: 'number',
          placeholder: '1200',
        },
        routerStructuredField(key, 'headers'),
      ]
    case 'clear_route_cache':
      return [{ name: 'value', label: 'Clear Route Cache On Reload', type: 'boolean' }]
    case 'model_selection':
      return [{ name: 'enabled', label: 'Enable Model Selection', type: 'boolean' }]
    case 'api':
      return [routerStructuredField(key, 'batch_classification')]
  }

  return []
}

function fieldsForKey(key: RouterSystemKey): FieldConfig[] {
  const curated = curatedFieldsForKey(key)
  if (key === 'admission') {
    return [
      generatedRouterValueField(
        ['global', ...CURATED_ROUTER_SECTIONS[key].path],
        'value',
        'Admission Policies',
      ),
    ]
  }
  if (key === 'clear_route_cache' || key === 'embedding_models') {
    return curated
  }

  const omit = key === 'router_core' ? ['clear_route_cache', 'model_selection', 'learning'] : []
  return mergeGeneratedRouterFields(['global', ...CURATED_ROUTER_SECTIONS[key].path], curated, {
    omit,
  })
}

function editDataForKey(key: RouterSystemKey, data: unknown): EditFormData {
  if (key === 'clear_route_cache') {
    return { value: Boolean(data) }
  }
  if (key === 'admission') {
    return { value: asObject(data) || cloneDefaultSection(key) }
  }
  if (key === 'router_core') {
    const router = asObject(data)
    return {
      ...(router || {}),
      config_source: router?.config_source,
      strategy: router?.strategy,
      list_backend_models: router?.list_backend_models,
      streamed_body: asObject(router?.streamed_body) || {},
    }
  }
  if (key === 'embedding_models') {
    return embeddingModelsEditData(data)
  }
  const objectData = asObject(data)
  return objectData ? { ...objectData } : asObject(cloneDefaultSection(key)) || {}
}

function saveForKey(key: RouterSystemKey, rawData: EditFormData): Partial<ConfigData> {
  if (key === 'embedding_models') {
    return buildNestedPatch(
      CURATED_ROUTER_SECTIONS[key].path.slice(0, -1),
      embeddingModelsPatch(rawData),
    ) as Partial<ConfigData>
  }
  const data = normalizeRouterStructuredFields(key, rawData)
  if (key === 'clear_route_cache') {
    return buildNestedPatch(
      CURATED_ROUTER_SECTIONS[key].path,
      Boolean(data.value),
    ) as Partial<ConfigData>
  }
  if (key === 'admission') {
    return buildNestedPatch(
      CURATED_ROUTER_SECTIONS[key].path,
      asObject(data.value) || {},
    ) as Partial<ConfigData>
  }
  if (key === 'router_core') {
    const routerCore: Record<string, unknown> = {
      ...data,
      config_source: data.config_source,
      strategy: data.strategy,
      list_backend_models: Boolean(data.list_backend_models),
      streamed_body: asObject(data.streamed_body) || {},
    }
    return buildNestedPatch(CURATED_ROUTER_SECTIONS[key].path, routerCore) as Partial<ConfigData>
  }
  return buildNestedPatch(CURATED_ROUTER_SECTIONS[key].path, data) as Partial<ConfigData>
}

export function buildEffectiveRouterConfig(
  routerDefaults: CanonicalGlobalConfig | null,
  config: ConfigData | null,
): RouterConfigSectionData {
  const effective: RouterConfigSectionData = {}
  for (const key of Object.keys(CURATED_ROUTER_SECTIONS) as RouterSystemKey[]) {
    effective[key] = getSectionValue(routerDefaults, key) ?? getSectionValue(config, key)
  }
  return effective
}

export function buildRouterSectionCards(ctx: RouterSectionContext): RouterSectionCard[] {
  const orderedKeys = Object.keys(CURATED_ROUTER_SECTIONS) as RouterSystemKey[]
  const decisionBindings = decisionBindingsState(
    ctx.routerConfig.system_models,
    ctx.config,
    ctx.routerDefaults,
  )

  const deployments =
    getNestedValue(ctx.routerDefaults, ['model_catalog', 'deployments']) ??
    getNestedValue(ctx.config?.global, ['model_catalog', 'deployments'])
  const curatedCards: RouterSectionCard[] = orderedKeys.map((key) => {
    const data = ctx.routerConfig[key]
    const view = key === 'embedding_models' ? embeddingModelsView(data, deployments) : data
    const meta = SECTION_META[key]
    const source = sourceBadge(key, ctx.routerDefaults, data)
    const card = {
      key,
      layer: CURATED_ROUTER_SECTIONS[key].layer,
      path: [...CURATED_ROUTER_SECTIONS[key].path],
      title: meta.title,
      eyebrow: meta.eyebrow,
      description: meta.description,
      data,
      sourceLabel: source.label,
      sourceTone: source.tone,
      status: statusForKey(data),
    }

    if (key === 'system_models') {
      return {
        ...card,
        badges: decisionBindingsBadges(decisionBindings),
        summary: decisionBindingsSummary(decisionBindings),
        editData: decisionBindingsEditData(decisionBindings),
        editFields: decisionBindingsFields(decisionBindings),
        save: (nextData: EditFormData) => decisionBindingsPatch(decisionBindings, nextData),
      }
    }
    return {
      ...card,
      badges: badgesForKey(key, view),
      summary: summaryForKey(key, view),
      editData: editDataForKey(key, view),
      editFields: fieldsForKey(key),
      save: (nextData: EditFormData) => saveForKey(key, nextData),
    }
  })

  const curatedPaths = new Set(
    Object.values(CURATED_ROUTER_SECTIONS).map(({ path }) => path.join('.')),
  )
  const generatedCards: RouterSectionCard[] = ROUTER_CONFIG_EXTENSION.global_sections
    // KB management is not a Dashboard surface; the canonical schema and raw editor retain it.
    .filter((surface) => surface.path.join('.') !== 'model_catalog.kbs')
    // Decision Model & Bindings edits the bindings.
    .filter((surface) => surface.path.join('.') !== 'model_catalog.bindings')
    .filter((surface) => !curatedPaths.has(surface.path.join('.')))
    .map((surface) => {
      const schemaPath = ['global', ...surface.path]
      const generatedFields = mergeGeneratedRouterFields(schemaPath, [])
      const objectForm = generatedFields.length > 0
      const defaultData = getNestedValue(ctx.routerDefaults, surface.path)
      const overrideData = getNestedValue(ctx.config?.global, surface.path)
      const data = defaultData ?? overrideData
      const editData = objectForm ? { ...(asObject(data) || {}) } : { value: data }
      const editFields = objectForm
        ? generatedFields
        : [generatedRouterValueField(schemaPath, 'value', surface.display_name)]
      const source =
        defaultData !== undefined
          ? { label: 'router effective defaults', tone: 'active' as const }
          : overrideData !== undefined
            ? { label: 'config.yaml override', tone: 'info' as const }
            : { label: 'Router default available', tone: 'inactive' as const }

      return {
        key: `schema:${surface.path.join('.')}`,
        layer: surface.layer,
        path: [...surface.path],
        title: surface.display_name,
        eyebrow: routerLayerMeta(surface.layer).title,
        description: 'Canonical configuration surfaced automatically from the Router schema.',
        data,
        sourceLabel: source.label,
        sourceTone: source.tone,
        status: statusForKey(data),
        badges: [],
        summary: Array.isArray(data)
          ? [{ label: 'Entries', value: `${data.length}` }]
          : asObject(data)
            ? [{ label: 'Fields', value: `${Object.keys(asObject(data) || {}).length}` }]
            : [{ label: 'Value', value: stringOrFallback(data) }],
        editData,
        editFields,
        save: (nextData) =>
          buildNestedPatch(
            surface.path,
            !objectForm ? nextData.value : nextData,
          ) as Partial<ConfigData>,
      }
    })

  return [...curatedCards, ...generatedCards]
}
