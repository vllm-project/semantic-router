import type { EditFormData, FieldConfig } from '../components/EditModal'
import { routerStructuredField } from './configPageRouterStructuredFields'
import { normalizeRouterStructuredFields } from './configPageRouterStructuredSchema'

const REMOTE_BACKEND = 'openai_compatible'
const DEFAULT_LOCAL_BACKEND = 'candle'
const DEFAULT_LOCAL_MODEL_TYPE = 'qwen3'
const LOCAL_PROVIDER_TYPE = 'local'
const REMOTE_PROVIDER_TYPE = 'remote'

type EmbeddingBackend = 'candle' | 'openvino' | 'openai_compatible'
type EmbeddingProviderType = 'local' | 'remote'

interface EmbeddingSummaryItem {
  label: string
  value: string
}

interface EmbeddingBadge {
  label: string
  tone: 'active' | 'inactive' | 'info'
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function trimmedString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function resolveBackend(embeddingConfig?: Record<string, unknown>): EmbeddingBackend {
  const backend = trimmedString(embeddingConfig?.backend).toLocaleLowerCase()
  if (backend === DEFAULT_LOCAL_BACKEND || backend === 'openvino' || backend === REMOTE_BACKEND) {
    return backend
  }
  if (trimmedString(embeddingConfig?.model_type).toLocaleLowerCase() === 'remote') {
    return REMOTE_BACKEND
  }
  return DEFAULT_LOCAL_BACKEND
}

function resolveProviderType(embeddingConfig?: Record<string, unknown>): EmbeddingProviderType {
  return resolveBackend(embeddingConfig) === REMOTE_BACKEND
    ? REMOTE_PROVIDER_TYPE
    : LOCAL_PROVIDER_TYPE
}

function providerTypeFromForm(data: EditFormData): EmbeddingProviderType {
  const providerType = trimmedString(data.provider_type).toLocaleLowerCase()
  if (providerType === REMOTE_PROVIDER_TYPE) return REMOTE_PROVIDER_TYPE
  if (providerType === LOCAL_PROVIDER_TYPE) return LOCAL_PROVIDER_TYPE
  return trimmedString(data.backend).toLocaleLowerCase() === REMOTE_BACKEND
    ? REMOTE_PROVIDER_TYPE
    : LOCAL_PROVIDER_TYPE
}

function isRemoteForm(data: object): boolean {
  return providerTypeFromForm(data as EditFormData) === REMOTE_PROVIDER_TYPE
}

function compactValue(value: unknown, fallback = 'Not set'): string {
  const text = trimmedString(value)
  if (!text) return typeof value === 'number' ? String(value) : fallback
  if (text.length <= 42) return text
  return `${text.slice(0, 18)}...${text.slice(-18)}`
}

function inferLocalModelType(semantic: Record<string, unknown>): string {
  if (trimmedString(semantic.mmbert_model_path)) return 'mmbert'
  if (trimmedString(semantic.qwen3_model_path)) return 'qwen3'
  if (trimmedString(semantic.gemma_model_path)) return 'gemma'
  if (trimmedString(semantic.multimodal_model_path)) return 'multimodal'
  if (trimmedString(semantic.bert_model_path)) return 'bert'
  return DEFAULT_LOCAL_MODEL_TYPE
}

function localModelPath(semantic: Record<string, unknown>, modelType: string): unknown {
  const pathByType: Record<string, unknown> = {
    qwen3: semantic.qwen3_model_path,
    gemma: semantic.gemma_model_path,
    mmbert: semantic.mmbert_model_path,
    multimodal: semantic.multimodal_model_path,
    bert: semantic.bert_model_path,
  }
  return pathByType[modelType] ?? semantic.mmbert_model_path ?? semantic.qwen3_model_path
}

export function embeddingModelsSummary(data: unknown): EmbeddingSummaryItem[] {
  const catalog = asRecord(data)
  const semantic = asRecord(catalog?.semantic) ?? {}
  const embeddingConfig = asRecord(semantic.embedding_config) ?? {}
  const endpoint = asRecord(semantic.endpoint) ?? {}
  const backend = resolveBackend(embeddingConfig)

  if (backend === REMOTE_BACKEND) {
    return [
      { label: 'Provider', value: 'Remote / OpenAI compatible' },
      { label: 'Model', value: compactValue(endpoint.model) },
      {
        label: 'Dimension',
        value: compactValue(endpoint.dimensions ?? embeddingConfig.target_dimension),
      },
    ]
  }

  const modelType = trimmedString(embeddingConfig.model_type) || inferLocalModelType(semantic)
  return [
    { label: 'Provider', value: `Local / ${backend}` },
    { label: 'Model', value: compactValue(localModelPath(semantic, modelType), modelType) },
    { label: 'Runtime', value: semantic.use_cpu ? 'CPU' : 'GPU' },
  ]
}

export function embeddingModelsBadges(data: unknown): EmbeddingBadge[] {
  const semantic = asRecord(asRecord(data)?.semantic) ?? {}
  const embeddingConfig = asRecord(semantic.embedding_config) ?? {}
  const backend = resolveBackend(embeddingConfig)
  const badges: EmbeddingBadge[] = [
    {
      label: backend === REMOTE_BACKEND ? 'Remote provider' : 'Local inference',
      tone: backend === REMOTE_BACKEND ? 'info' : 'active',
    },
  ]
  if (embeddingConfig.preload_embeddings !== undefined) {
    badges.push({
      label: embeddingConfig.preload_embeddings ? 'Preload embeddings' : 'Lazy embeddings',
      tone: embeddingConfig.preload_embeddings ? 'active' : 'info',
    })
  }
  return badges
}

export function embeddingModelsFields(): FieldConfig[] {
  const hideForRemote = (data: object) => isRemoteForm(data)
  const hideForLocal = (data: object) => !isRemoteForm(data)
  const endpointField = routerStructuredField('embedding_models', 'endpoint')

  return [
    {
      name: 'provider_type',
      label: 'Provider Type',
      type: 'select',
      options: [LOCAL_PROVIDER_TYPE, REMOTE_PROVIDER_TYPE],
      required: true,
      description:
        'Choose whether embeddings run in-process or through a remote API. Remote mode currently applies to text embedding consumers only.',
    },
    {
      name: 'local_backend',
      label: 'Local Backend',
      type: 'select',
      options: ['candle', 'openvino'],
      required: true,
      description: 'Local inference engine used to execute the selected embedding model family.',
      shouldHide: hideForRemote,
    },
    {
      name: 'remote_backend',
      label: 'API Protocol',
      type: 'select',
      options: [REMOTE_BACKEND],
      required: true,
      description:
        'Remote API adapter. OpenAI compatible is currently supported; additional protocols can be added here later.',
      shouldHide: hideForLocal,
    },
    {
      name: 'model_type',
      label: 'Local Model Type',
      type: 'text',
      placeholder: 'mmbert',
      description:
        'Embedding model family used by local consumers, such as mmbert, qwen3, gemma, multimodal, or bert.',
      shouldHide: hideForRemote,
    },
    {
      name: 'qwen3_model_path',
      label: 'Qwen3 Model Path',
      type: 'text',
      placeholder: 'models/mom-embedding-pro',
      shouldHide: hideForRemote,
    },
    {
      name: 'gemma_model_path',
      label: 'Gemma Model Path',
      type: 'text',
      placeholder: 'models/mom-embedding-flash',
      shouldHide: hideForRemote,
    },
    {
      name: 'mmbert_model_path',
      label: 'mmBERT Model Path',
      type: 'text',
      placeholder: 'models/mmbert-embed-32k-2d-matryoshka',
      shouldHide: hideForRemote,
    },
    {
      name: 'multimodal_model_path',
      label: 'Multimodal Model Path',
      type: 'text',
      placeholder: 'models/mom-embedding-multimodal',
      shouldHide: hideForRemote,
    },
    {
      name: 'bert_model_path',
      label: 'BERT Model Path',
      type: 'text',
      placeholder: 'models/mom-embedding-bert',
      shouldHide: hideForRemote,
    },
    { name: 'use_cpu', label: 'Use CPU', type: 'boolean', shouldHide: hideForRemote },
    { ...endpointField, shouldHide: hideForLocal },
    routerStructuredField('embedding_models', 'embedding_config'),
  ]
}

export function embeddingModelsEditData(data: unknown): EditFormData {
  const catalog = asRecord(data) ?? {}
  const semantic = asRecord(catalog.semantic) ?? {}
  const embeddingConfig = asRecord(semantic.embedding_config) ?? {}
  const backend = resolveBackend(embeddingConfig)
  const providerType = resolveProviderType(embeddingConfig)
  const configuredModelType = trimmedString(embeddingConfig.model_type)
  const localModelType =
    configuredModelType && configuredModelType !== 'remote'
      ? configuredModelType
      : inferLocalModelType(semantic)
  const optimization = { ...embeddingConfig }
  delete optimization.backend
  delete optimization.model_type

  return {
    ...semantic,
    provider_type: providerType,
    local_backend: providerType === LOCAL_PROVIDER_TYPE ? backend : DEFAULT_LOCAL_BACKEND,
    remote_backend: providerType === REMOTE_PROVIDER_TYPE ? backend : REMOTE_BACKEND,
    model_type: localModelType,
    embedding_config: optimization,
    endpoint: semantic.endpoint,
    bert: asRecord(catalog.bert) ?? {},
    __catalog: catalog,
    __embedding_config: embeddingConfig,
  }
}

function validateRemoteEndpoint(
  endpoint: Record<string, unknown> | undefined,
  optimization: Record<string, unknown>,
): void {
  if (!endpoint) throw new Error('Remote Endpoint is required for the selected provider.')
  if (!trimmedString(endpoint.base_url)) throw new Error('Remote Endpoint.Base URL is required.')
  if (!trimmedString(endpoint.model)) throw new Error('Remote Endpoint.Model is required.')

  const dimensions = endpoint.dimensions
  const targetDimension = optimization.target_dimension
  if (
    typeof dimensions === 'number' &&
    typeof targetDimension === 'number' &&
    dimensions !== targetDimension
  ) {
    throw new Error(
      'Remote Endpoint.Dimensions must match Embedding Optimization.Target Dimension.',
    )
  }
}

function normalizedRemoteEndpoint(endpoint: Record<string, unknown>): Record<string, unknown> {
  return {
    ...endpoint,
    base_url: trimmedString(endpoint.base_url),
    model: trimmedString(endpoint.model),
    ...(endpoint.api_key_env === undefined
      ? {}
      : { api_key_env: trimmedString(endpoint.api_key_env) }),
  }
}

export function embeddingModelsCatalogValue(rawData: EditFormData): Record<string, unknown> {
  const providerType = providerTypeFromForm(rawData)
  const remote = providerType === REMOTE_PROVIDER_TYPE
  const backend = remote
    ? trimmedString(rawData.remote_backend) || REMOTE_BACKEND
    : trimmedString(rawData.local_backend) || DEFAULT_LOCAL_BACKEND
  const dataForNormalization = remote ? rawData : { ...rawData, endpoint: undefined }
  const normalized = normalizeRouterStructuredFields('embedding_models', dataForNormalization)
  const rawModelType = normalized.model_type
  const rawOptimization = normalized.embedding_config
  const rawEndpoint = normalized.endpoint
  const bert = normalized.bert
  const catalogValue = normalized.__catalog
  const existingEmbeddingConfigValue = normalized.__embedding_config
  const semanticFields = { ...normalized }
  delete semanticFields.backend
  delete semanticFields.provider_type
  delete semanticFields.local_backend
  delete semanticFields.remote_backend
  delete semanticFields.model_type
  delete semanticFields.embedding_config
  delete semanticFields.endpoint
  delete semanticFields.bert
  delete semanticFields.__catalog
  delete semanticFields.__embedding_config
  const catalog = asRecord(catalogValue) ?? {}
  const existingSemantic = asRecord(catalog.semantic) ?? {}
  const existingEmbeddingConfig = asRecord(existingEmbeddingConfigValue) ?? {}
  const optimization = {
    ...existingEmbeddingConfig,
    ...(asRecord(rawOptimization) ?? {}),
    backend,
    model_type: remote
      ? 'remote'
      : trimmedString(rawModelType) || inferLocalModelType(semanticFields),
  }
  const endpoint = remote ? asRecord(rawEndpoint) : asRecord(rawData.endpoint)
  if (remote) validateRemoteEndpoint(endpoint, optimization)

  return {
    ...catalog,
    semantic: {
      ...existingSemantic,
      ...semanticFields,
      embedding_config: optimization,
      ...(endpoint ? { endpoint: remote ? normalizedRemoteEndpoint(endpoint) : endpoint } : {}),
    },
    bert: asRecord(bert) ?? {},
  }
}
