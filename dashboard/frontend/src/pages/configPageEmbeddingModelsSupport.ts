import type { EditFormData, FieldConfig } from '../components/EditModal'
import { routerStructuredField } from './configPageRouterStructuredFields'
import { normalizeRouterStructuredFields } from './configPageRouterStructuredSchema'

const REMOTE_BACKEND = 'openai_compatible'
// A remote embedding model is an openai_embeddings deployment that
// embeddings.default names.
const REMOTE_PROVIDER = 'openai_embeddings'
const DEFAULT_REMOTE_DEPLOYMENT = 'remote-embed'
// The card's data carries the catalog's deployments under this key, so it can
// show and edit the deployment embeddings.default names.
const DEPLOYMENTS_KEY = '__deployments'
const DEFAULT_LOCAL_BACKEND = 'model_runtime'
const DEFAULT_LOCAL_MODEL_TYPE = 'qwen3'
const LOCAL_PROVIDER_TYPE = 'local'
const REMOTE_PROVIDER_TYPE = 'remote'

type EmbeddingBackend = 'model_runtime' | 'openai_compatible'
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

/** The embeddings block with the catalog's deployments, which the card reads. */
export function embeddingModelsView(
  embeddings: unknown,
  deployments: unknown,
): Record<string, unknown> | undefined {
  const block = asRecord(embeddings)
  const declared = asRecord(deployments)
  if (!block || !declared) return block
  return { ...(block ?? {}), [DEPLOYMENTS_KEY]: declared }
}

function remoteDeployment(
  embeddings: Record<string, unknown>,
): { name: string; deployment: Record<string, unknown> } | undefined {
  const name = trimmedString(embeddings.default)
  const deployment = asRecord(asRecord(embeddings[DEPLOYMENTS_KEY])?.[name])
  if (!name || !deployment || trimmedString(deployment.provider) !== REMOTE_PROVIDER) {
    return undefined
  }
  return { name, deployment }
}

function positiveNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

// The remote endpoint form of the deployment embeddings.default names, else
// of the deprecated embeddings.semantic.endpoint.
function remoteEndpoint(embeddings: Record<string, unknown>): Record<string, unknown> | undefined {
  const remote = remoteDeployment(embeddings)
  if (!remote) return asRecord(asRecord(embeddings.semantic)?.endpoint)
  const { deployment } = remote
  const timeoutMs = positiveNumber(deployment.timeout_ms)
  const form: Record<string, unknown> = {
    base_url: deployment.endpoint,
    model: deployment.model,
    api_key_env: asRecord(deployment.auth)?.api_key_env,
    timeout_seconds: timeoutMs === undefined ? undefined : timeoutMs / 1000,
    max_retries: deployment.max_retries,
    max_response_bytes: asRecord(deployment.limits)?.max_response_bytes,
    dimensions: asRecord(deployment.capabilities)?.dimensions,
  }
  return Object.fromEntries(Object.entries(form).filter(([, value]) => value !== undefined))
}

function remoteEmbeddingDeployment(endpoint: Record<string, unknown>): Record<string, unknown> {
  const apiKeyEnv = trimmedString(endpoint.api_key_env)
  const timeout = positiveNumber(endpoint.timeout_seconds)
  const retries = positiveNumber(endpoint.max_retries)
  const maxResponseBytes = positiveNumber(endpoint.max_response_bytes)
  const dimensions = positiveNumber(endpoint.dimensions)
  // A null removes the field from a deployment saved before.
  return {
    provider: REMOTE_PROVIDER,
    endpoint: trimmedString(endpoint.base_url).replace(/\/+$/, ''),
    model: trimmedString(endpoint.model),
    auth: apiKeyEnv ? { api_key_env: apiKeyEnv } : null,
    timeout_ms: timeout === undefined ? null : Math.round(timeout * 1000),
    max_retries: retries ?? null,
    limits: maxResponseBytes === undefined ? null : { max_response_bytes: maxResponseBytes },
    capabilities: dimensions === undefined ? null : { dimensions },
  }
}

function resolveBackend(embeddings: Record<string, unknown>): EmbeddingBackend {
  if (remoteDeployment(embeddings)) return REMOTE_BACKEND
  const embeddingConfig = asRecord(asRecord(embeddings.semantic)?.embedding_config)
  const backend = trimmedString(embeddingConfig?.backend).toLocaleLowerCase()
  if (backend === DEFAULT_LOCAL_BACKEND || backend === REMOTE_BACKEND) {
    return backend
  }
  if (trimmedString(embeddingConfig?.model_type).toLocaleLowerCase() === 'remote') {
    return REMOTE_BACKEND
  }
  return DEFAULT_LOCAL_BACKEND
}

function resolveProviderType(embeddings: Record<string, unknown>): EmbeddingProviderType {
  return resolveBackend(embeddings) === REMOTE_BACKEND
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
  if (trimmedString(semantic.multimodal_model_path)) return 'multimodal'
  return DEFAULT_LOCAL_MODEL_TYPE
}

function localModelPath(semantic: Record<string, unknown>, modelType: string): unknown {
  const pathByType: Record<string, unknown> = {
    qwen3: semantic.qwen3_model_path,
    mmbert: semantic.mmbert_model_path,
    multimodal: semantic.multimodal_model_path,
  }
  return pathByType[modelType] ?? semantic.mmbert_model_path ?? semantic.qwen3_model_path
}

export function embeddingModelsSummary(data: unknown): EmbeddingSummaryItem[] {
  const catalog = asRecord(data) ?? {}
  const semantic = asRecord(catalog.semantic) ?? {}
  const embeddingConfig = asRecord(semantic.embedding_config) ?? {}
  const endpoint = remoteEndpoint(catalog) ?? {}
  const backend = resolveBackend(catalog)

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
  const catalog = asRecord(data) ?? {}
  const semantic = asRecord(catalog.semantic) ?? {}
  const embeddingConfig = asRecord(semantic.embedding_config) ?? {}
  const backend = resolveBackend(catalog)
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
        'Choose whether embeddings run in the built-in model runtime or through a remote API. Remote mode saves an openai_embeddings deployment and names it in embeddings.default; it applies to text embedding consumers only.',
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
        'Embedding model family used by local consumers: mmbert, qwen3 or multimodal.',
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
      name: 'mmbert_model_path',
      label: 'mmBERT Model Path',
      type: 'text',
      placeholder: 'models/Vela-1.0-Encoder-307M-Embedding',
      shouldHide: hideForRemote,
    },
    {
      name: 'multimodal_model_path',
      label: 'Multimodal Model Path',
      type: 'text',
      placeholder: 'models/vela-1.0-omni-nano',
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
  const backend = resolveBackend(catalog)
  const providerType = resolveProviderType(catalog)
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
    remote_backend: providerType === REMOTE_PROVIDER_TYPE ? backend : REMOTE_BACKEND,
    model_type: localModelType,
    embedding_config: optimization,
    endpoint: remoteEndpoint(catalog),
    __deployment_name: remoteDeployment(catalog)?.name,
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
    targetDimension > 0 &&
    dimensions !== targetDimension
  ) {
    throw new Error(
      'Remote Endpoint.Dimensions must match Embedding Optimization.Target Dimension.',
    )
  }
}

/**
 * The global.model_catalog patch the card saves: the embeddings block and, in
 * remote mode, the openai_embeddings deployment embeddings.default names. A
 * null removes a key, so the deprecated embeddings.semantic.endpoint and its
 * remote markers never survive a save.
 */
export function embeddingModelsPatch(rawData: EditFormData): Record<string, unknown> {
  const providerType = providerTypeFromForm(rawData)
  const remote = providerType === REMOTE_PROVIDER_TYPE
  const dataForNormalization = remote ? rawData : { ...rawData, endpoint: undefined }
  const normalized = normalizeRouterStructuredFields('embedding_models', dataForNormalization)
  const rawModelType = normalized.model_type
  const rawOptimization = normalized.embedding_config
  const rawEndpoint = normalized.endpoint
  const deploymentName = trimmedString(normalized.__deployment_name)
  const catalogValue = normalized.__catalog
  const existingEmbeddingConfigValue = normalized.__embedding_config
  const semanticFields = { ...normalized }
  for (const key of [
    'backend',
    'provider_type',
    'remote_backend',
    'model_type',
    'embedding_config',
    'endpoint',
    '__deployment_name',
    '__catalog',
    '__embedding_config',
  ]) {
    delete semanticFields[key]
  }
  const catalog = { ...(asRecord(catalogValue) ?? {}) }
  const existingRemote = remoteDeployment(catalog)
  delete catalog[DEPLOYMENTS_KEY]
  const existingSemantic = asRecord(catalog.semantic) ?? {}
  const existingEmbeddingConfig = asRecord(existingEmbeddingConfigValue) ?? {}
  const existingBackend = trimmedString(existingEmbeddingConfig.backend).toLocaleLowerCase()
  if (existingBackend && existingBackend !== DEFAULT_LOCAL_BACKEND && existingBackend !== REMOTE_BACKEND) {
    throw new Error(
      `Embedding backend ${existingBackend} is retired; run vllm-sr config migrate to move this config to the model runtime.`,
    )
  }
  const optimization = {
    ...existingEmbeddingConfig,
    ...(asRecord(rawOptimization) ?? {}),
  }

  if (remote) {
    const endpoint = asRecord(rawEndpoint)
    validateRemoteEndpoint(endpoint, optimization)
    const name = deploymentName || DEFAULT_REMOTE_DEPLOYMENT
    return {
      deployments: { [name]: remoteEmbeddingDeployment(endpoint ?? {}) },
      embeddings: {
        ...catalog,
        default: name,
        semantic: {
          ...existingSemantic,
          ...semanticFields,
          embedding_config: { ...optimization, backend: null, model_type: null },
          endpoint: null,
        },
      },
    }
  }
  return {
    embeddings: {
      ...catalog,
      ...(existingRemote ? { default: null } : {}),
      semantic: {
        ...existingSemantic,
        ...semanticFields,
        embedding_config: {
          ...optimization,
          backend: DEFAULT_LOCAL_BACKEND,
          model_type: trimmedString(rawModelType) || inferLocalModelType(semanticFields),
        },
        endpoint: null,
      },
    },
  }
}
