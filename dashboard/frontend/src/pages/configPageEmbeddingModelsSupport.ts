import type { EditFormData, FieldConfig } from '../components/EditModal'
import { routerStructuredField } from './configPageRouterStructuredFields'
import { normalizeRouterStructuredFields } from './configPageRouterStructuredSchema'

const REMOTE_BACKEND = 'openai_compatible'
// A remote embedding model is an openai_embeddings deployment that
// embeddings.default names.
const REMOTE_PROVIDER = 'openai_embeddings'
const DEFAULT_REMOTE_DEPLOYMENT = 'remote-embed'
// The pre-declared deployment that serves when embeddings.default is unset.
const DEFAULT_DEPLOYMENT = 'vela-embedding'
// The card's data carries the catalog's deployments under this key, so it can
// show and edit the deployment embeddings.default names.
const DEPLOYMENTS_KEY = '__deployments'
const LOCAL_PROVIDER_TYPE = 'local'
const REMOTE_PROVIDER_TYPE = 'remote'

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

function defaultDeployment(embeddings: Record<string, unknown>): {
  name: string
  deployment?: Record<string, unknown>
} {
  const name = trimmedString(embeddings.default) || DEFAULT_DEPLOYMENT
  return { name, deployment: asRecord(asRecord(embeddings[DEPLOYMENTS_KEY])?.[name]) }
}

function remoteDeployment(
  embeddings: Record<string, unknown>,
): { name: string; deployment: Record<string, unknown> } | undefined {
  const { name, deployment } = defaultDeployment(embeddings)
  if (!deployment || trimmedString(deployment.provider) !== REMOTE_PROVIDER) {
    return undefined
  }
  return { name, deployment }
}

function positiveNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

// The remote endpoint form of the deployment embeddings.default names.
function remoteEndpoint(embeddings: Record<string, unknown>): Record<string, unknown> | undefined {
  const remote = remoteDeployment(embeddings)
  if (!remote) return undefined
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

function isRemoteForm(data: object): boolean {
  return (
    trimmedString((data as EditFormData).provider_type).toLocaleLowerCase() === REMOTE_PROVIDER_TYPE
  )
}

function compactValue(value: unknown, fallback = 'Not set'): string {
  const text = trimmedString(value)
  if (!text) return typeof value === 'number' ? String(value) : fallback
  if (text.length <= 42) return text
  return `${text.slice(0, 18)}...${text.slice(-18)}`
}

export function embeddingModelsSummary(data: unknown): EmbeddingSummaryItem[] {
  const catalog = asRecord(data) ?? {}
  const multimodal = { label: 'Multimodal', value: compactValue(catalog.multimodal, 'Off') }
  const endpoint = remoteEndpoint(catalog)
  if (endpoint) {
    return [
      { label: 'Provider', value: 'Remote / OpenAI compatible' },
      { label: 'Model', value: compactValue(endpoint.model) },
      { label: 'Dimension', value: compactValue(endpoint.dimensions) },
      multimodal,
    ]
  }
  const { name, deployment } = defaultDeployment(catalog)
  const builtIn = name === DEFAULT_DEPLOYMENT ? 'Vela 1.0 Embedding' : 'Not set'
  return [
    { label: 'Provider', value: `Local / ${trimmedString(deployment?.provider) || 'model_runtime'}` },
    { label: 'Deployment', value: compactValue(name) },
    { label: 'Model', value: compactValue(deployment?.artifact, builtIn) },
    multimodal,
  ]
}

export function embeddingModelsBadges(data: unknown): EmbeddingBadge[] {
  const remote = remoteDeployment(asRecord(data) ?? {}) !== undefined
  return [
    {
      label: remote ? 'Remote provider' : 'Local inference',
      tone: remote ? 'info' : 'active',
    },
  ]
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
        'Choose whether embeddings.default names a model runtime deployment or a remote API. Remote mode saves an openai_embeddings deployment and names it in embeddings.default; it applies to text embedding consumers only.',
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
      name: 'default',
      label: 'Default Deployment',
      type: 'text',
      placeholder: DEFAULT_DEPLOYMENT,
      description:
        'Deployment in global.model_catalog.deployments that embeds for match.recall and /v1/embeddings; decisions never use it. Unset, vela-embedding serves.',
      shouldHide: hideForRemote,
    },
    {
      name: 'multimodal',
      label: 'Multimodal Deployment',
      type: 'text',
      placeholder: 'omni-nano',
      description: 'Deployment that embeds images and audio; unset, multimodal embedding is off.',
    },
    { ...endpointField, shouldHide: hideForLocal },
  ]
}

export function embeddingModelsEditData(data: unknown): EditFormData {
  const catalog = asRecord(data) ?? {}
  const remote = remoteDeployment(catalog)
  return {
    provider_type: remote ? REMOTE_PROVIDER_TYPE : LOCAL_PROVIDER_TYPE,
    remote_backend: REMOTE_BACKEND,
    default: remote ? '' : trimmedString(catalog.default),
    multimodal: trimmedString(catalog.multimodal),
    endpoint: remoteEndpoint(catalog),
    __deployment_name: remote?.name,
    __catalog: catalog,
  }
}

function validateRemoteEndpoint(endpoint: Record<string, unknown> | undefined): void {
  if (!endpoint) throw new Error('Remote Endpoint is required for the selected provider.')
  if (!trimmedString(endpoint.base_url)) throw new Error('Remote Endpoint.Base URL is required.')
  if (!trimmedString(endpoint.model)) throw new Error('Remote Endpoint.Model is required.')
}

/**
 * The global.model_catalog patch the card saves: the embeddings block and, in
 * remote mode, the openai_embeddings deployment embeddings.default names. A
 * null removes a key.
 */
export function embeddingModelsPatch(rawData: EditFormData): Record<string, unknown> {
  const remote = isRemoteForm(rawData)
  const normalized = normalizeRouterStructuredFields(
    'embedding_models',
    remote ? rawData : { ...rawData, endpoint: undefined },
  )
  const catalog = { ...(asRecord(normalized.__catalog) ?? {}) }
  delete catalog[DEPLOYMENTS_KEY]
  const multimodal = trimmedString(normalized.multimodal) || null

  if (remote) {
    const endpoint = asRecord(normalized.endpoint)
    validateRemoteEndpoint(endpoint)
    const name = trimmedString(normalized.__deployment_name) || DEFAULT_REMOTE_DEPLOYMENT
    return {
      deployments: { [name]: remoteEmbeddingDeployment(endpoint ?? {}) },
      embeddings: { ...catalog, default: name, multimodal },
    }
  }
  return {
    embeddings: { ...catalog, default: trimmedString(normalized.default) || null, multimodal },
  }
}
