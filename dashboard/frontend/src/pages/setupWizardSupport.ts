import { ROUTER_CONFIG_DOCUMENT_SCHEMA, configYaml } from '../lib/configYaml'
import {
  DEFAULT_SETUP_PROVIDER_ID,
  DEFAULT_SETUP_RUNTIME_BASE_URL,
  getSetupProviderOption,
  isSetupRuntimeProvider,
  type ProviderKind,
} from './setupWizardProviderCatalog'

export type SetupStep = 0 | 1 | 2 | 3
export type { ProviderKind } from './setupWizardProviderCatalog'
export type SetupValidationState = 'idle' | 'validating' | 'valid' | 'error'
export type SetupActivationState = 'idle' | 'activating' | 'persisted' | 'error'
export type SetupRoutingMode = 'scratch' | 'remote' | 'preset'
export type RemoteImportState = 'idle' | 'importing' | 'imported' | 'error'
export type PresetCatalogState = 'loading' | 'ready' | 'error'
export type PresetRequestState = 'idle' | 'loading' | 'ready' | 'importing' | 'imported' | 'error'

export interface ModelDraft {
  id: string
  name: string
  providerKind: ProviderKind
  baseUrl: string
  accessKey: string
  endpointName: string
}

export interface ModelDraftFieldErrors {
  name?: string
  baseUrl?: string
}

export interface RemovedModelSnapshot {
  model: ModelDraft
  index: number
  wasDefault: boolean
}

export interface SetupModelPage {
  items: ModelDraft[]
  page: number
  pageCount: number
  total: number
  startIndex: number
}

export interface SetupRequestGuard {
  begin: () => number
  invalidate: () => void
  isCurrent: (generation: number) => boolean
}

interface BuiltModel {
  name: string
  provider_model_id: string
  backend_refs: Array<{
    name: string
    weight: number
    /** host:port for a local runtime, or the provider's full URL. */
    endpoint: string
    protocol: 'http' | 'https'
    provider: ProviderKind
    api_key?: string
  }>
  api_format?: 'responses' | 'messages'
}

export interface SetupConfigCounts {
  models: number
  decisions: number
  signals: number
  canActivate: boolean
}

export interface ImportedSetupConfig {
  config: Record<string, unknown>
  sourceUrl: string
  counts: SetupConfigCounts
}

export const SETUP_STEP_LABELS: ReadonlyArray<[string, string]> = [
  ['1', 'Connect models'],
  ['2', 'Choose routing'],
  ['3', 'Decision model'],
  ['4', 'Review & activate'],
]

export const SETUP_STEP_DESCRIPTIONS: readonly string[] = [
  'Add the backends the router can send traffic to. The first one becomes the default.',
  'Start from a single catch-all route, a starter architecture, or a config.yaml you already have.',
  'The model that answers the router’s questions about every request: its signals and decision questions.',
  'Validate the generated configuration, then activate it as the router’s first architecture.',
]

export const SETUP_LAST_STEP = (SETUP_STEP_LABELS.length - 1) as SetupStep

export const DEFAULT_REMOTE_SETUP_CONFIG_URL =
  'https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/balance/config.yaml'

const DEFAULT_MODEL_NAME = 'qwen/qwen3.5-rocm'
export const SETUP_MODELS_PER_PAGE = 4

export function createSetupRequestGuard(): SetupRequestGuard {
  let currentGeneration = 0
  return {
    begin: () => {
      currentGeneration += 1
      return currentGeneration
    },
    invalidate: () => {
      currentGeneration += 1
    },
    isCurrent: (generation) => generation === currentGeneration,
  }
}

export function filterSetupModels(models: ModelDraft[], rawQuery: string): ModelDraft[] {
  const query = rawQuery.trim().toLowerCase()
  if (!query) {
    return models
  }

  return models.filter((model) =>
    [model.name, model.baseUrl, model.endpointName, model.providerKind].some((value) =>
      value.toLowerCase().includes(query),
    ),
  )
}

export function paginateSetupModels(
  models: ModelDraft[],
  requestedPage: number,
  pageSize = SETUP_MODELS_PER_PAGE,
): SetupModelPage {
  const safePageSize = Math.max(1, Math.floor(pageSize))
  const pageCount = Math.max(1, Math.ceil(models.length / safePageSize))
  const page = Math.min(Math.max(1, Math.floor(requestedPage)), pageCount)
  const startIndex = (page - 1) * safePageSize

  return {
    items: models.slice(startIndex, startIndex + safePageSize),
    page,
    pageCount,
    total: models.length,
    startIndex,
  }
}

export function removeSetupModel(
  models: ModelDraft[],
  modelId: string,
  defaultModelId: string,
): {
  models: ModelDraft[]
  defaultModelId: string
  removed: RemovedModelSnapshot | null
} {
  const index = models.findIndex((model) => model.id === modelId)
  if (index < 0 || models.length <= 1) {
    return { models, defaultModelId, removed: null }
  }

  const model = models[index]
  const nextModels = models.filter((candidate) => candidate.id !== modelId)
  const wasDefault = defaultModelId === modelId
  return {
    models: nextModels,
    defaultModelId: wasDefault ? (nextModels[0]?.id ?? '') : defaultModelId,
    removed: { model, index, wasDefault },
  }
}

export function restoreSetupModel(
  models: ModelDraft[],
  removed: RemovedModelSnapshot,
  defaultModelId: string,
): { models: ModelDraft[]; defaultModelId: string } {
  if (models.some((model) => model.id === removed.model.id)) {
    return { models, defaultModelId }
  }

  const restoredModels = [...models]
  restoredModels.splice(
    Math.min(Math.max(0, removed.index), restoredModels.length),
    0,
    removed.model,
  )
  return {
    models: restoredModels,
    defaultModelId: removed.wasDefault ? removed.model.id : defaultModelId,
  }
}

export function createSetupConfigCounts(
  overrides: Partial<SetupConfigCounts> = {},
): SetupConfigCounts {
  return {
    models: 0,
    decisions: 0,
    signals: 0,
    canActivate: false,
    ...overrides,
  }
}

export function createModelDraft(seed: number, existingModels: ModelDraft[] = []): ModelDraft {
  return {
    id: `model-${Date.now()}-${seed}`,
    name: existingModels.length === 0 ? DEFAULT_MODEL_NAME : '',
    providerKind: DEFAULT_SETUP_PROVIDER_ID,
    baseUrl: nextRuntimeBaseUrl(existingModels),
    accessKey: '',
    endpointName: 'primary',
  }
}

export function switchSetupModelProvider(
  model: ModelDraft,
  providerKind: ProviderKind,
): ModelDraft {
  const previousDefault = getSetupProviderOption(model.providerKind).initialBaseUrl
  const nextDefault = getSetupProviderOption(providerKind).initialBaseUrl
  const baseUrl = model.baseUrl.trim()
  const generatedVllmEndpoint = model.providerKind === 'vllm' && /^vllm:\d+$/.test(baseUrl)
  return {
    ...model,
    providerKind,
    baseUrl:
      !baseUrl || baseUrl === previousDefault || generatedVllmEndpoint
        ? nextDefault
        : model.baseUrl,
  }
}

function nextRuntimeBaseUrl(existingModels: ModelDraft[]): string {
  const usedEndpoints = new Set(
    existingModels
      .map((model) => normalizeRuntimeEndpoint(model))
      .filter((endpoint): endpoint is string => Boolean(endpoint)),
  )

  let port = 8000
  let candidate = DEFAULT_SETUP_RUNTIME_BASE_URL
  while (usedEndpoints.has(normalizeBaseUrl(candidate, DEFAULT_SETUP_PROVIDER_ID) ?? '')) {
    port += 1
    candidate = `vllm:${port}`
  }

  return candidate
}

function normalizeModelName(value: string): string {
  return value.trim().toLowerCase()
}

function normalizeRuntimeEndpoint(
  model: Pick<ModelDraft, 'baseUrl' | 'providerKind'>,
): string | null {
  if (!isSetupRuntimeProvider(model.providerKind)) {
    return null
  }

  return normalizeBaseUrl(model.baseUrl, model.providerKind)
}

function normalizeBaseUrl(rawValue: string, providerKind: ProviderKind): string | null {
  try {
    return parseBaseUrl(rawValue, providerKind).endpoint.toLowerCase()
  } catch {
    return null
  }
}

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function inferProtocol(endpoint: string, providerKind: ProviderKind): 'http' | 'https' {
  if (
    endpoint.startsWith('localhost') ||
    endpoint.startsWith('127.0.0.1') ||
    endpoint.startsWith('0.0.0.0') ||
    endpoint.startsWith('host.docker.internal')
  ) {
    return 'http'
  }

  if (endpoint.includes(':80')) {
    return 'http'
  }

  if (getSetupProviderOption(providerKind).initialBaseUrl.startsWith('http://')) {
    return 'http'
  }

  return 'https'
}

export function parseBaseUrl(
  rawValue: string,
  providerKind: ProviderKind,
): {
  protocol: 'http' | 'https'
  endpoint: string
} {
  const trimmed = rawValue.trim().replace(/\/$/, '')
  if (!trimmed) {
    throw new Error('Model base URL is required.')
  }

  const normalized = trimmed.includes('://')
    ? trimmed
    : `${inferProtocol(trimmed, providerKind)}://${trimmed}`

  let parsed: URL
  try {
    parsed = new URL(normalized)
  } catch {
    throw new Error(`Invalid model endpoint: ${rawValue}`)
  }

  const protocol = parsed.protocol.replace(':', '')
  if (protocol !== 'http' && protocol !== 'https') {
    throw new Error(`Unsupported protocol for model endpoint: ${rawValue}`)
  }

  const path = parsed.pathname && parsed.pathname !== '/' ? parsed.pathname.replace(/\/$/, '') : ''
  return {
    protocol,
    endpoint: `${parsed.host}${path}`,
  }
}

export function getModelDraftFieldErrors(
  models: ModelDraft[],
): Record<string, ModelDraftFieldErrors> {
  const errors: Record<string, ModelDraftFieldErrors> = {}
  const modelIdsByName = new Map<string, string[]>()
  const modelIdsByEndpoint = new Map<string, string[]>()
  const endpointLabels = new Map<string, string>()

  models.forEach((model) => {
    errors[model.id] = {}

    const trimmedName = model.name.trim()
    if (!trimmedName) {
      errors[model.id].name = 'Model name is required.'
    } else {
      const normalizedName = normalizeModelName(trimmedName)
      modelIdsByName.set(normalizedName, [...(modelIdsByName.get(normalizedName) ?? []), model.id])
    }

    if (!model.baseUrl.trim()) {
      errors[model.id].baseUrl = 'Base URL or host is required.'
      return
    }

    let parsedEndpoint = ''
    try {
      parsedEndpoint = parseBaseUrl(model.baseUrl, model.providerKind).endpoint
    } catch {
      errors[model.id].baseUrl = 'Enter a valid base URL or host before continuing.'
      return
    }

    if (!isSetupRuntimeProvider(model.providerKind)) {
      return
    }

    const endpointKey = parsedEndpoint.toLowerCase()
    endpointLabels.set(endpointKey, parsedEndpoint)
    modelIdsByEndpoint.set(endpointKey, [...(modelIdsByEndpoint.get(endpointKey) ?? []), model.id])
  })

  modelIdsByName.forEach((ids) => {
    if (ids.length < 2) {
      return
    }

    ids.forEach((id) => {
      const model = models.find((candidate) => candidate.id === id)
      errors[id].name = `Model name "${model?.name.trim() ?? ''}" is duplicated.`
    })
  })

  modelIdsByEndpoint.forEach((ids, endpointKey) => {
    if (ids.length < 2) {
      return
    }

    const endpoint = endpointLabels.get(endpointKey) ?? endpointKey
    ids.forEach((id) => {
      errors[id].baseUrl = `Endpoint "${endpoint}" is already used by another local model.`
    })
  })

  return errors
}

export function getStepOneErrors(models: ModelDraft[], defaultModelId: string): string[] {
  const errors: string[] = []

  if (models.length === 0) {
    errors.push('Add at least one model before continuing.')
    return errors
  }

  const names = new Set<string>()
  const localEndpoints = new Set<string>()
  let hasDefault = false

  models.forEach((model, index) => {
    const position = index + 1
    const trimmedName = model.name.trim()

    if (!trimmedName) {
      errors.push(`Model ${position} is missing a model name.`)
    } else {
      const normalizedName = normalizeModelName(trimmedName)
      if (names.has(normalizedName)) {
        errors.push(`Model name "${trimmedName}" is duplicated.`)
      }
      names.add(normalizedName)
    }

    if (!model.baseUrl.trim()) {
      errors.push(`Model ${position} is missing a base URL.`)
    } else {
      try {
        const parsedEndpoint = parseBaseUrl(model.baseUrl, model.providerKind).endpoint
        if (isSetupRuntimeProvider(model.providerKind)) {
          const endpointKey = parsedEndpoint.toLowerCase()
          if (localEndpoints.has(endpointKey)) {
            errors.push(`Endpoint "${parsedEndpoint}" is already used by another local model.`)
          }
          localEndpoints.add(endpointKey)
        }
      } catch (err) {
        errors.push(
          err instanceof Error ? err.message : `Model ${position} has an invalid base URL.`,
        )
      }
    }

    if (model.id === defaultModelId) {
      hasDefault = true
    }
  })

  if (!hasDefault) {
    errors.push('Choose a default model before continuing.')
  }

  return errors
}

export function countConfigSignals(rawSignals: unknown): number {
  if (!rawSignals || typeof rawSignals !== 'object' || Array.isArray(rawSignals)) {
    return 0
  }

  return Object.values(rawSignals as Record<string, unknown>).reduce<number>((total, value) => {
    return total + (Array.isArray(value) ? value.length : 0)
  }, 0)
}

export function summarizeSetupConfig(config: Record<string, unknown> | null): SetupConfigCounts {
  if (!config) {
    return createSetupConfigCounts()
  }

  const providers =
    config.providers && typeof config.providers === 'object'
      ? (config.providers as Record<string, unknown>)
      : {}
  const recipes = Array.isArray(config.recipes)
    ? config.recipes.filter(
        (recipe): recipe is Record<string, unknown> =>
          Boolean(recipe) && typeof recipe === 'object' && !Array.isArray(recipe),
      )
    : []
  const models = Array.isArray(providers.models) ? providers.models.length : 0
  const decisions = recipes.reduce(
    (total, recipe) => total + (Array.isArray(recipe.decisions) ? recipe.decisions.length : 0),
    0,
  )
  const signals = recipes.reduce((total, recipe) => total + countConfigSignals(recipe.signals), 0)

  return createSetupConfigCounts({
    models,
    decisions,
    signals,
    canActivate: models > 0 && decisions > 0,
  })
}

export function buildSetupConfig(
  models: ModelDraft[],
  defaultModelId: string,
): Record<string, unknown> {
  const builtModels: BuiltModel[] = models.map((model, index) => {
    const { protocol, endpoint } = parseBaseUrl(model.baseUrl, model.providerKind)
    const endpointName =
      model.endpointName.trim() || `${slugify(model.name) || `model-${index + 1}`}-primary`
    const apiKey = model.accessKey.trim() || undefined
    const trimmedBaseUrl = model.baseUrl.trim().replace(/\/$/, '')
    const provider = getSetupProviderOption(model.providerKind)
    const backendRef: BuiltModel['backend_refs'][number] = isSetupRuntimeProvider(
      model.providerKind,
    )
      ? {
          name: endpointName,
          weight: 100,
          endpoint,
          protocol,
          provider: model.providerKind,
          api_key: apiKey,
        }
      : {
          name: endpointName,
          weight: 100,
          protocol,
          endpoint: trimmedBaseUrl,
          provider: model.providerKind,
          api_key: apiKey,
        }

    return {
      name: model.name.trim(),
      provider_model_id: model.name.trim(),
      backend_refs: [backendRef],
      ...(provider.apiFormat === 'chat_completions' ? {} : { api_format: provider.apiFormat }),
    }
  })

  const defaultModel = builtModels.find((model) => {
    const draft = models.find((item) => item.id === defaultModelId)
    return draft?.name.trim() === model.name
  })

  if (!defaultModel) {
    throw new Error('Default model selection is invalid.')
  }

  const catchAllDecision = {
    name: 'default-route',
    description: 'Generated during setup to route all requests to the default model.',
    priority: 100,
    rules: {
      operator: 'AND',
      conditions: [],
    },
    modelRefs: [
      {
        model: defaultModel.name,
        use_reasoning: false,
      },
    ],
  }

  const config: Record<string, unknown> = {
    providers: {
      models: builtModels,
      defaults: {
        model: defaultModel.name,
      },
    },
    recipes: [
      {
        name: 'default',
        decisions: [catchAllDecision],
      },
    ],
  }

  return config
}

export interface PresetModel {
  name: string
  role: string
}

export interface PresetInfo {
  id: string
  label: string
  summary: string
  required_models: PresetModel[]
  recipe_url: string
}

export interface PresetDelta {
  preset_id: string
  configured_models: string[]
  missing_models: PresetModel[]
  ready: boolean
  recipe_url: string
}

export async function fetchPresets(): Promise<PresetInfo[]> {
  const resp = await fetch('/api/setup/presets')
  if (!resp.ok) {
    throw new Error(`Failed to fetch presets: ${resp.status}`)
  }
  return resp.json()
}

export async function fetchPresetDelta(presetId: string, models: string[]): Promise<PresetDelta> {
  const resp = await fetch('/api/setup/presets/delta', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset_id: presetId, models }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to fetch preset delta: ${resp.status}`)
  }
  return resp.json()
}

export function maskSecrets(config: Record<string, unknown> | null): string {
  if (!config) {
    return ''
  }

  return JSON.stringify(
    config,
    (key, value) => {
      if (
        (key === 'api_key' || key === 'access_key') &&
        typeof value === 'string' &&
        value.length > 0
      ) {
        return '••••••••'
      }
      return value
    },
    2,
  )
}

/** The draft config as YAML for review, with access keys masked. */
export function maskedSetupYaml(config: Record<string, unknown> | null): string {
  if (!config) return ''
  const masked = JSON.parse(maskSecrets(config)) as Record<string, unknown>
  return configYaml(masked, ROUTER_CONFIG_DOCUMENT_SCHEMA).trimEnd()
}

/** The model IDs a provider endpoint lists, through the dashboard's discovery helper. */
export async function discoverSetupModels(
  model: ModelDraft,
  signal?: AbortSignal,
): Promise<string[]> {
  const { protocol, endpoint } = parseBaseUrl(model.baseUrl, model.providerKind)
  const baseUrl = model.baseUrl.trim().includes('://')
    ? model.baseUrl.trim()
    : `${protocol}://${endpoint}`
  const response = await fetch('/api/models/discover', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ baseUrl, apiKey: model.accessKey.trim(), provider: model.providerKind }),
    signal,
  })
  let payload: { models?: unknown; error?: unknown } = {}
  try {
    payload = (await response.json()) as typeof payload
  } catch {
    // An empty body falls back to the status line below.
  }
  if (!response.ok) {
    throw new Error(
      typeof payload.error === 'string' && payload.error
        ? payload.error
        : `The endpoint answered HTTP ${response.status}.`,
    )
  }
  return Array.isArray(payload.models)
    ? payload.models.filter(
        (item): item is string => typeof item === 'string' && item.trim() !== '',
      )
    : []
}

const CPU_DECISION_MODELS: readonly RegExp[] = [
  /vela[\s._-]*3\.1[\s._-]*fast/i,
  /vela[\s._-]*3\.1.*pico.*150\s*m/i,
  /vela[\s._-]*3\.1.*pico.*380\s*m/i,
]

/** True when the dashboard runs without a GPU platform (neither ROCm nor CUDA). */
export function isCpuOnlyPlatform(platform: string | undefined): boolean {
  const value = (platform ?? '').trim().toLowerCase()
  return value !== 'rocm' && value !== 'cuda'
}

/** The memory the container runtime reports in GiB, or null when it is unknown. */
export function hostMemoryGiB(bytes: number | undefined): number | null {
  return bytes && Number.isFinite(bytes) && bytes > 0 ? bytes / 1024 ** 3 : null
}

/** Whether a model's memory floor fits the host; unknown memory or floor counts as fitting. */
export function decisionModelFits(
  option: { minMemoryGiB?: number | null },
  memoryGiB: number | null,
): boolean {
  return memoryGiB === null || option.minMemoryGiB == null || option.minMemoryGiB <= memoryGiB
}

/** A memory size in GiB with at most one decimal, such as 7.7 GiB. */
export function formatGiB(value: number): string {
  return `${Number(value.toFixed(1))} GiB`
}

/** The warning for a model whose memory floor exceeds the host, or null when it fits. */
export function memoryShortfall(
  option: { minMemoryGiB?: number | null },
  memoryGiB: number | null,
): string | null {
  if (memoryGiB === null || option.minMemoryGiB == null || decisionModelFits(option, memoryGiB))
    return null
  return `Needs ${formatGiB(option.minMemoryGiB)}; this host has ${formatGiB(memoryGiB)}`
}

/**
 * The decision models to recommend: on a CPU-only deployment the small Vela
 * 3.1 models the runtime catalog ships, otherwise the default. When the host
 * memory is known, only the CPU models that fit it are recommended, largest
 * first, so the first one is the largest model the host can run.
 */
export function recommendedDecisionModels<
  T extends { name: string; label: string; artifact: string; minMemoryGiB?: number | null },
>(
  options: readonly T[],
  cpuOnly: boolean,
  defaultName: string,
  memoryGiB: number | null = null,
): T[] {
  if (cpuOnly) {
    const matches = [
      ...new Set(
        CPU_DECISION_MODELS.flatMap((pattern) => {
          const found = options.find((option) =>
            pattern.test(`${option.name} ${option.label} ${option.artifact}`),
          )
          return found ? [found] : []
        }),
      ),
    ]
    const fitting = matches.filter((option) => decisionModelFits(option, memoryGiB))
    if (memoryGiB !== null) {
      fitting.sort((left, right) => (right.minMemoryGiB ?? 0) - (left.minMemoryGiB ?? 0))
    }
    if (fitting.length) return fitting
    if (matches.length) return matches
  }
  const fallback = options.find((option) => option.name === defaultName)
  return fallback ? [fallback] : []
}

/** The CPU decision models the catalog announces but the model runtime does not serve yet. */
export function upcomingCpuDecisionModels<
  T extends { name: string; label: string; artifact: string; servable: boolean },
>(catalog: readonly T[]): T[] {
  return CPU_DECISION_MODELS.flatMap((pattern) => {
    const found = catalog.find(
      (option) =>
        !option.servable && pattern.test(`${option.name} ${option.label} ${option.artifact}`),
    )
    return found ? [found] : []
  })
}
