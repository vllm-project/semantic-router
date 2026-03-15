export interface RouterRuntimeStatus {
  phase: string
  ready: boolean
  message?: string
  downloading_model?: string
  pending_models?: string[]
  ready_models?: number
  total_models?: number
}

export interface ServiceStatus {
  name: string
  status: string
  healthy: boolean
  message?: string
  component?: string
}

export interface RouterModelRegistryInfo {
  local_path?: string
  repo_id?: string
  purpose?: string
  description?: string
  parameter_size?: string
  embedding_dim?: number
  max_context_length?: number
  base_model_max_context?: number
  uses_lora?: boolean
  num_classes?: number
  tags?: string[]
  model_card_url?: string
  pipeline_tag?: string
  base_model?: string
  license?: string
  languages?: string[]
  datasets?: string[]
}

export interface RouterModelInfo {
  name: string
  type: string
  loaded: boolean
  state?: string
  model_path?: string
  resolved_model_path?: string
  categories?: string[]
  metadata?: Record<string, string>
  registry?: RouterModelRegistryInfo | null
  load_time?: string
  memory_usage?: string
}

export interface RouterModelsSummary {
  ready?: boolean
  phase?: string
  message?: string
  downloading_model?: string
  pending_models?: string[]
  loaded_models?: number
  total_models?: number
  updated_at?: string
}

export interface RouterModelsSystemInfo {
  go_version: string
  architecture: string
  os: string
  memory_usage: string
  gpu_available: boolean
}

export interface RouterModelsInfo {
  models: RouterModelInfo[]
  summary?: RouterModelsSummary
  system?: RouterModelsSystemInfo
}

export interface StatusWithRouterRuntime {
  overall?: string
  router_runtime?: RouterRuntimeStatus
}

export interface SystemStatus {
  overall: string
  deployment_type: string
  services: ServiceStatus[]
  version?: string
  router_runtime?: RouterRuntimeStatus
  models?: RouterModelsInfo
}

export type ModelStatusTone = 'ok' | 'warn' | 'down'

export interface ModelStatusSummary {
  value: string
  detail: string
  tone: ModelStatusTone
}

const ROUTER_MODEL_STATE_ORDER: Record<string, number> = {
  ready: 0,
  downloading: 1,
  pending: 2,
  initializing: 3,
  not_loaded: 4,
}

export function describeRouterRuntime(runtime: RouterRuntimeStatus): string {
  const readyModels = runtime.ready_models ?? 0
  const totalModels = runtime.total_models ?? 0

  switch (runtime.phase) {
    case 'downloading_models':
      if (totalModels > 0) {
        return `Downloading router models ${readyModels}/${totalModels}`
      }
      return 'Downloading required router models'
    case 'checking_models':
      return 'Checking required router models'
    case 'initializing_models':
      return 'Initializing router models'
    case 'starting':
      return 'Starting router services'
    case 'error':
      return runtime.message || 'Router startup failed'
    default:
      return runtime.message || 'Router startup in progress'
  }
}

export function getActiveRouterRuntime(
  status?: StatusWithRouterRuntime | null,
): RouterRuntimeStatus | null {
  const runtime = status?.router_runtime

  if (!runtime || runtime.ready || runtime.phase === 'setup_mode') {
    return null
  }

  if (status?.overall === 'not_running' || status?.overall === 'stopped') {
    return null
  }

  return runtime
}

export function getRouterModelState(model: RouterModelInfo): string {
  if (model.state) {
    return model.state
  }
  return model.loaded ? 'ready' : 'not_loaded'
}

export function getRouterModelStateLabel(model: RouterModelInfo): string {
  const state = getRouterModelState(model)

  switch (state) {
    case 'ready':
      return 'Ready'
    case 'downloading':
      return 'Downloading'
    case 'pending':
      return 'Pending'
    case 'initializing':
      return 'Initializing'
    default:
      return 'Not Loaded'
  }
}

export function getLoadedModelCount(modelsInfo?: RouterModelsInfo | null): number {
  if (!modelsInfo) return 0
  if (typeof modelsInfo.summary?.loaded_models === 'number') {
    return modelsInfo.summary.loaded_models
  }
  return modelsInfo.models.filter((model) => model.loaded).length
}

export function getTotalKnownModelCount(modelsInfo?: RouterModelsInfo | null): number {
  if (!modelsInfo) return 0
  if (typeof modelsInfo.summary?.total_models === 'number') {
    return modelsInfo.summary.total_models
  }
  return modelsInfo.models.length
}

export function sortRouterModels(models: RouterModelInfo[]): RouterModelInfo[] {
  return [...models].sort((left, right) => {
    const leftState = ROUTER_MODEL_STATE_ORDER[getRouterModelState(left)] ?? 99
    const rightState = ROUTER_MODEL_STATE_ORDER[getRouterModelState(right)] ?? 99
    if (leftState !== rightState) {
      return leftState - rightState
    }

    return left.name.localeCompare(right.name)
  })
}

export function getPreviewRouterModels(
  modelsInfo?: RouterModelsInfo | null,
  limit = 4,
): RouterModelInfo[] {
  if (!modelsInfo?.models?.length) {
    return []
  }

  const sorted = sortRouterModels(modelsInfo.models)
  const loaded = sorted.filter((model) => model.loaded)
  if (loaded.length > 0) {
    return loaded.slice(0, limit)
  }

  return sorted.slice(0, limit)
}

export function getRouterModelAnchor(model: Pick<RouterModelInfo, 'name'>): string {
  const slug = model.name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

  return `model-${slug || 'unknown'}`
}

export function getModelStatusSummary(
  status?: StatusWithRouterRuntime | null,
): ModelStatusSummary {
  if (!status) {
    return {
      value: 'Unknown',
      detail: 'Waiting for router status',
      tone: 'warn',
    }
  }

  if (status.overall === 'not_running' || status.overall === 'stopped') {
    return {
      value: 'Offline',
      detail: 'Router is not running',
      tone: 'down',
    }
  }

  const runtime = getActiveRouterRuntime(status)
  if (runtime) {
    if (runtime.phase === 'downloading_models') {
      const readyModels = runtime.ready_models ?? 0
      const totalModels = runtime.total_models ?? 0
      if (totalModels > 0) {
        return {
          value: `${readyModels}/${totalModels}`,
          detail: 'Downloading models',
          tone: 'warn',
        }
      }
      return {
        value: 'Syncing',
        detail: 'Downloading models',
        tone: 'warn',
      }
    }

    if (runtime.phase === 'checking_models') {
      return {
        value: 'Checking',
        detail: 'Verifying model assets',
        tone: 'warn',
      }
    }

    if (runtime.phase === 'error') {
      return {
        value: 'Error',
        detail: runtime.message || 'Router startup failed',
        tone: 'down',
      }
    }

    return {
      value: 'Starting',
      detail: describeRouterRuntime(runtime),
      tone: 'warn',
    }
  }

  if (status.overall === 'degraded') {
    return {
      value: 'Degraded',
      detail: 'Router is up with warnings',
      tone: 'warn',
    }
  }

  return {
    value: 'Ready',
    detail: 'All required models are ready',
    tone: 'ok',
  }
}
