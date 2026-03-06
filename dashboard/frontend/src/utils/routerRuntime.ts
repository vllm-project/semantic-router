export interface RouterRuntimeStatus {
  phase: string
  ready: boolean
  message?: string
  downloading_model?: string
  pending_models?: string[]
  ready_models?: number
  total_models?: number
}

export interface StatusWithRouterRuntime {
  overall?: string
  router_runtime?: RouterRuntimeStatus
}

export type ModelStatusTone = 'ok' | 'warn' | 'down'

export interface ModelStatusSummary {
  value: string
  detail: string
  tone: ModelStatusTone
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
