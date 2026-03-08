import { getRoutingPreset, type RoutingPresetId } from '../presets/routingPresets'

export type SetupStep = 0 | 1 | 2
export type ProviderKind = 'vllm' | 'openai-compatible' | 'anthropic'
export type SetupValidationState = 'idle' | 'validating' | 'valid' | 'error'
export type SetupActivationState = 'idle' | 'activating' | 'error'

export interface ModelDraft {
  id: string
  name: string
  providerKind: ProviderKind
  baseUrl: string
  accessKey: string
  endpointName: string
}

interface BuiltModel {
  name: string
  endpoints: Array<{
    name: string
    weight: number
    endpoint: string
    protocol: 'http' | 'https'
  }>
  access_key?: string
  api_format?: 'anthropic'
}

export interface ProviderOption {
  id: ProviderKind
  label: string
  description: string
  placeholder: string
}

export const PROVIDER_OPTIONS: ProviderOption[] = [
  {
    id: 'vllm',
    label: 'Local vLLM',
    description: 'Best for first-run with a local or self-hosted OpenAI-compatible endpoint.',
    placeholder: 'http://vllm-gpt-oss-120b:8000',
  },
  {
    id: 'openai-compatible',
    label: 'OpenAI-compatible API',
    description: 'Works for hosted endpoints that expose the OpenAI chat/completions surface.',
    placeholder: 'https://api.openai.com',
  },
  {
    id: 'anthropic',
    label: 'Anthropic Messages API',
    description: 'Uses Anthropic-compatible request translation inside the router.',
    placeholder: 'https://api.anthropic.com',
  },
]

export const SETUP_STEP_LABELS: ReadonlyArray<[string, string]> = [
  ['1', 'Connect model'],
  ['2', 'Choose routing'],
  ['3', 'Review & activate'],
]

export function createModelDraft(seed: number): ModelDraft {
  return {
    id: `model-${Date.now()}-${seed}`,
    name: 'openai/gpt-oss-120b',
    providerKind: 'vllm',
    baseUrl: 'http://vllm-gpt-oss-120b:8000',
    accessKey: '',
    endpointName: 'primary',
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
  if (providerKind === 'anthropic') {
    return 'https'
  }

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

  return 'https'
}

export function parseBaseUrl(rawValue: string, providerKind: ProviderKind): {
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

export function getStepOneErrors(models: ModelDraft[], defaultModelId: string): string[] {
  const errors: string[] = []

  if (models.length === 0) {
    errors.push('Add at least one model before continuing.')
    return errors
  }

  const names = new Set<string>()
  let hasDefault = false

  models.forEach((model, index) => {
    const position = index + 1
    const trimmedName = model.name.trim()

    if (!trimmedName) {
      errors.push(`Model ${position} is missing a model name.`)
    } else {
      const normalizedName = trimmedName.toLowerCase()
      if (names.has(normalizedName)) {
        errors.push(`Model name "${trimmedName}" is duplicated.`)
      }
      names.add(normalizedName)
    }

    if (!model.baseUrl.trim()) {
      errors.push(`Model ${position} is missing a base URL.`)
    } else {
      try {
        parseBaseUrl(model.baseUrl, model.providerKind)
      } catch (err) {
        errors.push(err instanceof Error ? err.message : `Model ${position} has an invalid base URL.`)
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

function mergeSignals(
  presetSignals?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  if (!presetSignals) {
    return undefined
  }

  const mergedEntries = Object.entries(presetSignals).filter(([, value]) => Array.isArray(value) && value.length > 0)
  if (mergedEntries.length === 0) {
    return undefined
  }

  return Object.fromEntries(mergedEntries)
}

export function buildSetupConfig(
  models: ModelDraft[],
  defaultModelId: string,
  selectedPresetId: RoutingPresetId | null,
): Record<string, unknown> {
  const builtModels: BuiltModel[] = models.map((model, index) => {
    const { protocol, endpoint } = parseBaseUrl(model.baseUrl, model.providerKind)
    const endpointName = model.endpointName.trim() || `${slugify(model.name) || `model-${index + 1}`}-primary`

    return {
      name: model.name.trim(),
      endpoints: [
        {
          name: endpointName,
          weight: 100,
          endpoint,
          protocol,
        },
      ],
      access_key: model.accessKey.trim() || undefined,
      api_format: model.providerKind === 'anthropic' ? 'anthropic' : undefined,
    }
  })

  const defaultModel = builtModels.find((model) => {
    const draft = models.find((item) => item.id === defaultModelId)
    return draft?.name.trim() === model.name
  })

  if (!defaultModel) {
    throw new Error('Default model selection is invalid.')
  }

  const preset = selectedPresetId ? getRoutingPreset(selectedPresetId) : null
  const presetFragment = preset ? preset.build(defaultModel.name) : null

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
      default_model: defaultModel.name,
    },
    decisions: [...(presetFragment?.decisions ?? []), catchAllDecision],
  }

  const signals = mergeSignals(presetFragment?.signals as Record<string, unknown> | undefined)
  if (signals) {
    config.signals = signals
  }

  return config
}

export function maskSecrets(config: Record<string, unknown> | null): string {
  if (!config) {
    return ''
  }

  return JSON.stringify(
    config,
    (key, value) => {
      if (key === 'access_key' && typeof value === 'string' && value.length > 0) {
        return '••••••••'
      }
      return value
    },
    2,
  )
}
