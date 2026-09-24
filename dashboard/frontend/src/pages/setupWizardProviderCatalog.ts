import { modelProviderCatalog, type ModelProviderPreset } from './modelProviderCatalog'

// The setup wizard intentionally starts with a short onboarding subset. The
// IDs are the only local UX choice; all provider facts come from the generated
// catalog projection shared with Add Model and Model Hub.
export const SETUP_PROVIDER_IDS = ['vllm', 'openai-compatible', 'anthropic'] as const

export type ProviderKind = (typeof SETUP_PROVIDER_IDS)[number]

export interface SetupProviderOption {
  id: ProviderKind
  label: string
  description: string
  placeholder: string
  initialBaseUrl: string
  apiFormat: ModelProviderPreset['apiFormat']
  supportTier: ModelProviderPreset['supportTier']
}

export const DEFAULT_SETUP_RUNTIME_BASE_URL = 'vllm:8000'
const DEFAULT_COMPATIBLE_PLACEHOLDER = 'https://api.example.com/v1'

function requireCatalogProvider(id: ProviderKind): ModelProviderPreset {
  const provider = modelProviderCatalog.find((candidate) => candidate.id === id)
  if (!provider) {
    throw new Error(`Setup provider ${id} is missing from the generated catalog.`)
  }
  return provider
}

function projectSetupProvider(id: ProviderKind): SetupProviderOption {
  const provider = requireCatalogProvider(id)
  const runtimeBaseUrl = provider.supportTier === 'runtime' ? DEFAULT_SETUP_RUNTIME_BASE_URL : ''
  const initialBaseUrl = provider.baseUrl || runtimeBaseUrl

  return {
    id,
    label: provider.name,
    description: provider.description,
    placeholder: initialBaseUrl || DEFAULT_COMPATIBLE_PLACEHOLDER,
    initialBaseUrl,
    apiFormat: provider.apiFormat,
    supportTier: provider.supportTier,
  }
}

export const SETUP_PROVIDER_OPTIONS: readonly SetupProviderOption[] =
  SETUP_PROVIDER_IDS.map(projectSetupProvider)

export const DEFAULT_SETUP_PROVIDER_ID: ProviderKind = SETUP_PROVIDER_IDS[0]

export function getSetupProviderOption(id: ProviderKind): SetupProviderOption {
  const provider = SETUP_PROVIDER_OPTIONS.find((candidate) => candidate.id === id)
  if (!provider) {
    throw new Error(`Unsupported setup provider: ${id}`)
  }
  return provider
}

export function isSetupRuntimeProvider(id: ProviderKind): boolean {
  return getSetupProviderOption(id).supportTier === 'runtime'
}
