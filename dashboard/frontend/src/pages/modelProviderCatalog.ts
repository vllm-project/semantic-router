import catalog from '../generated/modelCatalog.json'
import type { CatalogProvider } from '../types/modelCatalog'
import { monochromeModelProviderIcons, resolveModelCatalogIcon } from './modelProviderIcons'

export interface ModelProviderPreset {
  id: string
  name: string
  description: string
  category: 'Start here' | 'Model APIs' | 'Private runtimes'
  baseUrl: string
  apiFormat: string
  authStrategy: CatalogProvider['auth']['strategy']
  icon: string
  monogram: string
  supportTier: 'native' | 'compatible' | 'runtime'
  protocols: string[]
  supportsModelDiscovery: boolean
}

const categoryName = (category: CatalogProvider['category']): ModelProviderPreset['category'] => {
  if (category === 'start_here') return 'Start here'
  if (category === 'model_api') return 'Model APIs'
  return 'Private runtimes'
}

const apiFormat = (protocol: string): string => {
  if (protocol === 'anthropic/messages@1') return 'anthropic'
  if (protocol === 'openai/responses@1') return 'responses'
  return 'openai'
}

export const isMonochromeModelProviderIcon = (icon: string): boolean =>
  monochromeModelProviderIcons.has(icon)

// This is a projection, not a second inventory. Provider identity, order,
// support level, protocols, defaults, auth, and presentation all come from the
// generated repository catalog; package:* merely resolves bundled SVG assets.
export const modelProviderPresetsFromCatalog = (
  providers: readonly CatalogProvider[],
): ModelProviderPreset[] =>
  providers.map((provider) => ({
    id: provider.id,
    name: provider.display_name,
    description: provider.description,
    category: categoryName(provider.category),
    baseUrl: provider.default_base_url ?? '',
    apiFormat: apiFormat(provider.default_protocol),
    authStrategy: provider.auth.strategy,
    icon: resolveModelCatalogIcon(provider.presentation.logo),
    monogram: provider.presentation.monogram,
    supportTier: provider.support_tier,
    protocols: [...provider.protocols],
    supportsModelDiscovery: provider.supported_operations.includes(
      `${provider.default_protocol}#list_models`,
    ),
  }))

// Generated fallback keeps Add Model usable while the authenticated API is
// loading. Both projections are generated from the same repository source.
export const modelProviderCatalog = modelProviderPresetsFromCatalog(
  catalog.providers as CatalogProvider[],
)

interface ProviderLookupInput {
  backendName?: string
  baseUrl?: string
  apiFormat?: string
  providers?: readonly ModelProviderPreset[]
}

function normalizedProviderID(value?: string): string {
  return (value ?? '')
    .trim()
    .toLowerCase()
    .replace(/-primary$/, '')
}

function providerHost(value?: string): string {
  if (!value) return ''
  try {
    return new URL(value).hostname.toLowerCase()
  } catch {
    return ''
  }
}

export function findModelProviderPreset({
  backendName,
  baseUrl,
  apiFormat: format,
  providers = modelProviderCatalog,
}: ProviderLookupInput): ModelProviderPreset | undefined {
  const providerID = normalizedProviderID(backendName)
  const exact = providers.find((provider) => provider.id === providerID)
  if (exact) return exact

  const host = providerHost(baseUrl)
  if (host) {
    const hostMatch = providers.find(
      (provider) => providerHost(provider.baseUrl) === host,
    )
    if (hostMatch) return hostMatch
  }

  if (format === 'anthropic') {
    return providers.find((provider) => provider.id === 'anthropic')
  }
  if (format === 'openai' || format === 'responses') {
    return providers.find((provider) => provider.id === 'openai-compatible')
  }
  return undefined
}
