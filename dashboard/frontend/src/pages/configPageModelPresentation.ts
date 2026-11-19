import catalogMetadata from '../modelCatalogMetadata'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { formatInsightsCost } from '../utils/insightsCost'
import { DEFAULT_RECIPE_NAME, listRecipeProfiles } from './configPageEntrypointsRecipesSupport'
import type {
  CanonicalModelDeployment,
  ConfigData,
  DecisionConfig,
  ModelPricing,
  NormalizedModel,
} from './configPageSupport'

export const bundledCatalog = catalogMetadata as BuiltInModelCatalog

export function catalogProvider(providerId: string | undefined, catalog = bundledCatalog) {
  const id = providerId?.trim().toLowerCase()
  return id ? catalog.providers.find((provider) => provider.id === id) : undefined
}

/** A provider's catalog name, or its ID when the catalog does not know it. */
export function providerDisplayName(providerId: string, catalog = bundledCatalog): string {
  return catalogProvider(providerId, catalog)?.display_name ?? providerId
}

/** The bundled logo a provider shows, for components that take a logo key such as ProviderStack. */
export function providerLogoKey(providerId: string, catalog = bundledCatalog): string {
  const logo = catalogProvider(providerId, catalog)?.presentation.logo ?? ''
  return logo.startsWith('package:') ? logo.slice('package:'.length) : providerId
}

export function modelProviderIds(model: Pick<NormalizedModel, 'backend_refs'>): string[] {
  return [
    ...new Set(
      (model.backend_refs ?? [])
        .map((backend) => backend.provider?.trim())
        .filter((provider): provider is string => Boolean(provider)),
    ),
  ]
}

export function formatModelPrice(value: number | undefined, currency?: string): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return formatInsightsCost(value, currency || 'USD', { total: true })
}

export function modelPriceSummary(pricing?: ModelPricing) {
  return {
    input: formatModelPrice(pricing?.prompt_per_1m, pricing?.currency),
    output: formatModelPrice(pricing?.completion_per_1m, pricing?.currency),
  }
}

export const modalityLabel = (modalities?: { input: string[]; output: string[] }): string =>
  `${modalities?.input.join(', ') || 'text'} → ${modalities?.output.join(', ') || 'text'}`

export type ModelKind =
  | 'chat'
  | 'reasoning'
  | 'vision'
  | 'image'
  | 'audio'
  | 'embedding'
  | 'decision'
  | 'mixture'

export const MODEL_KIND_LABELS: Record<ModelKind, string> = {
  chat: 'Chat',
  reasoning: 'Reasoning',
  vision: 'Vision',
  image: 'Image',
  audio: 'Audio',
  embedding: 'Embedding',
  decision: 'Decision',
  mixture: 'Mixture',
}

/**
 * What a model is for, from its own declaration and else its catalog card:
 * what it produces first, then whether it reasons, then what it reads.
 */
export function modelKind(model: NormalizedModel, catalog = bundledCatalog): ModelKind {
  const card = model.catalog
    ? catalog.models.find((entry) => entry.id === model.catalog)
    : undefined
  const format = (model.api_format ?? '').toLowerCase()
  const capabilities = new Set(
    [...(model.capabilities ?? []), ...(card?.capabilities ?? [])].map((value) =>
      value.toLowerCase(),
    ),
  )
  const modalities = model.modalities ?? card?.modalities
  const input = modalities?.input ?? []
  const output = modalities?.output ?? []
  if (format === 'systemone') return 'decision'
  if (card?.kind === 'virtual') return 'mixture'
  if (format.includes('embedding') || capabilities.has('embedding') || output.includes('embedding'))
    return 'embedding'
  if (format === 'images' || output.includes('image')) return 'image'
  if (output.includes('audio') && !output.includes('text')) return 'audio'
  if (model.reasoning_family || model.reasoning?.type || capabilities.has('reasoning'))
    return 'reasoning'
  if (input.includes('image') || capabilities.has('vision')) return 'vision'
  return 'chat'
}

/** Who answers a model, on one line: its providers, a shared deployment, or nothing. */
export function modelServedBy(
  model: NormalizedModel,
  deployment: string | undefined,
  catalog = bundledCatalog,
): string | null {
  if (model.endpoints.length === 0 && deployment) return `Model runtime · ${deployment}`
  if (model.endpoints.length === 0) return null
  const providers = modelProviderIds(model).map((id) => providerDisplayName(id, catalog))
  const label = providers.length ? providers.join(', ') : 'Endpoint'
  return model.endpoints.length > 1 ? `${label} ×${model.endpoints.length}` : label
}

export function providerModelConfig(config: ConfigData | null, modelName: string) {
  return config?.providers?.models?.find((model) => model.name === modelName)
}

/** The shared deployment a model alias names instead of its own endpoint and key. */
export function modelDeploymentName(config: ConfigData | null, modelName: string): string {
  const entry = providerModelConfig(config, modelName) as { deployment?: unknown } | undefined
  return typeof entry?.deployment === 'string' ? entry.deployment.trim() : ''
}

export interface NamedDeployment {
  name: string
  deployment?: CanonicalModelDeployment
}

/**
 * The shared deployments a model is tied to: the one it names, and the
 * deployments whose `model` resolves to it (they bill at its rates).
 */
export function modelDeployments(config: ConfigData | null, modelName: string) {
  const declared = config?.global?.model_catalog?.deployments ?? {}
  const name = modelDeploymentName(config, modelName)
  return {
    referenced: name ? ({ name, deployment: declared[name] } as NamedDeployment) : null,
    naming: Object.entries(declared)
      .filter(
        ([deploymentName, deployment]) =>
          deployment?.model === modelName && deploymentName !== name,
      )
      .map(([deploymentName, deployment]) => ({ name: deploymentName, deployment })),
  }
}

export interface ModelDecisionReference {
  decision: DecisionConfig
  scope: string
  scopeLabel: string
}

/** Every decision, in every recipe, that can send traffic to this model. */
export function modelDecisionReferences(
  config: ConfigData | null,
  modelName: string,
  targets: (decision: DecisionConfig) => string[],
): ModelDecisionReference[] {
  return listRecipeProfiles(config).flatMap((recipe) =>
    (recipe.routing.decisions ?? [])
      .filter((decision) => targets(decision).includes(modelName))
      .map((decision) => ({
        decision,
        scope: recipe.name,
        scopeLabel: recipe.name === DEFAULT_RECIPE_NAME ? 'Default routing' : recipe.name,
      })),
  )
}

export function modelHasBackend(model: NormalizedModel, config: ConfigData | null): boolean {
  return (
    model.endpoints.some((endpoint) => endpoint.endpoint.trim()) ||
    Boolean(modelDeploymentName(config, model.name))
  )
}
