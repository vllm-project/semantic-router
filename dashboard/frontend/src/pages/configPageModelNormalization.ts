import type { Endpoint } from '../components/EndpointsEditor'
import bundledCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog, BuiltInModelMetadata } from '../types/modelCatalog'
import {
  normalizeEndpoint,
  normalizeProviderModelEndpoints,
  type ConfigData,
  type ModelConfigEntry,
  type NormalizedModel,
  type ProviderModelConfig,
  type RoutingModelCard,
  type VLLMEndpoint,
} from './configPageSupport'

const catalogRuntimeModality = (model?: BuiltInModelMetadata): string | undefined => {
  if (!model) return undefined
  if (model.capabilities.includes('image_generation')) return 'diffusion'
  if (model.modalities.input.some((input) => input !== 'text')) return 'omni'
  return 'ar'
}

const normalizedProviderModel = (
  model: ProviderModelConfig,
  cardByName: Map<string, RoutingModelCard>,
  builtInByName: Map<string, BuiltInModelMetadata>,
): NormalizedModel => {
  const cardID = model.catalog || model.name
  const override = cardByName.get(cardID)
  const builtIn = model.catalog ? builtInByName.get(model.catalog) : undefined
  return {
    name: model.name,
    catalog: model.catalog,
    reasoning: model.reasoning,
    reasoning_family: model.reasoning?.family || builtIn?.reasoning_family,
    provider_model_id: model.provider_model_id,
    api_format: model.api_format,
    external_model_ids: model.external_model_ids,
    backend_refs: model.backend_refs,
    endpoints: normalizeProviderModelEndpoints(model),
    param_size: override?.param_size ?? builtIn?.parameter_size,
    context_window_size: override?.context_window_size ?? builtIn?.limits?.context_window_size,
    description: override?.description ?? builtIn?.description,
    capabilities: override?.capabilities ?? builtIn?.capabilities,
    loras: override?.loras,
    tags: override?.tags ?? builtIn?.tags,
    evaluations: override?.evaluations,
    modality: override?.modality ?? catalogRuntimeModality(builtIn),
    card_override: override,
    pricing: model.pricing,
    reliability: model.reliability,
  }
}

const normalizedUnboundCard = (card: RoutingModelCard): NormalizedModel => ({
  name: card.name,
  card_override: card,
  endpoints: [],
  param_size: card.param_size,
  context_window_size: card.context_window_size,
  description: card.description,
  capabilities: card.capabilities,
  loras: card.loras,
  tags: card.tags,
  evaluations: card.evaluations,
  modality: card.modality,
})

const canonicalModels = (
  config: ConfigData,
  catalog?: BuiltInModelCatalog | null,
): NormalizedModel[] => {
  const providerModels = config.providers?.models ?? []
  const cards = config.routing?.modelCards ?? []
  const cardByName = new Map(cards.map((card) => [card.name, card]))
  const builtInByName = new Map((catalog?.models ?? []).map((model) => [model.id, model]))
  const models = providerModels.map((model) =>
    normalizedProviderModel(model, cardByName, builtInByName),
  )
  const boundCards = new Set(providerModels.map((model) => model.catalog || model.name))
  for (const card of cards) {
    if (!boundCards.has(card.name)) models.push(normalizedUnboundCard(card))
  }
  return models
}

const legacyEndpoints = (config: ConfigData, model: ModelConfigEntry): Endpoint[] =>
  model.preferred_endpoints
    ?.map((name: string) => {
      const endpoint = config.vllm_endpoints?.find((entry: VLLMEndpoint) => entry.name === name)
      return endpoint
        ? normalizeEndpoint(
            {
              name,
              weight: endpoint.weight || 1,
              endpoint: `${endpoint.address}:${endpoint.port}`,
              protocol: 'http',
            },
            0,
          )
        : null
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null) ?? []

const legacyModels = (config: ConfigData): NormalizedModel[] =>
  (Object.entries(config.model_config ?? {}) as [string, ModelConfigEntry][]).map(
    ([name, model]) => ({
      name,
      reasoning_family: model.reasoning_family,
      endpoints: legacyEndpoints(config, model),
      access_key: undefined,
      pricing: model.pricing,
    }),
  )

export const getNormalizedModels = (
  config: ConfigData | null,
  isPythonCLI: boolean,
  catalog: BuiltInModelCatalog | null = bundledCatalog as unknown as BuiltInModelCatalog,
): NormalizedModel[] => {
  if (!config) return []
  if (isPythonCLI && config.providers?.models) return canonicalModels(config, catalog)
  if (config.model_config) return legacyModels(config)
  return []
}
