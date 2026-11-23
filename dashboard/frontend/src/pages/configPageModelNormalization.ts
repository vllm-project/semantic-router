import {
  canonicalAPIFormat,
  effectiveModelAPIFormat,
  protocolForModelAPIFormat,
} from './configPageModelCatalogSupport'
import bundledCatalog from '../modelCatalogMetadata'
import type { BuiltInModelCatalog, BuiltInModelMetadata } from '../types/modelCatalog'
import {
  normalizeProviderModelEndpoints,
  type ConfigData,
  type NormalizedModel,
  type ProviderModelConfig,
  type RoutingModelCard,
} from './configPageSupport'

const intersectValues = <Value extends string>(
  current: Value[] | undefined,
  next: Value[] | undefined,
): Value[] | undefined => {
  if (!next) return current
  if (!current) return [...next]
  const allowed = new Set(next)
  return current.filter((value) => allowed.has(value))
}

const catalogReasoningConstraints = (
  model: ProviderModelConfig,
  catalog?: BuiltInModelCatalog | null,
): Pick<NormalizedModel, 'reasoning_modes' | 'reasoning_efforts'> => {
  if (!model.catalog || !catalog) return {}
  let reasoningModes: NormalizedModel['reasoning_modes']
  let reasoningEfforts: string[] | undefined
  for (const backend of model.backend_refs ?? []) {
    const provider = catalog.providers.find((candidate) => candidate.id === backend.provider)
    const binding = provider?.models?.find((candidate) => candidate.catalog === model.catalog)
    const protocol = protocolForModelAPIFormat(effectiveModelAPIFormat(model, catalog).format)
    const bindingEfforts = protocol
      ? (binding?.reasoning_efforts_by_protocol?.[protocol] ?? binding?.reasoning_efforts)
      : binding?.reasoning_efforts
    reasoningModes = intersectValues(reasoningModes, binding?.reasoning_modes)
    reasoningEfforts = intersectValues(reasoningEfforts, bindingEfforts)
  }
  return { reasoning_modes: reasoningModes, reasoning_efforts: reasoningEfforts }
}

const normalizedProviderModel = (
  model: ProviderModelConfig,
  cardByName: Map<string, RoutingModelCard>,
  builtInByName: Map<string, BuiltInModelMetadata>,
  catalog?: BuiltInModelCatalog | null,
): NormalizedModel => {
  const cardID = model.catalog || model.name
  const override = cardByName.get(cardID)
  const builtIn = model.catalog ? builtInByName.get(model.catalog) : undefined
  const reasoningConstraints = catalogReasoningConstraints(model, catalog)
  return {
    name: model.name,
    catalog: model.catalog,
    reasoning: model.reasoning,
    reasoning_family: model.reasoning?.family || builtIn?.reasoning_family,
    ...reasoningConstraints,
    provider_model_id: model.provider_model_id,
    api_format: effectiveModelAPIFormat(model, catalog).format,
    api_format_override: canonicalAPIFormat(model.api_format),
    external_model_ids: model.external_model_ids,
    backend_refs: model.backend_refs,
    endpoints: normalizeProviderModelEndpoints(model),
    param_size: override?.param_size ?? builtIn?.parameter_size,
    context_window_size: override?.context_window_size ?? builtIn?.limits?.context_window_size,
    description: override?.description ?? builtIn?.description,
    capabilities: override?.capabilities ?? builtIn?.capabilities,
    loras: override?.loras,
    tags: override?.tags ?? builtIn?.tags,
    modalities: override?.modalities ?? builtIn?.modalities,
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
  modalities: card.modalities,
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
    normalizedProviderModel(model, cardByName, builtInByName, catalog),
  )
  const boundCards = new Set(providerModels.map((model) => model.catalog || model.name))
  for (const card of cards) {
    if (!boundCards.has(card.name)) models.push(normalizedUnboundCard(card))
  }
  return models
}

export const getNormalizedModels = (
  config: ConfigData | null,
  catalog: BuiltInModelCatalog | null = bundledCatalog as unknown as BuiltInModelCatalog,
): NormalizedModel[] => (config?.providers?.models ? canonicalModels(config, catalog) : [])
