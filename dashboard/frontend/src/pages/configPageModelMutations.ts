import type { ConnectedModelInput } from './ConfigPageConnectModelsDialog'
import { buildConnectedProviderModel } from './configPageConnectModelSupport'
import {
  cloneConfigData,
  ensureProvidersConfig,
  removeModelCardDataIfUnreferenced,
  writeRoutingModelCard,
} from './configPageCanonicalization'
import { buildProviderModelPayload, effectiveModelCardFormData } from './configPageModelFormSupport'
import { validateNewModelName } from './configPageModelInventory'
import { modelCardPatch } from './configPageModelsSectionSupport'
import type { ConfigData, NormalizedModel, RoutingModelCard } from './configPageSupport'

type ProviderModel = NonNullable<ConfigData['providers']>['models'][number]

const catalogIDFromForm = (data: Record<string, unknown>, fallback: string): string =>
  typeof data.catalog === 'string' && data.catalog.trim() ? data.catalog.trim() : fallback

export function buildAddedModelConfig(
  config: ConfigData,
  modelName: string,
  data: Record<string, unknown>,
): ConfigData {
  const next = cloneConfigData(config)
  const providers = ensureProvidersConfig(next)
  const catalogID = catalogIDFromForm(data, modelName)
  writeRoutingModelCard(next, catalogID, modelCardPatch(data))
  providers.models.push(buildProviderModelPayload(modelName, data))
  return next
}

// Edit shows effective metadata, but only an actual user change becomes a new
// override. Preserve fields outside this form (for example max_output_tokens).
function editedModelCardPatch(
  model: NormalizedModel,
  data: Record<string, unknown>,
  existingCard?: RoutingModelCard,
): Omit<RoutingModelCard, 'name'> {
  const patch: Omit<RoutingModelCard, 'name'> = { ...existingCard }
  delete (patch as Partial<RoutingModelCard>).name
  const before = modelCardPatch(effectiveModelCardFormData(model))
  const after = modelCardPatch(data)
  for (const field of Object.keys(after) as Array<keyof typeof after>) {
    if (
      Object.prototype.hasOwnProperty.call(data, field) &&
      JSON.stringify(before[field]) !== JSON.stringify(after[field])
    ) {
      Object.assign(patch, { [field]: after[field] })
    }
  }
  return patch
}

export function buildEditedModelConfig(
  config: ConfigData,
  model: NormalizedModel,
  data: Record<string, unknown>,
): ConfigData {
  const next = cloneConfigData(config)
  if (next.providers?.models) {
    const providers = ensureProvidersConfig(next)
    const oldCardID = model.catalog || model.name
    const nextCatalog = catalogIDFromForm(data, '') || undefined
    const nextCardID = nextCatalog || model.name
    providers.models = providers.models.map((providerModel: ProviderModel) =>
      providerModel.name === model.name
        ? { ...providerModel, ...buildProviderModelPayload(model.name, data, providerModel) }
        : providerModel,
    )
    if (oldCardID !== nextCardID) removeModelCardDataIfUnreferenced(next, oldCardID)
    const existingCard = next.routing?.modelCards?.find((card) => card.name === nextCardID)
    const patch = editedModelCardPatch(model, data, existingCard)
    if (existingCard || Object.values(patch).some((value) => value !== undefined)) {
      writeRoutingModelCard(next, nextCardID, patch, { removeWhenEmpty: true })
    }
  }
  return next
}

export function buildConnectedModelsConfig(
  config: ConfigData,
  models: NormalizedModel[],
  input: ConnectedModelInput,
): ConfigData {
  const next = cloneConfigData(config)
  const providers = ensureProvidersConfig(next)
  for (const modelID of input.modelIds) {
    const modelName = validateNewModelName(input.modelNames[modelID] ?? modelID, models)
    const catalogID = input.catalogModels[modelID]
    providers.models.push(
      buildConnectedProviderModel({
        name: modelName,
        catalog: catalogID,
        providerModelID: modelID,
        catalogProviderModelID: input.providerModelIds[modelID],
        providerID: input.provider.id,
        providerAPIFormat: input.provider.apiFormat,
        baseURL: input.baseUrl,
        apiKey: input.apiKey,
        reasoningFamily: input.reasoningFamily,
        pricing: input.pricing,
        reliability: input.reliability,
      }),
    )
    writeRoutingModelCard(next, catalogID || modelName, input.metadata)
  }
  return next
}

export function buildDeletedModelsConfig(
  config: ConfigData,
  models: NormalizedModel[],
  modelNames: string[],
): ConfigData {
  const next = cloneConfigData(config)
  const namesToDelete = new Set(modelNames)
  if (next.providers?.models) {
    const providers = ensureProvidersConfig(next)
    providers.models = providers.models.filter(
      (providerModel: ProviderModel) => !namesToDelete.has(providerModel.name),
    )
    for (const modelName of namesToDelete) {
      const model = models.find((candidate) => candidate.name === modelName)
      removeModelCardDataIfUnreferenced(next, model?.catalog || modelName)
    }
  }
  return next
}
