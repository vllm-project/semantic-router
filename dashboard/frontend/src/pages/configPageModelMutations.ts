import type { ConnectedModelInput } from './ConfigPageConnectModelsDialog'
import { buildConnectedProviderModel } from './configPageConnectModelSupport'
import {
  cloneConfigData,
  ensureProvidersConfig,
  removeModelCardDataIfUnreferenced,
  writeRoutingModelCard,
} from './configPageCanonicalization'
import {
  buildProviderModelPayload,
  normalizeModelBackendRefs,
  normalizeModelPricing,
  normalizeModelStringMap,
} from './configPageModelFormSupport'
import { validateNewModelName } from './configPageModelInventory'
import { modelCardPatch } from './configPageModelsSectionSupport'
import type { ConfigData, NormalizedModel } from './configPageSupport'

type ProviderModel = NonNullable<ConfigData['providers']>['models'][number]

const catalogIDFromForm = (data: Record<string, unknown>, fallback: string): string =>
  typeof data.catalog === 'string' && data.catalog.trim() ? data.catalog.trim() : fallback

const legacyModelConfig = (data: Record<string, unknown>, fallbackModelID: string) => ({
  reasoning_family: typeof data.reasoning_family === 'string' ? data.reasoning_family : undefined,
  pricing: normalizeModelPricing(data.pricing),
  api_format: typeof data.api_format === 'string' ? data.api_format : undefined,
  external_model_ids: normalizeModelStringMap(data.external_model_ids),
  preferred_endpoints: normalizeModelBackendRefs(data.backend_refs)
    .map((backendRef) => backendRef.name || '')
    .filter(Boolean),
  model_id:
    typeof data.provider_model_id === 'string' && data.provider_model_id.trim()
      ? data.provider_model_id.trim()
      : fallbackModelID,
})

export function buildAddedModelConfig(
  config: ConfigData,
  modelName: string,
  data: Record<string, unknown>,
  isPythonCLI: boolean,
): ConfigData {
  const next = cloneConfigData(config)
  if (isPythonCLI) {
    const providers = ensureProvidersConfig(next)
    const catalogID = catalogIDFromForm(data, modelName)
    writeRoutingModelCard(next, catalogID, modelCardPatch(data))
    providers.models.push(buildProviderModelPayload(modelName, data))
  } else {
    next.model_config ??= {}
    next.model_config[modelName] = legacyModelConfig(data, modelName)
  }
  return next
}

export function buildEditedModelConfig(
  config: ConfigData,
  model: NormalizedModel,
  data: Record<string, unknown>,
  isPythonCLI: boolean,
): ConfigData {
  const next = cloneConfigData(config)
  if (isPythonCLI && next.providers?.models) {
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
    writeRoutingModelCard(next, nextCardID, modelCardPatch(data), {
      removeWhenEmpty: oldCardID === nextCardID,
    })
  } else if (next.model_config) {
    next.model_config[model.name] = {
      ...next.model_config[model.name],
      ...legacyModelConfig(data, model.name),
      model_id: typeof data.provider_model_id === 'string' ? data.provider_model_id : model.name,
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
  isPythonCLI: boolean,
): ConfigData {
  const next = cloneConfigData(config)
  const namesToDelete = new Set(modelNames)
  if (isPythonCLI && next.providers?.models) {
    const providers = ensureProvidersConfig(next)
    providers.models = providers.models.filter(
      (providerModel: ProviderModel) => !namesToDelete.has(providerModel.name),
    )
    for (const modelName of namesToDelete) {
      const model = models.find((candidate) => candidate.name === modelName)
      removeModelCardDataIfUnreferenced(next, model?.catalog || modelName)
    }
  } else if (next.model_config) {
    for (const modelName of namesToDelete) delete next.model_config[modelName]
  }
  return next
}
