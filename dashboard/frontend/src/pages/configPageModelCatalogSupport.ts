import type {
  BuiltInModelCatalog,
  BuiltInModelCatalogVersion,
  BuiltInModelMetadata,
} from '../types/modelCatalog'
import type { ProviderModel } from '../types/config'
import type { EntrypointConfig, ProviderModelConfig } from './configPageSupport'

export function modelCatalogVersionKey(catalog: BuiltInModelCatalogVersion): string {
  return `${catalog.catalog_version}:${catalog.channel}`
}

export function modelsForCatalogVersion(
  catalog: BuiltInModelCatalog,
  version: BuiltInModelCatalogVersion,
): BuiltInModelMetadata[] {
  const active = catalog.catalogs.some(
    (candidate) => modelCatalogVersionKey(candidate) === modelCatalogVersionKey(version),
  )
  return active ? catalog.models : []
}

export function catalogSnapshotsForEntrypoint(
  catalog: BuiltInModelCatalog | null,
  entrypoint: EntrypointConfig,
): BuiltInModelMetadata[] {
  if (!catalog) return []
  const publicNames = new Set(entrypoint.model_names)
  return catalog.models
    .filter(
      (model) =>
        (typeof model.entrypoint === 'string' && publicNames.has(model.entrypoint)) ||
        publicNames.has(model.id),
    )
    .sort((left, right) => left.id.localeCompare(right.id))
}

export function preferredCatalogModelForEntrypoint(
  catalog: BuiltInModelCatalog | null,
  entrypoint: EntrypointConfig,
): BuiltInModelMetadata | null {
  return catalogSnapshotsForEntrypoint(catalog, entrypoint)[0] ?? null
}

// Explicit formats follow the canonical providers.models input contract. The
// effective default follows catalog/compiler_model_provider.go, not a universal
// OpenAI fallback: a model binding can narrow a provider's supported protocols.
export const modelAPIFormats = [
  'openai',
  'responses',
  'anthropic',
] as const satisfies readonly NonNullable<ProviderModel['api_format']>[]

export const protocolForModelAPIFormat = (format?: string): string | undefined => {
  if (format === 'openai') return 'openai/chat-completions@1'
  if (format === 'responses') return 'openai/responses@1'
  if (format === 'anthropic') return 'anthropic/messages@1'
  return undefined
}

export function effectiveModelAPIFormat(
  model: ProviderModelConfig,
  catalog: BuiltInModelCatalog | null | undefined,
): { format?: NonNullable<ProviderModel['api_format']>; error?: string } {
  const explicitProtocol = protocolForModelAPIFormat(model.api_format)
  if (model.api_format && !explicitProtocol) return { error: 'Unsupported API format.' }
  if (!catalog) return { error: 'Model catalog is unavailable.' }
  const card = catalog.models.find((entry) => entry.id === model.catalog)
  if (model.catalog && !card) {
    return { error: 'Choose a known built-in model or leave Catalog Model empty.' }
  }
  let selected: string | undefined
  for (const backend of model.backend_refs ?? []) {
    const providerID = backend.provider || 'vllm'
    const provider = catalog.providers.find((entry) => entry.id === providerID)
    if (!provider) return { error: `Unknown provider: ${providerID}.` }
    const nativeID =
      model.external_model_ids?.[providerID] ||
      model.provider_model_id ||
      (model.catalog ? '' : model.name)
    const protocols = [
      ...new Set(
        (provider.models ?? [])
          .filter(
            (binding) =>
              binding.catalog === (model.catalog || model.name) &&
              (!nativeID ||
                binding.id === nativeID ||
                binding.restrictions?.provider_model_id_kind === 'deployment_name'),
          )
          .flatMap((binding) => binding.protocols),
      ),
    ]
    let protocol = explicitProtocol
    if (!protocol) {
      if (protocols.length === 0) {
        if (!nativeID && card?.kind === 'physical')
          return { error: `Set a provider model ID and API format for ${providerID}.` }
        protocol = selected || provider.default_protocol
      } else if (protocols.length === 1) {
        protocol = protocols[0]
      } else {
        const preferred = selected || provider.default_protocol
        if (!protocols.includes(preferred))
          return {
            error: `Choose an API format for ${providerID}; its model binding is ambiguous.`,
          }
        protocol = preferred
      }
    }
    if (
      !provider.protocols.includes(protocol) ||
      !provider.supported_operations.includes(`${protocol}#create`)
    ) {
      return { error: `${providerID} cannot create requests using this API format.` }
    }
    if (selected && selected !== protocol)
      return { error: 'All backends for one model must use the same API format.' }
    selected = protocol
  }
  const protocol = selected || explicitProtocol
  const format = modelAPIFormats.find(
    (candidate) => protocolForModelAPIFormat(candidate) === protocol,
  )
  return format
    ? { format }
    : {
        error: protocol
          ? 'This protocol is not available as a canonical API format.'
          : 'Add a provider backend to resolve the API format.',
      }
}
