import type {
  BuiltInModelCatalog,
  BuiltInModelCatalogVersion,
  BuiltInModelMetadata,
} from '../types/modelCatalog'
import type { EntrypointConfig } from './configPageSupport'

export function modelCatalogVersionKey(catalog: BuiltInModelCatalogVersion): string {
  return `${catalog.catalog_version}:${catalog.channel}`
}

export function modelsForCatalogVersion(
  catalog: BuiltInModelCatalog,
  version: BuiltInModelCatalogVersion,
): BuiltInModelMetadata[] {
  return catalog.models.filter(
    (model) =>
      model.catalog_version === version.catalog_version && model.channel === version.channel,
  )
}

export function catalogSnapshotsForEntrypoint(
  catalog: BuiltInModelCatalog | null,
  entrypoint: EntrypointConfig,
): BuiltInModelMetadata[] {
  if (!catalog) return []
  const publicNames = new Set(entrypoint.model_names)
  return catalog.models
    .filter((model) => publicNames.has(model.entrypoint) || publicNames.has(model.id))
    .sort((left, right) => {
      if (left.channel !== right.channel) return left.channel === 'latest' ? -1 : 1
      return right.catalog_version.localeCompare(left.catalog_version, undefined, { numeric: true })
    })
}

export function preferredCatalogModelForEntrypoint(
  catalog: BuiltInModelCatalog | null,
  entrypoint: EntrypointConfig,
): BuiltInModelMetadata | null {
  return catalogSnapshotsForEntrypoint(catalog, entrypoint)[0] ?? null
}
