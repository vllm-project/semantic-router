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
