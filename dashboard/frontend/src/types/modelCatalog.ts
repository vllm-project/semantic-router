export type ModelCatalogChannel = 'latest' | 'release'

export interface BuiltInModelCatalogVersion {
  catalog_version: string
  channel: ModelCatalogChannel
  default_model: string
  enabled_models: string[]
}

export interface BuiltInModelRole {
  name: string
  required: boolean
  minimum_candidates: number
  traits: string[]
  recommended_pool: string[]
}

export interface BuiltInModelVerification {
  status: 'verified' | 'unverified'
  authority: string
  asset_sha256: string
}

export interface BuiltInModelMetadata {
  id: string
  display_name: string
  description: string
  kind: string
  family: string
  generation: number
  policy_version: string
  entrypoint: string
  recipe: string
  protocols: string[]
  traits: string[]
  roles: BuiltInModelRole[]
  catalog_version: string
  channel: ModelCatalogChannel
  compatible: boolean
  compatibility_reason: string
  enabled_by_default: boolean
  default: boolean
  verification: BuiltInModelVerification
}

export interface BuiltInModelCatalog {
  catalogs: BuiltInModelCatalogVersion[]
  models: BuiltInModelMetadata[]
}
