// Named imports let the bundler omit benchmark results from ordinary routing
// and setup screens. This is a projection of the shared generated source, not
// a second catalog inventory. Full evidence is loaded by modelCatalogResource.
import {
  schema_version,
  catalogs,
  protocols,
  providers,
  reasoning_families,
  models,
  benchmarks,
  indices,
} from '../../../website/static/model-catalog/catalog.json'
import type { BuiltInModelCatalog } from './types/modelCatalog'

const modelCatalogMetadata = {
  schema_version,
  catalogs,
  protocols,
  providers,
  reasoning_families,
  models,
  benchmarks,
  indices,
  evaluations: [],
  index_results: [],
} as unknown as BuiltInModelCatalog

export default modelCatalogMetadata
