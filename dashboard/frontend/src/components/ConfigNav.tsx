export type ConfigSection =
  | 'signals' // config.yaml: recipes[].signals (keywords, domains, safety, systemone, etc.)
  | 'projections' // config.yaml: recipes[].projections (partitions, scores, mappings)
  | 'decisions' // config.yaml: recipes[].decisions (routing rules)
  | 'models' // config.yaml: providers.models
  | 'entrypoints-recipes' // config.yaml: top-level entrypoints + recipes
  | 'global-config' // config.yaml: global runtime overrides (stores, services, plugins, model catalog)
  | 'mcp' // MCP servers configuration
  | 'topology' // Separate page for visualization
