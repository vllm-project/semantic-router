export type RouterLayerKey = 'router' | 'services' | 'stores' | 'integrations' | 'model_catalog'

// Dashboard presentation policy for canonical global sections. The paths are
// owned by the generated contract; these entries only split selected paths
// into focused cards and assign them to visual layers.
export const CURATED_ROUTER_SECTIONS = {
  router_core: { path: ['router'], layer: 'router' },
  learning: { path: ['router', 'learning'], layer: 'router' },
  response_api: { path: ['services', 'response_api'], layer: 'services' },
  router_replay: { path: ['services', 'router_replay'], layer: 'services' },
  authz: { path: ['services', 'authz'], layer: 'services' },
  ratelimit: { path: ['services', 'ratelimit'], layer: 'services' },
  management_api: { path: ['services', 'management_api'], layer: 'services' },
  startup_status: { path: ['services', 'startup_status'], layer: 'services' },
  memory: { path: ['stores', 'memory'], layer: 'stores' },
  response_cache: { path: ['stores', 'response_cache'], layer: 'stores' },
  vector_store: { path: ['stores', 'vector_store'], layer: 'stores' },
  tools: { path: ['integrations', 'tools'], layer: 'integrations' },
  prompt_guard: {
    path: ['model_catalog', 'modules', 'prompt_guard'],
    layer: 'model_catalog',
  },
  classifier: {
    path: ['model_catalog', 'modules', 'classifier'],
    layer: 'model_catalog',
  },
  hallucination_mitigation: {
    path: ['model_catalog', 'modules', 'hallucination_mitigation'],
    layer: 'model_catalog',
  },
  feedback_detector: {
    path: ['model_catalog', 'modules', 'feedback_detector'],
    layer: 'model_catalog',
  },
  complexity: {
    path: ['model_catalog', 'modules', 'complexity'],
    layer: 'model_catalog',
  },
  external_models: { path: ['model_catalog', 'external'], layer: 'model_catalog' },
  knowledge_bases: { path: ['model_catalog', 'kbs'], layer: 'model_catalog' },
  admission: { path: ['model_catalog', 'admission'], layer: 'model_catalog' },
  system_models: { path: ['model_catalog', 'system'], layer: 'model_catalog' },
  embedding_models: { path: ['model_catalog', 'embeddings'], layer: 'model_catalog' },
  prompt_compression: {
    path: ['model_catalog', 'modules', 'prompt_compression'],
    layer: 'model_catalog',
  },
  modality_detector: {
    path: ['model_catalog', 'modules', 'modality_detector'],
    layer: 'model_catalog',
  },
  observability: { path: ['services', 'observability'], layer: 'services' },
  looper: { path: ['integrations', 'looper'], layer: 'integrations' },
  clear_route_cache: { path: ['router', 'clear_route_cache'], layer: 'router' },
  model_selection: { path: ['router', 'model_selection'], layer: 'router' },
  api: { path: ['services', 'api'], layer: 'services' },
} as const satisfies Record<string, { path: readonly string[]; layer: RouterLayerKey }>

export type RouterSystemKey = keyof typeof CURATED_ROUTER_SECTIONS

export function routerSectionSchemaPath(key: RouterSystemKey): string[] {
  return ['global', ...CURATED_ROUTER_SECTIONS[key].path]
}

export function routerStructuredFieldSchemaPath(key: RouterSystemKey, name: string): string[] {
  const section = routerSectionSchemaPath(key)
  if (key === 'model_selection' && ['knn', 'kmeans', 'svm'].includes(name)) {
    return [...section, 'ml', name]
  }
  return [...section, name]
}
