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
  budgets: { path: ['services', 'budgets'], layer: 'services' },
  management_api: { path: ['services', 'management_api'], layer: 'services' },
  startup_status: { path: ['services', 'startup_status'], layer: 'services' },
  masking_identity: { path: ['services', 'masking_identity'], layer: 'services' },
  memory: { path: ['stores', 'memory'], layer: 'stores' },
  response_cache: { path: ['stores', 'response_cache'], layer: 'stores' },
  vector_store: { path: ['stores', 'vector_store'], layer: 'stores' },
  admission: { path: ['model_catalog', 'admission'], layer: 'model_catalog' },
  system_models: { path: ['model_catalog', 'system'], layer: 'model_catalog' },
  embedding_models: { path: ['model_catalog', 'embeddings'], layer: 'model_catalog' },
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
  return [...routerSectionSchemaPath(key), name]
}

export type RouterSettingsLayer = RouterLayerKey | 'plugins'

export interface RouterSettingsRoot {
  /** Path below `global`. */
  path: readonly string[]
  /** `children` renders each child of the path as a field; `field` renders the path itself. */
  mode: 'children' | 'field'
  /** Children another card renders. */
  omit?: readonly string[]
}

export interface RouterSettingsCardPolicy {
  /** Stable card key; `#global-section-<key>` links to the card. */
  key: string
  layer: RouterSettingsLayer
  roots: readonly RouterSettingsRoot[]
  /** The boolean shown as the card's switch instead of a field. */
  master?: readonly string[]
  /** Cards whose form maps onto several paths through a support module. */
  custom?: 'decision-model' | 'embeddings'
}

// How the router settings page cuts the canonical global block into cards, in
// display order. A generated section that no card claims still renders, as a
// card of its own, so new contract sections never go missing.
export const ROUTER_SETTINGS_CARDS: readonly RouterSettingsCardPolicy[] = [
  {
    key: 'router_core',
    layer: 'router',
    roots: [{ path: ['router'], mode: 'children', omit: ['learning', 'fallback'] }],
  },
  {
    key: 'fallback',
    layer: 'router',
    roots: [{ path: ['router', 'fallback'], mode: 'children' }],
    master: ['router', 'fallback', 'enabled'],
  },
  {
    key: 'learning',
    layer: 'router',
    roots: [{ path: ['router', 'learning'], mode: 'children' }],
    master: ['router', 'learning', 'enabled'],
  },
  { key: 'api', layer: 'services', roots: [{ path: ['services', 'api'], mode: 'children' }] },
  {
    key: 'response_api',
    layer: 'services',
    roots: [{ path: ['services', 'response_api'], mode: 'children' }],
    master: ['services', 'response_api', 'enabled'],
  },
  {
    key: 'observability',
    layer: 'services',
    roots: [{ path: ['services', 'observability'], mode: 'children' }],
  },
  { key: 'authz', layer: 'services', roots: [{ path: ['services', 'authz'], mode: 'children' }] },
  {
    key: 'ratelimit',
    layer: 'services',
    roots: [{ path: ['services', 'ratelimit'], mode: 'children' }],
  },
  {
    key: 'budgets',
    layer: 'services',
    roots: [{ path: ['services', 'budgets'], mode: 'children' }],
  },
  {
    key: 'management_api',
    layer: 'services',
    roots: [{ path: ['services', 'management_api'], mode: 'children' }],
  },
  {
    key: 'router_replay',
    layer: 'services',
    roots: [{ path: ['services', 'router_replay'], mode: 'children' }],
  },
  {
    key: 'startup_status',
    layer: 'services',
    roots: [{ path: ['services', 'startup_status'], mode: 'children' }],
  },
  {
    key: 'masking_identity',
    layer: 'services',
    roots: [{ path: ['services', 'masking_identity'], mode: 'children' }],
  },
  {
    key: 'response_cache',
    layer: 'stores',
    roots: [{ path: ['stores', 'response_cache'], mode: 'children' }],
  },
  { key: 'memory', layer: 'stores', roots: [{ path: ['stores', 'memory'], mode: 'children' }] },
  {
    key: 'vector_store',
    layer: 'stores',
    roots: [{ path: ['stores', 'vector_store'], mode: 'children' }],
    master: ['stores', 'vector_store', 'enabled'],
  },
  {
    key: 'tool_sessions',
    layer: 'stores',
    roots: [{ path: ['stores', 'tool_sessions'], mode: 'children' }],
  },
  {
    key: 'guard_verdicts',
    layer: 'stores',
    roots: [{ path: ['stores', 'guard_verdicts'], mode: 'children' }],
  },
  {
    key: 'kv_transfer',
    layer: 'integrations',
    roots: [{ path: ['integrations', 'kv_transfer'], mode: 'children' }],
    master: ['integrations', 'kv_transfer', 'enabled'],
  },
  {
    key: 'looper',
    layer: 'integrations',
    roots: [{ path: ['integrations', 'looper'], mode: 'children' }],
  },
  {
    key: 'system_models',
    layer: 'model_catalog',
    custom: 'decision-model',
    roots: [
      { path: ['model_catalog', 'system'], mode: 'children' },
      { path: ['model_catalog', 'bindings'], mode: 'field' },
    ],
  },
  {
    key: 'embedding_models',
    layer: 'model_catalog',
    custom: 'embeddings',
    roots: [{ path: ['model_catalog', 'embeddings'], mode: 'children' }],
  },
  {
    key: 'admission',
    layer: 'model_catalog',
    roots: [{ path: ['model_catalog', 'admission'], mode: 'field' }],
  },
  {
    key: 'signal_limits',
    layer: 'model_catalog',
    roots: [
      { path: ['model_catalog', 'signal_timeout_ms'], mode: 'field' },
      { path: ['model_catalog', 'on_signal_overload'], mode: 'field' },
    ],
  },
  {
    key: 'deployments',
    layer: 'model_catalog',
    roots: [{ path: ['model_catalog', 'deployments'], mode: 'field' }],
  },
  { key: 'plugins', layer: 'plugins', roots: [{ path: ['plugins'], mode: 'field' }] },
]

/** Generated sections the router settings page leaves to another surface. */
export const ROUTER_SETTINGS_EXCLUDED_PATHS: readonly string[] = [
  // KB management is not a Dashboard surface; the canonical schema and raw YAML keep it.
  'model_catalog.kbs',
]
