import type { RouterLayerKey, RouterSystemKey } from './configPageRouterDefaultsSupport'

export const ROUTER_LAYER_META: Record<RouterLayerKey, { title: string; description: string }> = {
  router: {
    title: 'Router',
    description: 'Core router-engine controls, startup behavior, and model-selection strategy.',
  },
  services: {
    title: 'Services',
    description: 'APIs, replay, observability, and other router-owned service surfaces.',
  },
  stores: {
    title: 'Stores',
    description: 'Shared storage-backed capabilities such as semantic cache and memory.',
  },
  integrations: {
    title: 'Integrations',
    description: 'Auxiliary runtime integrations used by routing and tool selection.',
  },
  model_catalog: {
    title: 'Model Catalog',
    description: 'Router-owned embedding catalogs, external models, and model-backed modules.',
  },
}

export const GLOBAL_SECTION_PATHS: Record<RouterSystemKey, string[]> = {
  router_core: ['router'],
  response_api: ['services', 'response_api'],
  router_replay: ['services', 'router_replay'],
  authz: ['services', 'authz'],
  ratelimit: ['services', 'ratelimit'],
  memory: ['stores', 'memory'],
  semantic_cache: ['stores', 'semantic_cache'],
  vector_store: ['stores', 'vector_store'],
  tools: ['integrations', 'tools'],
  prompt_guard: ['model_catalog', 'modules', 'prompt_guard'],
  classifier: ['model_catalog', 'modules', 'classifier'],
  hallucination_mitigation: ['model_catalog', 'modules', 'hallucination_mitigation'],
  feedback_detector: ['model_catalog', 'modules', 'feedback_detector'],
  external_models: ['model_catalog', 'external'],
  system_models: ['model_catalog', 'system'],
  embedding_models: ['model_catalog', 'embeddings'],
  prompt_compression: ['model_catalog', 'modules', 'prompt_compression'],
  modality_detector: ['model_catalog', 'modules', 'modality_detector'],
  observability: ['services', 'observability'],
  looper: ['integrations', 'looper'],
  clear_route_cache: ['router', 'clear_route_cache'],
  model_selection: ['router', 'model_selection'],
  api: ['services', 'api'],
}

export const ROUTER_SECTION_LAYERS: Record<RouterSystemKey, RouterLayerKey> = {
  router_core: 'router',
  response_api: 'services',
  router_replay: 'services',
  authz: 'services',
  ratelimit: 'services',
  memory: 'stores',
  semantic_cache: 'stores',
  vector_store: 'stores',
  tools: 'integrations',
  prompt_guard: 'model_catalog',
  classifier: 'model_catalog',
  hallucination_mitigation: 'model_catalog',
  feedback_detector: 'model_catalog',
  external_models: 'model_catalog',
  system_models: 'model_catalog',
  embedding_models: 'model_catalog',
  prompt_compression: 'model_catalog',
  modality_detector: 'model_catalog',
  observability: 'services',
  looper: 'integrations',
  clear_route_cache: 'router',
  model_selection: 'router',
  api: 'services',
}
