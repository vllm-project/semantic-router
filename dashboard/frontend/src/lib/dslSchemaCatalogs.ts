import {
  PLUGIN_TYPES,
  ROUTER_CONFIG_EXTENSION,
  SIGNAL_TYPES,
  type SignalType,
} from '../generated/routerConfigContract'

export { PLUGIN_TYPES, SIGNAL_TYPES }
export type { SignalType }

export const PLUGIN_DESCRIPTIONS: Record<string, string> = Object.fromEntries(
  ROUTER_CONFIG_EXTENSION.plugins.map((surface) => [surface.type, surface.description]),
)

// Backend entities are Dashboard topology concepts rather than Router routing
// DSL discriminators. They remain local until the backend surface has its own
// generated registry.
export const BACKEND_TYPES = [
  'vllm_endpoint',
  'provider_profile',
  'embedding_model',
  'response_cache',
  'memory',
  'response_api',
  'vector_store',
] as const
