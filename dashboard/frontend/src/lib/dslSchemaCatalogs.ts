export const SIGNAL_TYPES = [
  'keyword',
  'embedding',
  'domain',
  'fact_check',
  'user_feedback',
  'reask',
  'preference',
  'language',
  'context',
  'structure',
  'complexity',
  'modality',
  'authz',
  'jailbreak',
  'pii',
  'kb',
  'conversation',
  'event',
] as const

export type SignalType = (typeof SIGNAL_TYPES)[number]

export const PLUGIN_TYPES = [
  'semantic_cache',
  'memory',
  'system_prompt',
  'header_mutation',
  'hallucination',
  'router_replay',
  'rag',
  'image_gen',
  'fast_response',
  'tools',
  'tool_selection',
  'request_params',
  'response_jailbreak',
] as const

export const PLUGIN_DESCRIPTIONS: Record<string, string> = {
  semantic_cache: 'Cache semantically similar queries to reduce latency and cost',
  memory: 'Persistent conversation memory with vector retrieval',
  system_prompt: 'Inject or replace system prompts for the model',
  header_mutation: 'Add, update, or remove HTTP headers on requests/responses',
  hallucination: 'Detect hallucinated content using NLI or other methods',
  router_replay: 'Record request/response pairs for replay and debugging',
  rag: 'Retrieval-Augmented Generation — inject retrieved context into prompts',
  image_gen: 'Route to image generation backends',
  fast_response: 'Short-circuit and return a fixed response without calling upstream models',
  tools: 'Route-local tool filtering and semantic tool selection',
  tool_selection: 'Semantic tool add/filter plugin for route-local tool catalogs',
  request_params: 'Mutate request parameters before forwarding to the model',
  response_jailbreak: 'Screen generated responses for jailbreak-like output before returning',
}

export const BACKEND_TYPES = [
  'vllm_endpoint',
  'provider_profile',
  'embedding_model',
  'semantic_cache',
  'memory',
  'response_api',
  'vector_store',
  'image_gen_backend',
] as const
