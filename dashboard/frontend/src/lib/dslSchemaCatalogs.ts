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
  'metadata',
  'classifier',
  'input_modality',
] as const

export type SignalType = (typeof SIGNAL_TYPES)[number]

export const PLUGIN_TYPES = [
  'response_cache',
  'memory',
  'system_prompt',
  'header_mutation',
  'hallucination',
  'router_replay',
  'rag',
  'fast_response',
  'tools',
  'tool_selection',
  'request_params',
  'response_jailbreak',
  'context_compression',
  'shadow_dispatch',
] as const

export const PLUGIN_DESCRIPTIONS: Record<string, string> = {
  response_cache: 'Reuse exact or semantically compatible responses to reduce latency and cost',
  memory: 'Persistent conversation memory with vector retrieval',
  system_prompt: 'Inject or replace system prompts for the model',
  header_mutation: 'Add, update, or remove HTTP headers on requests/responses',
  hallucination: 'Detect hallucinated content using NLI or other methods',
  router_replay: 'Record request/response pairs for replay and debugging',
  rag: 'Retrieval-Augmented Generation — inject retrieved context into prompts',
  fast_response: 'Short-circuit and return a fixed response without calling upstream models',
  tools: 'Route-local tool filtering and semantic tool selection',
  tool_selection: 'Semantic tool add/filter plugin for route-local tool catalogs',
  request_params: 'Mutate request parameters before forwarding to the model',
  response_jailbreak: 'Screen generated responses for jailbreak-like output before returning',
  context_compression: 'Compress large tool outputs before provider dispatch',
  shadow_dispatch: 'Send a bounded, sampled copy of the request to a secondary model without affecting the live response',
}

export const BACKEND_TYPES = [
  'vllm_endpoint',
  'provider_profile',
  'embedding_model',
  'response_cache',
  'memory',
  'response_api',
  'vector_store',
] as const
