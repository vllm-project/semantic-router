export interface FieldSchema {
  key: string
  label: string
  type: 'string' | 'number' | 'boolean' | 'string[]' | 'number[]' | 'select' | 'json'
  options?: string[]
  required?: boolean
  placeholder?: string
  description?: string
}

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

export function getSignalFieldSchema(signalType: string): FieldSchema[] {
  switch (signalType) {
    case 'keyword':
      return [
        {
          key: 'operator',
          label: 'Operator',
          type: 'select',
          options: ['any', 'all', 'OR', 'AND'],
          required: true,
        },
        {
          key: 'keywords',
          label: 'Keywords',
          type: 'string[]',
          required: true,
          placeholder: 'Add keyword...',
        },
        { key: 'method', label: 'Method', type: 'select', options: ['regex', 'bm25', 'ngram'] },
        { key: 'case_sensitive', label: 'Case Sensitive', type: 'boolean' },
        { key: 'fuzzy_match', label: 'Fuzzy Match', type: 'boolean' },
        { key: 'fuzzy_threshold', label: 'Fuzzy Threshold', type: 'number', placeholder: '2' },
        { key: 'bm25_threshold', label: 'BM25 Threshold', type: 'number' },
        { key: 'ngram_threshold', label: 'N-gram Threshold', type: 'number' },
        { key: 'ngram_arity', label: 'N-gram Arity', type: 'number' },
      ]
    case 'embedding':
      return [
        {
          key: 'threshold',
          label: 'Threshold',
          type: 'number',
          required: true,
          placeholder: '0.75',
        },
        {
          key: 'candidates',
          label: 'Candidates',
          type: 'string[]',
          required: true,
          placeholder: 'Add candidate...',
        },
        {
          key: 'aggregation_method',
          label: 'Aggregation',
          type: 'select',
          options: ['mean', 'max', 'any'],
        },
        {
          key: 'query_modality',
          label: 'Query Modality',
          type: 'select',
          options: ['text', 'image', 'audio'],
        },
      ]
    case 'domain':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
        {
          key: 'mmlu_categories',
          label: 'MMLU Categories',
          type: 'string[]',
          placeholder: 'Add category...',
        },
        { key: 'model_scores', label: 'Model Scores', type: 'json' },
      ]
    case 'fact_check':
      return [{ key: 'description', label: 'Description', type: 'string', required: true }]
    case 'user_feedback':
      return [{ key: 'description', label: 'Description', type: 'string', required: true }]
    case 'reask':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'threshold', label: 'Threshold', type: 'number', placeholder: '0.80' },
        { key: 'lookback_turns', label: 'Lookback Turns', type: 'number', placeholder: '1' },
      ]
    case 'preference':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
        { key: 'examples', label: 'Examples', type: 'string[]', placeholder: 'Add example...' },
        { key: 'threshold', label: 'Threshold', type: 'number', placeholder: '0.70' },
      ]
    case 'language':
      return [{ key: 'description', label: 'Description', type: 'string' }]
    case 'context':
      return [
        {
          key: 'min_tokens',
          label: 'Min Tokens',
          type: 'string',
          required: true,
          placeholder: '4K',
        },
        {
          key: 'max_tokens',
          label: 'Max Tokens',
          type: 'string',
          required: true,
          placeholder: '32K',
        },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'structure':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'feature',
          label: 'Feature',
          type: 'json',
          required: true,
          description:
            'Typed structure feature object, e.g. { type: "count", source: { type: "regex", pattern: "[?？]" } }',
        },
        {
          key: 'predicate',
          label: 'Predicate',
          type: 'json',
          description: 'Optional numeric predicate, e.g. { gte: 4 }',
        },
      ]
    case 'complexity':
      return [
        {
          key: 'threshold',
          label: 'Threshold',
          type: 'number',
          required: true,
          placeholder: '0.1',
        },
        {
          key: 'hard',
          label: 'Hard Examples',
          type: 'json',
          description: 'e.g. { candidates: ["..."] }',
        },
        {
          key: 'easy',
          label: 'Easy Examples',
          type: 'json',
          description: 'e.g. { candidates: ["..."] }',
        },
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'composer',
          label: 'Composer',
          type: 'json',
          description: '{ operator: "OR", conditions: [{ type: "domain", name: "..." }] }',
        },
      ]
    case 'modality':
      return [{ key: 'description', label: 'Description', type: 'string' }]
    case 'authz':
      return [
        {
          key: 'subjects',
          label: 'Subjects',
          type: 'json',
          required: true,
          description: '[{ kind: "Group", name: "..." }]',
        },
        { key: 'role', label: 'Role', type: 'string', required: true, placeholder: 'premium_tier' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'jailbreak':
      return [
        {
          key: 'method',
          label: 'Method',
          type: 'select',
          options: ['classifier', 'contrastive'],
          description: 'Detection algorithm',
        },
        {
          key: 'threshold',
          label: 'Threshold',
          type: 'number',
          required: true,
          placeholder: '0.9',
          description: 'Minimum score to trigger (0.0-1.0)',
        },
        {
          key: 'include_history',
          label: 'Include History',
          type: 'boolean',
          description: 'Include conversation history in detection',
        },
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'jailbreak_patterns',
          label: 'Jailbreak Patterns',
          type: 'string[]',
          placeholder: 'Add jailbreak example...',
          description: 'Contrastive mode: example jailbreak prompts',
        },
        {
          key: 'benign_patterns',
          label: 'Benign Patterns',
          type: 'string[]',
          placeholder: 'Add benign example...',
          description: 'Contrastive mode: example benign prompts',
        },
      ]
    case 'pii':
      return [
        {
          key: 'threshold',
          label: 'Threshold',
          type: 'number',
          required: true,
          placeholder: '0.8',
          description: 'Minimum confidence for PII detection (0.0-1.0)',
        },
        {
          key: 'pii_types_allowed',
          label: 'PII Types Allowed',
          type: 'string[]',
          placeholder: 'e.g. EMAIL_ADDRESS',
          description: 'PII types to allow through (others trigger signal)',
        },
        {
          key: 'include_history',
          label: 'Include History',
          type: 'boolean',
          description: 'Include conversation history in detection',
        },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'kb':
      return [
        {
          key: 'kb',
          label: 'Knowledge Base',
          type: 'string',
          required: true,
          placeholder: 'my_kb',
          description: 'Name of the knowledge base to query',
        },
        {
          key: 'target',
          label: 'Target',
          type: 'json',
          description: 'Match target, e.g. { kind: "group", value: "category" }',
        },
        {
          key: 'match',
          label: 'Match Strategy',
          type: 'select',
          options: ['best', 'all'],
          description: 'How to match against the KB',
        },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'conversation':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'feature',
          label: 'Feature',
          type: 'json',
          required: true,
          description: '{ type: "count", source: { type: "message", role: "user" } }',
        },
        { key: 'predicate', label: 'Predicate', type: 'json', description: '{ gte: 2 }' },
      ]
    case 'event':
      return [
        {
          key: 'event_types',
          label: 'Event Types',
          type: 'string[]',
          placeholder: 'payment_failed',
        },
        { key: 'severities', label: 'Severities', type: 'string[]', placeholder: 'critical' },
        {
          key: 'action_codes',
          label: 'Action Codes',
          type: 'string[]',
          placeholder: 'TXN_DECLINE',
        },
        { key: 'temporal', label: 'Temporal', type: 'boolean' },
      ]
    default:
      return [{ key: 'description', label: 'Description', type: 'string' }]
  }
}

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

export function getPluginFieldSchema(pluginType: string): FieldSchema[] {
  switch (pluginType) {
    case 'semantic_cache':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'similarity_threshold',
          label: 'Similarity Threshold',
          type: 'number',
          placeholder: '0.95',
          description: 'Minimum similarity for cache hit (0-1)',
        },
      ]
    case 'memory':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'retrieval_limit',
          label: 'Retrieval Limit',
          type: 'number',
          placeholder: '5',
          description: 'Max memories to retrieve',
        },
        {
          key: 'similarity_threshold',
          label: 'Similarity Threshold',
          type: 'number',
          placeholder: '0.7',
        },
        {
          key: 'auto_store',
          label: 'Auto Store',
          type: 'boolean',
          description: 'Automatically store conversation turns',
        },
      ]
    case 'system_prompt':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'system_prompt',
          label: 'System Prompt',
          type: 'string',
          required: true,
          placeholder: 'You are a helpful assistant...',
        },
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['', 'replace', 'insert'],
          description: 'Replace or insert before existing prompt',
        },
      ]
    case 'hallucination':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'use_nli',
          label: 'Use NLI',
          type: 'boolean',
          description: 'Use Natural Language Inference for detection',
        },
        {
          key: 'hallucination_action',
          label: 'Action',
          type: 'select',
          options: ['', 'header', 'body', 'none'],
          description: 'What to do when hallucination is detected',
        },
      ]
    case 'router_replay':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'max_records', label: 'Max Records', type: 'number', placeholder: '10000' },
        { key: 'capture_request_body', label: 'Capture Request Body', type: 'boolean' },
        { key: 'capture_response_body', label: 'Capture Response Body', type: 'boolean' },
        { key: 'max_body_bytes', label: 'Max Body Bytes', type: 'number', placeholder: '4096' },
      ]
    case 'rag':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'backend',
          label: 'Backend',
          type: 'string',
          required: true,
          placeholder: 'my_vector_store',
          description: 'Backend name for retrieval',
        },
        {
          key: 'top_k',
          label: 'Top K',
          type: 'number',
          placeholder: '5',
          description: 'Number of documents to retrieve',
        },
        {
          key: 'similarity_threshold',
          label: 'Similarity Threshold',
          type: 'number',
          placeholder: '0.7',
        },
        {
          key: 'injection_mode',
          label: 'Injection Mode',
          type: 'select',
          options: ['', 'tool_role', 'system_prompt'],
        },
        {
          key: 'on_failure',
          label: 'On Failure',
          type: 'select',
          options: ['', 'skip', 'block', 'warn'],
        },
      ]
    case 'header_mutation':
      return [
        {
          key: 'add',
          label: 'Add Headers',
          type: 'json',
          description: '[{ "name": "X-Custom", "value": "..." }]',
        },
        {
          key: 'update',
          label: 'Update Headers',
          type: 'json',
          description: '[{ "name": "X-Custom", "value": "..." }]',
        },
        {
          key: 'delete',
          label: 'Delete Headers',
          type: 'string[]',
          placeholder: 'Header name to delete',
        },
      ]
    case 'image_gen':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'backend',
          label: 'Backend',
          type: 'string',
          required: true,
          placeholder: 'my_image_gen_backend',
        },
      ]
    case 'fast_response':
      return [
        {
          key: 'message',
          label: 'Message',
          type: 'string',
          required: true,
          placeholder: 'I cannot help with that request.',
          description: 'The response message returned directly to the client',
        },
      ]
    case 'tools':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['passthrough', 'filtered', 'none'],
          required: true,
        },
        {
          key: 'semantic_selection',
          label: 'Semantic Selection',
          type: 'boolean',
          description: 'Run semantic tool selection from the global tools database',
        },
        {
          key: 'allow_tools',
          label: 'Allow Tools',
          type: 'string[]',
          placeholder: 'Tool name to allow',
        },
        {
          key: 'block_tools',
          label: 'Block Tools',
          type: 'string[]',
          placeholder: 'Tool name to block',
        },
      ]
    case 'tool_selection':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['', 'add', 'filter'],
          description: 'Add tools from a catalog or filter request-provided tools',
        },
        {
          key: 'tools_db_path',
          label: 'Tools DB Path',
          type: 'string',
          placeholder: 'config/tools_db.json',
        },
        { key: 'top_k', label: 'Top K', type: 'number', placeholder: '3' },
        {
          key: 'similarity_threshold',
          label: 'Similarity Threshold',
          type: 'number',
          placeholder: '0.7',
        },
        {
          key: 'strategy',
          label: 'Strategy',
          type: 'select',
          options: ['', 'default', 'weighted', 'hybrid_history'],
        },
        {
          key: 'relevance_threshold',
          label: 'Relevance Threshold',
          type: 'number',
          placeholder: '0.5',
        },
        { key: 'preserve_count', label: 'Preserve Count', type: 'number', placeholder: '0' },
      ]
    case 'request_params':
      return [
        {
          key: 'blocked_params',
          label: 'Blocked Params',
          type: 'string[]',
          placeholder: 'Parameter name to block',
          description: 'Request body parameters to strip before forwarding',
        },
        {
          key: 'max_tokens_limit',
          label: 'Max Tokens Limit',
          type: 'number',
          placeholder: '4096',
          description: 'Maximum allowed value for max_tokens',
        },
        {
          key: 'max_n',
          label: 'Max N',
          type: 'number',
          placeholder: '1',
          description: 'Maximum allowed value for n (number of completions)',
        },
        {
          key: 'strip_unknown',
          label: 'Strip Unknown',
          type: 'boolean',
          description: 'Remove fields not in the OpenAI spec',
        },
      ]
    case 'response_jailbreak':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'threshold',
          label: 'Threshold',
          type: 'number',
          placeholder: '0.8',
          description: 'Minimum classifier score required to flag the response',
        },
        {
          key: 'action',
          label: 'Action',
          type: 'select',
          options: ['', 'block', 'header', 'none'],
          description: 'Block the response, emit warning headers, or do nothing',
        },
      ]
    default:
      return [{ key: 'enabled', label: 'Enabled', type: 'boolean' }]
  }
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

export {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  getAlgorithmFieldSchema,
} from './dslAlgorithmSchemas'
export type { AlgorithmType } from './dslAlgorithmSchemas'
