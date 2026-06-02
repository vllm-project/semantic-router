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
  'keyword', 'embedding', 'domain', 'fact_check', 'user_feedback',
  'reask',
  'preference', 'language', 'context', 'structure', 'complexity', 'modality', 'authz',
  'jailbreak', 'pii', 'kb', 'conversation', 'event',
] as const

export type SignalType = typeof SIGNAL_TYPES[number]

export function getSignalFieldSchema(signalType: string): FieldSchema[] {
  switch (signalType) {
    case 'keyword':
      return [
        { key: 'operator', label: 'Operator', type: 'select', options: ['any', 'all', 'OR', 'AND'], required: true },
        { key: 'keywords', label: 'Keywords', type: 'string[]', required: true, placeholder: 'Add keyword...' },
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
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.75' },
        { key: 'candidates', label: 'Candidates', type: 'string[]', required: true, placeholder: 'Add candidate...' },
        { key: 'aggregation_method', label: 'Aggregation', type: 'select', options: ['mean', 'max', 'any'] },
        { key: 'query_modality', label: 'Query Modality', type: 'select', options: ['text', 'image', 'audio'] },
      ]
    case 'domain':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
        { key: 'mmlu_categories', label: 'MMLU Categories', type: 'string[]', placeholder: 'Add category...' },
        { key: 'model_scores', label: 'Model Scores', type: 'json' },
      ]
    case 'fact_check':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
      ]
    case 'user_feedback':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
      ]
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
      return [
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'context':
      return [
        { key: 'min_tokens', label: 'Min Tokens', type: 'string', required: true, placeholder: '4K' },
        { key: 'max_tokens', label: 'Max Tokens', type: 'string', required: true, placeholder: '32K' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'structure':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'feature', label: 'Feature', type: 'json', required: true, description: 'Typed structure feature object, e.g. { type: "count", source: { type: "regex", pattern: "[?？]" } }' },
        { key: 'predicate', label: 'Predicate', type: 'json', description: 'Optional numeric predicate, e.g. { gte: 4 }' },
      ]
    case 'complexity':
      return [
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.1' },
        { key: 'hard', label: 'Hard Examples', type: 'json', description: 'e.g. { candidates: ["..."] }' },
        { key: 'easy', label: 'Easy Examples', type: 'json', description: 'e.g. { candidates: ["..."] }' },
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'composer', label: 'Composer', type: 'json', description: '{ operator: "OR", conditions: [{ type: "domain", name: "..." }] }' },
      ]
    case 'modality':
      return [
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'authz':
      return [
        { key: 'subjects', label: 'Subjects', type: 'json', required: true, description: '[{ kind: "Group", name: "..." }]' },
        { key: 'role', label: 'Role', type: 'string', required: true, placeholder: 'premium_tier' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'jailbreak':
      return [
        { key: 'method', label: 'Method', type: 'select', options: ['classifier', 'contrastive'], description: 'Detection algorithm' },
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.9', description: 'Minimum score to trigger (0.0-1.0)' },
        { key: 'include_history', label: 'Include History', type: 'boolean', description: 'Include conversation history in detection' },
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'jailbreak_patterns', label: 'Jailbreak Patterns', type: 'string[]', placeholder: 'Add jailbreak example...', description: 'Contrastive mode: example jailbreak prompts' },
        { key: 'benign_patterns', label: 'Benign Patterns', type: 'string[]', placeholder: 'Add benign example...', description: 'Contrastive mode: example benign prompts' },
      ]
    case 'pii':
      return [
        { key: 'threshold', label: 'Threshold', type: 'number', required: true, placeholder: '0.8', description: 'Minimum confidence for PII detection (0.0-1.0)' },
        { key: 'pii_types_allowed', label: 'PII Types Allowed', type: 'string[]', placeholder: 'e.g. EMAIL_ADDRESS', description: 'PII types to allow through (others trigger signal)' },
        { key: 'include_history', label: 'Include History', type: 'boolean', description: 'Include conversation history in detection' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'kb':
      return [
        { key: 'kb', label: 'Knowledge Base', type: 'string', required: true, placeholder: 'my_kb', description: 'Name of the knowledge base to query' },
        { key: 'target', label: 'Target', type: 'json', description: 'Match target, e.g. { kind: "group", value: "category" }' },
        { key: 'match', label: 'Match Strategy', type: 'select', options: ['best', 'all'], description: 'How to match against the KB' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'conversation':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'feature', label: 'Feature', type: 'json', required: true, description: '{ type: "count", source: { type: "message", role: "user" } }' },
        { key: 'predicate', label: 'Predicate', type: 'json', description: '{ gte: 2 }' },
      ]
    case 'event':
      return [
        { key: 'event_types', label: 'Event Types', type: 'string[]', placeholder: 'payment_failed' },
        { key: 'severities', label: 'Severities', type: 'string[]', placeholder: 'critical' },
        { key: 'action_codes', label: 'Action Codes', type: 'string[]', placeholder: 'TXN_DECLINE' },
        { key: 'temporal', label: 'Temporal', type: 'boolean' },
      ]
    default:
      return [
        { key: 'description', label: 'Description', type: 'string' },
      ]
  }
}

export const PLUGIN_TYPES = [
  'semantic_cache', 'memory', 'system_prompt',
  'header_mutation', 'hallucination', 'router_replay', 'rag', 'image_gen',
  'fast_response', 'tools', 'tool_selection', 'request_params', 'response_jailbreak',
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
        { key: 'similarity_threshold', label: 'Similarity Threshold', type: 'number', placeholder: '0.95', description: 'Minimum similarity for cache hit (0-1)' },
      ]
    case 'memory':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'retrieval_limit', label: 'Retrieval Limit', type: 'number', placeholder: '5', description: 'Max memories to retrieve' },
        { key: 'similarity_threshold', label: 'Similarity Threshold', type: 'number', placeholder: '0.7' },
        { key: 'auto_store', label: 'Auto Store', type: 'boolean', description: 'Automatically store conversation turns' },
      ]
    case 'system_prompt':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'system_prompt', label: 'System Prompt', type: 'string', required: true, placeholder: 'You are a helpful assistant...' },
        { key: 'mode', label: 'Mode', type: 'select', options: ['', 'replace', 'insert'], description: 'Replace or insert before existing prompt' },
      ]
    case 'hallucination':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'use_nli', label: 'Use NLI', type: 'boolean', description: 'Use Natural Language Inference for detection' },
        { key: 'hallucination_action', label: 'Action', type: 'select', options: ['', 'header', 'body', 'none'], description: 'What to do when hallucination is detected' },
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
        { key: 'backend', label: 'Backend', type: 'string', required: true, placeholder: 'my_vector_store', description: 'Backend name for retrieval' },
        { key: 'top_k', label: 'Top K', type: 'number', placeholder: '5', description: 'Number of documents to retrieve' },
        { key: 'similarity_threshold', label: 'Similarity Threshold', type: 'number', placeholder: '0.7' },
        { key: 'injection_mode', label: 'Injection Mode', type: 'select', options: ['', 'tool_role', 'system_prompt'] },
        { key: 'on_failure', label: 'On Failure', type: 'select', options: ['', 'skip', 'block', 'warn'] },
      ]
    case 'header_mutation':
      return [
        { key: 'add', label: 'Add Headers', type: 'json', description: '[{ "name": "X-Custom", "value": "..." }]' },
        { key: 'update', label: 'Update Headers', type: 'json', description: '[{ "name": "X-Custom", "value": "..." }]' },
        { key: 'delete', label: 'Delete Headers', type: 'string[]', placeholder: 'Header name to delete' },
      ]
    case 'image_gen':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'backend', label: 'Backend', type: 'string', required: true, placeholder: 'my_image_gen_backend' },
      ]
    case 'fast_response':
      return [
        { key: 'message', label: 'Message', type: 'string', required: true, placeholder: 'I cannot help with that request.', description: 'The response message returned directly to the client' },
      ]
    case 'tools':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'mode', label: 'Mode', type: 'select', options: ['passthrough', 'filtered', 'none'], required: true },
        { key: 'semantic_selection', label: 'Semantic Selection', type: 'boolean', description: 'Run semantic tool selection from the global tools database' },
        { key: 'allow_tools', label: 'Allow Tools', type: 'string[]', placeholder: 'Tool name to allow' },
        { key: 'block_tools', label: 'Block Tools', type: 'string[]', placeholder: 'Tool name to block' },
      ]
    case 'tool_selection':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'mode', label: 'Mode', type: 'select', options: ['', 'add', 'filter'], description: 'Add tools from a catalog or filter request-provided tools' },
        { key: 'tools_db_path', label: 'Tools DB Path', type: 'string', placeholder: 'config/tools_db.json' },
        { key: 'top_k', label: 'Top K', type: 'number', placeholder: '3' },
        { key: 'similarity_threshold', label: 'Similarity Threshold', type: 'number', placeholder: '0.7' },
        { key: 'strategy', label: 'Strategy', type: 'select', options: ['', 'default', 'weighted', 'hybrid_history'] },
        { key: 'relevance_threshold', label: 'Relevance Threshold', type: 'number', placeholder: '0.5' },
        { key: 'preserve_count', label: 'Preserve Count', type: 'number', placeholder: '0' },
      ]
    case 'request_params':
      return [
        { key: 'blocked_params', label: 'Blocked Params', type: 'string[]', placeholder: 'Parameter name to block', description: 'Request body parameters to strip before forwarding' },
        { key: 'max_tokens_limit', label: 'Max Tokens Limit', type: 'number', placeholder: '4096', description: 'Maximum allowed value for max_tokens' },
        { key: 'max_n', label: 'Max N', type: 'number', placeholder: '1', description: 'Maximum allowed value for n (number of completions)' },
        { key: 'strip_unknown', label: 'Strip Unknown', type: 'boolean', description: 'Remove fields not in the OpenAI spec' },
      ]
    case 'response_jailbreak':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'threshold', label: 'Threshold', type: 'number', placeholder: '0.8', description: 'Minimum classifier score required to flag the response' },
        { key: 'action', label: 'Action', type: 'select', options: ['', 'block', 'header', 'none'], description: 'Block the response, emit warning headers, or do nothing' },
      ]
    default:
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
      ]
  }
}

export const BACKEND_TYPES = [
  'vllm_endpoint', 'provider_profile', 'embedding_model', 'semantic_cache',
  'memory', 'response_api', 'vector_store', 'image_gen_backend',
] as const

export const ALGORITHM_TYPES = [
  'confidence', 'ratings', 'remom', 'static', 'elo', 'router_dc', 'automix',
  'hybrid', 'rl_driven', 'gmtrouter', 'latency_aware', 'knn', 'kmeans', 'svm',
  'mlp', 'multi_factor', 'session_aware',
] as const

export type AlgorithmType = typeof ALGORITHM_TYPES[number]

export const ALGORITHM_DESCRIPTIONS: Record<string, string> = {
  confidence: 'Try smaller models first, escalate to larger models if confidence is low',
  ratings: 'Execute all models concurrently and return multiple choices for comparison',
  remom: 'Multi-round parallel reasoning with intelligent synthesis (ReMoM)',
  static: 'Use static scores from configuration (no extra fields)',
  elo: 'Elo rating system with Bradley-Terry model for model selection',
  router_dc: 'Dual-contrastive learning for query-model matching',
  automix: 'POMDP-based cost-quality optimization (arXiv:2310.12963)',
  hybrid: 'Combine multiple selection methods with configurable weights',
  rl_driven: 'Reinforcement learning with Thompson Sampling (arXiv:2506.09033)',
  gmtrouter: 'Heterogeneous graph learning for personalized routing',
  latency_aware: 'TPOT/TTFT percentile thresholds for latency-aware model selection',
  knn: 'K-Nearest Neighbors for query-based model selection (no extra fields)',
  kmeans: 'KMeans clustering for model selection (no extra fields)',
  svm: 'Support Vector Machine for model classification (no extra fields)',
  mlp: 'Neural model-selection classifier using shared ML settings (no extra fields)',
  multi_factor: 'Combine quality, latency, cost, and load into one SLO-aware score',
  session_aware: 'Agentic stay-vs-switch policy for multi-turn sessions',
}

export function getAlgorithmFieldSchema(algoType: string): FieldSchema[] {
  switch (algoType) {
    case 'confidence':
      return [
        { key: 'confidence_method', label: 'Confidence Method', type: 'select', options: ['avg_logprob', 'margin', 'hybrid', 'self_verify'], description: 'How to evaluate model confidence' },
        { key: 'threshold', label: 'Threshold', type: 'number', placeholder: '-1.0', description: 'Confidence threshold for escalation' },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
        { key: 'escalation_order', label: 'Escalation Order', type: 'select', options: ['', 'size', 'cost', 'automix'], description: 'How models are ordered for cascade' },
        { key: 'cost_quality_tradeoff', label: 'Cost/Quality Tradeoff', type: 'number', placeholder: '0.3', description: '0.0=quality, 1.0=cost (for automix order)' },
        { key: 'token_filter', label: 'Token Filter', type: 'string' },
        { key: 'verifier_server_url', label: 'Verifier Server URL', type: 'string', placeholder: 'http://automix-verifier:8080' },
        { key: 'verifier_timeout_seconds', label: 'Verifier Timeout', type: 'number', placeholder: '60' },
      ]
    case 'ratings':
      return [
        { key: 'max_concurrent', label: 'Max Concurrent', type: 'number', placeholder: '0 (no limit)', description: 'Limit concurrent model calls' },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
      ]
    case 'remom':
      return [
        { key: 'breadth_schedule', label: 'Breadth Schedule', type: 'number[]', required: true, placeholder: 'e.g. 32', description: 'Parallel calls per round, e.g. [4], [16], [32, 4]' },
        { key: 'model_distribution', label: 'Model Distribution', type: 'select', options: ['', 'weighted', 'equal', 'first_only'] },
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '1.0' },
        { key: 'include_reasoning', label: 'Include Reasoning', type: 'boolean', description: 'Include reasoning content in synthesis' },
        { key: 'compaction_strategy', label: 'Compaction Strategy', type: 'select', options: ['', 'full', 'last_n_tokens'] },
        { key: 'compaction_tokens', label: 'Compaction Tokens', type: 'number', placeholder: '1000', description: 'Tokens to keep (last_n_tokens strategy)' },
        { key: 'max_concurrent', label: 'Max Concurrent', type: 'number', placeholder: '0 (no limit)' },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
        { key: 'include_intermediate_responses', label: 'Include Intermediate', type: 'boolean', description: 'Save intermediate responses for dashboard' },
        { key: 'shuffle_seed', label: 'Shuffle Seed', type: 'number' },
        { key: 'max_responses_per_round', label: 'Max Responses/Round', type: 'number' },
      ]
    case 'elo':
      return [
        { key: 'initial_rating', label: 'Initial Rating', type: 'number', placeholder: '1500' },
        { key: 'k_factor', label: 'K Factor', type: 'number', placeholder: '32', description: 'Rating volatility' },
        { key: 'category_weighted', label: 'Category Weighted', type: 'boolean', description: 'Per-category Elo ratings' },
        { key: 'decay_factor', label: 'Decay Factor', type: 'number', placeholder: '0 (no decay)', description: 'Time decay 0-1' },
        { key: 'min_comparisons', label: 'Min Comparisons', type: 'number', placeholder: '5' },
        { key: 'cost_scaling_factor', label: 'Cost Scaling', type: 'number', placeholder: '0', description: '0 = ignore cost' },
        { key: 'storage_path', label: 'Storage Path', type: 'string', placeholder: '/tmp/elo' },
        { key: 'auto_save_interval', label: 'Auto-Save Interval', type: 'string', placeholder: '30s' },
      ]
    case 'router_dc':
      return [
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '0.07', description: 'Softmax scaling' },
        { key: 'dimension_size', label: 'Dimension Size', type: 'number', placeholder: '768' },
        { key: 'min_similarity', label: 'Min Similarity', type: 'number', placeholder: '0.3' },
        { key: 'use_query_contrastive', label: 'Query Contrastive', type: 'boolean' },
        { key: 'use_model_contrastive', label: 'Model Contrastive', type: 'boolean' },
        { key: 'require_descriptions', label: 'Require Descriptions', type: 'boolean' },
        { key: 'use_capabilities', label: 'Use Capabilities', type: 'boolean' },
      ]
    case 'automix':
      return [
        { key: 'verification_threshold', label: 'Verification Threshold', type: 'number', placeholder: '0.7' },
        { key: 'max_escalations', label: 'Max Escalations', type: 'number', placeholder: '2' },
        { key: 'cost_aware_routing', label: 'Cost-Aware Routing', type: 'boolean' },
        { key: 'cost_quality_tradeoff', label: 'Cost/Quality Tradeoff', type: 'number', placeholder: '0.3' },
        { key: 'discount_factor', label: 'Discount Factor', type: 'number', placeholder: '0.95', description: 'POMDP value iteration' },
        { key: 'use_logprob_verification', label: 'Logprob Verification', type: 'boolean' },
      ]
    case 'hybrid':
      return [
        { key: 'elo_weight', label: 'Elo Weight', type: 'number', placeholder: '0.3' },
        { key: 'router_dc_weight', label: 'RouterDC Weight', type: 'number', placeholder: '0.3' },
        { key: 'automix_weight', label: 'AutoMix Weight', type: 'number', placeholder: '0.2' },
        { key: 'cost_weight', label: 'Cost Weight', type: 'number', placeholder: '0.2' },
        { key: 'quality_gap_threshold', label: 'Quality Gap Threshold', type: 'number', placeholder: '0.1' },
        { key: 'normalize_scores', label: 'Normalize Scores', type: 'boolean' },
      ]
    case 'rl_driven':
      return [
        { key: 'exploration_rate', label: 'Exploration Rate', type: 'number', placeholder: '0.3', description: '0-1' },
        { key: 'exploration_decay', label: 'Exploration Decay', type: 'number', placeholder: '0.99', description: '0-1' },
        { key: 'min_exploration', label: 'Min Exploration', type: 'number', placeholder: '0.05', description: '0-1' },
        { key: 'use_thompson_sampling', label: 'Thompson Sampling', type: 'boolean' },
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        { key: 'personalization_blend', label: 'Personalization Blend', type: 'number', placeholder: '0.7', description: 'Global vs user-specific (0-1)' },
        { key: 'session_context_weight', label: 'Session Context Weight', type: 'number', placeholder: '0.3' },
        { key: 'implicit_feedback_weight', label: 'Implicit Feedback Weight', type: 'number', placeholder: '0.5' },
        { key: 'cost_awareness', label: 'Cost Awareness', type: 'boolean' },
        { key: 'cost_weight', label: 'Cost Weight', type: 'number', placeholder: '0.2' },
        { key: 'storage_path', label: 'Storage Path', type: 'string', placeholder: 'state/rl-driven.json' },
        { key: 'auto_save_interval', label: 'Auto-Save Interval', type: 'string', placeholder: '30s' },
        { key: 'use_router_r1_rewards', label: 'Router-R1 Rewards', type: 'boolean' },
        { key: 'cost_reward_alpha', label: 'Cost Reward Alpha', type: 'number', placeholder: '0.3' },
        { key: 'format_reward_penalty', label: 'Format Penalty', type: 'number', placeholder: '-1.0' },
        { key: 'enable_llm_routing', label: 'LLM Routing', type: 'boolean' },
        { key: 'router_r1_server_url', label: 'Router-R1 Server URL', type: 'string', placeholder: 'http://router-r1:8080' },
        { key: 'llm_routing_fallback', label: 'LLM Routing Fallback', type: 'string', placeholder: 'thompson' },
        { key: 'enable_multi_round_aggregation', label: 'Multi-Round Aggregation', type: 'boolean' },
        { key: 'max_aggregation_rounds', label: 'Max Aggregation Rounds', type: 'number', placeholder: '3' },
      ]
    case 'gmtrouter':
      return [
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        { key: 'history_sample_size', label: 'History Sample Size', type: 'number', placeholder: '5' },
        { key: 'embedding_dimension', label: 'Embedding Dimension', type: 'number', placeholder: '768' },
        { key: 'num_gnn_layers', label: 'GNN Layers', type: 'number', placeholder: '2' },
        { key: 'attention_heads', label: 'Attention Heads', type: 'number', placeholder: '8' },
        { key: 'min_interactions_for_personalization', label: 'Min Interactions', type: 'number' },
        { key: 'max_interactions_per_user', label: 'Max Interactions/User', type: 'number', placeholder: '100' },
        { key: 'feedback_types', label: 'Feedback Types', type: 'string[]', placeholder: 'rating, ranking' },
        { key: 'model_path', label: 'Model Path', type: 'string', placeholder: '/models/gmt' },
        { key: 'storage_path', label: 'Storage Path', type: 'string', placeholder: 'state/gmtrouter.db' },
      ]
    case 'latency_aware':
      return [
        { key: 'tpot_percentile', label: 'TPOT Percentile', type: 'number', required: true, placeholder: '20', description: 'Time Per Output Token (1-100)' },
        { key: 'ttft_percentile', label: 'TTFT Percentile', type: 'number', required: true, placeholder: '20', description: 'Time To First Token (1-100)' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'multi_factor':
      return [
        { key: 'weights', label: 'Weights', type: 'json', placeholder: '{ "quality": 0.4, "latency": 0.3, "cost": 0.2, "load": 0.1 }', description: 'Per-signal weights for quality, latency, cost, and load' },
        { key: 'slo', label: 'SLO Ceilings', type: 'json', placeholder: '{ "max_tpot_ms": 200, "max_ttft_ms": 800 }', description: 'Hard ceilings that prune unsafe candidates before scoring' },
        { key: 'latency_percentile', label: 'Latency Percentile', type: 'number', placeholder: '95' },
        { key: 'on_no_candidates', label: 'No Candidates Policy', type: 'select', options: ['', 'cheapest', 'first', 'fail'] },
      ]
    case 'session_aware':
      return [
        { key: 'base_method', label: 'Base Method', type: 'select', options: ['', 'static', 'elo', 'router_dc', 'automix', 'hybrid', 'multi_factor', 'latency_aware'] },
        { key: 'idle_timeout_seconds', label: 'Idle Timeout Seconds', type: 'number', placeholder: '600' },
        { key: 'min_turns_before_switch', label: 'Min Turns Before Switch', type: 'number', placeholder: '2' },
        { key: 'switch_margin', label: 'Switch Margin', type: 'number', placeholder: '0.1' },
        { key: 'stay_bias', label: 'Stay Bias', type: 'number', placeholder: '0.2' },
        { key: 'tool_loop_hard_lock', label: 'Tool Loop Hard Lock', type: 'boolean' },
        { key: 'context_portability_hard_lock', label: 'Context Portability Hard Lock', type: 'boolean' },
        { key: 'decision_drift_reset', label: 'Decision Drift Reset', type: 'boolean' },
        { key: 'tool_loop_stay_bias', label: 'Tool Loop Stay Bias', type: 'number' },
        { key: 'prefix_cache_weight', label: 'Prefix Cache Weight', type: 'number' },
        { key: 'handoff_penalty_weight', label: 'Handoff Penalty Weight', type: 'number' },
        { key: 'default_handoff_penalty', label: 'Default Handoff Penalty', type: 'number' },
        { key: 'quality_gap_multiplier', label: 'Quality Gap Multiplier', type: 'number' },
        { key: 'max_cache_cost_multiplier', label: 'Max Cache Cost Multiplier', type: 'number', placeholder: '1.0' },
        { key: 'switch_history_weight', label: 'Switch History Weight', type: 'number' },
        { key: 'remaining_turn_prior_weight', label: 'Remaining Turn Prior Weight', type: 'number' },
        { key: 'remaining_turn_prior_horizon', label: 'Remaining Turn Prior Horizon', type: 'number' },
        { key: 'min_remaining_turn_prior_samples', label: 'Min Remaining Turn Samples', type: 'number' },
      ]
    default:
      return []
  }
}
