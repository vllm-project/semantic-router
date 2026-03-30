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
  'preference', 'language', 'context', 'structure', 'complexity', 'modality', 'authz',
  'jailbreak', 'pii', 'kb',
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
    case 'preference':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
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
        { key: 'composer', label: 'Composer', type: 'string' },
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
    default:
      return [
        { key: 'description', label: 'Description', type: 'string' },
      ]
  }
}

export const PLUGIN_TYPES = [
  'semantic_cache', 'memory', 'system_prompt',
  'header_mutation', 'hallucination', 'router_replay', 'rag', 'image_gen',
  'fast_response', 'tools', 'request_params', 'response_jailbreak',
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
  request_params: 'Mutate request parameters before forwarding to the model',
  response_jailbreak: 'Detect jailbreak content in model responses',
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
        { key: 'injection_mode', label: 'Injection Mode', type: 'select', options: ['', 'system', 'user', 'context'] },
        { key: 'on_failure', label: 'On Failure', type: 'select', options: ['', 'skip', 'fail'] },
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
    case 'request_params':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
      ]
    case 'response_jailbreak':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'threshold', label: 'Threshold', type: 'number', placeholder: '0.8', description: 'Minimum confidence for response jailbreak detection' },
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
      ]
    case 'router_dc':
      return [
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '0.07', description: 'Softmax scaling' },
        { key: 'dimension_size', label: 'Dimension Size', type: 'number', placeholder: '768' },
        { key: 'min_similarity', label: 'Min Similarity', type: 'number', placeholder: '0.3' },
        { key: 'use_query_contrastive', label: 'Query Contrastive', type: 'boolean' },
        { key: 'use_model_contrastive', label: 'Model Contrastive', type: 'boolean' },
      ]
    case 'automix':
      return [
        { key: 'verification_threshold', label: 'Verification Threshold', type: 'number', placeholder: '0.7' },
        { key: 'max_escalations', label: 'Max Escalations', type: 'number', placeholder: '2' },
        { key: 'cost_aware_routing', label: 'Cost-Aware Routing', type: 'boolean' },
        { key: 'cost_quality_tradeoff', label: 'Cost/Quality Tradeoff', type: 'number', placeholder: '0.3' },
        { key: 'discount_factor', label: 'Discount Factor', type: 'number', placeholder: '0.95', description: 'POMDP value iteration' },
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
        { key: 'use_thompson_sampling', label: 'Thompson Sampling', type: 'boolean' },
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        { key: 'personalization_blend', label: 'Personalization Blend', type: 'number', placeholder: '0.3', description: 'Global vs user-specific (0-1)' },
        { key: 'cost_awareness', label: 'Cost Awareness', type: 'boolean' },
      ]
    case 'gmtrouter':
      return [
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        { key: 'history_sample_size', label: 'History Sample Size', type: 'number', placeholder: '5' },
        { key: 'min_interactions_for_personalization', label: 'Min Interactions', type: 'number' },
        { key: 'max_interactions_per_user', label: 'Max Interactions/User', type: 'number', placeholder: '100' },
        { key: 'model_path', label: 'Model Path', type: 'string', placeholder: '/models/gmt' },
      ]
    case 'latency_aware':
      return [
        { key: 'tpot_percentile', label: 'TPOT Percentile', type: 'number', required: true, placeholder: '20', description: 'Time Per Output Token (1-100)' },
        { key: 'ttft_percentile', label: 'TTFT Percentile', type: 'number', required: true, placeholder: '20', description: 'Time To First Token (1-100)' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    default:
      return []
  }
}
