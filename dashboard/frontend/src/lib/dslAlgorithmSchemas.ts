import type { FieldSchema } from './dslSchemas'

export const ALGORITHM_TYPES = [
  'confidence',
  'ratings',
  'remom',
  'fusion',
  'static',
  'router_dc',
  'automix',
  'hybrid',
  'latency_aware',
  'knn',
  'kmeans',
  'svm',
  'mlp',
  'multi_factor',
] as const

export type AlgorithmType = (typeof ALGORITHM_TYPES)[number]

export const ALGORITHM_DESCRIPTIONS: Record<string, string> = {
  confidence: 'Try smaller models first, escalate to larger models if confidence is low',
  ratings: 'Execute all models concurrently and return multiple choices for comparison',
  remom: 'Multi-round parallel reasoning with intelligent synthesis (ReMoM)',
  fusion: 'Parallel panel deliberation with judge analysis and final synthesis',
  static: 'Use static scores from configuration (no extra fields)',
  router_dc: 'Dual-contrastive learning for query-model matching',
  automix: 'POMDP-based cost-quality optimization (arXiv:2310.12963)',
  hybrid: 'Combine multiple selection methods with configurable weights',
  latency_aware: 'TPOT/TTFT percentile thresholds for latency-aware model selection',
  knn: 'K-Nearest Neighbors for query-based model selection (no extra fields)',
  kmeans: 'KMeans clustering for model selection (no extra fields)',
  svm: 'Support Vector Machine for model classification (no extra fields)',
  mlp: 'Neural model-selection classifier using shared ML settings (no extra fields)',
  multi_factor: 'Combine quality, latency, cost, and load into one SLO-aware score',
}

export function getAlgorithmFieldSchema(algoType: string): FieldSchema[] {
  switch (algoType) {
    case 'confidence':
      return [
        {
          key: 'confidence_method',
          label: 'Confidence Method',
          type: 'select',
          options: ['avg_logprob', 'margin', 'hybrid', 'self_verify'],
          description: 'How to evaluate model confidence',
        },
        {
          key: 'threshold',
          label: 'Threshold',
          type: 'number',
          placeholder: '-1.0',
          description: 'Confidence threshold for escalation',
        },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
        {
          key: 'escalation_order',
          label: 'Escalation Order',
          type: 'select',
          options: ['', 'size', 'cost', 'automix'],
          description: 'How models are ordered for cascade',
        },
        {
          key: 'cost_quality_tradeoff',
          label: 'Cost/Quality Tradeoff',
          type: 'number',
          placeholder: '0.3',
          description: '0.0=quality, 1.0=cost (for automix order)',
        },
        { key: 'token_filter', label: 'Token Filter', type: 'string' },
        {
          key: 'verifier_server_url',
          label: 'Verifier Server URL',
          type: 'string',
          placeholder: 'http://automix-verifier:8080',
        },
        {
          key: 'verifier_timeout_seconds',
          label: 'Verifier Timeout',
          type: 'number',
          placeholder: '60',
        },
      ]
    case 'ratings':
      return [
        {
          key: 'max_concurrent',
          label: 'Max Concurrent',
          type: 'number',
          placeholder: '0 (no limit)',
          description: 'Limit concurrent model calls',
        },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
      ]
    case 'remom':
      return [
        {
          key: 'breadth_schedule',
          label: 'Breadth Schedule',
          type: 'number[]',
          required: true,
          placeholder: 'e.g. 32',
          description: 'Parallel calls per round, e.g. [4], [16], [32, 4]',
        },
        {
          key: 'model_distribution',
          label: 'Model Distribution',
          type: 'select',
          options: ['', 'weighted', 'equal', 'first_only'],
        },
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '1.0' },
        {
          key: 'include_reasoning',
          label: 'Include Reasoning',
          type: 'boolean',
          description: 'Include reasoning content in synthesis',
        },
        {
          key: 'compaction_strategy',
          label: 'Compaction Strategy',
          type: 'select',
          options: ['', 'full', 'last_n_tokens'],
        },
        {
          key: 'compaction_tokens',
          label: 'Compaction Tokens',
          type: 'number',
          placeholder: '1000',
          description: 'Tokens to keep (last_n_tokens strategy)',
        },
        {
          key: 'max_concurrent',
          label: 'Max Concurrent',
          type: 'number',
          placeholder: '0 (no limit)',
        },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
        {
          key: 'include_intermediate_responses',
          label: 'Include Intermediate',
          type: 'boolean',
          description: 'Save intermediate responses for dashboard',
        },
        { key: 'shuffle_seed', label: 'Shuffle Seed', type: 'number' },
        { key: 'max_responses_per_round', label: 'Max Responses/Round', type: 'number' },
      ]
    case 'fusion':
      return [
        {
          key: 'model',
          label: 'Judge Model',
          type: 'string',
          placeholder: 'qwen3-32b',
          description: 'Judge/calling model for analysis and final synthesis',
        },
        {
          key: 'analysis_models',
          label: 'Analysis Models',
          type: 'string[]',
          placeholder: 'Add panel model...',
          description: 'Override route modelRefs with a dedicated panel',
        },
        {
          key: 'max_concurrent',
          label: 'Max Concurrent',
          type: 'number',
          placeholder: '0 (panel size)',
        },
        {
          key: 'max_completion_tokens',
          label: 'Max Completion Tokens',
          type: 'number',
          placeholder: '512',
        },
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '0.2' },
        {
          key: 'include_analysis',
          label: 'Include Analysis',
          type: 'boolean',
          description: 'Return structured judge analysis in the Fusion trace',
        },
        {
          key: 'include_intermediate_responses',
          label: 'Include Responses',
          type: 'boolean',
          description: 'Return panel responses in the Fusion trace',
        },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
        {
          key: 'judge_prompt_version',
          label: 'Prompt Version',
          type: 'string',
          placeholder: 'fusion-v1',
        },
      ]
    case 'router_dc':
      return [
        {
          key: 'temperature',
          label: 'Temperature',
          type: 'number',
          placeholder: '0.07',
          description: 'Softmax scaling',
        },
        { key: 'dimension_size', label: 'Dimension Size', type: 'number', placeholder: '768' },
        { key: 'min_similarity', label: 'Min Similarity', type: 'number', placeholder: '0.3' },
        { key: 'use_query_contrastive', label: 'Query Contrastive', type: 'boolean' },
        { key: 'use_model_contrastive', label: 'Model Contrastive', type: 'boolean' },
        { key: 'require_descriptions', label: 'Require Descriptions', type: 'boolean' },
        { key: 'use_capabilities', label: 'Use Capabilities', type: 'boolean' },
      ]
    case 'automix':
      return [
        {
          key: 'verification_threshold',
          label: 'Verification Threshold',
          type: 'number',
          placeholder: '0.7',
        },
        { key: 'max_escalations', label: 'Max Escalations', type: 'number', placeholder: '2' },
        { key: 'cost_aware_routing', label: 'Cost-Aware Routing', type: 'boolean' },
        {
          key: 'cost_quality_tradeoff',
          label: 'Cost/Quality Tradeoff',
          type: 'number',
          placeholder: '0.3',
        },
        {
          key: 'discount_factor',
          label: 'Discount Factor',
          type: 'number',
          placeholder: '0.95',
          description: 'POMDP value iteration',
        },
        { key: 'use_logprob_verification', label: 'Logprob Verification', type: 'boolean' },
      ]
    case 'hybrid':
      return [
        { key: 'experience_weight', label: 'Experience Weight', type: 'number', placeholder: '0.3' },
        { key: 'router_dc_weight', label: 'RouterDC Weight', type: 'number', placeholder: '0.3' },
        { key: 'automix_weight', label: 'AutoMix Weight', type: 'number', placeholder: '0.2' },
        { key: 'cost_weight', label: 'Cost Weight', type: 'number', placeholder: '0.2' },
        {
          key: 'quality_gap_threshold',
          label: 'Quality Gap Threshold',
          type: 'number',
          placeholder: '0.1',
        },
        { key: 'normalize_scores', label: 'Normalize Scores', type: 'boolean' },
      ]
    case 'latency_aware':
      return [
        {
          key: 'tpot_percentile',
          label: 'TPOT Percentile',
          type: 'number',
          required: true,
          placeholder: '20',
          description: 'Time Per Output Token (1-100)',
        },
        {
          key: 'ttft_percentile',
          label: 'TTFT Percentile',
          type: 'number',
          required: true,
          placeholder: '20',
          description: 'Time To First Token (1-100)',
        },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'multi_factor':
      return [
        {
          key: 'weights',
          label: 'Weights',
          type: 'json',
          placeholder: '{ "quality": 0.4, "latency": 0.3, "cost": 0.2, "load": 0.1 }',
          description: 'Per-signal weights for quality, latency, cost, and load',
        },
        {
          key: 'slo',
          label: 'SLO Ceilings',
          type: 'json',
          placeholder: '{ "max_tpot_ms": 200, "max_ttft_ms": 800 }',
          description: 'Hard ceilings that prune unsafe candidates before scoring',
        },
        {
          key: 'latency_percentile',
          label: 'Latency Percentile',
          type: 'number',
          placeholder: '95',
        },
        {
          key: 'on_no_candidates',
          label: 'No Candidates Policy',
          type: 'select',
          options: ['', 'cheapest', 'first', 'fail'],
        },
      ]
    default:
      return []
  }
}
