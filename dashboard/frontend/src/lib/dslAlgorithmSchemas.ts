import type { FieldSchema } from './dslSchemas'

export const ALGORITHM_TYPES = [
  'confidence',
  'ratings',
  'remom',
  'fusion',
  'static',
  'elo',
  'router_dc',
  'automix',
  'hybrid',
  'rl_driven',
  'gmtrouter',
  'latency_aware',
  'knn',
  'kmeans',
  'svm',
  'mlp',
  'multi_factor',
  'session_aware',
] as const

export type AlgorithmType = (typeof ALGORITHM_TYPES)[number]

export const ALGORITHM_DESCRIPTIONS: Record<string, string> = {
  confidence: 'Try smaller models first, escalate to larger models if confidence is low',
  ratings: 'Execute all models concurrently and return multiple choices for comparison',
  remom: 'Multi-round parallel reasoning with intelligent synthesis (ReMoM)',
  fusion: 'Parallel panel deliberation with judge analysis and final synthesis',
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
    case 'elo':
      return [
        { key: 'initial_rating', label: 'Initial Rating', type: 'number', placeholder: '1500' },
        {
          key: 'k_factor',
          label: 'K Factor',
          type: 'number',
          placeholder: '32',
          description: 'Rating volatility',
        },
        {
          key: 'category_weighted',
          label: 'Category Weighted',
          type: 'boolean',
          description: 'Per-category Elo ratings',
        },
        {
          key: 'decay_factor',
          label: 'Decay Factor',
          type: 'number',
          placeholder: '0 (no decay)',
          description: 'Time decay 0-1',
        },
        { key: 'min_comparisons', label: 'Min Comparisons', type: 'number', placeholder: '5' },
        {
          key: 'cost_scaling_factor',
          label: 'Cost Scaling',
          type: 'number',
          placeholder: '0',
          description: '0 = ignore cost',
        },
        { key: 'storage_path', label: 'Storage Path', type: 'string', placeholder: '/tmp/elo' },
        {
          key: 'auto_save_interval',
          label: 'Auto-Save Interval',
          type: 'string',
          placeholder: '30s',
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
        { key: 'elo_weight', label: 'Elo Weight', type: 'number', placeholder: '0.3' },
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
    case 'rl_driven':
      return [
        {
          key: 'exploration_rate',
          label: 'Exploration Rate',
          type: 'number',
          placeholder: '0.3',
          description: '0-1',
        },
        {
          key: 'exploration_decay',
          label: 'Exploration Decay',
          type: 'number',
          placeholder: '0.99',
          description: '0-1',
        },
        {
          key: 'min_exploration',
          label: 'Min Exploration',
          type: 'number',
          placeholder: '0.05',
          description: '0-1',
        },
        { key: 'use_thompson_sampling', label: 'Thompson Sampling', type: 'boolean' },
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        {
          key: 'personalization_blend',
          label: 'Personalization Blend',
          type: 'number',
          placeholder: '0.7',
          description: 'Global vs user-specific (0-1)',
        },
        {
          key: 'session_context_weight',
          label: 'Session Context Weight',
          type: 'number',
          placeholder: '0.3',
        },
        {
          key: 'implicit_feedback_weight',
          label: 'Implicit Feedback Weight',
          type: 'number',
          placeholder: '0.5',
        },
        { key: 'cost_awareness', label: 'Cost Awareness', type: 'boolean' },
        { key: 'cost_weight', label: 'Cost Weight', type: 'number', placeholder: '0.2' },
        {
          key: 'storage_path',
          label: 'Storage Path',
          type: 'string',
          placeholder: 'state/rl-driven.json',
        },
        {
          key: 'auto_save_interval',
          label: 'Auto-Save Interval',
          type: 'string',
          placeholder: '30s',
        },
        { key: 'use_router_r1_rewards', label: 'Router-R1 Rewards', type: 'boolean' },
        {
          key: 'cost_reward_alpha',
          label: 'Cost Reward Alpha',
          type: 'number',
          placeholder: '0.3',
        },
        {
          key: 'format_reward_penalty',
          label: 'Format Penalty',
          type: 'number',
          placeholder: '-1.0',
        },
        { key: 'enable_llm_routing', label: 'LLM Routing', type: 'boolean' },
        {
          key: 'router_r1_server_url',
          label: 'Router-R1 Server URL',
          type: 'string',
          placeholder: 'http://router-r1:8080',
        },
        {
          key: 'llm_routing_fallback',
          label: 'LLM Routing Fallback',
          type: 'string',
          placeholder: 'thompson',
        },
        {
          key: 'enable_multi_round_aggregation',
          label: 'Multi-Round Aggregation',
          type: 'boolean',
        },
        {
          key: 'max_aggregation_rounds',
          label: 'Max Aggregation Rounds',
          type: 'number',
          placeholder: '3',
        },
      ]
    case 'gmtrouter':
      return [
        { key: 'enable_personalization', label: 'Personalization', type: 'boolean' },
        {
          key: 'history_sample_size',
          label: 'History Sample Size',
          type: 'number',
          placeholder: '5',
        },
        {
          key: 'embedding_dimension',
          label: 'Embedding Dimension',
          type: 'number',
          placeholder: '768',
        },
        { key: 'num_gnn_layers', label: 'GNN Layers', type: 'number', placeholder: '2' },
        { key: 'attention_heads', label: 'Attention Heads', type: 'number', placeholder: '8' },
        { key: 'min_interactions_for_personalization', label: 'Min Interactions', type: 'number' },
        {
          key: 'max_interactions_per_user',
          label: 'Max Interactions/User',
          type: 'number',
          placeholder: '100',
        },
        {
          key: 'feedback_types',
          label: 'Feedback Types',
          type: 'string[]',
          placeholder: 'rating, ranking',
        },
        { key: 'model_path', label: 'Model Path', type: 'string', placeholder: '/models/gmt' },
        {
          key: 'storage_path',
          label: 'Storage Path',
          type: 'string',
          placeholder: 'state/gmtrouter.db',
        },
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
    case 'session_aware':
      return [
        {
          key: 'base_method',
          label: 'Base Method',
          type: 'select',
          options: [
            '',
            'static',
            'elo',
            'router_dc',
            'automix',
            'hybrid',
            'multi_factor',
            'latency_aware',
          ],
        },
        {
          key: 'idle_timeout_seconds',
          label: 'Idle Timeout Seconds',
          type: 'number',
          placeholder: '600',
        },
        {
          key: 'min_turns_before_switch',
          label: 'Min Turns Before Switch',
          type: 'number',
          placeholder: '2',
        },
        { key: 'switch_margin', label: 'Switch Margin', type: 'number', placeholder: '0.1' },
        { key: 'stay_bias', label: 'Stay Bias', type: 'number', placeholder: '0.2' },
        { key: 'tool_loop_hard_lock', label: 'Tool Loop Hard Lock', type: 'boolean' },
        {
          key: 'context_portability_hard_lock',
          label: 'Context Portability Hard Lock',
          type: 'boolean',
        },
        { key: 'decision_drift_reset', label: 'Decision Drift Reset', type: 'boolean' },
        { key: 'tool_loop_stay_bias', label: 'Tool Loop Stay Bias', type: 'number' },
        { key: 'prefix_cache_weight', label: 'Prefix Cache Weight', type: 'number' },
        { key: 'handoff_penalty_weight', label: 'Handoff Penalty Weight', type: 'number' },
        { key: 'default_handoff_penalty', label: 'Default Handoff Penalty', type: 'number' },
        { key: 'quality_gap_multiplier', label: 'Quality Gap Multiplier', type: 'number' },
        {
          key: 'max_cache_cost_multiplier',
          label: 'Max Cache Cost Multiplier',
          type: 'number',
          placeholder: '1.0',
        },
        { key: 'switch_history_weight', label: 'Switch History Weight', type: 'number' },
        {
          key: 'remaining_turn_prior_weight',
          label: 'Remaining Turn Prior Weight',
          type: 'number',
        },
        {
          key: 'remaining_turn_prior_horizon',
          label: 'Remaining Turn Prior Horizon',
          type: 'number',
        },
        {
          key: 'min_remaining_turn_prior_samples',
          label: 'Min Remaining Turn Samples',
          type: 'number',
        },
      ]
    default:
      return []
  }
}
