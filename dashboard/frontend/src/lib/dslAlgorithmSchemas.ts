import {
  ALGORITHM_TYPES,
  ROUTER_CONFIG_EXTENSION,
  type AlgorithmType,
} from '../generated/routerConfigContract'
import { algorithmFieldsFromRouterSchema, mergeRouterFieldSchemas } from './routerConfigSchema'
import type { FieldSchema } from './dslSchemaTypes'

export { ALGORITHM_TYPES }
export type { AlgorithmType }

export const ALGORITHM_DESCRIPTIONS: Record<string, string> = Object.fromEntries(
  ROUTER_CONFIG_EXTENSION.algorithms.map((surface) => [surface.type, surface.description]),
)

const COMMON_ALGORITHM_FIELDS: FieldSchema[] = [
  {
    key: 'minimum_candidates',
    label: 'Minimum Candidates',
    type: 'number',
    min: 1,
    placeholder: '1',
    description: 'Minimum distinct decision candidates required after model assignment',
  },
]

export function getAlgorithmFieldSchema(algoType: string): FieldSchema[] {
  return mergeRouterFieldSchemas(algorithmFieldsFromRouterSchema(algoType), [
    ...COMMON_ALGORITHM_FIELDS,
    ...getAlgorithmSpecificFieldSchema(algoType),
  ])
}

function getAlgorithmSpecificFieldSchema(algoType: string): FieldSchema[] {
  switch (algoType) {
    case 'confidence':
      return [
        {
          key: 'confidence_method',
          label: 'Confidence Method',
          type: 'select',
          options: ['avg_logprob', 'margin', 'hybrid', 'self_verify', 'automix_entailment'],
          description: 'How to evaluate model confidence',
        },
        {
          key: 'threshold',
          label: 'Threshold',
          type: 'number',
          placeholder: '-1.0',
          description: 'Confidence threshold for escalation',
        },
        {
          key: 'hybrid_weights',
          label: 'Hybrid Weights',
          type: 'object',
          description: 'Weights used only by the hybrid confidence method',
          fields: [
            { key: 'logprob_weight', label: 'Logprob Weight', type: 'number', min: 0, max: 1 },
            { key: 'margin_weight', label: 'Margin Weight', type: 'number', min: 0, max: 1 },
          ],
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
        {
          key: 'max_response_bytes',
          label: 'Max Response Bytes',
          type: 'number',
          placeholder: '33554432',
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
          options: ['', 'weighted', 'equal', 'round_robin', 'first_only'],
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
          key: 'synthesis_template',
          label: 'Synthesis Template',
          type: 'string',
          description: 'Optional prompt template used to synthesize each round',
        },
        {
          key: 'synthesis_model',
          label: 'Synthesis Model',
          type: 'string',
          placeholder: 'model from modelRefs',
          description: 'Optional modelRef to use for the final ReMoM synthesis round',
        },
        {
          key: 'max_concurrent',
          label: 'Max Concurrent',
          type: 'number',
          placeholder: '0 (no limit)',
        },
        {
          key: 'max_completion_tokens',
          label: 'Max Completion Tokens',
          type: 'number',
          min: 1,
          placeholder: '1024',
          description: 'Apply a completion limit to every ReMoM subrequest',
        },
        {
          key: 'round_timeout_seconds',
          label: 'Round Timeout',
          type: 'number',
          placeholder: '0 (wait for all)',
          description: 'Stop waiting for a round after this many seconds',
        },
        {
          key: 'min_successful_responses',
          label: 'Min Successful',
          type: 'number',
          placeholder: '0 (all calls)',
          description: 'Return from a round after this many successful responses',
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
          key: 'analysis_overrides',
          label: 'Analysis Overrides',
          type: 'object[]',
          description: 'Per-panel-model sampling overrides',
          addLabel: 'Add override',
          emptyLabel: 'No model-specific overrides configured.',
          itemLabel: 'Model override',
          itemLabelKey: 'model',
          fields: [
            { key: 'model', label: 'Model', type: 'string', required: true },
            { key: 'temperature', label: 'Temperature', type: 'number', min: 0 },
            {
              key: 'max_completion_tokens',
              label: 'Max Completion Tokens',
              type: 'number',
              min: 1,
            },
          ],
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
        {
          key: 'round_timeout_seconds',
          label: 'Round Timeout',
          type: 'number',
          placeholder: '0 (wait for all)',
          description: 'Stop waiting for panel responses after this many seconds',
        },
        {
          key: 'min_successful_responses',
          label: 'Min Successful',
          type: 'number',
          placeholder: '0 (all calls)',
          description: 'Continue once this many panel responses succeed',
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
          key: 'analysis_template',
          label: 'Analysis Template',
          type: 'string',
          description: 'Optional prompt template for panel analysis',
        },
        {
          key: 'synthesis_template',
          label: 'Synthesis Template',
          type: 'string',
          description: 'Optional prompt template for the judge synthesis',
        },
        {
          key: 'judge_prompt_version',
          label: 'Prompt Version',
          type: 'string',
          placeholder: 'fusion-v1',
        },
        {
          key: 'grounding',
          label: 'Grounding',
          type: 'object',
          description: 'Score panel responses for faithfulness before synthesis',
          fields: [
            { key: 'enabled', label: 'Enabled', type: 'boolean' },
            {
              key: 'reference',
              label: 'Reference',
              type: 'select',
              options: ['', 'hybrid', 'context', 'panel'],
            },
            {
              key: 'policy',
              label: 'Policy',
              type: 'select',
              options: ['', 'weight', 'annotate', 'filter'],
            },
            { key: 'min_score', label: 'Minimum Score', type: 'number', min: 0, max: 1 },
            { key: 'min_keep', label: 'Minimum Responses', type: 'number', min: 0 },
            {
              key: 'nli_contradiction_penalty',
              label: 'NLI Contradiction Penalty',
              type: 'number',
              min: 0,
              max: 1,
            },
            {
              key: 'on_error',
              label: 'On Error',
              type: 'select',
              options: ['', 'skip', 'fail'],
            },
          ],
        },
      ]
    case 'workflows':
      return [
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['', 'static', 'dynamic'],
          description: 'Use dynamic when a planner model should generate the execution flow',
        },
        {
          key: 'template',
          label: 'Template',
          type: 'string',
          placeholder: 'micro_agent',
        },
        {
          key: 'planner',
          label: 'Planner',
          type: 'object',
          description: 'Required for dynamic mode; the model must also be configured globally',
          fields: [
            {
              key: 'model',
              label: 'Planner Model',
              type: 'string',
              required: true,
              placeholder: 'qwen-coordinator',
            },
            {
              key: 'max_completion_tokens',
              label: 'Planner Token Limit',
              type: 'number',
              placeholder: '1024',
            },
          ],
        },
        {
          key: 'roles',
          label: 'Static Roles',
          type: 'object[]',
          description: 'Required for static mode; role models must be in route models',
          addLabel: 'Add role',
          emptyLabel: 'No static roles configured.',
          itemLabel: 'Role',
          itemLabelKey: 'name',
          fields: [
            { key: 'name', label: 'Role Name', type: 'string', required: true },
            {
              key: 'models',
              label: 'Models',
              type: 'string[]',
              required: true,
              placeholder: 'qwen-worker',
            },
            { key: 'prompt', label: 'Role Prompt', type: 'string' },
            {
              key: 'access_list',
              label: 'Accessible Roles',
              type: 'string[]',
              placeholder: 'reviewer',
            },
          ],
        },
        {
          key: 'final',
          label: 'Static Final',
          type: 'object',
          description: 'Optional static final synthesis model and prompt',
          fields: [
            { key: 'model', label: 'Final Model', type: 'string', placeholder: 'qwen-worker' },
            { key: 'prompt', label: 'Final Prompt', type: 'string' },
          ],
        },
        {
          key: 'max_steps',
          label: 'Max Steps',
          type: 'number',
          placeholder: '6',
        },
        {
          key: 'max_parallel',
          label: 'Max Parallel',
          type: 'number',
          placeholder: '3',
        },
        {
          key: 'max_completion_tokens',
          label: 'Max Completion Tokens',
          type: 'number',
          placeholder: '1024',
        },
        {
          key: 'round_timeout_seconds',
          label: 'Round Timeout',
          type: 'number',
          placeholder: '0 (wait for all)',
          description:
            'Stop waiting for a workflow step or final synthesis after this many seconds',
        },
        {
          key: 'min_successful_responses',
          label: 'Min Successful',
          type: 'number',
          placeholder: '0 (all calls)',
          description: 'Continue once this many parallel workers succeed',
        },
        { key: 'temperature', label: 'Temperature', type: 'number', placeholder: '0.2' },
        {
          key: 'include_intermediate_responses',
          label: 'Include Trace',
          type: 'boolean',
          description: 'Return plan and worker responses in the Flow trace',
        },
        { key: 'on_error', label: 'On Error', type: 'select', options: ['', 'skip', 'fail'] },
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
        {
          key: 'experience_weight',
          label: 'Experience Weight',
          type: 'number',
          placeholder: '0.3',
        },
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
          key: 'objective',
          label: 'Objective',
          type: 'object',
          description: 'Use weighted for balance or ordered priorities for quality/cost first',
          fields: [
            {
              key: 'strategy',
              label: 'Strategy',
              type: 'select',
              options: ['weighted', 'lexicographic'],
            },
            {
              key: 'priorities',
              label: 'Priorities',
              type: 'object[]',
              addLabel: 'Add priority',
              emptyLabel: 'No ordered priorities. Weighted strategy uses the weights below.',
              itemLabel: 'Priority',
              itemLabelKey: 'factor',
              fields: [
                {
                  key: 'factor',
                  label: 'Factor',
                  type: 'select',
                  required: true,
                  options: ['quality', 'latency', 'cost', 'load'],
                },
                {
                  key: 'tolerance',
                  label: 'Relative Tolerance',
                  type: 'number',
                  min: 0,
                  max: 1,
                  placeholder: '0.02',
                  description: 'Retain values within this relative gap before the next priority',
                },
              ],
            },
          ],
        },
        {
          key: 'weights',
          label: 'Weights',
          type: 'object',
          description: 'Per-signal weights for quality, latency, cost, and load',
          fields: [
            { key: 'quality', label: 'Quality', type: 'number', min: 0, placeholder: '0.4' },
            { key: 'latency', label: 'Latency', type: 'number', min: 0, placeholder: '0.3' },
            { key: 'cost', label: 'Cost', type: 'number', min: 0, placeholder: '0.2' },
            { key: 'load', label: 'Load', type: 'number', min: 0, placeholder: '0.1' },
          ],
        },
        {
          key: 'slo',
          label: 'SLO Ceilings',
          type: 'object',
          description: 'Hard ceilings that prune unsafe candidates before scoring',
          fields: [
            { key: 'max_tpot_ms', label: 'Max TPOT (ms)', type: 'number', placeholder: '200' },
            { key: 'max_ttft_ms', label: 'Max TTFT (ms)', type: 'number', placeholder: '800' },
            {
              key: 'max_cost_per_1m',
              label: 'Max Cost / 1M',
              type: 'number',
              placeholder: '5.0',
            },
            { key: 'max_inflight', label: 'Max Inflight', type: 'number', placeholder: '50' },
          ],
        },
        {
          key: 'quality',
          label: 'Quality Evidence',
          type: 'object',
          description: 'Versioned catalog index used as the quality signal',
          fields: [
            {
              key: 'index',
              label: 'Index',
              type: 'string',
              required: true,
              placeholder: 'vllm-sr/intelligence@1.0.0',
            },
            {
              key: 'on_missing',
              label: 'Missing Evidence',
              type: 'select',
              options: ['exclude', 'disable_quality'],
            },
            {
              key: 'min_coverage',
              label: 'Minimum Coverage',
              type: 'number',
              min: 0,
              max: 1,
              placeholder: '1.0',
              description: 'Treat lower-coverage index results as missing',
            },
            {
              key: 'min_score',
              label: 'Minimum Score',
              type: 'number',
              placeholder: '40',
              description: 'Eligibility floor; requires missing evidence to be excluded',
            },
          ],
        },
        {
          key: 'latency_percentile',
          label: 'Latency Percentile',
          type: 'number',
          min: 1,
          max: 100,
          placeholder: '95',
        },
        {
          key: 'on_no_candidates',
          label: 'No Candidates Policy',
          type: 'select',
          options: ['', 'cheapest', 'first', 'fail'],
        },
      ]
    case 'prompt':
      return [
        {
          key: 'prompt',
          label: 'Prompt Selector',
          type: 'object',
          required: true,
          fields: [
            {
              key: 'model',
              label: 'Helper Model',
              type: 'string',
              required: true,
            },
            {
              key: 'instructions',
              label: 'Instructions',
              type: 'string',
              required: true,
            },
            {
              key: 'timeout_seconds',
              label: 'Timeout Seconds',
              type: 'number',
              placeholder: '5',
            },
          ],
        },
        {
          key: 'on_error',
          label: 'On Error',
          type: 'select',
          options: ['', 'fallback'],
        },
      ]
    default:
      return []
  }
}
