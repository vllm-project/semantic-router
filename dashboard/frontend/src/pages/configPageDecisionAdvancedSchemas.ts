import type { FieldSchema } from '../lib/dslSchemas'
import { mergeRouterFieldSchemas, routerConfigFieldsForRef } from '../lib/routerConfigSchema'

const DECISION_ACTION_OVERRIDES: FieldSchema[] = [
  { key: 'type', label: 'Type', type: 'select', options: ['route'], required: true },
  { key: 'destination', label: 'Destination Model', type: 'string', required: true },
]
export const DECISION_ACTION_SCHEMA = mergeRouterFieldSchemas(
  routerConfigFieldsForRef('#/$defs/DecisionAction'),
  DECISION_ACTION_OVERRIDES,
)

const DECISION_ADAPTATIONS_OVERRIDES: FieldSchema[] = [
  {
    key: 'mode',
    label: 'Decision Mode',
    type: 'select',
    options: ['', 'apply', 'observe', 'bypass'],
  },
  {
    key: 'adaptation',
    label: 'Learning Adaptation',
    type: 'object',
    fields: [
      { key: 'mode', label: 'Mode', type: 'select', options: ['', 'apply', 'observe', 'bypass'] },
      {
        key: 'candidate_set',
        label: 'Candidate Set',
        type: 'select',
        options: ['', 'decision', 'tier', 'global'],
      },
    ],
  },
  {
    key: 'protection',
    label: 'Switch Protection',
    type: 'object',
    fields: [
      { key: 'mode', label: 'Mode', type: 'select', options: ['', 'apply', 'observe', 'bypass'] },
      { key: 'stability_weight', label: 'Stability Weight', type: 'number', min: 0 },
      { key: 'switch_margin', label: 'Switch Margin', type: 'number', min: 0 },
    ],
  },
]
export const DECISION_ADAPTATIONS_SCHEMA = mergeRouterFieldSchemas(
  routerConfigFieldsForRef('#/$defs/DecisionAdaptationsConfig'),
  DECISION_ADAPTATIONS_OVERRIDES,
)

const NATIVE_ONLY = 'Honored in standalone mode only; Envoy cannot apply it to one request.'
const DECISION_RELIABILITY_OVERRIDES: FieldSchema[] = [
  {
    key: 'total_timeout',
    label: 'Total Timeout',
    type: 'string',
    placeholder: '600s',
    description: "Replaces the provider model's route timeout; 0s disables it.",
  },
  {
    key: 'per_try_timeout',
    label: 'Per-Try Timeout',
    type: 'string',
    placeholder: '300s',
    description: 'Bounds each attempt until its response starts; 0s disables it.',
  },
  {
    key: 'idle_timeout',
    label: 'Idle Timeout',
    type: 'string',
    placeholder: '60s',
    description: NATIVE_ONLY,
  },
  {
    key: 'first_byte_timeout',
    label: 'First-Byte Timeout',
    type: 'string',
    placeholder: '30s',
    description: NATIVE_ONLY,
  },
  { key: 'retry_count', label: 'Retry Count', type: 'number', min: 0, max: 5 },
  {
    key: 'retry_on',
    label: 'Retry On',
    type: 'string',
    placeholder: 'reset,connect-failure',
    description: "Envoy retry_on conditions, added to the provider model's.",
  },
  {
    key: 'retriable_status_codes',
    label: 'Retriable Status Codes',
    type: 'number[]',
    description: "Added to the provider model's; retried under retriable-status-codes.",
  },
  {
    key: 'retry_back_off_base',
    label: 'Retry Back-off Base',
    type: 'string',
    placeholder: '25ms',
    description: NATIVE_ONLY,
  },
  {
    key: 'retry_back_off_max',
    label: 'Retry Back-off Max',
    type: 'string',
    placeholder: '250ms',
    description: NATIVE_ONLY,
  },
  {
    key: 'retry_after_max',
    label: 'Retry-After Max',
    type: 'string',
    placeholder: '30s',
    description: NATIVE_ONLY,
  },
]
export const DECISION_RELIABILITY_SCHEMA = mergeRouterFieldSchemas(
  routerConfigFieldsForRef('#/$defs/DecisionReliability'),
  DECISION_RELIABILITY_OVERRIDES,
)

const DECISION_FALLBACK_OVERRIDES: FieldSchema[] = [
  {
    key: 'enabled',
    label: 'Enabled',
    type: 'boolean',
    description: "Turns cross-model fallback on or off for this decision; unset keeps the recipe's.",
  },
  {
    key: 'max_attempts',
    label: 'Max Attempts',
    type: 'number',
    min: 0,
    description: 'Attempts across candidates, the first included.',
  },
  {
    key: 'total_timeout',
    label: 'Total Timeout',
    type: 'string',
    placeholder: '30s',
    description: 'Bounds the whole fallback chain.',
  },
  {
    key: 'per_attempt_timeout',
    label: 'Per-Attempt Timeout',
    type: 'string',
    placeholder: '10s',
    description: 'Bounds each candidate; it cannot exceed the total timeout.',
  },
  {
    key: 'retryable_status_codes',
    label: 'Retryable Status Codes',
    type: 'number[]',
    description: "Statuses that move on to the next candidate; replaces the recipe's list.",
  },
]
export const DECISION_FALLBACK_SCHEMA = mergeRouterFieldSchemas(
  routerConfigFieldsForRef('#/$defs/FallbackOverride'),
  DECISION_FALLBACK_OVERRIDES,
)

const DECISION_OUTPUT_CONTRACT_OVERRIDES: FieldSchema[] = [
  {
    key: 'type',
    label: 'Type',
    type: 'select',
    options: ['', 'choice', 'structured_json', 'reference_selection'],
  },
  {
    key: 'choice_set',
    label: 'Choice Set',
    type: 'object',
    fields: [{ key: 'values', label: 'Values', type: 'string[]' }],
  },
  {
    key: 'json_schema',
    label: 'JSON Schema',
    type: 'object',
    fields: [{ key: 'schema_ref', label: 'Schema Ref', type: 'string' }],
  },
  {
    key: 'reference',
    label: 'Reference Selection',
    type: 'object',
    fields: [
      { key: 'source', label: 'Source', type: 'string' },
      {
        key: 'id_format',
        label: 'ID Format',
        type: 'select',
        options: ['', 'index', 'reference_number'],
      },
    ],
  },
  {
    key: 'render',
    label: 'Rendering',
    type: 'object',
    fields: [
      { key: 'mode', label: 'Mode', type: 'select', options: ['', 'value', 'template'] },
      { key: 'template', label: 'Template', type: 'string' },
    ],
  },
  {
    key: 'extract',
    label: 'Extraction',
    type: 'object',
    fields: [
      { key: 'mode', label: 'Mode', type: 'select', options: ['', 'exact', 'json_object'] },
      { key: 'sources', label: 'Sources', type: 'string[]' },
    ],
  },
  {
    key: 'normalize',
    label: 'Normalization',
    type: 'object',
    fields: [{ key: 'defaults', label: 'Defaults', type: 'key-value' }],
  },
  {
    key: 'postprocess',
    label: 'Post-processing',
    type: 'object[]',
    fields: [{ key: 'type', label: 'Type', type: 'string', required: true }],
    itemLabel: 'Step',
    itemLabelKey: 'type',
  },
]
export const DECISION_OUTPUT_CONTRACT_SCHEMA = mergeRouterFieldSchemas(
  routerConfigFieldsForRef('#/$defs/OutputContractSpec'),
  DECISION_OUTPUT_CONTRACT_OVERRIDES,
)

const DECISION_DECLARATIVE_OVERRIDES: FieldSchema[] = [
  {
    key: 'candidateIterations',
    label: 'Candidate Iterations',
    type: 'object[]',
    itemLabel: 'Iteration',
    itemLabelKey: 'variable',
    fields: [
      { key: 'variable', label: 'Variable', type: 'string', required: true },
      { key: 'source', label: 'Source', type: 'string', required: true },
      {
        key: 'models',
        label: 'Models',
        type: 'object[]',
        fields: [
          { key: 'model', label: 'Model', type: 'string', required: true },
          { key: 'use_reasoning', label: 'Use Reasoning', type: 'boolean' },
          { key: 'weight', label: 'Weight', type: 'number' },
        ],
      },
      {
        key: 'outputs',
        label: 'Outputs',
        type: 'object[]',
        fields: [
          { key: 'type', label: 'Type', type: 'string', required: true },
          { key: 'value', label: 'Value', type: 'string' },
        ],
      },
    ],
  },
  {
    key: 'emits',
    label: 'Emits',
    type: 'object[]',
    itemLabel: 'Directive',
    itemLabelKey: 'kind',
    fields: [
      { key: 'kind', label: 'Kind', type: 'select', options: ['retention'], required: true },
      {
        key: 'retention',
        label: 'Retention',
        type: 'object',
        fields: [
          { key: 'drop', label: 'Drop', type: 'boolean' },
          { key: 'ttl_turns', label: 'TTL Turns', type: 'number', min: 0 },
          { key: 'keep_current_model', label: 'Keep Current Model', type: 'boolean' },
          {
            key: 'prefer_prefix_retention',
            label: 'Prefer Prefix Retention',
            type: 'boolean',
          },
        ],
      },
    ],
  },
  { key: 'annotations', label: 'Annotations', type: 'key-value' },
]
const declarativeKeys = new Set(['candidateIterations', 'emits', 'annotations'])
export const DECISION_DECLARATIVE_SCHEMA = mergeRouterFieldSchemas(
  routerConfigFieldsForRef('#/$defs/Decision').filter((field) => declarativeKeys.has(field.key)),
  DECISION_DECLARATIVE_OVERRIDES,
)
