import { getPolicySignalFieldSchema } from './dslPolicySignalSchemas'
import {
  mergeRouterFieldSchemas,
  pluginFieldsFromRouterSchema,
  signalFieldsFromRouterSchema,
} from './routerConfigSchema'
import { getCuratedPluginFieldSchema } from './dslPluginSchemas'
import type { FieldSchema } from './dslSchemaTypes'
export type { FieldSchema } from './dslSchemaTypes'

// getCuratedPluginFieldSchema (the per-plugin switch getPluginFieldSchema
// below merges router-schema-derived fields on top of) lives in
// dslPluginSchemas.ts: it grew past this repo's function-length structure
// cap. resolveCapabilityPluginFieldSchema moved there with it, since it's
// only ever called from within that switch.

const NUMERIC_PREDICATE_FIELDS: FieldSchema[] = [
  { key: 'gt', label: 'Greater Than', type: 'number' },
  { key: 'gte', label: 'Greater Than or Equal', type: 'number' },
  { key: 'lt', label: 'Less Than', type: 'number' },
  { key: 'lte', label: 'Less Than or Equal', type: 'number' },
]

function getCuratedSignalFieldSchema(signalType: string): FieldSchema[] {
  const policyFields = getPolicySignalFieldSchema(signalType)
  if (policyFields) return policyFields
  switch (signalType) {
    case 'keyword':
      return [
        { key: 'description', label: 'Description', type: 'string' },
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
        {
          key: 'include_history',
          label: 'Include History',
          type: 'boolean',
          description: 'Also match earlier user messages, not just the latest one',
        },
        {
          key: 'count',
          label: 'Count',
          type: 'object',
          description:
            'An OR rule with literal or regex keywords matches on how many times they occur.',
          fields: NUMERIC_PREDICATE_FIELDS,
        },
        {
          key: 'density',
          label: 'Density',
          type: 'object',
          description:
            'An OR rule with literal or regex keywords matches on occurrences per 1,000 characters. Set count or density, not both.',
          fields: NUMERIC_PREDICATE_FIELDS,
        },
      ]
    case 'domain':
      return [
        { key: 'description', label: 'Description', type: 'string', required: true },
        {
          key: 'model_scores',
          label: 'Model Scores',
          type: 'object[]',
          addLabel: 'Add model score',
          emptyLabel: 'No model scores configured.',
          itemLabel: 'Model Score',
          itemLabelKey: 'model',
          fields: [
            { key: 'model', label: 'Model', type: 'string', required: true },
            { key: 'score', label: 'Score', type: 'number', required: true },
            { key: 'use_reasoning', label: 'Use Reasoning', type: 'boolean' },
          ],
        },
      ]
    case 'fact_check':
      return [{ key: 'description', label: 'Description', type: 'string', required: true }]
    case 'user_feedback':
      return [{ key: 'description', label: 'Description', type: 'string', required: true }]
    case 'language':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'threshold',
          label: 'Confidence Threshold',
          type: 'number',
          min: 0,
          max: 1,
          placeholder: '0.3',
        },
      ]
    case 'context':
      return [
        {
          key: 'min_tokens',
          label: 'Min Tokens',
          type: 'string',
          placeholder: '4K (defaults to 0)',
          description: 'Inclusive lower bound. Defaults to 0 when empty.',
        },
        {
          key: 'max_tokens',
          label: 'Max Tokens',
          type: 'string',
          placeholder: '32K (leave empty for no upper bound)',
          description: 'Inclusive upper bound. Leave empty for an open-ended band.',
        },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'modality':
      return [{ key: 'description', label: 'Description', type: 'string' }]
    case 'authz':
      return [
        {
          key: 'subjects',
          label: 'Subjects',
          type: 'object[]',
          required: true,
          description: 'Users, groups, or other identities assigned to this role.',
          addLabel: 'Add subject',
          emptyLabel: 'No subjects configured.',
          itemLabel: 'Subject',
          itemLabelKey: 'name',
          fields: [
            { key: 'kind', label: 'Kind', type: 'string', required: true, placeholder: 'Group' },
            { key: 'name', label: 'Name', type: 'string', required: true },
          ],
        },
        { key: 'role', label: 'Role', type: 'string', required: true, placeholder: 'premium_tier' },
        { key: 'description', label: 'Description', type: 'string' },
      ]
    case 'hallucination':
      return [{ key: 'description', label: 'Description', type: 'string' }]
    case 'conversation':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'feature',
          label: 'Feature',
          type: 'object',
          required: true,
          description: 'Count or detect a conversation source such as messages or tool activity.',
          fields: [
            {
              key: 'type',
              label: 'Feature Type',
              type: 'select',
              options: ['count', 'exists'],
              required: true,
            },
            {
              key: 'source',
              label: 'Source',
              type: 'object',
              required: true,
              fields: [
                {
                  key: 'type',
                  label: 'Source Type',
                  type: 'select',
                  options: [
                    'message',
                    'tool_definition',
                    'tool_choice_required',
                    'tool_choice_none',
                    'assistant_tool_call',
                    'assistant_tool_cycle',
                    'active_tool_loop',
                    'flow_tool_state',
                  ],
                  required: true,
                },
                {
                  key: 'role',
                  label: 'Message Role',
                  type: 'select',
                  // 'non_user' is a computed aggregate, not an OpenAI role
                  // (classifier_signal_conversation.go:139). Valid per
                  // validator_conversation.go:25 and used by config/config.yaml:497.
                  options: ['', 'system', 'developer', 'user', 'assistant', 'tool', 'non_user'],
                  description: 'Only used when the source type is message.',
                },
              ],
            },
          ],
        },
        {
          key: 'predicate',
          label: 'Predicate',
          type: 'object',
          description: 'Optional numeric comparison for count features.',
          fields: NUMERIC_PREDICATE_FIELDS,
        },
      ]
    default:
      return [{ key: 'description', label: 'Description', type: 'string' }]
  }
}

export function getSignalFieldSchema(signalType: string): FieldSchema[] {
  return mergeRouterFieldSchemas(
    signalFieldsFromRouterSchema(signalType),
    getCuratedSignalFieldSchema(signalType),
  )
}

export function getPluginFieldSchema(pluginType: string): FieldSchema[] {
  return mergeRouterFieldSchemas(
    pluginFieldsFromRouterSchema(pluginType),
    getCuratedPluginFieldSchema(pluginType),
  )
}

export {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  getAlgorithmFieldSchema,
} from './dslAlgorithmSchemas'
export type { AlgorithmType } from './dslAlgorithmSchemas'
export { BACKEND_TYPES, PLUGIN_DESCRIPTIONS, PLUGIN_TYPES, SIGNAL_TYPES } from './dslSchemaCatalogs'
export type { SignalType } from './dslSchemaCatalogs'
