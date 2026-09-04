import { getPolicySignalFieldSchema } from './dslPolicySignalSchemas'
import type { FieldSchema } from './dslSchemaTypes'
export type { FieldSchema } from './dslSchemaTypes'

// getPluginFieldSchema moved to dslPluginSchemas.ts: the combined
// per-plugin switch it used to hold here grew past this repo's
// function-length structure cap. Re-exported so existing importers of
// dslSchemas don't need to change their import path.
export { getPluginFieldSchema } from './dslPluginSchemas'

export function getSignalFieldSchema(signalType: string): FieldSchema[] {
  const policyFields = getPolicySignalFieldSchema(signalType)
  if (policyFields) return policyFields
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
    case 'structure':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'feature',
          label: 'Feature',
          type: 'object',
          required: true,
          description:
            'Choose the feature operation and configure its regex, keyword-set, or sequence source.',
          fields: [
            {
              key: 'type',
              label: 'Feature Type',
              type: 'select',
              options: ['exists', 'count', 'density', 'sequence'],
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
                  options: ['regex', 'keyword_set', 'sequence'],
                  required: true,
                },
                { key: 'pattern', label: 'Regex Pattern', type: 'string' },
                {
                  key: 'keywords',
                  label: 'Keywords',
                  type: 'string[]',
                  placeholder: 'Add keyword...',
                },
                { key: 'case_sensitive', label: 'Case Sensitive', type: 'boolean' },
                {
                  key: 'sequences',
                  label: 'Sequences',
                  type: 'string[][]',
                  addLabel: 'Add sequence',
                  emptyLabel: 'No marker sequences configured.',
                },
              ],
            },
          ],
        },
        {
          key: 'predicate',
          label: 'Predicate',
          type: 'object',
          description: 'Optional numeric comparison applied to the extracted feature.',
          fields: [
            { key: 'gt', label: 'Greater Than', type: 'number' },
            { key: 'gte', label: 'Greater Than or Equal', type: 'number' },
            { key: 'lt', label: 'Less Than', type: 'number' },
            { key: 'lte', label: 'Less Than or Equal', type: 'number' },
          ],
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
          type: 'object',
          description: 'Text and image examples associated with the hard side of the rule.',
          fields: [
            {
              key: 'candidates',
              label: 'Text Candidates',
              type: 'string[]',
              placeholder: 'Add hard example...',
            },
            {
              key: 'image_candidates',
              label: 'Image Candidates',
              type: 'string[]',
              placeholder: 'Add image example...',
            },
          ],
        },
        {
          key: 'easy',
          label: 'Easy Examples',
          type: 'object',
          description: 'Text and image examples associated with the easy side of the rule.',
          fields: [
            {
              key: 'candidates',
              label: 'Text Candidates',
              type: 'string[]',
              placeholder: 'Add easy example...',
            },
            {
              key: 'image_candidates',
              label: 'Image Candidates',
              type: 'string[]',
              placeholder: 'Add image example...',
            },
          ],
        },
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'composer',
          label: 'Composer',
          type: 'rule',
          description: 'Optional recursive AND, OR, or NOT composition over other signals.',
        },
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
          type: 'object',
          description: 'Knowledge-base group or label to match.',
          fields: [
            {
              key: 'kind',
              label: 'Target Kind',
              type: 'select',
              options: ['group', 'label'],
              required: true,
            },
            { key: 'value', label: 'Target Value', type: 'string', required: true },
          ],
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
                    'image_content', // validator_conversation.go:16 -- #3001
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
          fields: [
            { key: 'gt', label: 'Greater Than', type: 'number' },
            { key: 'gte', label: 'Greater Than or Equal', type: 'number' },
            { key: 'lt', label: 'Less Than', type: 'number' },
            { key: 'lte', label: 'Less Than or Equal', type: 'number' },
          ],
        },
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

export {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  getAlgorithmFieldSchema,
} from './dslAlgorithmSchemas'
export type { AlgorithmType } from './dslAlgorithmSchemas'
export { BACKEND_TYPES, PLUGIN_DESCRIPTIONS, PLUGIN_TYPES, SIGNAL_TYPES } from './dslSchemaCatalogs'
export type { SignalType } from './dslSchemaCatalogs'
