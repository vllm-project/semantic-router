export interface FieldSchema {
  key: string
  label: string
  type:
    | 'string'
    | 'number'
    | 'boolean'
    | 'string[]'
    | 'number[]'
    | 'string[][]'
    | 'select'
    | 'object'
    | 'object[]'
    | 'key-value'
    | 'rule'
  options?: string[]
  required?: boolean
  placeholder?: string
  description?: string
  fields?: FieldSchema[]
  addLabel?: string
  emptyLabel?: string
  itemLabel?: string
  itemLabelKey?: string
  keyLabel?: string
  valueLabel?: string
}

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
                    'assistant_tool_call',
                    'assistant_tool_cycle',
                    'active_tool_loop',
                  ],
                  required: true,
                },
                {
                  key: 'role',
                  label: 'Message Role',
                  type: 'select',
                  options: ['', 'system', 'developer', 'user', 'assistant', 'tool'],
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
          type: 'object[]',
          description: 'Headers inserted when they are not already present.',
          addLabel: 'Add header',
          emptyLabel: 'No headers to add.',
          itemLabel: 'Header',
          itemLabelKey: 'name',
          fields: [
            { key: 'name', label: 'Header Name', type: 'string', required: true },
            { key: 'value', label: 'Header Value', type: 'string', required: true },
          ],
        },
        {
          key: 'update',
          label: 'Update Headers',
          type: 'object[]',
          description: 'Headers overwritten before forwarding the request.',
          addLabel: 'Add header update',
          emptyLabel: 'No headers to update.',
          itemLabel: 'Header',
          itemLabelKey: 'name',
          fields: [
            { key: 'name', label: 'Header Name', type: 'string', required: true },
            { key: 'value', label: 'Header Value', type: 'string', required: true },
          ],
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

export {
  ALGORITHM_DESCRIPTIONS,
  ALGORITHM_TYPES,
  getAlgorithmFieldSchema,
} from './dslAlgorithmSchemas'
export type { AlgorithmType } from './dslAlgorithmSchemas'
export { BACKEND_TYPES, PLUGIN_DESCRIPTIONS, PLUGIN_TYPES, SIGNAL_TYPES } from './dslSchemaCatalogs'
export type { SignalType } from './dslSchemaCatalogs'
