import { X_VSR_COMPRESSION_CONTROL } from '../generated/vsrHeaders'
import type { FieldSchema } from './dslSchemaTypes'

function matchField(recallMethods: string[], judgePreset: string): FieldSchema {
  return {
    key: 'match',
    label: 'Match',
    type: 'object',
    description: 'A written block replaces the default whole',
    fields: [
      {
        key: 'recall',
        label: 'Recall',
        type: 'object',
        fields: [
          { key: 'method', label: 'Method', type: 'select', options: recallMethods },
          { key: 'top_k', label: 'Top K', type: 'number' },
          { key: 'min_similarity', label: 'Min Similarity', type: 'number' },
        ],
      },
      {
        key: 'judge',
        label: 'Judge',
        type: 'object',
        fields: [
          { key: 'preset', label: 'Preset', type: 'string', placeholder: judgePreset },
          { key: 'deployment', label: 'Deployment', type: 'string' },
          { key: 'top_k', label: 'Top K', type: 'number' },
          { key: 'min_ratio', label: 'Min Ratio', type: 'number' },
          { key: 'min_score', label: 'Min Score', type: 'number' },
        ],
      },
    ],
  }
}

const sourceFields: FieldSchema[] = [
  { key: 'type', label: 'Type', type: 'select', options: ['rag', 'memory'], required: true },
  {
    ...matchField(['embedding'], 'relevance'),
    description:
      'rag: recalls 20 at 0.3 and keeps the 6 the relevance judge scores 0.5 or more; memory: recalls with the store defaults',
  },
  {
    key: 'backend',
    label: 'Backend (rag)',
    type: 'select',
    options: ['', 'vectorstore', 'milvus', 'qdrant', 'external_api', 'mcp', 'openai', 'hybrid'],
  },
  { key: 'backend_config', label: 'Backend Config (rag)', type: 'json' },
  {
    key: 'max_context_length',
    label: 'Max Context Length (rag)',
    type: 'number',
    placeholder: '10000',
  },
  {
    key: 'on_failure',
    label: 'On Failure (rag)',
    type: 'select',
    options: ['', 'skip', 'warn', 'block'],
  },
  { key: 'cache_results', label: 'Cache Results (rag)', type: 'boolean' },
  { key: 'cache_ttl_seconds', label: 'Cache TTL Seconds (rag)', type: 'number', placeholder: '3600' },
  {
    key: 'store',
    label: 'Store (memory)',
    type: 'object',
    fields: [
      {
        key: 'auto',
        label: 'Write After The Response',
        type: 'boolean',
        description: 'Unset follows global.stores.memory.auto_store',
      },
    ],
  },
  { key: 'hybrid_search', label: 'Hybrid Search (memory)', type: 'boolean' },
  {
    key: 'hybrid_mode',
    label: 'Hybrid Mode (memory)',
    type: 'select',
    options: ['', 'weighted', 'rrf'],
  },
  {
    key: 'reflection',
    label: 'Reflection (memory)',
    type: 'object',
    description: "Overrides the memory store's reflection gate for this decision",
    fields: [
      { key: 'enabled', label: 'Enabled', type: 'boolean' },
      { key: 'algorithm', label: 'Algorithm', type: 'select', options: ['', 'heuristic', 'noop'] },
      { key: 'max_inject_tokens', label: 'Max Inject Tokens', type: 'number' },
      { key: 'recency_decay_days', label: 'Recency Decay Days', type: 'number' },
      { key: 'dedup_threshold', label: 'Dedup Threshold', type: 'number' },
      { key: 'block_patterns', label: 'Block Patterns', type: 'string[]' },
    ],
  },
]

const segmentModes = ['preserve', 'extractive', 'recoverable']

const compressTargetsField: FieldSchema = {
  key: 'targets',
  label: 'Targets',
  type: 'object',
  fields: [
    {
      key: 'tool_outputs',
      label: 'Tool Outputs',
      type: 'object',
      fields: [
        { key: 'mode', label: 'Mode', type: 'select', options: segmentModes },
        { key: 'min_tokens', label: 'Minimum Tokens', type: 'number', placeholder: '2000' },
        { key: 'target_tokens', label: 'Target Tokens', type: 'number', placeholder: '1000' },
      ],
    },
    ...['history', 'rag', 'memory'].map((key) => ({
      key,
      label: key === 'rag' ? 'RAG' : key[0].toUpperCase() + key.slice(1),
      type: 'object' as const,
      fields: [{ key: 'mode', label: 'Mode', type: 'select' as const, options: segmentModes }],
    })),
    {
      key: 'current_user',
      label: 'Current User',
      type: 'object',
      fields: [{ key: 'mode', label: 'Mode', type: 'select', options: ['preserve', 'truncate'] }],
    },
  ],
}

const compressFields: FieldSchema[] = [
  { key: 'mode', label: 'Mode', type: 'select', options: ['auto', 'always'] },
  compressTargetsField,
  {
    ...matchField(['bm25'], 'relevance'),
    description: 'Default: the relevance judge ranks the segments; recall bm25 ranks them lexically',
  },
  {
    key: 'recovery',
    label: 'Recoverable Compression',
    type: 'object',
    fields: [
      { key: 'enabled', label: 'Enabled', type: 'boolean' },
      {
        key: 'store',
        label: 'Shared Store',
        type: 'select',
        options: ['', 'response_cache', 'redis', 'valkey'],
      },
      { key: 'ttl_seconds', label: 'TTL Seconds', type: 'number' },
      { key: 'max_bytes_per_request', label: 'Max Bytes Per Request', type: 'number' },
      { key: 'max_total_bytes', label: 'Max Total Bytes', type: 'number' },
      { key: 'max_retrievals', label: 'Max Retrievals', type: 'number' },
    ],
  },
  {
    key: 'request_controls',
    label: 'Request Controls',
    type: 'object',
    fields: [
      { key: 'enabled', label: 'Enabled', type: 'boolean' },
      { key: 'header', label: 'Header', type: 'string', placeholder: X_VSR_COMPRESSION_CONTROL },
      { key: 'allowed', label: 'Allowed Directives', type: 'string[]' },
      { key: 'max_target_tokens', label: 'Maximum Target Tokens', type: 'number' },
    ],
  },
  { key: 'failure_mode', label: 'Failure Mode', type: 'select', options: ['fail_open', 'fail_closed'] },
]

export const contextPluginFields: FieldSchema[] = [
  { key: 'enabled', label: 'Enabled', type: 'boolean' },
  {
    key: 'sources',
    label: 'Sources',
    type: 'object[]',
    description: 'Retrieved in order; at most one rag and one memory source',
    addLabel: 'Add source',
    itemLabel: 'Source',
    itemLabelKey: 'type',
    fields: sourceFields,
  },
  {
    key: 'compress',
    label: 'Compress',
    type: 'object',
    description: 'Shrinks tool outputs, history and injected context within the budget',
    fields: compressFields,
  },
  {
    key: 'budget',
    label: 'Budget',
    type: 'object',
    description: 'Needs compress',
    fields: [
      { key: 'max_tokens', label: 'Max Input Tokens', type: 'string', placeholder: 'auto' },
      {
        key: 'reserve_output_tokens',
        label: 'Reserved Output Tokens',
        type: 'string',
        placeholder: 'auto',
      },
    ],
  },
  {
    key: 'injection',
    label: 'Injection',
    type: 'object',
    fields: [
      { key: 'mode', label: 'Mode', type: 'select', options: ['', 'tool_role', 'system_prompt'] },
    ],
  },
]
