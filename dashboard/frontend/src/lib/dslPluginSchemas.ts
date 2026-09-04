import { getCapabilityPluginFieldSchema } from './dslCapabilityPluginSchemas'
import type { FieldSchema } from './dslSchemaTypes'

// Per-plugin field schemas for the decision-editor DSL. Extracted out of
// dslSchemas.ts (which re-exports getPluginFieldSchema) because that file's
// combined switch grew past the repo's function-length structure cap —
// see dslSchemas.ts's own file-level note. Each plugin case below is its
// own small helper instead of one large switch, matching this directory's
// narrow-module convention.
export function getPluginFieldSchema(pluginType: string): FieldSchema[] {
  const capabilityFields = getCapabilityPluginFieldSchema(pluginType)
  if (capabilityFields) return capabilityFields
  switch (pluginType) {
    case 'memory':
      return getMemoryPluginFieldSchema()
    case 'system_prompt':
      return getSystemPromptPluginFieldSchema()
    case 'hallucination':
      return getHallucinationPluginFieldSchema()
    case 'router_replay':
      return getRouterReplayPluginFieldSchema()
    case 'rag':
      return getRAGPluginFieldSchema()
    case 'header_mutation':
      return getHeaderMutationPluginFieldSchema()
    case 'fast_response':
      return getFastResponsePluginFieldSchema()
    case 'tools':
      return getToolsPluginFieldSchema()
    case 'tool_selection':
      return getToolSelectionPluginFieldSchema()
    case 'request_params':
      return getRequestParamsPluginFieldSchema()
    case 'response_jailbreak':
      return getResponseJailbreakPluginFieldSchema()
    default:
      return [{ key: 'enabled', label: 'Enabled', type: 'boolean' }]
  }
}

function getMemoryPluginFieldSchema(): FieldSchema[] {
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
}

function getSystemPromptPluginFieldSchema(): FieldSchema[] {
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
}

function getHallucinationPluginFieldSchema(): FieldSchema[] {
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
}

function getRouterReplayPluginFieldSchema(): FieldSchema[] {
  return [
    { key: 'enabled', label: 'Enabled', type: 'boolean' },
    { key: 'max_records', label: 'Max Records', type: 'number', placeholder: '10000' },
    { key: 'capture_request_body', label: 'Capture Request Body', type: 'boolean' },
    { key: 'capture_response_body', label: 'Capture Response Body', type: 'boolean' },
    { key: 'max_body_bytes', label: 'Max Body Bytes', type: 'number', placeholder: '4096' },
  ]
}

function getRAGPluginFieldSchema(): FieldSchema[] {
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
}

function getHeaderMutationPluginFieldSchema(): FieldSchema[] {
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
}

function getFastResponsePluginFieldSchema(): FieldSchema[] {
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
}

function getToolsPluginFieldSchema(): FieldSchema[] {
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
      key: 'strip_tool_history',
      label: 'Strip Tool History',
      type: 'boolean',
      description:
        'With mode none, remove prior tool calls and results from the provider-bound body',
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
    {
      key: 'strategy',
      label: 'Retrieval Strategy',
      type: 'string',
      placeholder: 'default',
    },
    {
      key: 'dynamic_retrieval',
      label: 'Dynamic Retrieval',
      type: 'object',
      fields: [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'strategy',
          label: 'Strategy',
          type: 'select',
          options: ['semantic_only', 'hybrid_history'],
        },
        { key: 'history_window', label: 'History Window', type: 'number' },
        {
          key: 'weights',
          label: 'Weights',
          type: 'object',
          fields: [
            { key: 'semantic', label: 'Semantic', type: 'number' },
            { key: 'history', label: 'History', type: 'number' },
            { key: 'decision_prior', label: 'Decision Prior', type: 'number' },
            { key: 'repetition_penalty', label: 'Repetition Penalty', type: 'number' },
          ],
        },
        {
          key: 'min_history_confidence',
          label: 'Minimum History Confidence',
          type: 'number',
        },
        {
          key: 'fallback_on_low_confidence',
          label: 'Fallback on Low Confidence',
          type: 'boolean',
        },
      ],
    },
  ]
}

function getToolSelectionPluginFieldSchema(): FieldSchema[] {
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
    {
      key: 'sticky',
      label: 'Session-Scoped Sticky Selection',
      type: 'object',
      fields: [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'max_tools', label: 'Max Tools', type: 'number', placeholder: '16' },
        {
          key: 'max_new_tools_per_turn',
          label: 'Max New Tools Per Turn',
          type: 'number',
          placeholder: '2',
        },
        { key: 'pin_called_tools', label: 'Pin Called Tools', type: 'boolean' },
      ],
    },
  ]
}

function getRequestParamsPluginFieldSchema(): FieldSchema[] {
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
}

function getResponseJailbreakPluginFieldSchema(): FieldSchema[] {
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
}
