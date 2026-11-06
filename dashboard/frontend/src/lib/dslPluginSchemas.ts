import { SIGNAL_TYPES } from '../generated/routerConfigContract'
import { resolveCapabilityPluginFieldSchema } from './dslCapabilityPluginSchemas'
import type { FieldSchema } from './dslSchemaTypes'

// Per-plugin field schemas for the decision-editor DSL. Extracted out of
// dslSchemas.ts (which imports getCuratedPluginFieldSchema and merges it
// with router-schema-derived fields as the exported getPluginFieldSchema)
// because this combined switch grew past the repo's function-length
// structure cap — see dslSchemas.ts's own file-level note. Each plugin
// case below is its own small helper instead of one large switch, matching
// this directory's narrow-module convention.
export function getCuratedPluginFieldSchema(pluginType: string): FieldSchema[] {
  const capabilityFields = resolveCapabilityPluginFieldSchema(pluginType)
  if (capabilityFields) return capabilityFields
  switch (pluginType) {
    case 'memory':
      return getMemoryPluginFieldSchema()
    case 'system_prompt':
      return getSystemPromptPluginFieldSchema()
    case 'response_guard':
      return getResponseGuardPluginFieldSchema()
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
    case 'request_params':
      return getRequestParamsPluginFieldSchema()
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
      description: "Replace the caller's system prompt (default) or insert before it",
    },
  ]
}

function getResponseGuardPluginFieldSchema(): FieldSchema[] {
  return [
    {
      key: 'rules',
      label: 'Rules',
      type: 'object[]',
      required: true,
      description: 'Read in order over each answer; the first rule that blocks ends the read.',
      addLabel: 'Add rule',
      emptyLabel: 'No rules.',
      itemLabel: 'Rule',
      itemLabelKey: 'action',
      fields: [
        {
          key: 'when',
          label: 'When',
          type: 'rule',
          required: true,
          options: [...SIGNAL_TYPES],
          description:
            'Signals by name: jailbreak, safety and systemone rules with stage: response, and every hallucination and refusal rule, read the answer; any other reads the request.',
        },
        {
          key: 'action',
          label: 'Action',
          type: 'select',
          required: true,
          options: ['block', 'annotate', 'none'],
          description: 'Refuse the answer with 403, deliver it with x-vsr-response-warnings, or only record it',
        },
        {
          key: 'details',
          label: 'Details',
          type: 'boolean',
          description: 'With annotate, also write a note at the head of the answer',
        },
      ],
    },
    {
      key: 'on_unknown',
      label: 'On Unknown',
      type: 'select',
      options: ['', 'block', 'annotate', 'none'],
      description:
        'What a rule does when a response-stage signal it reads could not be read (default annotate)',
    },
  ]
}

function getRouterReplayPluginFieldSchema(): FieldSchema[] {
  return [
    { key: 'enabled', label: 'Enabled', type: 'boolean' },
    { key: 'max_records', label: 'Max Records', type: 'number', placeholder: '10000' },
    { key: 'capture_request_body', label: 'Capture Request Body', type: 'boolean' },
    { key: 'capture_response_body', label: 'Capture Response Body', type: 'boolean' },
    {
      key: 'capture_tool_results',
      label: 'Capture Tool Results',
      type: 'boolean',
      description: 'Store client tool-result text; requires personal data capture',
    },
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
      key: 'policy',
      label: 'Policy',
      type: 'object',
      description: 'Allow, block or remove tools by name; runs before the selection',
      fields: [
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['passthrough', 'filtered', 'none'],
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
          key: 'strip_tool_history',
          label: 'Strip Tool History',
          type: 'boolean',
          description:
            'With mode none, remove prior tool calls and results from the provider-bound body',
        },
      ],
    },
    {
      key: 'select',
      label: 'Select',
      type: 'object',
      description:
        'The decision model chooses the tools the request needs, among its own (filter) or a catalog (add)',
      fields: [
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['filter', 'add'],
          required: true,
        },
        {
          key: 'tools_db_path',
          label: 'Tool Catalog',
          type: 'string',
          placeholder: 'config/tools_db.json',
          description: 'Add mode: the catalog the decision model adds tools from',
        },
        {
          key: 'match',
          label: 'Match',
          type: 'object',
          fields: [
            {
              key: 'judge',
              label: 'Judge',
              type: 'object',
              fields: [
                { key: 'preset', label: 'Preset', type: 'string', placeholder: 'tool_rank' },
                { key: 'deployment', label: 'Deployment', type: 'string' },
                { key: 'top_k', label: 'Top K', type: 'number', placeholder: '3' },
                { key: 'min_ratio', label: 'Min Ratio', type: 'number', placeholder: '0.1' },
                { key: 'min_score', label: 'Min Score', type: 'number' },
              ],
            },
          ],
        },
        {
          key: 'need_threshold',
          label: 'Need Threshold',
          type: 'number',
          placeholder: '0.5',
          description: 'Below this P(tool_need), no tools are kept or added',
        },
        {
          key: 'preserve_count',
          label: 'Preserve Count',
          type: 'number',
          description: "Filter mode: the request's first tools are always kept",
        },
        {
          key: 'on_error',
          label: 'On Error',
          type: 'select',
          options: ['keep', 'strip', 'skip', 'fail'],
          description: 'Filter: keep, strip or fail; add: skip or fail',
        },
        {
          key: 'on_overflow',
          label: 'On Overflow',
          type: 'select',
          options: ['keep', 'reject'],
          description: 'Filter mode, more than 255 tools',
        },
        {
          key: 'input',
          label: 'Input',
          type: 'object',
          fields: [{ key: 'history', label: 'Read Earlier User Turns', type: 'boolean' }],
        },
        { key: 'timeout_ms', label: 'Timeout (ms)', type: 'number', placeholder: '1000' },
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
