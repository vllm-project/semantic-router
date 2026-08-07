import type { FieldSchema } from './dslSchemas'

export function getCapabilityPluginFieldSchema(pluginType: string): FieldSchema[] | null {
  switch (pluginType) {
    case 'response_cache':
    case 'response-cache':
    case 'semantic_cache':
    case 'semantic-cache':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['semantic', 'exact', 'exact_then_semantic'],
        },
        {
          key: 'scope',
          label: 'Scope',
          type: 'select',
          options: ['user', 'team', 'tenant', 'global'],
        },
        {
          key: 'semantic',
          label: 'Semantic Lookup',
          type: 'object',
          fields: [
            {
              key: 'similarity_threshold',
              label: 'Similarity Threshold',
              type: 'number',
              placeholder: '0.95',
            },
          ],
        },
        { key: 'ttl_seconds', label: 'TTL Seconds', type: 'number', placeholder: '3600' },
        {
          key: 'request_controls',
          label: 'Request Controls',
          type: 'object',
          fields: [
            { key: 'enabled', label: 'Enabled', type: 'boolean' },
            {
              key: 'header',
              label: 'Header',
              type: 'string',
              placeholder: 'x-vsr-cache-control',
            },
            {
              key: 'allowed',
              label: 'Allowed Directives',
              type: 'string[]',
            },
            {
              key: 'max_ttl_seconds',
              label: 'Maximum Request TTL',
              type: 'number',
            },
          ],
        },
        {
          key: 'personalized',
          label: 'Personalized Context',
          type: 'object',
          fields: [
            {
              key: 'mode',
              label: 'Mode',
              type: 'select',
              options: ['disabled', 'exact'],
            },
          ],
        },
        {
          key: 'revision',
          label: 'Revision Identity',
          type: 'object',
          fields: [
            { key: 'cache_epoch', label: 'Cache Epoch', type: 'string' },
            { key: 'model_revision', label: 'Model Revision', type: 'string' },
            { key: 'prompt_revision', label: 'Prompt Revision', type: 'string' },
            { key: 'policy_revision', label: 'Policy Revision', type: 'string' },
          ],
        },
      ]
    case 'provider_prompt_cache':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'system', label: 'Cache System Prefix', type: 'boolean' },
        { key: 'tools', label: 'Cache Tool Definitions', type: 'boolean' },
        { key: 'last_user', label: 'Cache Latest User Block', type: 'boolean' },
        { key: 'ttl', label: 'TTL', type: 'select', options: ['', '5m', '1h'] },
        { key: 'allow_request_controls', label: 'Allow Request Controls', type: 'boolean' },
        {
          key: 'control_header',
          label: 'Control Header',
          type: 'string',
          placeholder: 'x-vsr-provider-cache-control',
        },
      ]
    case 'context_compression':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'min_tokens', label: 'Minimum Tokens', type: 'number', placeholder: '2000' },
        { key: 'target_tokens', label: 'Target Tokens', type: 'number', placeholder: '1000' },
        {
          key: 'compress_rag',
          label: 'Compress RAG Tool Results',
          type: 'boolean',
          description: 'Opt in to compressing tool messages injected by the RAG plugin',
        },
        {
          key: 'bypass_header',
          label: 'Bypass Header',
          type: 'string',
          placeholder: 'x-vsr-compression-bypass',
        },
      ]
    default:
      return null
  }
}
