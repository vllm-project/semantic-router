import type { FieldSchema } from './dslSchemas'

export function getCapabilityPluginFieldSchema(pluginType: string): FieldSchema[] | null {
  switch (pluginType) {
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
        { key: 'allow_request_controls', label: 'Allow Request Controls', type: 'boolean' },
        {
          key: 'control_header',
          label: 'Control Header',
          type: 'string',
          placeholder: 'x-vsr-cache-control',
        },
        {
          key: 'similarity_threshold',
          label: 'Similarity Threshold',
          type: 'number',
          placeholder: '0.95',
          description: 'Minimum similarity for cache hit (0-1)',
        },
        { key: 'ttl_seconds', label: 'TTL Seconds', type: 'number', placeholder: '3600' },
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
