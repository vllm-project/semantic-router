import { X_VSR_CACHE_CONTROL } from '../generated/vsrHeaders'
import { contextPluginFields } from './dslContextPluginSchemas'
import type { FieldSchema } from './dslSchemas'

const promptCacheFields: FieldSchema[] = [
  { key: 'enabled', label: 'Enabled', type: 'boolean' },
  {
    key: 'ttl',
    label: 'Marker TTL',
    type: 'select',
    options: ['5m', '1h'],
  },
  {
    key: 'targets',
    label: 'Stable Targets',
    type: 'string[]',
  },
  {
    key: 'on_unsupported',
    label: 'Unsupported Target',
    type: 'select',
    options: ['skip', 'reject'],
  },
]

const additionalCapabilityPluginFields = new Map<string, FieldSchema[]>([
  ['prompt_cache', promptCacheFields],
])
// Plugin schemas added after getCapabilityPluginFieldSchema reached the
// structure ratchet live in this table so that function never grows.
const CAPABILITY_PLUGIN_FIELD_SCHEMAS: Record<string, FieldSchema[]> = {
  context: contextPluginFields,
  shadow_dispatch: [
    { key: 'enabled', label: 'Enabled', type: 'boolean' },
    { key: 'model', label: 'Shadow Model', type: 'string', placeholder: 'candidate-model' },
    { key: 'sample_rate', label: 'Sample Rate', type: 'number', placeholder: '0.05' },
    { key: 'max_concurrency', label: 'Max Concurrency', type: 'number', placeholder: '2' },
    { key: 'max_queue_depth', label: 'Max Queue Depth', type: 'number', placeholder: '8' },
    { key: 'timeout_seconds', label: 'Timeout Seconds', type: 'number', placeholder: '30' },
    {
      key: 'max_response_bytes',
      label: 'Max Response Bytes',
      type: 'number',
      placeholder: '1048576',
    },
    { key: 'max_retries', label: 'Max Retries', type: 'number', placeholder: '0' },
    { key: 'capture_response_body', label: 'Capture Response Body', type: 'boolean' },
    { key: 'max_capture_bytes', label: 'Max Capture Bytes', type: 'number', placeholder: '4096' },
    { key: 'tls_skip_verify', label: 'Skip TLS Verification', type: 'boolean' },
    {
      key: 'forward_headers',
      label: 'Forwarded Decision Headers',
      type: 'string[]',
      placeholder: 'x-tenant',
      description:
        'Decision header mutations the shadow copy may carry; credential headers are never forwarded',
    },
  ],
}

export function resolveCapabilityPluginFieldSchema(pluginType: string): FieldSchema[] | null {
  if (Object.prototype.hasOwnProperty.call(CAPABILITY_PLUGIN_FIELD_SCHEMAS, pluginType)) {
    return CAPABILITY_PLUGIN_FIELD_SCHEMAS[pluginType]
  }
  return getCapabilityPluginFieldSchema(pluginType)
}

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
              placeholder: X_VSR_CACHE_CONTROL,
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
    default:
      return additionalCapabilityPluginFields.get(pluginType) ?? null
  }
}
