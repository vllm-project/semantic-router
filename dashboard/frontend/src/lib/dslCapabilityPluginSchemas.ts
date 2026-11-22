import { X_VSR_CACHE_CONTROL } from '../generated/vsrHeaders'
import { contextPluginFields, matchField } from './dslContextPluginSchemas'
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
        { key: 'ttl_seconds', label: 'TTL Seconds', type: 'number', placeholder: '3600' },
        {
          key: 'semantic',
          label: 'Semantic Reuse',
          type: 'object',
          fields: [matchField(['embedding'], 'same_request')],
        },
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
