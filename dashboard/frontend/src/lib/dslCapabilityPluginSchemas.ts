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
    case 'shadow_dispatch':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        { key: 'model', label: 'Shadow Model', type: 'string', placeholder: 'candidate-model' },
        { key: 'sample_rate', label: 'Sample Rate', type: 'number', placeholder: '0.05' },
        { key: 'max_concurrency', label: 'Max Concurrency', type: 'number', placeholder: '2' },
        { key: 'max_queue_depth', label: 'Max Queue Depth', type: 'number', placeholder: '8' },
        { key: 'timeout_seconds', label: 'Timeout Seconds', type: 'number', placeholder: '30' },
        { key: 'max_response_bytes', label: 'Max Response Bytes', type: 'number', placeholder: '1048576' },
        { key: 'max_retries', label: 'Max Retries', type: 'number', placeholder: '0' },
        { key: 'capture_response_body', label: 'Capture Response Body', type: 'boolean' },
        { key: 'max_capture_bytes', label: 'Max Capture Bytes', type: 'number', placeholder: '4096' },
        { key: 'tls_skip_verify', label: 'Skip TLS Verification', type: 'boolean' },
      ]
    case 'context_compression':
      return [
        { key: 'enabled', label: 'Enabled', type: 'boolean' },
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: ['auto', 'always'],
        },
        {
          key: 'budget',
          label: 'Request Budget',
          type: 'object',
          fields: [
            { key: 'trigger_tokens', label: 'Trigger Tokens', type: 'string', placeholder: 'auto' },
            { key: 'target_tokens', label: 'Target Tokens', type: 'string', placeholder: 'auto' },
            {
              key: 'reserve_output_tokens',
              label: 'Reserved Output Tokens',
              type: 'string',
              placeholder: 'auto',
            },
          ],
        },
        {
          key: 'targets',
          label: 'Compression Targets',
          type: 'object',
          fields: [
            {
              key: 'tool_outputs',
              label: 'Tool Outputs',
              type: 'object',
              fields: [
                {
                  key: 'mode',
                  label: 'Mode',
                  type: 'select',
                  options: ['preserve', 'extractive', 'recoverable'],
                },
                { key: 'min_tokens', label: 'Minimum Tokens', type: 'number' },
                { key: 'target_tokens', label: 'Target Tokens', type: 'number' },
              ],
            },
            ...['history', 'rag', 'memory'].map((key) => ({
              key,
              label: key === 'rag' ? 'RAG' : key[0].toUpperCase() + key.slice(1),
              type: 'object' as const,
              fields: [
                {
                  key: 'mode',
                  label: 'Mode',
                  type: 'select' as const,
                  options: ['preserve', 'extractive', 'recoverable'],
                },
              ],
            })),
          ],
        },
        {
          key: 'scoring',
          label: 'Relevance Scoring',
          type: 'object',
          fields: [
            {
              key: 'method',
              label: 'Method',
              type: 'select',
              options: ['bm25', 'embedding', 'hybrid'],
            },
            { key: 'embedding_model_ref', label: 'Embedding Model Ref', type: 'string' },
          ],
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
            {
              key: 'header',
              label: 'Header',
              type: 'string',
              placeholder: 'x-vsr-compression-control',
            },
            { key: 'allowed', label: 'Allowed Directives', type: 'string[]' },
            { key: 'max_target_tokens', label: 'Maximum Target Tokens', type: 'number' },
          ],
        },
        {
          key: 'failure_mode',
          label: 'Failure Mode',
          type: 'select',
          options: ['fail_open', 'fail_closed'],
        },
      ]
    default:
      return null
  }
}
