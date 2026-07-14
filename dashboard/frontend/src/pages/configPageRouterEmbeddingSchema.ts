import {
  boolean,
  number,
  object,
  select,
  text,
  type RouterStructuredFieldDefinition,
} from './configPageRouterStructuredSchemaPrimitives'

// This is a credential binding name, not a credential. Keeping one dedicated
// source prevents a dashboard-authored endpoint from repurposing unrelated
// process secrets as outbound provider credentials.
export const REMOTE_EMBEDDING_API_KEY_ENV = 'VLLM_SR_EMBEDDING_API_KEY'

export const embeddingRouterStructuredFields: Record<string, RouterStructuredFieldDefinition> = {
  embedding_config: {
    label: 'Embedding Optimization',
    description: 'Embedding model, layer, dimension, and soft-matching controls.',
    schema: object('Embedding Optimization', {
      backend: select('Backend', ['candle', 'openvino', 'openai_compatible']),
      model_type: text('Model Type'),
      preload_embeddings: boolean('Preload Embeddings'),
      target_dimension: number('Target Dimension', { min: 1 }),
      target_layer: number('Target Layer', { min: 0 }),
      enable_soft_matching: boolean('Enable Soft Matching'),
      top_k: number('Top K', { min: 1 }),
      min_score_threshold: number('Min Score Threshold', { min: 0, max: 1, step: 0.01 }),
    }),
  },
  endpoint: {
    label: 'Remote Embedding Endpoint',
    description: `OpenAI-compatible embedding endpoint. Credentials are read only from ${REMOTE_EMBEDDING_API_KEY_ENV}, never from config.`,
    schema: {
      ...object('Remote Embedding Endpoint', {
        base_url: text('Base URL', { required: true }),
        model: text('Model', { required: true }),
        api_key_env: {
          ...select('API Key Environment Variable', [REMOTE_EMBEDDING_API_KEY_ENV]),
          description: 'Leave Not set for endpoints that do not require an API key.',
        },
        timeout_seconds: number('Timeout Seconds', { min: 0, max: 3600 }),
        max_retries: number('Max Retries', { min: 0, max: 10 }),
        dimensions: number('Dimensions', { min: 1 }),
      }),
      removable: true,
      removeLabel: 'Remove endpoint',
    },
  },
}
