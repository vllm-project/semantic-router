import {
  boolean,
  number,
  object,
  text,
  type RouterStructuredFieldDefinition,
} from './configPageRouterStructuredSchemaPrimitives'

export const EMBEDDING_MODELS_STRUCTURED_FIELDS: Record<string, RouterStructuredFieldDefinition> = {
  embedding_config: {
    label: 'Embedding Optimization',
    description:
      'Layer, dimension, preload, and soft-matching controls shared by embedding consumers.',
    schema: object('Embedding Optimization', {
      preload_embeddings: boolean('Preload Embeddings'),
      target_dimension: number('Target Dimension', { min: 1 }),
      target_layer: number('Target Layer', { min: 0 }),
      enable_soft_matching: boolean('Enable Soft Matching'),
      top_k: number('Top K', { min: 1 }),
      min_score_threshold: number('Min Score Threshold', { min: 0, max: 1, step: 0.01 }),
    }),
  },
  endpoint: {
    label: 'Remote Endpoint',
    description:
      'OpenAI-compatible embedding endpoint. Credentials are resolved from the named environment variable.',
    schema: object('Remote Endpoint', {
      base_url: text('Base URL', {
        required: true,
        placeholder: 'https://embedding.example.com/v1',
      }),
      model: text('Model', { required: true, placeholder: 'text-embedding-3-small' }),
      api_key_env: text('API Key Environment Variable', { placeholder: 'OPENAI_API_KEY' }),
      timeout_seconds: number('Timeout Seconds', { min: 0 }),
      max_retries: number('Max Retries', { min: 0 }),
      dimensions: number('Dimensions', { min: 1 }),
    }),
  },
}
