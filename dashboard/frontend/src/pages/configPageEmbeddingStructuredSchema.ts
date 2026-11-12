import {
  number,
  object,
  text,
  type RouterStructuredFieldDefinition,
} from './configPageRouterStructuredSchemaPrimitives'

// A store indexed with a model other than embeddings.default names that
// deployment; the dimension is the width its vectors carry.
export const STORE_EMBEDDING_FIELD: RouterStructuredFieldDefinition = {
  label: 'Embedding',
  description:
    'Deployment that embeds this store, when it was indexed with another model than embeddings.default, and the dimension of its vectors.',
  schema: object('Embedding', {
    deployment: text('Deployment', { placeholder: 'vela-embedding' }),
    dimension: number('Dimension', { min: 1 }),
  }),
}

export const EMBEDDING_MODELS_STRUCTURED_FIELDS: Record<string, RouterStructuredFieldDefinition> = {
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
      max_response_bytes: number('Max Response Bytes', { min: 0 }),
      dimensions: number('Dimensions', { min: 1 }),
    }),
  },
}
