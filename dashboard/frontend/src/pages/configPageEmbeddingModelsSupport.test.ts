import { describe, expect, it } from 'vitest'

import {
  embeddingModelsBadges,
  embeddingModelsEditData,
  embeddingModelsFields,
  embeddingModelsPatch,
  embeddingModelsSummary,
  embeddingModelsView,
} from './configPageEmbeddingModelsSupport'

type Patch = {
  deployments?: Record<string, Record<string, unknown>>
  embeddings: Record<string, unknown> & { semantic: Record<string, unknown> }
}

const save = (data: Parameters<typeof embeddingModelsPatch>[0]) => embeddingModelsPatch(data) as Patch

// The card's view after a save: the saved embeddings block with the saved
// deployments, nulls dropped as the Router's merge patch drops them.
function afterSave(patch: Patch): Record<string, unknown> | undefined {
  const dropNulls = (value: unknown): unknown =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(
          Object.entries(value)
            .filter(([, item]) => item !== null)
            .map(([key, item]) => [key, dropNulls(item)]),
        )
      : value
  return embeddingModelsView(dropNulls(patch.embeddings), dropNulls(patch.deployments ?? {}))
}

const localCatalog = {
  semantic: {
    mmbert_model_path: 'models/mmbert-embed-32k-2d-matryoshka',
    use_cpu: true,
    embedding_config: {
      backend: 'model_runtime',
      model_type: 'mmbert',
      target_dimension: 768,
      future_optimizer: 'preserved',
    },
    endpoint: {
      base_url: 'https://embedding.example.com/v1',
      model: 'text-embedding-3-small',
      dimensions: 1536,
      future_header_mode: 'preserved',
    },
    future_semantic_option: true,
  },
  future_catalog_option: { enabled: true },
}

describe('embedding models config support', () => {
  it('gives an explicit local backend precedence over legacy model_type inference', () => {
    const editData = embeddingModelsEditData({
      semantic: {
        embedding_config: { backend: 'model_runtime', model_type: 'remote' },
      },
    })

    expect(editData.provider_type).toBe('local')
  })

  it('refuses to save a retired local backend instead of migrating it', () => {
    const editData = embeddingModelsEditData({
      semantic: { embedding_config: { backend: 'candle', model_type: 'mmbert' } },
    })

    expect(editData.provider_type).toBe('local')
    expect(() => embeddingModelsPatch(editData)).toThrow(/candle is retired.*vllm-sr config migrate/)
  })

  it('offers only the embedding families the model runtime serves', () => {
    const names = embeddingModelsFields().map((field) => field.name)

    expect(names).toEqual(expect.arrayContaining(['qwen3_model_path', 'mmbert_model_path', 'multimodal_model_path']))
    expect(names).not.toContain('gemma_model_path')
    expect(names).not.toContain('bert_model_path')
    expect(save(embeddingModelsEditData(localCatalog)).embeddings).not.toHaveProperty('bert')
  })

  it('shows provider-specific fields for local and remote modes', () => {
    const fields = embeddingModelsFields()
    const localPath = fields.find((field) => field.name === 'mmbert_model_path')
    const endpoint = fields.find((field) => field.name === 'endpoint')
    const apiProtocol = fields.find((field) => field.name === 'remote_backend')

    expect(localPath?.shouldHide?.({ provider_type: 'local' })).toBe(false)
    expect(endpoint?.shouldHide?.({ provider_type: 'local' })).toBe(true)
    expect(apiProtocol?.shouldHide?.({ provider_type: 'local' })).toBe(true)
    expect(localPath?.shouldHide?.({ provider_type: 'remote' })).toBe(true)
    expect(endpoint?.shouldHide?.({ provider_type: 'remote' })).toBe(false)
    expect(apiProtocol?.shouldHide?.({ provider_type: 'remote' })).toBe(false)
  })

  it('saves a remote provider as an openai_embeddings deployment that embeddings.default names', () => {
    const editData = embeddingModelsEditData(localCatalog)
    const saved = save({
      ...editData,
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: {
        ...(editData.endpoint as Record<string, unknown>),
        base_url: ' https://embedding.example.com/v1/ ',
        model: ' text-embedding-3-small ',
        api_key_env: ' OPENAI_API_KEY ',
        timeout_seconds: 5,
        max_retries: 2,
      },
      embedding_config: {
        ...(editData.embedding_config as Record<string, unknown>),
        target_dimension: 1536,
      },
    })

    expect(saved.deployments).toEqual({
      'remote-embed': {
        provider: 'openai_embeddings',
        endpoint: 'https://embedding.example.com/v1',
        model: 'text-embedding-3-small',
        auth: { api_key_env: 'OPENAI_API_KEY' },
        timeout_ms: 5000,
        max_retries: 2,
        limits: null,
        capabilities: { dimensions: 1536 },
      },
    })
    expect(saved.embeddings).toEqual(
      expect.objectContaining({
        default: 'remote-embed',
        future_catalog_option: { enabled: true },
        semantic: expect.objectContaining({
          future_semantic_option: true,
          endpoint: null,
          embedding_config: expect.objectContaining({
            backend: null,
            model_type: null,
            target_dimension: 1536,
            future_optimizer: 'preserved',
          }),
        }),
      }),
    )
  })

  it('edits the deployment embeddings.default names', () => {
    const view = embeddingModelsView(
      { default: 'bge', semantic: { embedding_config: { target_dimension: 1024 } } },
      {
        bge: {
          provider: 'openai_embeddings',
          endpoint: 'https://embedding.example.com/v1',
          model: 'BAAI/bge-m3',
          auth: { api_key_env: 'EMBEDDING_API_KEY' },
          timeout_ms: 5000,
          capabilities: { dimensions: 1024 },
        },
      },
    )
    const editData = embeddingModelsEditData(view)

    expect(editData.provider_type).toBe('remote')
    expect(editData.endpoint).toEqual({
      base_url: 'https://embedding.example.com/v1',
      model: 'BAAI/bge-m3',
      api_key_env: 'EMBEDDING_API_KEY',
      timeout_seconds: 5,
      dimensions: 1024,
    })
    const saved = save(editData)
    expect(Object.keys(saved.deployments ?? {})).toEqual(['bge'])
    expect(saved.embeddings.default).toBe('bge')
    expect(saved.embeddings).not.toHaveProperty('__deployments')
  })

  it('switches back to local inference and keeps the remote deployment for reuse', () => {
    const remote = save({
      ...embeddingModelsEditData(localCatalog),
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: localCatalog.semantic.endpoint as Record<string, unknown>,
      embedding_config: { target_dimension: 1536 },
    })
    const local = save({
      ...embeddingModelsEditData(afterSave(remote)),
      provider_type: 'local',
      model_type: 'mmbert',
    })

    expect(local.deployments).toBeUndefined()
    expect(local.embeddings.default).toBeNull()
    expect(local.embeddings.semantic.endpoint).toBeNull()
    expect(local.embeddings.semantic.embedding_config).toEqual(
      expect.objectContaining({ backend: 'model_runtime', model_type: 'mmbert' }),
    )
  })

  it('rejects remote dimensions that disagree with the shared target dimension', () => {
    expect(() =>
      embeddingModelsPatch({
        ...embeddingModelsEditData(localCatalog),
        provider_type: 'remote',
        remote_backend: 'openai_compatible',
        endpoint: { base_url: 'https://example.com/v1', model: 'embed', dimensions: 1536 },
        embedding_config: { target_dimension: 768 },
      }),
    ).toThrow(/must match/i)
  })

  it('accepts the endpoint dimension when the target requests the full model dimension', () => {
    expect(() =>
      embeddingModelsPatch({
        ...embeddingModelsEditData(localCatalog),
        provider_type: 'remote',
        remote_backend: 'openai_compatible',
        endpoint: { base_url: 'https://example.com/v1', model: 'embed', dimensions: 1536 },
        embedding_config: { target_dimension: 0 },
      }),
    ).not.toThrow()
  })

  it('saves the Router default top_k of 0, which keeps every accepted match', () => {
    const saved = save(embeddingModelsEditData({ semantic: { embedding_config: { top_k: 0 } } }))

    expect(saved.embeddings.semantic.embedding_config).toEqual(
      expect.objectContaining({ top_k: 0 }),
    )
  })

  it('summarizes provider mode and remote model without exposing credentials', () => {
    const remote = afterSave(save({
      ...embeddingModelsEditData(localCatalog),
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: {
        base_url: 'https://example.com/v1',
        model: 'text-embedding-3-small',
        api_key_env: 'OPENAI_API_KEY',
        dimensions: 1536,
      },
      embedding_config: { target_dimension: 1536 },
    }))

    expect(embeddingModelsSummary(remote)).toEqual([
      { label: 'Provider', value: 'Remote / OpenAI compatible' },
      { label: 'Model', value: 'text-embedding-3-small' },
      { label: 'Dimension', value: '1536' },
    ])
    expect(embeddingModelsBadges(remote)).toContainEqual({ label: 'Remote provider', tone: 'info' })
    expect(JSON.stringify(embeddingModelsSummary(remote))).not.toContain('OPENAI_API_KEY')
  })
})
