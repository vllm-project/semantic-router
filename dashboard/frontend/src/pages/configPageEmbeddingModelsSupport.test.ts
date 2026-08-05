import { describe, expect, it } from 'vitest'

import {
  embeddingModelsBadges,
  embeddingModelsCatalogValue,
  embeddingModelsEditData,
  embeddingModelsFields,
  embeddingModelsSummary,
} from './configPageEmbeddingModelsSupport'

const localCatalog = {
  semantic: {
    mmbert_model_path: 'models/mmbert-embed-32k-2d-matryoshka',
    use_cpu: true,
    embedding_config: {
      backend: 'candle',
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
        embedding_config: { backend: 'candle', model_type: 'remote' },
      },
    })

    expect(editData.provider_type).toBe('local')
    expect(editData.local_backend).toBe('candle')
  })

  it('shows provider-specific fields for local and remote modes', () => {
    const fields = embeddingModelsFields()
    const localPath = fields.find((field) => field.name === 'mmbert_model_path')
    const endpoint = fields.find((field) => field.name === 'endpoint')
    const localBackend = fields.find((field) => field.name === 'local_backend')
    const apiProtocol = fields.find((field) => field.name === 'remote_backend')

    expect(localPath?.shouldHide?.({ provider_type: 'local' })).toBe(false)
    expect(localBackend?.shouldHide?.({ provider_type: 'local' })).toBe(false)
    expect(endpoint?.shouldHide?.({ provider_type: 'local' })).toBe(true)
    expect(apiProtocol?.shouldHide?.({ provider_type: 'local' })).toBe(true)
    expect(localPath?.shouldHide?.({ provider_type: 'remote' })).toBe(true)
    expect(localBackend?.shouldHide?.({ provider_type: 'remote' })).toBe(true)
    expect(endpoint?.shouldHide?.({ provider_type: 'remote' })).toBe(false)
    expect(apiProtocol?.shouldHide?.({ provider_type: 'remote' })).toBe(false)
  })

  it('switches local config to a canonical remote provider without losing extensions', () => {
    const editData = embeddingModelsEditData(localCatalog)
    const saved = embeddingModelsCatalogValue({
      ...editData,
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: {
        ...(editData.endpoint as Record<string, unknown>),
        base_url: ' https://embedding.example.com/v1 ',
        model: ' text-embedding-3-small ',
        api_key_env: ' OPENAI_API_KEY ',
      },
      embedding_config: {
        ...(editData.embedding_config as Record<string, unknown>),
        target_dimension: 1536,
      },
    })

    expect(saved).toEqual(
      expect.objectContaining({
        future_catalog_option: { enabled: true },
        semantic: expect.objectContaining({
          future_semantic_option: true,
          embedding_config: expect.objectContaining({
            backend: 'openai_compatible',
            model_type: 'remote',
            target_dimension: 1536,
            future_optimizer: 'preserved',
          }),
          endpoint: expect.objectContaining({
            base_url: 'https://embedding.example.com/v1',
            model: 'text-embedding-3-small',
            api_key_env: 'OPENAI_API_KEY',
            dimensions: 1536,
            future_header_mode: 'preserved',
          }),
        }),
      }),
    )
  })

  it('switches back to local inference while preserving the remote endpoint for reuse', () => {
    const remote = embeddingModelsCatalogValue({
      ...embeddingModelsEditData(localCatalog),
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: localCatalog.semantic.endpoint as Record<string, unknown>,
      embedding_config: { target_dimension: 1536 },
    })
    const local = embeddingModelsCatalogValue({
      ...embeddingModelsEditData(remote),
      provider_type: 'local',
      local_backend: 'openvino',
      model_type: 'mmbert',
    })

    expect((local.semantic as Record<string, unknown>).embedding_config).toEqual(
      expect.objectContaining({ backend: 'openvino', model_type: 'mmbert' }),
    )
    expect((local.semantic as Record<string, unknown>).endpoint).toEqual(
      expect.objectContaining({
        base_url: 'https://embedding.example.com/v1',
        model: 'text-embedding-3-small',
      }),
    )
  })

  it('rejects remote dimensions that disagree with the shared target dimension', () => {
    expect(() =>
      embeddingModelsCatalogValue({
        ...embeddingModelsEditData(localCatalog),
        provider_type: 'remote',
        remote_backend: 'openai_compatible',
        endpoint: { base_url: 'https://example.com/v1', model: 'embed', dimensions: 1536 },
        embedding_config: { target_dimension: 768 },
      }),
    ).toThrow(/must match/i)
  })

  it('summarizes provider mode and remote model without exposing credentials', () => {
    const remote = embeddingModelsCatalogValue({
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
    })

    expect(embeddingModelsSummary(remote)).toEqual([
      { label: 'Provider', value: 'Remote / OpenAI compatible' },
      { label: 'Model', value: 'text-embedding-3-small' },
      { label: 'Dimension', value: '1536' },
    ])
    expect(embeddingModelsBadges(remote)).toContainEqual({ label: 'Remote provider', tone: 'info' })
    expect(JSON.stringify(embeddingModelsSummary(remote))).not.toContain('OPENAI_API_KEY')
  })
})
