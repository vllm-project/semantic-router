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
  embeddings: Record<string, unknown>
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
  default: { deployment: 'qwen3-embedding' },
  future_catalog_option: { enabled: true },
}

const remoteEndpoint = {
  base_url: 'https://embedding.example.com/v1',
  model: 'text-embedding-3-small',
  dimensions: 1536,
}

describe('embedding models config support', () => {
  it('edits the deployment embeddings.default names, as {deployment}', () => {
    const editData = embeddingModelsEditData(localCatalog)
    expect(editData).toEqual(expect.objectContaining({ provider_type: 'local', default: 'qwen3-embedding' }))

    const saved = save({ ...editData, default: ' bge-local ' })
    expect(saved.deployments).toBeUndefined()
    expect(saved.embeddings).toEqual({
      default: { deployment: 'bge-local' },
      future_catalog_option: { enabled: true },
    })
  })

  it('offers deployments, not embedding model keywords, paths or a multimodal deployment', () => {
    const names = embeddingModelsFields().map((field) => field.name)

    expect(names).toEqual(['provider_type', 'remote_backend', 'default', 'endpoint'])
    for (const removed of ['model_type', 'mmbert_model_path', 'qwen3_model_path', 'use_cpu', 'embedding_config', 'multimodal']) {
      expect(names).not.toContain(removed)
    }
  })

  it('shows provider-specific fields for local and remote modes', () => {
    const fields = embeddingModelsFields()
    const deployment = fields.find((field) => field.name === 'default')
    const endpoint = fields.find((field) => field.name === 'endpoint')
    const apiProtocol = fields.find((field) => field.name === 'remote_backend')

    expect(deployment?.shouldHide?.({ provider_type: 'local' })).toBe(false)
    expect(endpoint?.shouldHide?.({ provider_type: 'local' })).toBe(true)
    expect(apiProtocol?.shouldHide?.({ provider_type: 'local' })).toBe(true)
    expect(deployment?.shouldHide?.({ provider_type: 'remote' })).toBe(true)
    expect(endpoint?.shouldHide?.({ provider_type: 'remote' })).toBe(false)
    expect(apiProtocol?.shouldHide?.({ provider_type: 'remote' })).toBe(false)
  })

  it('saves a remote provider as an openai_embeddings deployment that embeddings.default names', () => {
    const saved = save({
      ...embeddingModelsEditData(localCatalog),
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: {
        base_url: ' https://embedding.example.com/v1/ ',
        model: ' text-embedding-3-small ',
        api_key_env: ' OPENAI_API_KEY ',
        timeout_seconds: 5,
        max_retries: 2,
        dimensions: 1536,
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
    expect(saved.embeddings).toEqual({
      default: { deployment: 'remote-embed' },
      future_catalog_option: { enabled: true },
    })
  })

  it('edits the deployment embeddings.default names', () => {
    const view = embeddingModelsView(
      { default: { deployment: 'bge' } },
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
    expect(saved.embeddings.default).toEqual({ deployment: 'bge' })
    expect(saved.embeddings).not.toHaveProperty('__deployments')
  })

  it('keeps the auth fields it does not edit, such as a delegated credential', () => {
    const view = embeddingModelsView(
      { default: { deployment: 'bge' } },
      {
        bge: {
          provider: 'openai_embeddings',
          endpoint: 'https://embedding.example.com/v1',
          model: 'BAAI/bge-m3',
          auth: { credential: 'delegated', header: 'Authorization' },
        },
      },
    )
    const saved = save(embeddingModelsEditData(view))

    // The save is a merge patch: a null removes only the key the card edits.
    expect(saved.deployments?.bge.auth).toEqual({ api_key_env: null })
  })

  it('switches back to local inference and keeps the remote deployment for reuse', () => {
    const remote = save({
      ...embeddingModelsEditData(localCatalog),
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: remoteEndpoint,
    })
    const local = save({ ...embeddingModelsEditData(afterSave(remote)), provider_type: 'local' })

    expect(local.deployments).toBeUndefined()
    expect(local.embeddings.default).toBeNull()
  })

  it('requires the remote endpoint and model', () => {
    expect(() =>
      embeddingModelsPatch({
        provider_type: 'remote',
        endpoint: { base_url: 'https://embedding.example.com/v1' },
      }),
    ).toThrow(/Remote Endpoint\.model is required/i)
  })

  it('summarizes the default deployment, and a remote model without credentials', () => {
    expect(embeddingModelsSummary({})).toEqual([
      { label: 'Provider', value: 'Local / model_runtime' },
      { label: 'Deployment', value: 'vela-embedding' },
      { label: 'Model', value: 'Vela 1.0 Embedding' },
    ])

    const remote = afterSave(
      save({
        ...embeddingModelsEditData(localCatalog),
        provider_type: 'remote',
        remote_backend: 'openai_compatible',
        endpoint: { ...remoteEndpoint, api_key_env: 'OPENAI_API_KEY' },
      }),
    )
    expect(embeddingModelsSummary(remote)).toEqual([
      { label: 'Provider', value: 'Remote / OpenAI compatible' },
      { label: 'Model', value: 'text-embedding-3-small' },
      { label: 'Dimension', value: '1536' },
    ])
    expect(embeddingModelsBadges(remote)).toContainEqual({ label: 'Remote provider', tone: 'info' })
    expect(JSON.stringify(embeddingModelsSummary(remote))).not.toContain('OPENAI_API_KEY')
  })

  it('reads a default binding written as a deployment name', () => {
    expect(
      embeddingModelsSummary({
        default: 'qwen3-embedding',
        __deployments: {
          'qwen3-embedding': { provider: 'model_runtime', artifact: 'Qwen/Qwen3-Embedding-0.6B' },
        },
      }),
    ).toEqual([
      { label: 'Provider', value: 'Local / model_runtime' },
      { label: 'Deployment', value: 'qwen3-embedding' },
      { label: 'Model', value: 'Qwen/Qwen3-Embedding-0.6B' },
    ])
  })
})
