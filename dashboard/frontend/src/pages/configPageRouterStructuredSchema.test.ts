import { describe, expect, it } from 'vitest'

import { buildRouterSectionCards } from './configPageRouterDefaultsSupport'
import {
  normalizeRouterStructuredFields,
  normalizeRouterStructuredValue,
  REMOTE_EMBEDDING_API_KEY_ENV,
  ROUTER_STRUCTURED_FIELDS,
} from './configPageRouterStructuredSchema'

describe('router defaults structured schemas', () => {
  it('normalizes typed lists and objects while preserving advanced keys', () => {
    const normalized = normalizeRouterStructuredFields('router_core', {
      auto_model_names: [' vllm-sr/auto ', 'MoM'],
      streamed_body: {
        enabled: true,
        max_bytes: 1024,
        timeout_sec: 10,
        future_limit: 7,
      },
      skip_processing: { enabled: true },
    })

    expect(normalized.auto_model_names).toEqual(['vllm-sr/auto', 'MoM'])
    expect(normalized.streamed_body).toEqual({
      enabled: true,
      max_bytes: 1024,
      timeout_sec: 10,
      future_limit: 7,
    })
    expect(normalized.skip_processing).toEqual({ enabled: true })
  })

  it('round-trips nested provider and rule object arrays', () => {
    const normalized = normalizeRouterStructuredFields('ratelimit', {
      fail_open: false,
      providers: [
        {
          type: 'redis',
          address: 'redis:6379',
          future_provider_option: 'preserved',
          rules: [
            {
              name: 'premium',
              match: { group: 'premium-tier', future_match: true },
              requests_per_unit: 120,
              unit: 'minute',
            },
          ],
        },
      ],
    })

    expect(normalized.providers).toEqual([
      expect.objectContaining({
        future_provider_option: 'preserved',
        rules: [
          expect.objectContaining({
            name: 'premium',
            match: { group: 'premium-tier', future_match: true },
          }),
        ],
      }),
    ])
  })

  it('round-trips typed tool filtering and classifier modules with unknown keys', () => {
    const tools = normalizeRouterStructuredFields('tools', {
      advanced_filtering: {
        enabled: true,
        retrieval_strategy: 'hybrid_history',
        allow_tools: [' docs.search '],
        hybrid_history: {
          history_horizon: 8,
          future_history_weight: 0.25,
        },
        future_retriever: 'preserved',
      },
    })
    expect(tools.advanced_filtering).toEqual(
      expect.objectContaining({
        allow_tools: ['docs.search'],
        future_retriever: 'preserved',
        hybrid_history: {
          history_horizon: 8,
          future_history_weight: 0.25,
        },
      }),
    )

    const classifier = normalizeRouterStructuredFields('classifier', {
      preference: {
        use_contrastive: true,
        prototype_scoring: {
          enabled: true,
          max_prototypes: 8,
          future_bank_mode: 'adaptive',
        },
      },
    })
    expect(classifier.preference).toEqual(
      expect.objectContaining({
        prototype_scoring: {
          enabled: true,
          max_prototypes: 8,
          future_bank_mode: 'adaptive',
        },
      }),
    )
  })

  it('rejects duplicate list values and invalid typed numbers', () => {
    const aliases = ROUTER_STRUCTURED_FIELDS.router_core?.auto_model_names.schema
    const streamedBody = ROUTER_STRUCTURED_FIELDS.router_core?.streamed_body.schema
    expect(aliases).toBeDefined()
    expect(streamedBody).toBeDefined()
    expect(() => normalizeRouterStructuredValue(aliases!, ['auto', 'AUTO'])).toThrow(/unique/i)
    expect(() =>
      normalizeRouterStructuredValue(streamedBody!, { enabled: true, max_bytes: 0 }),
    ).toThrow(/at least 1/i)
  })

  it('builds custom controls for every migrated field and preserves model-selection extensions', () => {
    const selection = {
      enabled: true,
      method: 'hybrid',
      momentum: { enabled: true, attack: 0.7 },
      ml: {
        models_path: 'models/selection',
        knn: { k: 5 },
        tuning: { switch_margin: 0.05 },
      },
    }
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: {
        model_selection: selection,
        classifier: {
          preference: {
            use_contrastive: true,
            prototype_scoring: { enabled: true, future_bank_mode: 'adaptive' },
          },
          future_classifier_module: { enabled: true },
        },
        embedding_models: {
          semantic: {
            future_semantic_option: 'preserved',
            embedding_config: { backend: 'openai_compatible', model_type: 'remote' },
            endpoint: {
              base_url: 'https://embeddings.example/v1',
              model: 'embedding-model',
              api_key_env: REMOTE_EMBEDDING_API_KEY_ENV,
              timeout_seconds: 30,
              max_retries: 2,
              dimensions: 1024,
              future_transport_option: 'preserved',
            },
          },
        },
      },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })

    const routerCore = cards.find((card) => card.key === 'router_core')
    expect(routerCore?.editFields.find((field) => field.name === 'auto_model_names')?.type).toBe(
      'custom',
    )
    const selectionCard = cards.find((card) => card.key === 'model_selection')
    expect(
      selectionCard?.editFields
        .filter((field) => ['knn', 'router_dc', 'hybrid'].includes(field.name))
        .every((field) => field.type === 'custom'),
    ).toBe(true)

    const remainingJsonFields = cards.flatMap((card) =>
      card.editFields
        .filter((field) => (field.type as string) === 'json')
        .map((field) => `${card.key}.${field.name}`),
    )
    expect(remainingJsonFields).toEqual([])

    const embeddingsCard = cards.find((card) => card.key === 'embedding_models')
    expect(embeddingsCard?.editFields.find((field) => field.name === 'endpoint')?.type).toBe(
      'custom',
    )
    const embeddingPatch = embeddingsCard?.save(embeddingsCard.editData) as {
      model_catalog?: {
        embeddings?: { semantic?: Record<string, unknown> }
      }
    }
    expect(embeddingPatch.model_catalog?.embeddings?.semantic).toEqual(
      expect.objectContaining({
        embedding_config: { backend: 'openai_compatible', model_type: 'remote' },
        endpoint: expect.objectContaining({
          base_url: 'https://embeddings.example/v1',
          model: 'embedding-model',
          api_key_env: REMOTE_EMBEDDING_API_KEY_ENV,
          future_transport_option: 'preserved',
        }),
      }),
    )

    const endpointSchema = ROUTER_STRUCTURED_FIELDS.embedding_models?.endpoint.schema
    expect(endpointSchema?.fields?.api_key_env?.options).toEqual([REMOTE_EMBEDDING_API_KEY_ENV])
    expect(endpointSchema?.removable).toBe(true)
    const clearedEndpoint = {
      ...(embeddingsCard?.editData.endpoint as Record<string, unknown>),
    }
    delete clearedEndpoint.timeout_seconds
    delete clearedEndpoint.max_retries
    delete clearedEndpoint.dimensions
    const clearedEmbeddingPatch = embeddingsCard?.save({
      ...embeddingsCard.editData,
      endpoint: clearedEndpoint,
    }) as {
      model_catalog?: {
        embeddings?: { semantic?: { endpoint?: Record<string, unknown> } }
      }
    }
    expect(clearedEmbeddingPatch.model_catalog?.embeddings?.semantic?.endpoint).toEqual(
      expect.objectContaining({
        api_key_env: REMOTE_EMBEDDING_API_KEY_ENV,
        timeout_seconds: null,
        max_retries: null,
        dimensions: null,
        future_transport_option: 'preserved',
      }),
    )
    const arbitraryCredentialPatch = embeddingsCard?.save({
      ...embeddingsCard.editData,
      endpoint: {
        ...(embeddingsCard.editData.endpoint as Record<string, unknown>),
        api_key_env: 'ARBITRARY_PROCESS_SECRET',
      },
    }) as {
      model_catalog?: {
        embeddings?: { semantic?: { endpoint?: Record<string, unknown> } }
      }
    }
    expect(
      arbitraryCredentialPatch.model_catalog?.embeddings?.semantic?.endpoint?.api_key_env,
    ).toBeNull()
    const removedEndpointPatch = embeddingsCard?.save({
      ...embeddingsCard.editData,
      embedding_config: { backend: 'candle', model_type: 'mmbert' },
      endpoint: undefined,
    }) as {
      model_catalog?: {
        embeddings?: { semantic?: Record<string, unknown> }
      }
    }
    expect(removedEndpointPatch.model_catalog?.embeddings?.semantic).toEqual(
      expect.objectContaining({
        future_semantic_option: 'preserved',
        embedding_config: { backend: 'candle', model_type: 'mmbert' },
        endpoint: null,
      }),
    )

    const patch = selectionCard?.save({
      ...selectionCard.editData,
      knn: { k: 9 },
    }) as { router?: { model_selection?: Record<string, unknown> } }
    const saved = patch.router?.model_selection
    expect(saved?.momentum).toEqual({ enabled: true, attack: 0.7 })
    expect(saved?.ml).toEqual(
      expect.objectContaining({
        models_path: 'models/selection',
        knn: { k: 9 },
        tuning: { switch_margin: 0.05 },
      }),
    )

    const classifierCard = cards.find((card) => card.key === 'classifier')
    const classifierPatch = classifierCard?.save(classifierCard.editData) as {
      model_catalog?: { modules?: { classifier?: Record<string, unknown> } }
    }
    expect(classifierPatch.model_catalog?.modules?.classifier).toEqual(
      expect.objectContaining({
        future_classifier_module: { enabled: true },
        preference: expect.objectContaining({
          prototype_scoring: expect.objectContaining({ future_bank_mode: 'adaptive' }),
        }),
      }),
    )
  })

  it('preserves an unauthenticated HTTP embedding endpoint without adding a credential', () => {
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: {
        embedding_models: {
          semantic: {
            embedding_config: { backend: 'openai_compatible', model_type: 'remote' },
            endpoint: {
              base_url: 'http://embeddings.internal/v1',
              model: 'internal-embedding-model',
              future_transport_option: 'preserved',
            },
          },
        },
      },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })
    const embeddingsCard = cards.find((card) => card.key === 'embedding_models')
    const patch = embeddingsCard?.save(embeddingsCard.editData) as {
      model_catalog?: {
        embeddings?: { semantic?: { endpoint?: Record<string, unknown> } }
      }
    }
    const endpoint = patch.model_catalog?.embeddings?.semantic?.endpoint

    expect(endpoint).toEqual(
      expect.objectContaining({
        base_url: 'http://embeddings.internal/v1',
        model: 'internal-embedding-model',
        future_transport_option: 'preserved',
      }),
    )
    expect(endpoint).not.toHaveProperty('api_key_env')
  })
})
