import { describe, expect, it } from 'vitest'

import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import type { FieldSchema } from '../lib/dslSchemas'
import { routerConfigFieldAtPath } from '../lib/routerConfigSchema'
import { buildRouterSectionCards } from './configPageRouterDefaultsSupport'
import { getRouterStructuredFieldDefinition } from './configPageRouterStructuredFields'
import {
  routerStructuredFieldSchemaPath,
  type RouterSystemKey,
} from './configPageRouterSectionCatalog'
import {
  normalizeRouterStructuredFields,
  normalizeRouterStructuredValue,
  ROUTER_STRUCTURED_FIELDS,
  type RouterStructuredSchema,
} from './configPageRouterStructuredSchema'

function expectGeneratedChildrenRendered(generated: FieldSchema, rendered: RouterStructuredSchema) {
  if (generated.type === 'object') {
    for (const child of generated.fields ?? []) {
      if (rendered.omit?.includes(child.key)) {
        expect(rendered.fields?.[child.key], `omitted field ${child.key}`).toBeUndefined()
        continue
      }
      const renderedChild = rendered.fields?.[child.key]
      expect(renderedChild, `missing generated field ${child.key}`).toBeDefined()
      expectGeneratedChildrenRendered(child, renderedChild!)
    }
  }
  if (generated.type === 'object[]' && generated.fields) {
    expect(rendered.item).toBeDefined()
    expectGeneratedChildrenRendered(
      { key: generated.key, label: generated.label, type: 'object', fields: generated.fields },
      rendered.item!,
    )
  }
}

describe('router defaults structured schemas', () => {
  it('edits global Replay capture defaults without losing explicit false or zero values', () => {
    const replay = {
      enabled: true,
      store_backend: 'memory',
      capture_request_body: false,
      capture_response_body: true,
      capture_personal_data: false,
      capture_tool_results: false,
      max_body_bytes: 0,
      max_tool_trace_steps: 0,
    }
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: { router_replay: replay },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })
    const card = cards.find((item) => item.key === 'router_replay')!
    expect(card.editFields.map((field) => field.name)).toEqual(
      expect.arrayContaining([
        'capture_request_body',
        'capture_response_body',
        'capture_personal_data',
        'capture_tool_results',
      ]),
    )
    expect(card.save(card.editData)).toMatchObject({
      services: { router_replay: replay },
    })
  })

  it('normalizes typed lists and objects while preserving advanced keys', () => {
    const normalized = normalizeRouterStructuredFields('router_core', {
      list_backend_models: true,
      streamed_body: {
        enabled: true,
        max_bytes: 1024,
        timeout_sec: 10,
        future_limit: 7,
      },
      skip_processing: { enabled: true },
    })

    expect(normalized.list_backend_models).toBe(true)
    expect(normalized.streamed_body).toEqual({
      enabled: true,
      max_bytes: 1024,
      timeout_sec: 10,
      future_limit: 7,
    })
    expect(normalized.skip_processing).toEqual({ enabled: true })
  })

  it('does not reintroduce removed automatic model fields', () => {
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: { router_core: { strategy: 'priority' } },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })
    const routerCore = cards.find((card) => card.key === 'router_core')
    const patch = routerCore?.save(routerCore.editData) as {
      router?: Record<string, unknown>
    }

    expect(patch.router).not.toHaveProperty('auto_model_names')
  })

  it('shows deprecated external models without offering to declare more', () => {
    const external = [
      {
        name: 'external-safety',
        llm_provider: '',
        model_role: 'classification',
        llm_endpoint: { address: 'safety-service', port: 8080, protocol: 'http' },
      },
      { name: 'external-hazard', model_role: 'classification' },
      { name: 'external-classifier', llm_provider: 'openai', model_role: 'classification' },
    ]
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: { external_models: external },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })
    const externalModels = cards.find((card) => card.key === 'external_models')!

    expect(externalModels.summary).toContainEqual({ label: 'Configured models', value: '3' })
    expect(externalModels.deprecation).toMatch(/classify, openai_chat or openai_embeddings/)
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
    const aliases = ROUTER_STRUCTURED_FIELDS.prompt_compression?.skip_signals.schema
    const streamedBody = ROUTER_STRUCTURED_FIELDS.router_core?.streamed_body.schema
    expect(aliases).toBeDefined()
    expect(streamedBody).toBeDefined()
    expect(() => normalizeRouterStructuredValue(aliases!, ['jailbreak', 'jailbreak'])).toThrow(
      /unique/i,
    )
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
      },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })

    const routerCore = cards.find((card) => card.key === 'router_core')
    expect(routerCore?.editFields.find((field) => field.name === 'list_backend_models')?.type).toBe(
      'boolean',
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

    const embeddingCards = buildRouterSectionCards({
      config: null,
      routerConfig: {
        embedding_models: {
          semantic: {
            mmbert_model_path: 'models/mmbert-embed-32k-2d-matryoshka',
            embedding_config: { backend: 'model_runtime', model_type: 'mmbert' },
          },
        },
      },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })
    const embeddingCard = embeddingCards.find((card) => card.key === 'embedding_models')
    expect(embeddingCard?.editFields.find((field) => field.name === 'provider_type')).toEqual(
      expect.objectContaining({ type: 'select', required: true }),
    )
    expect(embeddingCard?.editFields.find((field) => field.name === 'remote_backend')).toEqual(
      expect.objectContaining({ type: 'select', options: ['openai_compatible'] }),
    )
    expect(embeddingCard?.editFields.find((field) => field.name === 'endpoint')?.type).toBe(
      'custom',
    )
    const embeddingPatch = embeddingCard?.save({
      ...embeddingCard.editData,
      provider_type: 'remote',
      remote_backend: 'openai_compatible',
      endpoint: {
        base_url: 'https://embedding.example.com/v1',
        model: 'text-embedding-3-small',
        api_key_env: 'OPENAI_API_KEY',
        dimensions: 1536,
      },
      embedding_config: { target_dimension: 1536 },
    }) as {
      model_catalog?: {
        deployments?: Record<string, Record<string, unknown>>
        embeddings?: { default?: string; semantic?: Record<string, unknown> }
      }
    }
    expect(embeddingPatch.model_catalog?.deployments?.['remote-embed']).toEqual(
      expect.objectContaining({
        provider: 'openai_embeddings',
        endpoint: 'https://embedding.example.com/v1',
        model: 'text-embedding-3-small',
        auth: { api_key_env: 'OPENAI_API_KEY' },
        capabilities: { dimensions: 1536 },
      }),
    )
    expect(embeddingPatch.model_catalog?.embeddings?.default).toBe('remote-embed')
    expect(embeddingPatch.model_catalog?.embeddings?.semantic).toEqual(
      expect.objectContaining({
        embedding_config: expect.objectContaining({ backend: null, model_type: null }),
        endpoint: null,
      }),
    )
  })

  it('surfaces canonical global capabilities while leaving KBs to the raw config', () => {
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: {
        management_api: { bind_address: '127.0.0.1', port: 8080 },
        startup_status: { store_backend: 'redis' },
        admission: { default: { max_concurrency: 8 } },
      },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })

    expect(cards.map((card) => card.key)).toEqual(
      expect.arrayContaining(['management_api', 'startup_status', 'admission']),
    )
    // The decision model answers complexity rules; no module card configures it.
    expect(cards.map((card) => card.key)).not.toContain('complexity')
    expect(
      cards.find((card) => card.key === 'management_api')?.editFields.map((field) => field.name),
    ).toEqual(expect.arrayContaining(['bind_address', 'port', 'remote_exposure', 'auth']))

    const admission = cards.find((card) => card.key === 'admission')
    expect(admission?.save({ value: { default: { max_concurrency: 16 } } })).toEqual({
      model_catalog: { admission: { default: { max_concurrency: 16 } } },
    })

    const renderedPaths = new Set(cards.map((card) => card.path.join('.')))
    expect(renderedPaths).not.toContain('model_catalog.kbs')
    expect(renderedPaths).not.toContain('model_catalog.bindings')
    expect(
      cards.find((card) => card.key === 'system_models')?.editFields.map((field) => field.name),
    ).toContain('binding:domain')
    for (const section of ROUTER_CONFIG_EXTENSION.global_sections) {
      if (['model_catalog.kbs', 'model_catalog.bindings'].includes(section.path.join('.'))) continue
      expect(renderedPaths).toContain(section.path.join('.'))
    }
  })

  it('recursively augments specialized controls with every generated nested field', () => {
    for (const [key, fields] of Object.entries(ROUTER_STRUCTURED_FIELDS)) {
      for (const name of Object.keys(fields ?? {})) {
        const generated = routerConfigFieldAtPath(
          routerStructuredFieldSchemaPath(key as RouterSystemKey, name),
        )
        if (!generated) continue
        const rendered = getRouterStructuredFieldDefinition(key as RouterSystemKey, name).schema
        expectGeneratedChildrenRendered(generated, rendered)
      }
    }
  })
})
