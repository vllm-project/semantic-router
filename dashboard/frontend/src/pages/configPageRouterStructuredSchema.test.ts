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

  it('preserves omitted auto aliases instead of turning them into an explicit empty list', () => {
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
            embedding_config: { backend: 'candle', model_type: 'mmbert' },
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
        embeddings?: { semantic?: Record<string, unknown> }
      }
    }
    expect(embeddingPatch.model_catalog?.embeddings?.semantic).toEqual(
      expect.objectContaining({
        embedding_config: expect.objectContaining({
          backend: 'openai_compatible',
          model_type: 'remote',
        }),
        endpoint: expect.objectContaining({
          base_url: 'https://embedding.example.com/v1',
          model: 'text-embedding-3-small',
        }),
      }),
    )
  })

  it('surfaces every current canonical global capability from the generated schema', () => {
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: {
        management_api: { bind_address: '127.0.0.1', port: 8080 },
        startup_status: { store_backend: 'redis' },
        complexity: { backend: { base_url: 'http://classifier' } },
        knowledge_bases: [{ name: 'docs' }],
        admission: { default: { max_concurrency: 8 } },
      },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })

    expect(cards.map((card) => card.key)).toEqual(
      expect.arrayContaining([
        'management_api',
        'startup_status',
        'complexity',
        'knowledge_bases',
        'admission',
      ]),
    )
    expect(
      cards.find((card) => card.key === 'management_api')?.editFields.map((field) => field.name),
    ).toEqual(expect.arrayContaining(['bind_address', 'port', 'remote_exposure', 'auth']))
    expect(
      cards.find((card) => card.key === 'complexity')?.editFields.map((field) => field.name),
    ).toEqual(expect.arrayContaining(['prototype_scoring', 'backend']))

    const knowledgeBases = cards.find((card) => card.key === 'knowledge_bases')
    expect(knowledgeBases?.save({ items: [{ name: 'docs' }, { name: 'runbooks' }] })).toEqual({
      model_catalog: { kbs: [{ name: 'docs' }, { name: 'runbooks' }] },
    })
    const admission = cards.find((card) => card.key === 'admission')
    expect(admission?.save({ value: { default: { max_concurrency: 16 } } })).toEqual({
      model_catalog: { admission: { default: { max_concurrency: 16 } } },
    })

    const renderedPaths = new Set(cards.map((card) => card.path.join('.')))
    for (const section of ROUTER_CONFIG_EXTENSION.global_sections) {
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
