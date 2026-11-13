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
  it('offers no Replay, memory or response cache switch, which v0.5 reads from global.plugins', () => {
    const replay = { store_backend: 'memory', ttl_seconds: 3600, async_writes: false }
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: { router_replay: replay },
      routerDefaults: null,
    })
    const fields = (key: string) =>
      cards.find((item) => item.key === key)!.editFields.map((field) => field.name)
    for (const [key, removed] of [
      ['router_replay', 'enabled'],
      ['router_replay', 'capture_request_body'],
      ['router_replay', 'capture_response_body'],
      ['router_replay', 'prompt_capture'],
      ['router_replay', 'capture_personal_data'],
      ['router_replay', 'capture_tool_results'],
      ['memory', 'enabled'],
      ['response_cache', 'enabled'],
      ['response_cache', 'similarity_threshold'],
    ]) {
      expect(fields(key), `${key}.${removed}`).not.toContain(removed)
    }
    const card = cards.find((item) => item.key === 'router_replay')!
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

  it('rejects duplicate list values and invalid typed numbers', () => {
    const formats = ROUTER_STRUCTURED_FIELDS.vector_store?.supported_formats.schema
    const streamedBody = ROUTER_STRUCTURED_FIELDS.router_core?.streamed_body.schema
    expect(formats).toBeDefined()
    expect(streamedBody).toBeDefined()
    expect(() => normalizeRouterStructuredValue(formats!, ['.md', '.md'])).toThrow(/unique/i)
    expect(() =>
      normalizeRouterStructuredValue(streamedBody!, { enabled: true, max_bytes: 0 }),
    ).toThrow(/at least 1/i)
  })

  it('builds custom controls for every migrated field and preserves model-selection extensions', () => {
    const selection = {
      enabled: true,
      momentum: { enabled: true, attack: 0.7 },
    }
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: {
        model_selection: selection,
      },
      routerDefaults: null,
    })

    const routerCore = cards.find((card) => card.key === 'router_core')
    expect(routerCore?.editFields.find((field) => field.name === 'list_backend_models')?.type).toBe(
      'boolean',
    )
    const selectionCard = cards.find((card) => card.key === 'model_selection')
    expect(selectionCard?.editFields.map((field) => field.name)).toEqual(['enabled'])

    const remainingJsonFields = cards.flatMap((card) =>
      card.editFields
        .filter((field) => (field.type as string) === 'json')
        .map((field) => `${card.key}.${field.name}`),
    )
    expect(remainingJsonFields).toEqual([])

    const patch = selectionCard?.save({
      ...selectionCard.editData,
      enabled: false,
    }) as { router?: { model_selection?: Record<string, unknown> } }
    expect(patch.router?.model_selection).toEqual({
      enabled: false,
      momentum: { enabled: true, attack: 0.7 },
    })

    const embeddingCards = buildRouterSectionCards({
      config: null,
      routerConfig: {
        embedding_models: { default: { deployment: 'vela-embedding' } },
      },
      routerDefaults: null,
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
    }) as {
      model_catalog?: {
        deployments?: Record<string, Record<string, unknown>>
        embeddings?: { default?: { deployment: string } }
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
    expect(embeddingPatch.model_catalog?.embeddings?.default).toEqual({ deployment: 'remote-embed' })
    expect(embeddingPatch.model_catalog?.embeddings).not.toHaveProperty('semantic')
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
