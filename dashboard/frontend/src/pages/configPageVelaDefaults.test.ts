import { describe, expect, it } from 'vitest'

import { DEFAULT_SECTIONS } from './configPageRouterDefaultsCatalog'
import {
  buildRouterSectionCards,
  editedRouterSectionPatch,
  type RouterSystemKey,
} from './configPageRouterDefaultsSupport'
import type {
  CanonicalModelCatalogConfig,
  CanonicalSystemModels,
  ConfigData,
} from './configPageSupport'
import {
  embeddingModelsEditData,
  embeddingModelsPatch,
} from './configPageEmbeddingModelsSupport'

describe('Vela defaults and explicit legacy models', () => {
  const systemModelCard = (
    system?: CanonicalSystemModels,
    authored?: CanonicalModelCatalogConfig,
  ) =>
    buildRouterSectionCards({
      config: authored ? ({ global: { model_catalog: authored } } as ConfigData) : null,
      routerConfig: { system_models: system },
      routerDefaults: null,
    }).find((card) => card.key === 'system_models')!

  it('shows the decision model and per-signal bindings instead of system lines', () => {
    const card = systemModelCard({ decision_model: { deployment: 'main-judge' } })

    expect(card.title).toBe('Decision Model & Bindings')
    expect(card.summary).toEqual([
      { label: 'Decision Model', value: 'main-judge' },
      { label: 'Jailbreak', value: 'Follows main-judge' },
      { label: 'Domain', value: 'Follows main-judge' },
      { label: 'PII', value: 'Follows main-judge' },
    ])
    expect(card.badges).toContainEqual({ label: '0 explicit bindings', tone: 'inactive' })
    expect(card.editFields.map((field) => field.name)).toEqual([
      'decision_model',
      'binding:domain',
      'binding:fact_check',
      'binding:user_feedback',
      'binding:jailbreak',
      'binding:safety',
      'binding:pii',
      'binding:hallucination',
      'binding:refusal',
      'binding:turn_relation',
      'binding:complexity',
      'binding:modality',
    ])
  })

  it('shows the default decision deployment', () => {
    const defaults = systemModelCard()
    expect(defaults.summary[0]).toEqual({
      label: 'Decision Model',
      value: 'vela3-fast (default)',
    })
    expect(defaults.editData.decision_model).toBe('primary')
    expect(defaults.editFields.find((field) => field.name === 'decision_model')?.options).toEqual([
      'primary',
    ])
  })

  describe('with bindings', () => {
    const authored: CanonicalModelCatalogConfig = {
      deployments: {
        'judge-a': { provider: 'model_runtime', artifact: 'vllm-sr/Vela-2.0-4B' },
        'd3-remote': {
          provider: 'systemone',
          endpoint: 'https://systemone.example.com/v1/systemone',
          model: 'vllm-sr/d3-ultra',
        },
      },
      bindings: { domain: { deployment: 'judge-a' }, pii: 'd3-remote' },
      system: { decision_model: { deployment: 'main-judge' } },
    }
    const card = () => systemModelCard({ decision_model: { deployment: 'main-judge' } }, authored)

    it('shows each binding by signal type', () => {
      expect(card().summary).toEqual([
        { label: 'Decision Model', value: 'main-judge' },
        { label: 'Jailbreak', value: 'Follows main-judge' },
        { label: 'Domain', value: 'judge-a' },
        { label: 'PII', value: 'd3-remote' },
      ])
      expect(card().badges).toEqual([{ label: '2 explicit bindings', tone: 'active' }])
      expect(card().editData).toEqual({
        decision_model: 'main-judge',
        'binding:domain': 'judge-a',
        'binding:fact_check': '',
        'binding:user_feedback': '',
        'binding:jailbreak': '',
        'binding:safety': '',
        'binding:pii': 'd3-remote',
        'binding:hallucination': '',
        'binding:refusal': '',
        'binding:turn_relation': '',
        'binding:complexity': '',
        'binding:modality': '',
      })
      expect(
        card().editFields.find((field) => field.name === 'binding:jailbreak')?.options,
      ).toEqual(['', 'd3-remote', 'judge-a', 'main-judge'])
    })

    it('writes only the decision model when no binding changes', () => {
      expect(card().save(card().editData)).toEqual({
        model_catalog: { system: { decision_model: { deployment: 'main-judge' } } },
      })
    })

    it('writes changed bindings as deployment names and clears unset ones', () => {
      const patch = card().save({
        ...card().editData,
        decision_model: 'd3-remote',
        'binding:domain': 'd3-remote',
        'binding:jailbreak': 'judge-a',
        'binding:pii': '',
      })
      expect(patch).toEqual({
        model_catalog: {
          system: { decision_model: { deployment: 'd3-remote' } },
          bindings: { domain: 'd3-remote', jailbreak: 'judge-a', pii: null },
        },
      })
    })
  })

  it('leaves store widths to the embedding deployment and names deployments, not keywords', () => {
    // Vela Embedding serves 768, 512, 256, 128 and 64; an unset width takes the model's own.
    expect(DEFAULT_SECTIONS.memory).toMatchObject({ milvus: { collection: 'agentic_memory' } })
    expect((DEFAULT_SECTIONS.memory as { milvus: object }).milvus).not.toHaveProperty('dimension')
    for (const store of ['memory', 'response_cache', 'vector_store'] as const) {
      expect(DEFAULT_SECTIONS[store]).not.toHaveProperty('embedding')
      expect(DEFAULT_SECTIONS[store]).not.toHaveProperty('embedding_model')
    }

    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: { vector_store: { enabled: true } },
      routerDefaults: null,
    })
    const vectorStore = cards.find((card) => card.key === 'vector_store')
    const names = vectorStore?.editFields.map((field) => field.name)
    expect(names).toContain('embedding')
    expect(names).not.toContain('embedding_model')
    expect(names).not.toContain('embedding_dimension')
  })

  it('keeps the embeddings block through an untouched editor save', () => {
    expect(DEFAULT_SECTIONS.embedding_models).toEqual({ default: { deployment: 'vela-embedding' } })
    const original = { default: { deployment: 'qwen3-embedding' } }
    const saved = embeddingModelsPatch(embeddingModelsEditData(original))
    expect(saved.embeddings).toEqual(original)
  })
})

describe('Global Config section saves', () => {
  const sectionCard = (key: RouterSystemKey, effective: unknown) =>
    buildRouterSectionCards({
      config: null,
      routerConfig: { [key]: effective },
      routerDefaults: null,
    }).find((card) => card.key === key)!
  const responseCache = sectionCard('response_cache', {
    enabled: true,
    backend_type: 'memory',
    similarity_threshold: 0.8,
    max_entries: 1000,
    ttl_seconds: 3600,
    eviction_policy: 'fifo',
  })
  const admission = sectionCard('admission', {
    hazard: { max_concurrency: 8, max_queue: 8, on_overflow: 'wait' },
    safety: { max_concurrency: 8, max_queue: 8, on_overflow: 'wait' },
  })
  const system = sectionCard('system_models', { decision_model: { deployment: 'main-judge' } })

  it('writes nothing when the shown values are saved unchanged', () => {
    expect(editedRouterSectionPatch(responseCache, { ...responseCache.editData })).toBeUndefined()
    expect(editedRouterSectionPatch(admission, { ...admission.editData })).toBeUndefined()
    expect(editedRouterSectionPatch(system, { ...system.editData })).toBeUndefined()
  })

  it('writes only the fields the user changed', () => {
    expect(
      editedRouterSectionPatch(responseCache, {
        ...responseCache.editData,
        similarity_threshold: 0.9,
      }),
    ).toEqual({ stores: { response_cache: { similarity_threshold: 0.9 } } })

    const policies = admission.editData.value as Record<string, Record<string, unknown>>
    expect(
      editedRouterSectionPatch(admission, {
        value: { ...policies, hazard: { ...policies.hazard, max_queue: 16 } },
      }),
    ).toEqual({ model_catalog: { admission: { hazard: { max_queue: 16 } } } })
  })

  it('sends the whole edited section when the shown values do not validate', () => {
    // The shown remote deployment has no model, which the form requires.
    const embeddings = buildRouterSectionCards({
      config: null,
      routerConfig: { embedding_models: { default: { deployment: 'remote-embed' } } },
      routerDefaults: {
        model_catalog: {
          deployments: {
            'remote-embed': {
              provider: 'openai_embeddings',
              endpoint: 'https://embedding.example.com/v1',
            },
          },
        },
      },
    }).find((card) => card.key === 'embedding_models')!
    const endpoint = embeddings.editData.endpoint as Record<string, unknown>
    expect(
      editedRouterSectionPatch(embeddings, {
        ...embeddings.editData,
        endpoint: { ...endpoint, model: 'text-embedding-3-small' },
      }),
    ).toEqual({
      model_catalog: {
        deployments: {
          'remote-embed': {
            provider: 'openai_embeddings',
            endpoint: 'https://embedding.example.com/v1',
            model: 'text-embedding-3-small',
            auth: null,
            timeout_ms: null,
            max_retries: null,
            limits: null,
            capabilities: null,
          },
        },
        embeddings: { default: { deployment: 'remote-embed' } },
      },
    })
  })
})
