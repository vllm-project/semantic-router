import type { ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { DEFAULT_SECTIONS } from './configPageRouterDefaultsCatalog'
import { buildRouterSectionCards } from './configPageRouterDefaultsSupport'
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
    // Rules that ask their own question name their deployment on the rule.
    expect(card.editFields.map((field) => field.name)).toEqual([
      'decision_model',
      'binding:domain',
      'binding:jailbreak',
      'binding:pii',
      'binding:fact_check',
      'binding:user_feedback',
      'binding:modality',
      'binding:hallucination',
      'binding:preference',
      'binding:reask',
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

  describe('with bindings and a deprecated system line', () => {
    const authored: CanonicalModelCatalogConfig = {
      deployments: {
        'judge-a': { provider: 'model_runtime', artifact: 'vllm-sr/Vela-2.0-4B' },
        'd3-remote': {
          provider: 'systemone',
          endpoint: 'https://systemone.example.com/v1/systemone',
          model: 'vllm-sr/d3-ultra',
        },
      },
      bindings: {
        domain_classifier: { deployment: 'judge-a', contract: 'label_distribution.v1' },
        'classifier.risk': 'judge-a',
      },
      system: { decision_model: { deployment: 'main-judge' }, pii_classifier: 'models/custom-pii' },
    }
    const card = () => systemModelCard({ decision_model: { deployment: 'main-judge' } }, authored)

    it('shows each binding by signal type and deprecated forms read-only', () => {
      expect(card().summary).toEqual([
        { label: 'Decision Model', value: 'main-judge' },
        { label: 'Jailbreak', value: 'Follows main-judge' },
        { label: 'Domain', value: 'judge-a' },
        { label: 'PII', value: 'models/custom-pii (deprecated system.pii_classifier)' },
      ])
      expect(card().badges).toEqual([
        { label: '1 explicit binding', tone: 'active' },
        { label: '1 deprecated model path', tone: 'info' },
        { label: '1 deprecated binding', tone: 'info' },
      ])
      expect(card().editData).toEqual({
        decision_model: 'main-judge',
        'binding:domain': 'judge-a',
        'binding:jailbreak': '',
        'binding:pii': 'models/custom-pii (deprecated system.pii_classifier)',
        'binding:fact_check': '',
        'binding:user_feedback': '',
        'binding:modality': '',
        'binding:hallucination': '',
        'binding:preference': '',
        'binding:reask': '',
      })
      expect(
        card().editFields.find((field) => field.name === 'binding:jailbreak')?.options,
      ).toEqual(['', 'd3-remote', 'judge-a', 'main-judge'])
      const deprecated = card().editFields.find((field) => field.name === 'deprecated_bindings')
      expect(deprecated?.type).toBe('custom')
      expect(
        renderToStaticMarkup(deprecated!.customRender!(undefined, () => {}) as ReactElement),
      ).toContain('classifier.risk: judge-a')
    })

    it('writes only the decision model when no binding changes', () => {
      expect(card().save(card().editData)).toEqual({
        model_catalog: { system: { decision_model: { deployment: 'main-judge' } } },
      })
    })

    it('writes changed bindings as deployment names and clears what they replace', () => {
      const patch = card().save({
        ...card().editData,
        decision_model: 'd3-remote',
        'binding:domain': 'd3-remote',
        'binding:jailbreak': 'judge-a',
        'binding:pii': '',
      })
      const modelFields = { model_id: null, model_ref: null, backend: null }
      expect(patch).toEqual({
        model_catalog: {
          system: {
            decision_model: { deployment: 'd3-remote' },
            domain_classifier: null,
            prompt_guard: null,
            pii_classifier: null,
          },
          bindings: {
            domain: 'd3-remote',
            domain_classifier: null,
            jailbreak: 'judge-a',
            prompt_guard: null,
            pii: null,
            pii_classifier: null,
          },
          modules: {
            classifier: { domain: modelFields, pii: modelFields },
            prompt_guard: modelFields,
          },
        },
      })
      expect(JSON.stringify(patch)).not.toContain('contract')
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
    const original = { default: 'qwen3-embedding', multimodal: 'omni-nano' }
    const saved = embeddingModelsPatch(embeddingModelsEditData(original))
    expect(saved.embeddings).toEqual(original)
  })
})
