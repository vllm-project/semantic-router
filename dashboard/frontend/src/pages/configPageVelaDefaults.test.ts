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

  it('leaves module thresholds to the model each module runs', () => {
    // The modules follow the decision model; a per-module line would pin one.
    expect(DEFAULT_SECTIONS.system_models).toEqual({ decision_model: { deployment: 'primary' } })
    expect(JSON.stringify(DEFAULT_SECTIONS)).not.toMatch(/model_ref|model_id/)
    const sections = DEFAULT_SECTIONS as Record<string, Record<string, unknown>>
    expect(sections.prompt_guard).not.toHaveProperty('threshold')
    expect(sections.classifier.domain).not.toHaveProperty('threshold')
    expect(sections.classifier.pii).not.toHaveProperty('threshold')
    expect(sections.hallucination_mitigation.fact_check).not.toHaveProperty('threshold')
    expect(sections.feedback_detector).not.toHaveProperty('threshold')
    expect(DEFAULT_SECTIONS.hallucination_mitigation).toMatchObject({
      detector: {
        threshold: 0.5,
        min_span_length: 1,
        min_span_confidence: 0,
      },
    })
    expect(DEFAULT_SECTIONS.embedding_models).toMatchObject({
      multimodal_model_path: 'models/vela-1.0-omni-nano',
      embedding_config: { target_dimension: 0 },
    })
    expect(DEFAULT_SECTIONS.feedback_detector).not.toHaveProperty('max_sequence_length')
  })

  it('saves an untouched seeded module section without a threshold', () => {
    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: {},
      routerDefaults: null,
    })
    const saved = (key: string) => {
      const card = cards.find((candidate) => candidate.key === key)!
      return card.save(card.editData)
    }

    expect(saved('prompt_guard')).toEqual({
      model_catalog: {
        modules: { prompt_guard: { enabled: true, use_cpu: true } },
      },
    })
    const classifier = saved('classifier') as {
      model_catalog: { modules: { classifier: Record<string, object> } }
    }
    expect(classifier.model_catalog.modules.classifier.domain).not.toHaveProperty('threshold')
    expect(classifier.model_catalog.modules.classifier.pii).not.toHaveProperty('threshold')
    const hallucination = saved('hallucination_mitigation') as {
      model_catalog: { modules: { hallucination_mitigation: { fact_check: object } } }
    }
    expect(
      hallucination.model_catalog.modules.hallucination_mitigation.fact_check,
    ).not.toHaveProperty('threshold')
    expect(saved('feedback_detector')).toEqual({
      model_catalog: {
        modules: {
          feedback_detector: { enabled: true, use_cpu: true },
        },
      },
    })
  })

  it('never saves the deprecated model fields the Router still reports for a module', () => {
    const cards = buildRouterSectionCards({
      config: {
        global: { model_catalog: { bindings: { jailbreak: 'guard-runtime' } } },
      } as ConfigData,
      routerConfig: {
        prompt_guard: {
          enabled: true,
          model_ref: 'prompt_guard',
          model_id: 'models/x',
          threshold: 0.9,
        },
        classifier: {
          domain: { model_ref: 'domain_classifier', threshold: 0.4 },
          pii: { model_id: 'm' },
        },
      },
      routerDefaults: null,
    })
    const card = (key: string) => cards.find((candidate) => candidate.key === key)!
    const saved = (key: string) => card(key).save(card(key).editData)

    expect(saved('prompt_guard')).toEqual({
      model_catalog: { modules: { prompt_guard: { enabled: true, threshold: 0.9 } } },
    })
    expect(saved('classifier')).toMatchObject({
      model_catalog: { modules: { classifier: { domain: { threshold: 0.4 }, pii: {} } } },
    })
    expect(JSON.stringify(saved('classifier'))).not.toMatch(/model_ref|model_id/)
    for (const key of ['prompt_guard', 'feedback_detector']) {
      const names = card(key).editFields.map((field) => field.name)
      expect(names).not.toContain('model_ref')
      expect(names).not.toContain('model_id')
      expect(names).not.toContain('backend')
    }
    expect(card('prompt_guard').summary[0]).toEqual({ label: 'Model', value: 'guard-runtime' })
    expect(card('feedback_detector').summary[0]).toEqual({
      label: 'Model',
      value: 'Follows vela3-fast',
    })
  })

  it('saves a cleared module threshold as no threshold, not 0', () => {
    const card = buildRouterSectionCards({
      config: null,
      routerConfig: { prompt_guard: { enabled: true, threshold: 0.9 } },
      routerDefaults: null,
    }).find((candidate) => candidate.key === 'prompt_guard')!

    expect(card.save({ ...card.editData, threshold: '' })).toEqual({
      model_catalog: { modules: { prompt_guard: { enabled: true } } },
    })
    expect(card.save({ ...card.editData, threshold: 0.9 })).toEqual({
      model_catalog: { modules: { prompt_guard: { enabled: true, threshold: 0.9 } } },
    })
  })

  it('leaves store widths to the embedding model and offers only runtime embedding models', () => {
    // Vela Embedding serves 768, 512, 256, 128 and 64; an unset width takes the model's own.
    expect(DEFAULT_SECTIONS.memory).toMatchObject({ milvus: { collection: 'agentic_memory' } })
    expect((DEFAULT_SECTIONS.memory as { milvus: object }).milvus).not.toHaveProperty('dimension')
    expect(DEFAULT_SECTIONS.vector_store).not.toHaveProperty('embedding_dimension')

    const cards = buildRouterSectionCards({
      config: null,
      routerConfig: { vector_store: { enabled: true } },
      routerDefaults: null,
    })
    const vectorStore = cards.find((card) => card.key === 'vector_store')
    expect(
      vectorStore?.editFields.find((field) => field.name === 'embedding_model')?.options,
    ).toEqual(['mmbert', 'qwen3', 'multimodal'])
  })

  it.each([false, true])(
    'preserves the explicit old embedding and full_context=%s through editor save',
    (fullContext) => {
      const original = {
        semantic: {
          mmbert_model_path: 'models/mmbert-embed-32k-2d-matryoshka',
          use_cpu: true,
          embedding_config: {
            backend: 'model_runtime',
            model_type: 'mmbert',
            target_dimension: 256,
            target_layer: 6,
            full_context: fullContext,
          },
        },
      }
      const saved = embeddingModelsPatch(embeddingModelsEditData(original))
      expect(saved.embeddings).toMatchObject(original)
    },
  )
})
