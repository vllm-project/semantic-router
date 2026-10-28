import { describe, expect, it } from 'vitest'

import { DEFAULT_SECTIONS } from './configPageRouterDefaultsCatalog'
import { buildRouterSectionCards } from './configPageRouterDefaultsSupport'
import type { CanonicalSystemModels } from './configPageSupport'
import {
  embeddingModelsCatalogValue,
  embeddingModelsEditData,
} from './configPageEmbeddingModelsSupport'

describe('Vela defaults and explicit legacy models', () => {
  const systemModelCard = (system?: CanonicalSystemModels) =>
    buildRouterSectionCards({
      config: null,
      routerConfig: { system_models: system },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    }).find((card) => card.key === 'system_models')!

  it('shows the decision model and inherited bindings instead of missing models', () => {
    const card = systemModelCard({ decision_model: { deployment: 'main-judge' } })

    expect(card.title).toBe('Decision Model & Bindings')
    expect(card.summary).toEqual([
      { label: 'Decision Model', value: 'main-judge' },
      { label: 'Prompt Guard', value: 'Follows main-judge' },
      { label: 'Domain', value: 'Follows main-judge' },
      { label: 'PII', value: 'Follows main-judge' },
    ])
    expect(card.badges).toContainEqual({ label: '0 explicit bindings', tone: 'inactive' })
  })

  it('shows the default decision deployment', () => {
    const defaults = systemModelCard()
    expect(defaults.summary[0]).toEqual({
      label: 'Decision Model',
      value: 'vela3-fast (default)',
    })
    expect(defaults.editData.decision_model).toEqual({ deployment: 'primary' })
  })

  it('preserves explicit bindings when the configured decision model is changed', () => {
    const card = systemModelCard({
      decision_model: { deployment: 'main-judge' },
      pii_classifier: 'models/custom-pii',
    })

    expect(card.summary).toContainEqual({ label: 'PII', value: 'models/custom-pii' })
    expect(card.badges).toContainEqual({ label: '1 explicit binding', tone: 'active' })
    expect(card.save({ ...card.editData, decision_model: { deployment: 'second-judge' } })).toEqual(
      {
        model_catalog: {
          system: {
            decision_model: { deployment: 'second-judge' },
            pii_classifier: 'models/custom-pii',
          },
        },
      },
    )
  })

  it('leaves module thresholds to the model each module runs', () => {
    // The modules follow the decision model; a per-module line would pin one.
    expect(DEFAULT_SECTIONS.system_models).toEqual({ decision_model: { deployment: 'primary' } })
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
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    })
    const saved = (key: string) => {
      const card = cards.find((candidate) => candidate.key === key)!
      return card.save(card.editData)
    }

    expect(saved('prompt_guard')).toEqual({
      model_catalog: {
        modules: { prompt_guard: { enabled: true, model_ref: 'prompt_guard', use_cpu: true } },
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
          feedback_detector: { enabled: true, model_ref: 'feedback_detector', use_cpu: true },
        },
      },
    })
  })

  it('saves a cleared module threshold as no threshold, not 0', () => {
    const card = buildRouterSectionCards({
      config: null,
      routerConfig: { prompt_guard: { enabled: true, threshold: 0.9 } },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
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
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
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
      const saved = embeddingModelsCatalogValue(embeddingModelsEditData(original))
      expect(saved).toMatchObject(original)
    },
  )
})
