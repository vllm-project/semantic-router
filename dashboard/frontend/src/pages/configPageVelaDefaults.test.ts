import { describe, expect, it } from 'vitest'

import { DEFAULT_SECTIONS } from './configPageRouterDefaultsCatalog'
import {
  buildRouterSectionCards,
  editedRouterSectionPatch,
  type RouterSystemKey,
} from './configPageRouterDefaultsSupport'
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
      value: 'Vela-2.0-0.3B (default)',
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

  it('keeps the fallback editor defaults aligned with the Vela 2.0 0.3B defaults', () => {
    // The modules follow the decision model; a per-module line would pin one.
    expect(DEFAULT_SECTIONS.system_models).toEqual({ decision_model: { deployment: 'primary' } })
    expect(DEFAULT_SECTIONS.hallucination_mitigation).toMatchObject({
      fact_check: { threshold: 0.93 },
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
    expect(DEFAULT_SECTIONS.feedback_detector).toMatchObject({ threshold: 0.37 })
    expect(DEFAULT_SECTIONS.feedback_detector).not.toHaveProperty('max_sequence_length')
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

describe('Global Config section saves', () => {
  const sectionCard = (key: RouterSystemKey, effective: unknown) =>
    buildRouterSectionCards({
      config: null,
      routerConfig: { [key]: effective },
      routerDefaults: null,
      toolsData: [],
      toolsLoading: false,
      toolsError: null,
    }).find((card) => card.key === key)!
  const promptGuard = sectionCard('prompt_guard', {
    enabled: true,
    model_ref: 'prompt_guard',
    model_id: 'models/Vela-2.0-0.3B',
    threshold: 0.75,
    use_cpu: true,
  })
  const admission = sectionCard('admission', {
    hazard: { max_concurrency: 8, max_queue: 8, on_overflow: 'wait' },
    safety: { max_concurrency: 8, max_queue: 8, on_overflow: 'wait' },
  })

  it('writes nothing when the shown values are saved unchanged', () => {
    expect(editedRouterSectionPatch(promptGuard, { ...promptGuard.editData })).toBeUndefined()
    expect(editedRouterSectionPatch(admission, { ...admission.editData })).toBeUndefined()
  })

  it('writes only the fields the user changed', () => {
    expect(
      editedRouterSectionPatch(promptGuard, { ...promptGuard.editData, threshold: 0.8 }),
    ).toEqual({ model_catalog: { modules: { prompt_guard: { threshold: 0.8 } } } })

    const policies = admission.editData.value as Record<string, Record<string, unknown>>
    expect(
      editedRouterSectionPatch(admission, {
        value: { ...policies, hazard: { ...policies.hazard, max_queue: 16 } },
      }),
    ).toEqual({ model_catalog: { admission: { hazard: { max_queue: 16 } } } })
  })

  it('sends the whole edited section when the shown values do not validate', () => {
    const system = sectionCard('system_models', {
      decision_model: {},
      pii_classifier: 'models/custom-pii',
    })
    expect(
      editedRouterSectionPatch(system, {
        ...system.editData,
        decision_model: { deployment: 'main-judge' },
      }),
    ).toEqual({
      model_catalog: {
        system: {
          decision_model: { deployment: 'main-judge' },
          pii_classifier: 'models/custom-pii',
        },
      },
    })
  })
})
