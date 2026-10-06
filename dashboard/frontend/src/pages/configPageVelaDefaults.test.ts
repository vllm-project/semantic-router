import { describe, expect, it } from 'vitest'

import { DEFAULT_SECTIONS } from './configPageRouterDefaultsCatalog'
import { buildRouterSectionCards } from './configPageRouterDefaultsSupport'
import {
  embeddingModelsCatalogValue,
  embeddingModelsEditData,
} from './configPageEmbeddingModelsSupport'

describe('Vela defaults and explicit legacy models', () => {
  it('keeps the fallback editor defaults aligned without promoting unpublished heads', () => {
    expect(DEFAULT_SECTIONS.system_models).toMatchObject({
      domain_classifier: 'models/Vela-1.0-Encoder-307M-Domain',
      pii_classifier: 'models/Vela-1.0-Encoder-307M-PII',
      fact_check_classifier: 'models/Vela-1.0-Encoder-307M-FactCheck',
      feedback_detector: 'models/Vela-1.0-Encoder-307M-Feedback',
      prompt_guard: 'models/Vela-1.0-Encoder-307M-Guard',
      hallucination_detector: 'models/Vela-1.0-Encoder-307M-Halu',
    })
    expect(DEFAULT_SECTIONS.system_models).not.toHaveProperty('safety')
    expect(DEFAULT_SECTIONS.system_models).not.toHaveProperty('hazard')
    expect(DEFAULT_SECTIONS.hallucination_mitigation).toMatchObject({
      fact_check: { threshold: 0.85 },
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
    expect(DEFAULT_SECTIONS.feedback_detector).toMatchObject({ threshold: 0.7 })
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
