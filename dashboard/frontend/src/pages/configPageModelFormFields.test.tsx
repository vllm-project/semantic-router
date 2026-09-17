import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { modelAPIFormatField, modelReasoningFamilyField } from './configPageModelFormFields'
import {
  editModelFormData,
  modelDialogFields,
  newModelFormData,
} from './configPageModelsSectionSupport'
import { getNormalizedModels } from './configPageModelNormalization'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog

const renderField = (
  field: ReturnType<typeof modelAPIFormatField>,
  value: unknown,
  data: Record<string, unknown>,
) => renderToStaticMarkup(<>{field.customRender!(value, vi.fn(), data)}</>)

describe('model form effective fields', () => {
  it('shows the actual inherited API format and only canonical override choices', () => {
    const field = modelAPIFormatField(catalog)
    const data = newModelFormData()
    const markup = renderField(field, '', data)
    expect(markup).toContain('value="" selected="">Inherit (openai)')
    expect(markup).toContain('value="openai"')
    expect(markup).toContain('value="responses"')
    expect(markup).toContain('value="anthropic"')
    expect(markup).not.toContain('value="images"')
  })

  it('recomputes defaults from the current backend rather than freezing the opening value', () => {
    const field = modelAPIFormatField(catalog)
    const markup = renderField(field, '', {
      ...newModelFormData(),
      backend_refs: [{ provider: 'anthropic' }],
    })
    expect(markup).toContain('Inherit (anthropic)')
    expect(markup).not.toContain('Inherit (openai)')
    expect(renderField(field, 'responses', newModelFormData())).toContain(
      'value="responses" selected=""',
    )
  })

  it('uses the real edit alias for a custom model native-ID fallback', () => {
    const provider = catalog.providers.find((entry) => entry.id === 'vllm')!
    const customCatalog: BuiltInModelCatalog = {
      ...catalog,
      providers: [
        {
          ...provider,
          id: 'test',
          default_protocol: 'openai/chat-completions@1',
          protocols: ['openai/chat-completions@1', 'openai/responses@1'],
          supported_operations: ['openai/chat-completions@1#create', 'openai/responses@1#create'],
          models: [
            {
              ...provider.models![0],
              catalog: 'custom-alias',
              id: 'custom-alias',
              protocols: ['openai/responses@1'],
            },
          ],
        },
      ],
    }
    const model = getNormalizedModels(
      { providers: { models: [{ name: 'custom-alias', backend_refs: [{ provider: 'test' }] }] } },
      true,
      customCatalog,
    )[0]
    const form = editModelFormData(model)
    expect(form.model_name).toBe('custom-alias')
    expect(model.api_format).toBe('responses')
    expect(renderField(modelAPIFormatField(customCatalog), form.api_format, form)).toContain(
      'Inherit (responses)',
    )
  })

  it('reports unresolved bindings without presenting a fabricated default', () => {
    const markup = renderField(modelAPIFormatField(catalog), '', {
      backend_refs: [{ provider: 'unknown' }],
    })
    expect(markup).toContain('Inherit (unresolved)')
    expect(markup).toContain('Unknown provider: unknown.')
  })

  it('shows GLM inherited reasoning instead of the first custom family option', () => {
    const model = getNormalizedModels(
      {
        providers: {
          models: [
            { name: 'local', catalog: 'zai/glm-5.3-flash', backend_refs: [{ provider: 'vllm' }] },
          ],
        },
      },
      true,
      catalog,
    )[0]
    const data = editModelFormData(model)
    const markup = renderField(
      modelReasoningFamilyField(['deepseek', 'glm-5.3'], catalog),
      data.reasoning_family,
      data,
    )
    expect(markup).toContain('glm-5.3 (inherited)')
    expect(markup).toContain('readonly=""')
    expect(markup).not.toContain('<select')
    expect(data.reasoning_family).toBe('')
    expect(
      modelDialogFields([], 'edit', catalog)
        .filter((field) => field.name.startsWith('reasoning_') && field.name !== 'reasoning_family')
        .every((field) => field.shouldHide?.(data)),
    ).toBe(true)
  })

  it('keeps custom reasoning unset visibly and exposes inline settings', () => {
    const data = newModelFormData()
    const markup = renderField(modelReasoningFamilyField(['deepseek'], catalog), '', data)
    expect(markup).toContain('value="" selected="">None / inline settings')
    expect(
      modelDialogFields([], 'add', catalog)
        .filter((field) => field.name.startsWith('reasoning_') && field.name !== 'reasoning_family')
        .every((field) => !field.shouldHide?.(data)),
    ).toBe(true)
  })
})
