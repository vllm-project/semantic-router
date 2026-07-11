import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { normalizeModelLoras } from './configPageModelFormSupport'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  ModelExternalIdsEditor,
  ModelLorasEditor,
  ModelPricingEditor,
  ModelTagsEditor,
} from './configPageModelStructuredEditors'

describe('model structured editors', () => {
  it('renders model tags as discrete values', () => {
    const markup = renderToStaticMarkup(
      createElement(ModelTagsEditor, {
        value: ['premium', 'tool-use'],
        readOnly: true,
      }),
    )

    expect(markup).toContain('premium')
    expect(markup).toContain('tool-use')
    expect(markup).not.toContain('premium,tool-use')
  })

  it('masks backend addresses for restricted read-only views', () => {
    const markup = renderToStaticMarkup(
      createElement(ModelBackendRefsEditor, {
        value: [
          {
            name: 'primary',
            endpoint: '10.0.0.4:8000',
            protocol: 'http',
            weight: 1,
            api_key: 'secret-key',
          },
        ],
        readOnly: true,
        maskSensitive: true,
      }),
    )

    expect(markup).toContain('primary')
    expect(markup).toContain('••••••••')
    expect(markup).not.toContain('10.0.0.4:8000')
    expect(markup).not.toContain('secret-key')
  })

  it('renders capabilities, LoRAs, provider IDs, and pricing without JSON input', () => {
    const markup = renderToStaticMarkup(
      createElement(
        'div',
        null,
        createElement(ModelCapabilitiesEditor, { value: ['tools', 'vision'], readOnly: true }),
        createElement(ModelLorasEditor, {
          value: [{ name: 'code-expert', description: 'Code specialization' }],
          readOnly: true,
        }),
        createElement(ModelExternalIdsEditor, {
          value: { openai: 'gpt-4.1' },
          readOnly: true,
        }),
        createElement(ModelPricingEditor, {
          value: { currency: 'USD', prompt_per_1m: 0.5, completion_per_1m: 1.5 },
          readOnly: true,
        }),
      ),
    )

    expect(markup).toContain('tools')
    expect(markup).toContain('code-expert')
    expect(markup).toContain('openai')
    expect(markup).toContain('Prompt / 1M tokens')
    expect(markup).not.toContain('JSON')
  })

  it('normalizes LoRA rows without changing their schema', () => {
    expect(
      normalizeModelLoras([
        { name: ' code-expert ', description: ' Code specialization ' },
        { name: ' ' },
      ]),
    ).toEqual([{ name: 'code-expert', description: 'Code specialization' }])
  })
})
