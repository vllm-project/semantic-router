import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { normalizeModelLoras } from './configPageModelFormSupport'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  EvaluationRecordsEditor,
  ModelExternalIdsEditor,
  ModelLorasEditor,
  ModelPricingEditor,
  ModelReliabilityEditor,
  ModelTagsEditor,
} from './configPageModelStructuredEditors'

describe('model structured editors', () => {
  it('shows the retry count the gateways apply', () => {
    const render = (value: object) =>
      renderToStaticMarkup(createElement(ModelReliabilityEditor, { value, readOnly: true }))

    expect(render({ retry_count: 0 })).toContain('0 retries')
    expect(render({})).toContain('2 retries by default')
    // retry_on without a count retries nothing
    const conditionsOnly = render({ retry_on: '5xx' })
    expect(conditionsOnly).toContain('0 retries')
    expect(conditionsOnly).not.toContain('by default')
  })

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

  it.each(['delegated', 'caller'] as const)(
    'keeps a %s credential when a backend is edited',
    (credential) => {
      const backend = {
        name: 'claude',
        endpoint: 'https://api.anthropic.com',
        provider: 'anthropic',
        auth: { credential },
      }
      let saved: unknown
      const editor = ModelBackendRefsEditor({
        value: [backend],
        onChange: (value) => {
          saved = value
        },
      })

      expect(renderToStaticMarkup(editor)).toContain(`<option value="${credential}" selected="">`)
      editor.props.onChange(editor.props.value)
      expect(saved).toEqual([backend])
    },
  )

  it('keeps the stored key file when a backend is edited, and a new inline key replaces it', () => {
    const backend = {
      name: 'openai-primary',
      endpoint: 'https://api.openai.com/v1',
      provider: 'openai',
      auth: { api_key_file: '/app/.vllm-sr/provider-keys/key-1' },
    }
    let saved: unknown
    const editor = ModelBackendRefsEditor({
      value: [backend],
      onChange: (value) => {
        saved = value
      },
    })

    editor.props.onChange(editor.props.value)
    expect(saved).toEqual([backend])

    editor.props.onChange([{ ...editor.props.value[0], api_key: 'replacement-key' }])
    expect(saved).toEqual([
      {
        name: 'openai-primary',
        endpoint: 'https://api.openai.com/v1',
        provider: 'openai',
        api_key: 'replacement-key',
      },
    ])
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

  it('renders benchmark and metric evidence as structured fields', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationRecordsEditor, {
        value: [
          {
            model: 'private-reasoner',
            benchmark: 'idavidrein/gpqa-diamond@1.0.0',
            benchmark_profile: 'published-standard',
            reasoning_effort: 'high',
            metrics: { pass_at_1: 0.72 },
          },
        ],
        readOnly: true,
      }),
    )

    expect(markup).toContain('private-reasoner')
    expect(markup).toContain('idavidrein/gpqa-diamond@1.0.0')
    expect(markup).toContain('published-standard')
    expect(markup).toContain('high')
    expect(markup).toContain('Benchmark profile')
    expect(markup).toContain('Reasoning effort')
    expect(markup).toContain('pass_at_1')
    expect(markup).toContain('0.72')
  })
})
