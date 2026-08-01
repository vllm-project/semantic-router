import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ChatComposerModelSelect from './ChatComposerModelSelect'

const models = [
  {
    id: 'amd/rocm-v1-balanced',
    description: 'Balanced AMD Mixture-of-Models profile',
  },
  {
    id: 'amd/rocm-v1-flash',
    description: 'Latency-first AMD Mixture-of-Models profile',
  },
]

describe('ChatComposerModelSelect', () => {
  it('renders a compact MoM dropdown beside the composer add button', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComposerModelSelect, {
        models,
        onChange: vi.fn(),
        value: models[0].id,
      }),
    )

    expect(markup).toContain('data-testid="playground-composer-model-select"')
    expect(markup).toContain('aria-haspopup="listbox"')
    expect(markup).toContain('aria-expanded="false"')
    expect(markup).toContain('AMD')
    expect(markup).toContain('amd/rocm-v1-balanced')
  })

  it('disables selection while model discovery is unavailable', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComposerModelSelect, {
        disabled: true,
        models: [],
        onChange: vi.fn(),
        value: 'amd/rocm-v1-balanced',
      }),
    )

    expect(markup).toContain('disabled=""')
  })
})
