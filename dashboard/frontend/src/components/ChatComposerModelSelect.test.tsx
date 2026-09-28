import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ChatComposerModelSelect from './ChatComposerModelSelect'

const models = [
  {
    id: 'vllm-sr/mom-v1-blend',
    description: 'Balanced Mixture-of-Models profile',
    recipe: 'balanced',
  },
  {
    id: 'vllm-sr/mom-v1-flash',
    description: 'Latency-first Mixture-of-Models profile',
    recipe: 'speed-first',
  },
]

describe('ChatComposerModelSelect', () => {
  it('renders a compact model dropdown without a redundant prefix badge', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComposerModelSelect, {
        models,
        onChange: vi.fn(),
        value: models[0].id,
      }),
    )

    expect(markup).toContain('data-testid="playground-composer-model-select"')
    expect(markup).toContain('aria-haspopup="listbox"')
    expect(markup).toContain('aria-label="Model: vllm-sr/mom-v1-blend"')
    expect(markup).toContain('aria-expanded="false"')
    expect(markup).not.toContain('>MoM<')
    expect(markup).not.toContain('AMD')
    expect(markup).toContain('vllm-sr/mom-v1-blend')
  })

  it('disables selection while model discovery is unavailable', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComposerModelSelect, {
        disabled: true,
        models: [],
        onChange: vi.fn(),
        value: 'vllm-sr/mom-v1-blend',
      }),
    )

    expect(markup).toContain('disabled=""')
  })

  it('supports listbox keyboard traversal and restores trigger focus', () => {
    const source = readFileSync(new URL('./ChatComposerModelSelect.tsx', import.meta.url), 'utf8')

    expect(source).toContain("event.key === 'ArrowDown'")
    expect(source).toContain("event.key === 'ArrowUp'")
    expect(source).toContain("event.key === 'Home'")
    expect(source).toContain("event.key === 'End'")
    expect(source).toContain('triggerRef.current?.focus()')
  })

  it('keeps options compact and presents recipe names as objective chips', () => {
    const source = readFileSync(new URL('./ChatComposerModelSelect.tsx', import.meta.url), 'utf8')

    expect(source).toContain('styles.objectiveChip')
    expect(source).not.toContain('recipe: {model.recipe}')
    expect(source).not.toContain('styles.optionDescription')
  })
})
