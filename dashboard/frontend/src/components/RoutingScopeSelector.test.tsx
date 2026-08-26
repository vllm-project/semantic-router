import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import RoutingScopeSelector from './RoutingScopeSelector'

const scopes = [
  {
    id: 'recipe-balanced',
    label: 'Balanced',
    description: 'Balanced routing',
    entrypointModelNames: ['vllm-sr/blend'],
    document: {},
  },
  {
    id: 'recipe-fast',
    label: 'Speed first',
    description: 'Fast routing',
    entrypointModelNames: [],
    document: {},
  },
]

describe('RoutingScopeSelector', () => {
  it('uses one accessible Recipe dropdown instead of a growing tab row', () => {
    const markup = renderToStaticMarkup(
      createElement(RoutingScopeSelector, {
        scopes,
        value: 'recipe-balanced',
        onChange: vi.fn(),
      }),
    )

    expect(markup).toMatch(/<label[^>]*for="[^"]+"[^>]*>Recipe<\/label>/)
    expect(markup).toContain('<select')
    expect(markup).toContain('aria-describedby=')
    expect(markup).toContain('value="recipe-balanced" selected=""')
    expect(markup).toContain('vllm-sr/blend')
    expect(markup).not.toContain('aria-pressed=')
    expect(markup).not.toContain('role="group"')
  })

  it('is the single shared Recipe switcher across every routing editor', () => {
    for (const page of [
      '../pages/ConfigPageSignalsSection.tsx',
      '../pages/ConfigPageDecisionsSection.tsx',
      '../pages/ConfigPageProjectionsSection.tsx',
    ]) {
      const source = readFileSync(new URL(page, import.meta.url), 'utf8')
      expect(source.match(/<RoutingScopeSelector/g)).toHaveLength(1)
      expect(source).not.toContain('aria-pressed')
    }
  })

  it('falls back to the selected Recipe description when it has no Entrypoint', () => {
    const markup = renderToStaticMarkup(
      createElement(RoutingScopeSelector, {
        scopes,
        value: 'recipe-fast',
        onChange: vi.fn(),
      }),
    )

    expect(markup).toContain('Fast routing')
    expect(markup).toContain('value="recipe-fast" selected=""')
  })
})
