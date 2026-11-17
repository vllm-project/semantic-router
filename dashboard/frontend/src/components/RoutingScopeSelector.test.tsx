import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { RoutingScope } from '../utils/routingScopes'
import RoutingScopeSelector from './RoutingScopeSelector'

const scopes: RoutingScope[] = [
  {
    id: 'default',
    label: 'default',
    entrypointModelNames: ['vllm-sr/auto'],
    routing: {
      decisions: [{}, {}],
      signals: { keywords: [{}, {}, {}], domains: [{}] },
    },
    isDefault: true,
    source: 'routing',
  },
  {
    id: 'privacy',
    label: 'privacy',
    entrypointModelNames: [],
    description: 'Keep private traffic local.',
    routing: { decisions: [{}], signals: { pii: [{}] } },
    isDefault: false,
    source: 'recipe',
  },
]

describe('RoutingScopeSelector', () => {
  it('keeps a labelled native select whose options carry recipe counts', () => {
    const markup = renderToStaticMarkup(
      <RoutingScopeSelector label="Recipe" scopes={scopes} value="privacy" onChange={() => {}} />,
    )

    expect(markup).toMatch(/<select[^>]+aria-label="Recipe"/)
    expect(markup).toContain('<option value="default">default · 2 decisions · 4 signals</option>')
    expect(markup).toMatch(
      /<option value="privacy" selected="">privacy · 1 decision · 1 signal<\/option>/,
    )
    expect(markup).toContain('title="Keep private traffic local."')
  })

  it('shows the selected recipe in a Select-style trigger and falls back to the first scope', () => {
    const markup = renderToStaticMarkup(
      <RoutingScopeSelector scopes={scopes} value="missing" onChange={() => {}} />,
    )

    expect(markup).toMatch(/aria-label="Routing profile"/)
    expect(markup).toContain('lucide-layers')
    expect(markup).toMatch(/>Recipe<\/span><span[^>]*>default<\/span>/)
    expect(markup).toContain('title="vllm-sr/auto"')
  })

  it('renders nothing without scopes', () => {
    expect(
      renderToStaticMarkup(<RoutingScopeSelector scopes={[]} value="" onChange={() => {}} />),
    ).toBe('')
  })
})
