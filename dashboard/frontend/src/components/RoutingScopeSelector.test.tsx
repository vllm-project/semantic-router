import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { RoutingScope } from '../utils/routingScopes'
import RoutingScopeSelector from './RoutingScopeSelector'
import { routingScopeDetail, routingScopeSummary } from './routingScopeSelectorSupport'

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

const trigger = (markup: string) => markup.match(/<button[^>]*>/)?.[0] ?? ''

describe('RoutingScopeSelector', () => {
  it('draws a labelled design-system Select trigger rather than a native select', () => {
    const markup = renderToStaticMarkup(
      <RoutingScopeSelector label="Recipe" scopes={scopes} value="privacy" onChange={() => {}} />,
    )

    expect(trigger(markup)).toContain('role="combobox"')
    expect(trigger(markup)).toContain('aria-label="Recipe"')
    expect(trigger(markup)).toContain('data-slot="select-trigger"')
    expect(trigger(markup)).toContain('title="Keep private traffic local."')
    expect(markup).not.toMatch(/<select[^>]+aria-label="Recipe"/)
    expect(markup).toMatch(/>Recipe<\/span><span[^>]*><span[^>]*>privacy<\/span><\/span>/)
  })

  it('leaves the label to a visible heading and falls back to the first scope', () => {
    const markup = renderToStaticMarkup(
      <RoutingScopeSelector
        showLabel={false}
        scopes={scopes}
        value="missing"
        onChange={() => {}}
        className="w-full"
      />,
    )

    expect(trigger(markup)).toContain('aria-label="Recipe"')
    expect(trigger(markup)).toContain('title="vllm-sr/auto"')
    expect(trigger(markup)).toMatch(/class="[^"]*\bw-full\b/)
    expect(trigger(markup)).not.toMatch(/\bw-fit\b/)
    expect(markup).not.toContain('>Recipe</span>')
    expect(markup).toContain('lucide-layers')
    expect(markup).toMatch(/<span[^>]*>default<\/span>/)
  })

  it('renders nothing without scopes', () => {
    expect(
      renderToStaticMarkup(<RoutingScopeSelector scopes={[]} value="" onChange={() => {}} />),
    ).toBe('')
  })
})

describe('routingScopeSummary', () => {
  it('counts decisions and signals, with a singular noun for one', () => {
    expect(routingScopeSummary(scopes[0])).toBe('2 decisions · 4 signals')
    expect(routingScopeSummary(scopes[1])).toBe('1 decision · 1 signal')
    expect(routingScopeSummary({ ...scopes[1], routing: {} })).toBe('0 decisions · 0 signals')
  })
})

describe('routingScopeDetail', () => {
  it('names the public models, else the description, else the default purpose', () => {
    expect(routingScopeDetail(scopes[0])).toBe('vllm-sr/auto')
    expect(routingScopeDetail({ ...scopes[0], entrypointModelNames: ['a', 'b'] })).toBe('a, b')
    expect(routingScopeDetail(scopes[1])).toBe('Keep private traffic local.')
    expect(routingScopeDetail({ ...scopes[1], description: undefined })).toBe(
      'Top-level automatic routing',
    )
  })
})
