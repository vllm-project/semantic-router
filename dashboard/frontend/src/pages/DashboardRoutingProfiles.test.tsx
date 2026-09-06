import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { RouterConfig } from './dashboardPageTypes'
import DashboardRoutingProfiles from './DashboardRoutingProfiles'

describe('DashboardRoutingProfiles', () => {
  it('renders entrypoint aliases and scoped routing counts', () => {
    const config: RouterConfig = {
      entrypoints: [
        { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
        { model_names: ['vllm-sr/private'], recipe: 'privacy' },
      ],
      recipes: [
        {
          name: 'balanced',
          description: 'Balanced objective',
          routing: {
            signals: { keywords: [{ name: 'intent' }] },
            projections: { scores: [{ name: 'score' }], mappings: [{ name: 'map' }] },
            decisions: [{ name: 'route' }],
          },
        },
        {
          name: 'privacy',
          routing: {
            signals: { pii: [{ name: 'private' }] },
            decisions: [{ name: 'private-route' }],
          },
        },
      ],
    }

    const markup = renderToStaticMarkup(
      createElement(DashboardRoutingProfiles, {
        config,
        onOpenTopology: vi.fn(),
      }),
    )

    expect(markup).toContain('2 profiles')
    expect(markup).toContain('vllm-sr/balanced')
    expect(markup).toContain('Balanced objective')
    expect(markup).toContain('View balanced topology')
  })
})
