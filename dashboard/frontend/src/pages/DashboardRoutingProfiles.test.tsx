import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { ManagedRoutingSummary } from '../utils/managedRoutingSnapshot'
import DashboardRoutingProfiles from './DashboardRoutingProfiles'

describe('DashboardRoutingProfiles', () => {
  it('renders Entrypoint aliases and native Recipe document counts', () => {
    const config = {
      models: [],
      entrypoints: [
        {
          id: 'entrypoint-balanced',
          name: 'vllm-sr/balanced',
          aliases: ['vllm-sr/balanced'],
          rules: [{ recipeId: 'recipe-balanced' }],
        },
        {
          id: 'entrypoint-private',
          name: 'vllm-sr/private',
          aliases: ['vllm-sr/private'],
          rules: [{ recipeId: 'recipe-private' }],
        },
      ],
      recipes: [
        {
          id: 'recipe-balanced',
          name: 'balanced',
          description: 'Balanced objective',
          document: {
            signals: { keywords: [{ name: 'intent' }] },
            projections: { scores: [{ name: 'score' }], mappings: [{ name: 'map' }] },
            decisions: [{ name: 'route' }],
          },
        },
        {
          id: 'recipe-private',
          name: 'privacy',
          document: {
            signals: { pii: [{ name: 'private' }] },
            decisions: [{ name: 'private-route' }],
          },
        },
      ],
    } as unknown as ManagedRoutingSummary

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
