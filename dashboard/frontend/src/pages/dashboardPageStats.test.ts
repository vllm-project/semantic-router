import { describe, expect, it } from 'vitest'

import type { RouterConfig } from './dashboardPageTypes'
import {
  categorizeDecisions,
  countDecisions,
  countPlugins,
  countSignals,
  getAllDecisions,
} from './dashboardPageStats'

const config: RouterConfig = {
  routing: { decisions: [], signals: {} },
  recipes: [
    {
      name: 'balanced',
      routing: {
        signals: {
          keywords: [{ name: 'balanced-keyword' }],
          context: [{ name: 'balanced-context' }],
        },
        decisions: [
          {
            name: 'balanced-route',
            priority: 100,
            plugins: [{ type: 'semantic-cache' }],
          },
        ],
      },
    },
    {
      name: 'privacy',
      routing: {
        signals: {
          pii: [{ name: 'private-pii' }],
        },
        decisions: [{ name: 'private-route', priority: 1000 }],
      },
    },
  ],
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
  ],
}

describe('recipe-aware dashboard stats', () => {
  it('counts signals, decisions, and plugins across recipes', () => {
    expect(countSignals(config)).toEqual({
      total: 3,
      byType: { keywords: 1, context: 1, pii: 1 },
    })
    expect(countDecisions(config)).toBe(2)
    expect(countPlugins(config)).toBe(1)
  })

  it('keeps recipe scope metadata on overview decisions', () => {
    expect(getAllDecisions(config)).toEqual([
      expect.objectContaining({
        name: 'balanced-route',
        routingScope: 'balanced',
        routingEntrypoints: ['vllm-sr/balanced'],
      }),
      expect.objectContaining({
        name: 'private-route',
        routingScope: 'privacy',
        routingEntrypoints: ['vllm-sr/private'],
      }),
    ])
    expect(categorizeDecisions(config).guardrails).toHaveLength(1)
  })
})
