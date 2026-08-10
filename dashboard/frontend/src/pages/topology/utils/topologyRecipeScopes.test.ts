import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  projectConfigForRoutingScope,
  type RoutingScopedConfigLike,
} from '../../../utils/routingScopes'
import type { ConfigData } from '../types'
import { testQueryDryRun } from './api'
import { parseConfigToTopology } from './topologyParser'

const config = {
  providers: {
    defaults: { default_model: 'model-a' },
    models: [{ name: 'model-a' }],
  },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
  ],
  recipes: [
    {
      name: 'balanced',
      routing: {
        strategy: 'confidence',
        signals: {
          keywords: [
            {
              name: 'balanced-keyword',
              operator: 'OR',
              keywords: ['balanced'],
            },
          ],
        },
        projections: {
          scores: [{ name: 'balanced-score', inputs: [] }],
          mappings: [
            {
              name: 'balanced-map',
              source: 'balanced-score',
              outputs: [{ name: 'balanced-standard' }],
            },
          ],
        },
        decisions: [
          {
            name: 'balanced-route',
            priority: 100,
            rules: {
              operator: 'AND',
              conditions: [{ type: 'keyword', name: 'balanced-keyword' }],
            },
            modelRefs: [{ model: 'model-a' }],
          },
        ],
      },
    },
    {
      name: 'privacy',
      routing: {
        signals: {
          pii: [{ name: 'private-pii', threshold: 0.8 }],
        },
        decisions: [
          {
            name: 'private-route',
            priority: 100,
            rules: {
              operator: 'AND',
              conditions: [{ type: 'pii', name: 'private-pii' }],
            },
            modelRefs: [{ model: 'model-a' }],
          },
        ],
      },
    },
  ],
} as ConfigData

describe('recipe-aware topology', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('parses only the selected recipe signal, projection, and decision graph', () => {
    const projected = projectConfigForRoutingScope(
      config as ConfigData & RoutingScopedConfigLike,
      'balanced',
    )
    const topology = parseConfigToTopology(projected)

    expect(topology.signals.map((signal) => signal.name)).toEqual(
      expect.arrayContaining(['balanced-keyword', 'balanced-standard']),
    )
    expect(topology.signals.map((signal) => signal.name)).not.toContain('private-pii')
    expect(topology.decisions.map((decision) => decision.name)).toEqual(['balanced-route'])
    expect(topology.strategy).toBe('confidence')
  })

  it('forwards the selected entrypoint to backend topology evaluation', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query: 'hello',
        mode: 'dry-run',
        matchedSignals: [],
        matchedDecision: 'balanced-route',
        matchedModels: ['model-a'],
        highlightedPath: [],
        isAccurate: true,
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await testQueryDryRun('hello', 'vllm-sr/balanced')

    const request = JSON.parse(fetchMock.mock.calls[0][1].body as string)
    expect(request).toMatchObject({
      query: 'hello',
      mode: 'dry-run',
      model: 'vllm-sr/balanced',
    })
  })
})
