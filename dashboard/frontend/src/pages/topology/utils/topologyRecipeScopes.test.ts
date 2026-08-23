import { describe, expect, it } from 'vitest'

import type { ManagedTopologyConfig } from '../types'
import { parseConfigToTopology } from './topologyParser'
import { simulateSignalMatching } from './signalMatcher'

const config = {
  models: [{ name: 'model-a', card: {} }],
  document: {
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
} as ManagedTopologyConfig

describe('recipe-aware topology', () => {
  it('parses only the selected recipe signal, projection, and decision graph', () => {
    const topology = parseConfigToTopology(config)

    expect(topology.signals.map((signal) => signal.name)).toEqual(
      expect.arrayContaining(['balanced-keyword', 'balanced-standard']),
    )
    expect(topology.signals.map((signal) => signal.name)).not.toContain('private-pii')
    expect(topology.decisions.map((decision) => decision.name)).toEqual(['balanced-route'])
    expect(topology.strategy).toBe('confidence')
  })

  it('keeps local previews honest when a recipe needs Router-only signals', async () => {
    const result = await simulateSignalMatching('balanced', parseConfigToTopology(config))

    expect(result.mode).toBe('simulate')
    expect(result.isAccurate).toBe(false)
    expect(result.matchedDecision).toBeNull()
    expect(result.matchedModels).toEqual([])
    expect(result.warning).toBe('Local preview covers keyword and language signals only.')
  })
})
