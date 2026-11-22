import { describe, expect, it } from 'vitest'

import type { TestQueryResult } from './types'
import { parseConfigToTopology } from './utils/topologyParser'
import {
  DEFAULT_ID,
  ENTRY_ID,
  MORE_ID,
  buildTopology,
  getDecisionReachability,
  ruleSummary,
  topologyModel,
  type Layer,
} from './topologyGraph'

const topology = parseConfigToTopology({
  routing: {
    signals: {
      keywords: [{ name: 'code_terms', operator: 'OR', keywords: ['bug'] }],
      domains: [{ name: 'math' }],
    },
    projections: {
      scores: [
        {
          name: 'difficulty',
          method: 'weighted_sum',
          inputs: [{ type: 'keyword', name: 'code_terms', weight: 1 }],
        },
      ],
      mappings: [
        {
          name: 'difficulty_band',
          source: 'difficulty',
          method: 'threshold_bands',
          outputs: [
            { name: 'hard', gte: 0.5 },
            { name: 'easy', lt: 0.5 },
          ],
        },
      ],
    },
    decisions: [
      {
        name: 'hard_code',
        priority: 200,
        rules: { operator: 'AND', conditions: [{ type: 'projection', name: 'hard' }] },
        modelRefs: [{ model: 'big' }],
      },
      {
        name: 'math_route',
        priority: 100,
        rules: {
          operator: 'OR',
          conditions: [
            { type: 'domain', name: 'math' },
            { type: 'keyword', name: 'code_terms' },
          ],
        },
        modelRefs: [{ model: 'small' }, { model: 'big' }],
      },
    ],
  },
  providers: { defaults: { model: 'small' }, models: [{ name: 'small' }, { name: 'big' }] },
})

const allLayers: Record<Layer, boolean> = {
  signals: true,
  projections: true,
  decisions: true,
  models: true,
}

describe('topology graph', () => {
  const model = topologyModel(topology, ['acme/auto'], {
    density: 'comfortable',
    showAllDecisions: false,
  })

  it('lays out public model, signal families, projections, decisions and models', () => {
    const { nodes } = buildTopology(model, {
      density: 'comfortable',
      layers: allLayers,
      result: null,
      focus: null,
    })
    const ids = nodes.filter((node) => node.type !== 'lane').map((node) => node.id)
    expect(ids).toEqual(
      expect.arrayContaining([
        ENTRY_ID,
        'signal-group-keyword',
        'signal-group-domain',
        'projection-group-difficulty_band',
        'decision-hard_code',
        'decision-math_route',
        DEFAULT_ID,
        'model-big',
        'model-small',
      ]),
    )
    const projection = nodes.find((node) => node.id === 'projection-group-difficulty_band')
    expect(projection?.data.sublabel).toBe('hard, easy')
  })

  it('keeps paths connected when a layer is hidden', () => {
    const { edges } = buildTopology(model, {
      density: 'comfortable',
      layers: { ...allLayers, projections: false },
      result: null,
      focus: null,
    })
    expect(edges.map((edge) => edge.id)).toContain('signal-group-keyword>decision-hard_code')
  })

  it('lights the matched path from a Router preview', () => {
    const result: TestQueryResult = {
      query: 'fix this bug',
      mode: 'dry-run',
      matchedSignals: [{ type: 'keyword', name: 'code_terms', matched: true }],
      matchedDecision: 'math_route',
      matchedModels: ['small', 'big'],
      selectedModel: 'small',
      highlightedPath: [],
      isAccurate: true,
    }
    const { nodes, edges, pathBounds } = buildTopology(model, {
      density: 'comfortable',
      layers: allLayers,
      result,
      focus: null,
    })
    const state = (id: string) => nodes.find((node) => node.id === id)?.data.state
    expect(state('decision-math_route')).toBe('active')
    expect(state('model-small')).toBe('active')
    expect(state('model-big')).toBe('secondary')
    expect(state('decision-hard_code')).toBe('dimmed')
    expect(
      edges.find((edge) => edge.id === 'signal-group-keyword>decision-math_route')?.data?.state,
    ).toBe('active')
    expect(pathBounds).not.toBeNull()
  })

  it('folds low-priority decisions into one node until every decision is shown', () => {
    const many = parseConfigToTopology({
      routing: {
        decisions: Array.from({ length: 13 }, (_, index) => ({
          name: `d${index}`,
          priority: 100 - index,
          rules: { operator: 'AND', conditions: [] },
          modelRefs: [{ model: 'm' }],
        })),
      },
    })
    const folded = topologyModel(many, [], { density: 'comfortable', showAllDecisions: false })
    expect(folded.shown).toHaveLength(10)
    const { nodes } = buildTopology(folded, {
      density: 'comfortable',
      layers: allLayers,
      result: null,
      focus: null,
    })
    expect(nodes.find((node) => node.id === MORE_ID)?.data.label).toBe('3 more decisions')
    expect(
      topologyModel(many, [], { density: 'comfortable', showAllDecisions: true }).hidden,
    ).toHaveLength(0)
  })

  it('summarises rules and reachability', () => {
    expect(ruleSummary(topology.decisions[1].rules)).toBe('domain("math") OR keyword("code_terms")')
    expect(
      getDecisionReachability(topology.decisions[0], new Set(['projection:hard'])).isUnreachable,
    ).toBe(false)
  })
})
