import { describe, expect, it } from 'vitest'
import { SIGNAL_TYPES } from '../constants'
import type { CollapseState, ParsedTopology, TestQueryResult } from '../types'
import { buildLayoutGraph } from './layoutGraphBuilder'

const topology: ParsedTopology = {
  globalPlugins: [], decisions: [], models: [], strategy: 'priority',
  signals: [
    {
      type: 'projection', name: 'balance_direct_workload', latency: '<1ms',
      config: { mapping: 'workload', source: 'workload_score', method: 'weighted_sum', upstreamSignals: [] },
    },
    {
      type: 'projection', name: 'no_recovery_needed', latency: '<1ms',
      config: { mapping: 'recovery', source: 'recovery_score', method: 'weighted_sum', upstreamSignals: [] },
    },
  ],
}
const collapseState: CollapseState = {
  signalGroups: Object.fromEntries(SIGNAL_TYPES.map(type => [type, false])) as CollapseState['signalGroups'],
  decisions: {}, pluginChains: {},
}
const preview = (names: string[]): TestQueryResult => ({
  query: 'hello', mode: 'dry-run', isAccurate: true,
  matchedSignals: names.map(name => ({ type: 'projection', name, matched: true })),
  matchedDecision: null, matchedModels: [],
  highlightedPath: names.map(name => `signal-projection-${name}`),
})
const graph = (result: TestQueryResult) => buildLayoutGraph(
  topology, collapseState, result.highlightedPath, result, undefined, 'balanced',
  id => result.highlightedPath.includes(id),
)

describe('preview projection nodes', () => {
  it('highlights configured mappings without duplicating them as unconfigured ML signals', () => {
    const { nodes } = graph(preview(['balance_direct_workload', 'no_recovery_needed']))
    const projections = nodes.filter(node => node.data.signalType === 'projection')
    expect(projections.map(node => node.id)).toEqual(['projection-group-workload', 'projection-group-recovery'])
    expect(projections.every(node => node.data.isHighlighted)).toBe(true)
    expect(projections.some(node => node.data.isDynamic)).toBe(false)
  })

  it('keeps genuinely unknown outputs visible without copying known projections into the dynamic group', () => {
    const { nodes } = graph(preview(['balance_direct_workload', 'new_output']))
    const dynamic = nodes.find(node => node.id === 'signal-group-projection')
    expect(dynamic?.data.signals.map((signal: { name: string }) => signal.name)).toEqual(['new_output'])
    expect(nodes.find(node => node.id === 'projection-group-workload')?.data.isHighlighted).toBe(true)
    expect(nodes.find(node => node.id === 'projection-group-recovery')?.data.isHighlighted).toBe(false)
  })
})
