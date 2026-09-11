import { describe, expect, it } from 'vitest'
import {
  ALGORITHM_TYPES,
  ALGORITHM_ICONS,
  PLUGIN_ICONS,
  PLUGIN_TYPES,
  SIGNAL_TYPES,
} from './topology/constants'
import {
  ALGORITHM_TYPES as ROUTER_ALGORITHM_TYPES,
  PLUGIN_TYPES as ROUTER_PLUGIN_TYPES,
  SIGNAL_TYPES as ROUTER_SIGNAL_TYPES,
} from '../generated/routerConfigContract'
import { groupSignalsByType, parseConfigToTopology } from './topology/utils/topologyParser'
import type { ConfigData } from './topology/types'

describe('topology v0.3 surface alignment', () => {
  it('extracts v0.3 signals, algorithms, and plugins from routing config', () => {
    const config: ConfigData = {
      routing: {
        signals: {
          conversation: [
            {
              name: 'deep_tool_loop',
              feature: { type: 'turn_count', source: { type: 'role', role: 'tool' } },
              predicate: { gte: 3 },
            },
          ],
          events: [
            {
              name: 'incident',
              event_types: ['service_incident'],
              severities: ['critical'],
              temporal: true,
            },
          ],
          metadata: [
            {
              name: 'tenant',
              key: 'tenant',
              predicate: { equals: 'research' },
            },
          ],
          classifiers: [
            {
              name: 'risk',
              type: 'local',
              labels: ['SAFE', 'RISKY'],
            },
          ],
          input_modality: [{ name: 'vision', modality: 'image' }],
        },
        decisions: [
          {
            name: 'agentic',
            rules: {
              operator: 'OR',
              conditions: [{ type: 'conversation', name: 'deep_tool_loop' }],
            },
            algorithm: {
              type: 'multi_factor',
              multi_factor: { latency_percentile: 95 },
            },
            modelRefs: [{ model: 'fast' }],
            plugins: [{ type: 'tool_selection', enabled: true }],
          },
        ],
      },
      providers: {
        models: [{ name: 'fast' }],
      },
    }

    const topology = parseConfigToTopology(config)

    expect(topology.signals.map((signal) => signal.type)).toEqual(
      expect.arrayContaining(['conversation', 'event', 'metadata', 'classifier', 'input_modality']),
    )
    expect(topology.decisions[0].algorithm?.type).toBe('multi_factor')
    expect(topology.decisions[0].algorithm?.multi_factor).toEqual({ latency_percentile: 95 })
    expect(topology.decisions[0].plugins?.[0].type).toBe('tool_selection')
  })

  it('declares display metadata for v0.3 topology surfaces', () => {
    expect(SIGNAL_TYPES).toEqual([...ROUTER_SIGNAL_TYPES, 'projection'])
    expect(PLUGIN_TYPES).toEqual(ROUTER_PLUGIN_TYPES)
    expect(ALGORITHM_TYPES).toEqual(ROUTER_ALGORITHM_TYPES)
    expect(ALGORITHM_ICONS).toMatchObject({
      fusion: 'FU',
      workflows: 'FL',
      mlp: 'MLP',
      multi_factor: 'MF',
    })
    expect(PLUGIN_ICONS).toMatchObject({
      tool_selection: 'TS',
    })
  })

  it('initializes topology groups from the generated signal inventory', () => {
    const groups = groupSignalsByType([])

    expect(Object.keys(groups)).toEqual(SIGNAL_TYPES)
    SIGNAL_TYPES.forEach((signalType) => expect(groups[signalType]).toEqual([]))
  })

  it('preserves the PII source in topology signal config', () => {
    const topology = parseConfigToTopology({
      signals: {
        pii: [{ name: 'tool-data', threshold: 0.8, source: 'tool_result' }],
      },
    })

    expect(topology.signals).toContainEqual(
      expect.objectContaining({
        type: 'pii',
        name: 'tool-data',
        config: expect.objectContaining({ source: 'tool_result' }),
      }),
    )
  })
})
