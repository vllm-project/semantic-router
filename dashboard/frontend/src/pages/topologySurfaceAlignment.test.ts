import { describe, expect, it } from 'vitest'
import { ALGORITHM_ICONS, PLUGIN_ICONS, SIGNAL_TYPES } from './topology/constants'
import { parseConfigToTopology } from './topology/utils/topologyParser'
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
        },
        decisions: [
          {
            name: 'agentic',
            rules: {
              operator: 'OR',
              conditions: [{ type: 'conversation', name: 'deep_tool_loop' }],
            },
            algorithm: {
              type: 'session_aware',
              session_aware: { base_method: 'multi_factor' },
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
      expect.arrayContaining(['conversation', 'event']),
    )
    expect(topology.decisions[0].algorithm?.type).toBe('session_aware')
    expect(topology.decisions[0].algorithm?.session_aware).toEqual({ base_method: 'multi_factor' })
    expect(topology.decisions[0].plugins?.[0].type).toBe('tool_selection')
  })

  it('declares display metadata for v0.3 topology surfaces', () => {
    expect(SIGNAL_TYPES).toEqual(expect.arrayContaining(['conversation', 'event']))
    expect(ALGORITHM_ICONS).toMatchObject({
      fusion: 'FU',
      mlp: 'MLP',
      multi_factor: 'MF',
      session_aware: 'SA',
    })
    expect(PLUGIN_ICONS).toMatchObject({
      tool_selection: 'TS',
    })
  })
})
