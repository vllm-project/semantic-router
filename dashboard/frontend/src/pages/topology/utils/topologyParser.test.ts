import { describe, expect, it } from 'vitest'

import { getDecisionReachability } from './layoutGraphBuilderSupport'
import { parseConfigToTopology } from './topologyParser'

describe('parseConfigToTopology rules', () => {
  it('keeps a single root condition instead of drawing a fallback', () => {
    const topology = parseConfigToTopology({
      routing: {
        signals: { jailbreak: [{ name: 'prompt_injection' }] },
        decisions: [
          {
            name: 'guard',
            priority: 120,
            rules: { type: 'jailbreak', name: 'prompt_injection' },
            modelRefs: [{ model: 'safe' }],
          },
          { name: 'default', priority: 1, rules: {}, modelRefs: [{ model: 'safe' }] },
        ],
      },
      providers: { models: [{ name: 'safe' }] },
    })
    const signals = new Set(topology.signals.map((signal) => `${signal.type}:${signal.name}`))
    const [guard, fallback] = topology.decisions

    expect(guard.rules).toEqual({
      operator: 'AND',
      conditions: [{ type: 'jailbreak', name: 'prompt_injection' }],
    })
    expect(getDecisionReachability(guard, signals)).toEqual({
      isFallback: false,
      isUnreachable: false,
    })
    expect(fallback.rules).toEqual({ operator: 'AND', conditions: [] })
    expect(getDecisionReachability(fallback, signals).isFallback).toBe(true)
  })
})
