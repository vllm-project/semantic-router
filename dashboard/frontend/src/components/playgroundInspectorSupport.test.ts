import { describe, expect, it } from 'vitest'

import { groupConversations } from './chatComponentSupport'
import { readInspectedRoute, sortedHeaderEntries } from './playgroundInspectorSupport'
import { expandRoutingHeaders } from './vsrTrace'

describe('readInspectedRoute', () => {
  it('reads the decision, model, signals and timings the chat collects', () => {
    const route = readInspectedRoute({
      ...expandRoutingHeaders({
        'x-vsr-selected-decision': 'computer_science_route',
        'x-vsr-selected-model': 'qwen/qwen3-32b',
        'x-vsr-selected-recipe': 'default',
        'x-vsr-replay-id': 'replay-1',
        'x-vsr-cost': '0.00031',
        'x-vsr-cost-currency': 'USD',
        'x-vsr-warnings': 'hallucination',
        'x-vsr-trace':
          'algorithm=multi_factor, confidence=0.86, routing_latency_ms=38, effective_input_tokens=1412, effective_max_output_tokens=32000, signals=("domain:computer science" "keyword:code_keywords" "keyword:urgent"), truncated',
      }),
      'x-vsr-latency-ms': '1412',
      'x-vsr-ttft-ms': '238',
    })

    expect(route.decision).toEqual({ id: 'computer_science_route', label: 'Computer Science' })
    expect(route).toMatchObject({
      recipe: 'default',
      model: 'qwen/qwen3-32b',
      algorithm: 'Multi-Factor',
      confidence: '0.86',
      replayId: 'replay-1',
      warnings: ['hallucination'],
      truncated: true,
    })
    expect(route.signals).toEqual([
      { type: 'domain', label: 'Domain', names: ['computer science'] },
      { type: 'keyword', label: 'Keywords', names: ['code_keywords', 'urgent'] },
    ])
    expect(Object.fromEntries(route.metrics.map((metric) => [metric.id, metric.value]))).toEqual({
      latency: '1.41 s',
      ttft: '238 ms',
      routing: '38 ms',
      input: '1,412',
      'output-limit': '32,000',
      cost: '0.00031 USD',
    })
  })

  it('names the looper models and its final model', () => {
    const route = readInspectedRoute(
      expandRoutingHeaders({
        'x-vsr-selected-model': 'a',
        'x-vsr-trace':
          'looper=("a" "b");algorithm=escalate;model="b";iterations=2;latency_ms=74;total_tokens=117',
      }),
    )
    expect(route.model).toBe('b')
    expect(route.algorithm).toBe('Escalate')
    expect(route.looper).toEqual({ models: ['a', 'b'], iterations: '2' })
    expect(route.metrics.map((metric) => metric.id)).toEqual(['looper-latency', 'looper-tokens'])
  })

  it('lists public headers before signals and trace details', () => {
    expect(
      sortedHeaderEntries({
        'trace:algorithm': 'static',
        'signal:domain': 'law',
        'x-vsr-selected-model': 'm',
      }).map(([key]) => key),
    ).toEqual(['x-vsr-selected-model', 'signal:domain', 'trace:algorithm'])
  })
})

describe('groupConversations', () => {
  it('groups conversations by day, newest first, and drops empty groups', () => {
    const now = new Date('2026-11-18T12:00:00').getTime()
    const day = 86_400_000
    const groups = groupConversations(
      [
        { id: 'a', updatedAt: now - 1000, preview: 'a' },
        { id: 'b', updatedAt: now - day, preview: 'b' },
        { id: 'c', updatedAt: now - 30 * day, preview: 'c' },
      ],
      now,
    )
    expect(groups.map((group) => [group.label, group.items.map((item) => item.id)])).toEqual([
      ['Today', ['a']],
      ['Yesterday', ['b']],
      ['Older', ['c']],
    ])
  })
})
