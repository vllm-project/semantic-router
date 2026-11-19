import { describe, expect, it } from 'vitest'

import type { InsightsRecord } from '../pages/insightsPageTypes'
import { groupConversations } from './chatComponentSupport'
import { readInspectedRoute, sortedHeaderEntries } from './playgroundInspectorSupport'
import { expandRoutingHeaders } from './vsrTrace'

const headers = {
  ...expandRoutingHeaders({
    'x-vsr-selected-decision': 'computer_science_route',
    'x-vsr-selected-model': 'qwen/qwen3-32b',
    'x-vsr-selected-recipe': 'default',
    'x-vsr-replay-id': 'replay-1',
    'x-vsr-cost': '0.00031',
    'x-vsr-cost-currency': 'USD',
    'x-vsr-warnings': 'hallucination',
    'x-vsr-trace':
      'algorithm=multi_factor, confidence=0.86, routing_latency_ms=38, effective_input_tokens=1412, effective_max_output_tokens=32000, signals=("domain:computer science" "keyword:code_keywords" "keyword:urgent"), prompt_cache=hit, ranking=fallback_general;reason=priority, truncated',
  }),
  'x-vsr-latency-ms': '1412',
  'x-vsr-ttft-ms': '238',
}

describe('readInspectedRoute', () => {
  it('reads the decision, model, signals and timings the chat collects', () => {
    const route = readInspectedRoute(headers)

    expect(route.decision).toEqual({ id: 'computer_science_route', label: 'Computer Science' })
    expect(route).toMatchObject({
      recipe: 'default',
      model: 'qwen/qwen3-32b',
      algorithm: 'Multi-Factor',
      confidence: '0.86',
      runnerUp: 'fallback_general: priority',
      replayId: 'replay-1',
      warnings: ['hallucination'],
      truncated: true,
      latencyMs: 1412,
      cost: { value: 0.00031, currency: 'USD' },
      candidates: [],
    })
    expect(route.signals.map((signal) => [signal.key, signal.typeLabel, signal.matched])).toEqual([
      ['domain:computer science', 'Domain', true],
      ['keyword:code_keywords', 'Keywords', true],
      ['keyword:urgent', 'Keywords', true],
    ])
    expect(route.signals.every((signal) => signal.confidence === undefined)).toBe(true)
    expect(route.plugins).toEqual([
      { id: 'prompt_cache', label: 'Prompt cache', detail: 'hit', tone: 'rose' },
    ])
    expect(Object.fromEntries(route.metrics.map((metric) => [metric.id, metric.value]))).toEqual({
      latency: '1.41 s',
      ttft: '238 ms',
      routing: '38 ms',
      cost: '$0.0003',
      input: '1,412',
      'output-limit': '32,000',
    })
  })

  it('adds confidences, scored candidates and plugins from the replay record', () => {
    const record = {
      id: 'replay-1',
      signals: { domain: ['computer science'], keyword: ['code_keywords'] },
      signal_confidences: {
        'domain:computer science': 0.91,
        'domain:law': 0.12,
        'pii:restricted': 0.02,
      },
      learning: {
        adaptation: {
          scores: { 'qwen/qwen3-32b': { score: 0.82 }, 'qwen/qwen3-8b': { score: 0.64 } },
        },
      },
      from_cache: false,
      cache_similarity: 0.42,
      pii_enabled: true,
      prompt_tokens: 400,
      completion_tokens: 212,
    } as unknown as InsightsRecord

    const route = readInspectedRoute(headers, record)

    expect(route.signals.map((signal) => [signal.key, signal.matched, signal.confidence])).toEqual([
      ['domain:computer science', true, 0.91],
      ['keyword:code_keywords', true, undefined],
      ['keyword:urgent', true, undefined],
      ['domain:law', false, 0.12],
      ['pii:restricted', false, 0.02],
    ])
    expect(route.candidates).toEqual([
      { model: 'qwen/qwen3-32b', score: 0.82, chosen: true },
      { model: 'qwen/qwen3-8b', score: 0.64, chosen: false },
    ])
    expect(route.plugins.map((plugin) => [plugin.id, plugin.detail])).toEqual([
      ['prompt_cache', 'hit'],
      ['semantic_cache', 'miss · 0.42'],
      ['pii', undefined],
    ])
    expect(route.metrics.find((metric) => metric.id === 'tokens')?.value).toBe('612')
  })

  it('takes the session policy scores or the surviving candidates when Learning has none', () => {
    const policy = readInspectedRoute(headers, {
      session_policy: { final_scores: { a: 0.3, 'qwen/qwen3-32b': 0.7 } },
    } as unknown as InsightsRecord)
    expect(policy.candidates.map((candidate) => candidate.model)).toEqual(['qwen/qwen3-32b', 'a'])

    const survivors = readInspectedRoute(headers, {
      route_diagnostics: {
        selection_trace: {
          stages: [],
          final_survivors: [{ Model: 'qwen/qwen3-32b' }, { Model: 'b' }],
        },
      },
    } as unknown as InsightsRecord)
    expect(survivors.candidates).toEqual([
      { model: 'qwen/qwen3-32b', chosen: true },
      { model: 'b', chosen: false },
    ])
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
