import { afterEach, describe, expect, it, vi } from 'vitest'

import { testQueryDryRun } from './api'

describe('testQueryDryRun', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('threads the eval trace, recipe, requested model, and algorithm through unchanged', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      statusText: 'OK',
      json: async () => ({
        query: 'debug this',
        mode: 'dry-run',
        requestedModel: 'Auto',
        recipe: 'balanced',
        matchedSignals: [],
        matchedDecision: 'coding_decision',
        algorithm: 'priority',
        matchedModels: ['gpt-4', 'gpt-4-mini'],
        highlightedPath: ['client', 'decision-coding_decision'],
        isAccurate: true,
        evalTrace: [
          {
            decision_name: 'coding_decision',
            matched: true,
            confidence: 0.9,
            root_trace: {
              node_type: 'leaf',
              signal_type: 'keyword',
              signal_name: 'coding',
              matched: true,
              confidence: 1,
            },
          },
        ],
      }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    const result = await testQueryDryRun('debug this', 'Auto')

    expect(result.requestedModel).toBe('Auto')
    expect(result.recipe).toBe('balanced')
    expect(result.algorithm).toBe('priority')
    expect(result.matchedModels).toEqual(['gpt-4', 'gpt-4-mini'])
    expect(result.evalTrace).toHaveLength(1)
    expect(result.evalTrace?.[0].root_trace?.signal_name).toBe('coding')
  })

  it('forwards optional messages, metadata, and tools alongside simple text', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => ({
      ok: true,
      statusText: 'OK',
      json: async () => ({
        query: 'hi',
        mode: 'dry-run',
        matchedSignals: [],
        matchedDecision: null,
        matchedModels: [],
        highlightedPath: [],
        isAccurate: true,
      }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    await testQueryDryRun('hi', 'Auto', {
      messages: [{ role: 'user', content: 'hi' }],
      metadata: { tenant: 'A' },
      tools: [{ type: 'function' }],
    })

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [, init] = fetchMock.mock.calls[0]
    const body = JSON.parse((init as RequestInit).body as string)
    expect(body.messages).toEqual([{ role: 'user', content: 'hi' }])
    expect(body.metadata).toEqual({ tenant: 'A' })
    expect(body.tools).toEqual([{ type: 'function' }])
  })
})
