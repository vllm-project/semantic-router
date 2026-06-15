import { describe, expect, it } from 'vitest'
import { buildDecisionPreviewRows, buildSignalBreakdownRows } from './dashboardPageOverview'

describe('dashboard overview row models', () => {
  it('sorts signal breakdown rows and scales them against the largest count', () => {
    expect(buildSignalBreakdownRows({
      pii: 2,
      custom: 1,
      keywords: 4,
    })).toEqual([
      { type: 'keywords', count: 4, percent: 100, color: '#4EC9B0' },
      { type: 'pii', count: 2, percent: 50, color: '#FF6B6B' },
      { type: 'custom', count: 1, percent: 25, color: '#999' },
    ])
  })

  it('omits empty signal breakdown entries', () => {
    expect(buildSignalBreakdownRows({
      embeddings: 0,
      keywords: 0,
    })).toEqual([])
  })

  it('builds stable decision preview rows for guardrails, routes, and fallbacks', () => {
    expect(buildDecisionPreviewRows([
      {
        name: 'PII guard',
        description: 'Block sensitive data',
        priority: 1000,
        modelRefs: [{ model: 'guard-model' }],
      },
      {
        name: 'Default path',
        priority: 50,
      },
      {
        priority: 300,
        modelRefs: [{ model: 'balanced' }, { model: 'fast' }],
      },
    ])).toEqual([
      {
        key: 'PII guard-0',
        name: 'PII guard',
        title: 'Block sensitive data',
        priorityLabel: 1000,
        category: 'guardrail',
        typeLabel: 'Guard',
        modelNames: 'guard-model',
      },
      {
        key: 'Default path-1',
        name: 'Default path',
        title: 'Default path',
        priorityLabel: 50,
        category: 'fallback',
        typeLabel: 'Default',
        modelNames: '—',
      },
      {
        key: 'Decision 3-2',
        name: 'Decision 3',
        title: '',
        priorityLabel: 300,
        category: 'routing',
        typeLabel: 'Route',
        modelNames: 'balanced, fast',
      },
    ])
  })

  it('limits decision preview rows without mutating the source list', () => {
    const decisions = [
      { name: 'one' },
      { name: 'two' },
      { name: 'three' },
    ]

    expect(buildDecisionPreviewRows(decisions, 2).map((row) => row.name)).toEqual(['one', 'two'])
    expect(decisions).toHaveLength(3)
  })
})
