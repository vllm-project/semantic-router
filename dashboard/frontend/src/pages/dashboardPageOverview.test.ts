import { describe, expect, it } from 'vitest'
import { buildDecisionPreviewRows } from './dashboardPageOverview'

describe('dashboard overview row models', () => {
  it('builds stable decision preview rows for guardrails, routes, and fallbacks', () => {
    expect(
      buildDecisionPreviewRows([
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
      ]),
    ).toEqual([
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
    const decisions = [{ name: 'one' }, { name: 'two' }, { name: 'three' }]

    expect(buildDecisionPreviewRows(decisions, 2).map((row) => row.name)).toEqual(['one', 'two'])
    expect(decisions).toHaveLength(3)
  })
})
