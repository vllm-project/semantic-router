import { describe, expect, it } from 'vitest'

import { buildInsightsRecordSections } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

describe('Insights outcome presentation', () => {
  it('shows feedback source, target, verdict, and time on the replay record', () => {
    const record = {
      id: 'replay-1',
      signals: {},
      outcomes: [
        {
          timestamp: '2026-09-11T16:30:00Z',
          source: 'operator',
          target: 'model',
          verdict: 'good_fit',
        },
      ],
    } as InsightsRecord

    const section = buildInsightsRecordSections(record, { isReadonly: true }).find(
      (candidate) => candidate.title === 'Outcomes',
    )

    expect(section?.fields).toHaveLength(1)
    expect(section?.fields[0]?.label).toBe('Outcome 1')
    expect(section?.fields[0]?.value).toContain('operator → model')
    expect(section?.fields[0]?.value).toContain('good_fit')
    expect(section?.fields[0]?.value).not.toContain('Unknown time')
  })
})
