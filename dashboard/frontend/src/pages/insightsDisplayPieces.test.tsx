import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { formatSeconds } from '../utils/dateTime'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import InsightsSystemOneAnswers from './InsightsSystemOneAnswers'
import { decisionTone } from './insightsDecisionTone'
import type { InsightsRecord } from './insightsPageTypes'
import { systemOneAnswers } from './insightsSystemOneAnswers'

// What a Router listener recorded for one request: a four-way and a two-way choice and a noul.
const VALUES = {
  'systemone:difficulty': 0.7287,
  'systemone:difficulty:d0': 0.7287,
  'systemone:difficulty:d1': 0.2049,
  'systemone:difficulty:d2': 0.0475,
  'systemone:difficulty:d3': 0.0189,
  'systemone:injection': 0.9184,
  'systemone:injection:benign': 0.9184,
  'systemone:injection:jailbreak': 0.0816,
  'systemone:needs_debugging': 0.0189,
  'keyword:privacy': 1,
}

describe('Insights display pieces', () => {
  it('shows a recorded duration in seconds to the millisecond', () => {
    expect(formatSeconds(1489)).toBe('1.489s')
    expect(formatSeconds(42258)).toBe('42.258s')
    expect(formatSeconds(0)).toBe('0.000s')
  })

  it.each([
    ['shield_injection', 'guard'],
    ['mask_pii', 'privacy'],
    ['think_slow', 'deep'],
    ['think_fast', 'fast'],
    ['tools', 'route'],
  ])('reads decision %s as the %s tone', (decision, tone) => {
    expect(decisionTone(decision)).toBe(tone)
  })

  it('draws a decision as a badge that keeps its recorded name as the title', () => {
    const html = renderToStaticMarkup(
      <InsightsDecisionBadge decision="shield_injection" label="Shield Injection" />,
    )
    expect(html).toContain('data-tone="guard"')
    expect(html).toContain('title="shield_injection"')
    expect(html).toContain('Shield Injection')
  })

  it('groups the System One answers by question and keeps every other signal out', () => {
    expect(systemOneAnswers(VALUES)).toEqual([
      {
        question: 'difficulty',
        kind: 'choice',
        options: [
          { key: 'd0', probability: 0.7287 },
          { key: 'd1', probability: 0.2049 },
          { key: 'd2', probability: 0.0475 },
          { key: 'd3', probability: 0.0189 },
        ],
      },
      {
        question: 'injection',
        kind: 'choice',
        options: [
          { key: 'benign', probability: 0.9184 },
          { key: 'jailbreak', probability: 0.0816 },
        ],
      },
      { question: 'needs_debugging', kind: 'value', value: 0.0189 },
    ])
    expect(systemOneAnswers(undefined)).toEqual([])
  })

  it('lights the answer a choice gave and states every option in the title', () => {
    const record = { signal_values: VALUES } as unknown as InsightsRecord
    const html = renderToStaticMarkup(<InsightsSystemOneAnswers record={record} />)
    expect(html).toContain('System One answers')
    expect(html).toContain('<strong>d0</strong> 73%')
    expect(html).toContain('title="d0 73% · d1 20% · d2 5% · d3 2%"')
    expect(html).toContain('<strong>benign</strong> 92%')
    expect(html).toContain('<strong>0.02</strong>')
    expect(html).not.toContain('keyword')
  })
})
