import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { formatSeconds } from '../utils/dateTime'
import InsightsDecisionBadge from './InsightsDecisionBadge'
import InsightsSystemOneAnswers from './InsightsSystemOneAnswers'
import { buildRoutingMetadataFields } from './insightsRoutingMetadata'
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
    expect(systemOneAnswers(VALUES, VALUES)).toEqual([
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
      { question: 'needs_debugging', kind: 'probability', value: 0.0189 },
    ])
    expect(systemOneAnswers(undefined, undefined)).toEqual([])
  })

  it('lights the answer a choice gave and states every option in the title', () => {
    const record = {
      signal_values: VALUES,
      signal_confidences: VALUES,
    } as unknown as InsightsRecord
    const html = renderToStaticMarkup(<InsightsSystemOneAnswers record={record} />)
    expect(html).toContain('System One answers')
    expect(html).toContain('<strong>d0</strong> 73%')
    expect(html).toContain('title="d0 73% · d1 20% · d2 5% · d3 2%"')
    expect(html).toContain('<strong>benign</strong> 92%')
    expect(html).toContain('<strong>0.02</strong>')
    expect(html).not.toContain('keyword')
  })

  it('keeps an option name with a colon, and tells label sets and scores from choices', () => {
    const values = {
      'systemone:topic:lang:python': 0.7,
      'systemone:topic:lang:go': 0.3,
      'systemone:topics:code': 0.9,
      'systemone:topics:math': 0.6,
      'systemone:effort': 1.3,
      'systemone:steps': 0.7,
      'systemone:risk': -0.2,
    }
    // a score's confidence is its likeliest level, not its expected level
    const confidences = { 'systemone:effort': 0.55, 'systemone:steps': 0.5 }
    expect(systemOneAnswers(values, confidences)).toEqual([
      { question: 'effort', kind: 'score', value: 1.3 },
      { question: 'risk', kind: 'score', value: -0.2 },
      { question: 'steps', kind: 'score', value: 0.7 },
      {
        question: 'topic',
        kind: 'choice',
        options: [
          { key: 'lang:python', probability: 0.7 },
          { key: 'lang:go', probability: 0.3 },
        ],
      },
      {
        question: 'topics',
        kind: 'labels',
        options: [
          { key: 'code', probability: 0.9 },
          { key: 'math', probability: 0.6 },
        ],
      },
    ])
  })

  it('draws a label set by its top label and a score as its level, with no bar', () => {
    const record = {
      signal_values: {
        'systemone:topics:code': 0.9,
        'systemone:topics:math': 0.6,
        'systemone:effort': 1.3,
      },
      signal_confidences: { 'systemone:effort': 0.55 },
    } as unknown as InsightsRecord
    const html = renderToStaticMarkup(<InsightsSystemOneAnswers record={record} />)
    expect(html).toContain('<strong>code</strong> 90%')
    expect(html).toContain('aria-label="code 90% · math 60%"')
    expect(html).toContain('level <strong>1.30</strong>')
  })

  it.each([
    ['unblock_requests', 'route'],
    ['fast_track_guard', 'guard'],
    ['blocklist', 'route'],
  ])('reads whole words only: %s is %s', (decision, tone) => {
    expect(decisionTone(decision)).toBe(tone)
  })

  it('says not recorded for a duration that cannot be one', () => {
    expect(formatSeconds(-5)).toBe('Not recorded')
    expect(formatSeconds(Number.NaN)).toBe('Not recorded')
    expect(formatSeconds(Number.POSITIVE_INFINITY)).toBe('Not recorded')
  })

  it('folds only the raw System One values and keeps other signals and error matches in view', () => {
    const record = {
      signal_values: { ...VALUES, 'pii:personal_data': 0.4 },
      signal_confidences: VALUES,
      signal_error_matches: { 'pii:personal_data': true },
    } as unknown as InsightsRecord
    const [field] = buildRoutingMetadataFields(record)
    const html = renderToStaticMarkup(<>{field.value}</>)
    const folded = html.slice(html.indexOf('<details'))
    expect(html.slice(0, html.indexOf('<details'))).toContain('pii:personal_data')
    expect(html.slice(0, html.indexOf('<details'))).toContain('Error match')
    expect(folded).toContain('Raw System One values (9)')
    expect(folded).toContain('systemone:difficulty:d0')
    expect(folded).not.toContain('pii:personal_data')
  })
})
