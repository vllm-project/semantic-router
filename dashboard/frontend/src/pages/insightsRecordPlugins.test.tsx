import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { buildInsightsPluginFields } from './insightsRecordPlugins'
import type { InsightsRecord } from './insightsPageTypes'

const base: InsightsRecord = {
  id: 'plugin-record',
  timestamp: '2026-09-01T00:00:00Z',
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  signals: {},
}
function field(record: Partial<InsightsRecord>, label = 'Guardrails') {
  const value = buildInsightsPluginFields({ ...base, ...record }).find(
    (entry) => entry.label === label,
  )?.value
  return renderToStaticMarkup(<>{value}</>)
}

describe('recorded plugin evidence', () => {
  it('shows detections independently of plugin flags', () => {
    const html = field({
      jailbreak_detected: true,
      jailbreak_type: 'unsafe',
      jailbreak_score_available: true,
      jailbreak_confidence: 0,
    })
    expect(html).toContain('Request jailbreak: unsafe')
    expect(html).toContain('0.0%')
    expect(html).not.toMatch(/Clean|Disabled/)
  })

  it('does not invent a score for unmarked historical detections', () => {
    const html = field({ jailbreak_detected: true, jailbreak_confidence: 1 })
    expect(html).toContain('Score unavailable')
    expect(html).not.toContain('100.0%')
  })

  it('does not call configured checks with no recorded result clean', () => {
    const html = field({ guardrails_enabled: true, jailbreak_enabled: true })
    expect(html).toContain('No request jailbreak result recorded')
    expect(html).not.toMatch(/Clean|Not detected/)
  })

  it('preserves mixed response outcomes, measured zero and how a streamed answer was delivered', () => {
    const html = field({
      outcomes: [
        {
          source: 'router',
          target: 'jailbreak:guard_a',
          verdict: 'unavailable',
          reason: 'classifier_failed',
          metadata: {
            signal: 'jailbreak',
            stage: 'response',
            score_available: 'false',
            action: 'block',
            stream: 'buffered',
          },
        },
        {
          source: 'router',
          target: 'jailbreak:guard_b',
          verdict: 'not_detected',
          metadata: {
            signal: 'jailbreak',
            stage: 'response',
            score_available: 'true',
            stream: 'unbuffered',
          },
        },
        {
          source: 'user',
          target: 'ignored-feedback',
          verdict: 'detected',
          metadata: { signal: 'jailbreak', stage: 'response' },
        },
      ],
    })
    expect(html).toContain('1 unavailable · 1 not detected')
    expect(html).toContain('classifier_failed')
    expect(html).toContain('Score: 0.0000')
    expect(html).toContain('Action: block')
    expect(html).toContain('Stream held for the guard')
    expect(html).toContain('Streamed before the check')
    expect(html).not.toContain('ignored-feedback')
    expect(html).not.toContain('open=""')
  })

  it('uses recorded hallucination applicability over enabled flags', () => {
    const html = field(
      {
        hallucination_enabled: true,
        outcomes: [
          {
            source: 'router',
            target: 'hallucination:grounding',
            verdict: 'not_applicable',
            reason: 'fact_check_not_needed',
            metadata: { stage: 'response', signal: 'hallucination' },
          },
        ],
      },
      'Hallucination Detection',
    )
    expect(html).toContain('1 not applicable')
    expect(html).toContain('fact_check_not_needed')
    expect(html).not.toContain('Not detected')
  })

  it('preserves hallucination score semantics and explicit availability', () => {
    const unscored = field(
      { hallucination_detected: true, hallucination_confidence: 0.8 },
      'Hallucination Detection',
    )
    expect(unscored).toContain('Score unavailable')
    const measured = field(
      {
        hallucination_detected: true,
        hallucination_score_available: true,
        hallucination_confidence: 0.8,
        hallucination_score_kind: 'max_hallucinated_token_score',
      },
      'Hallucination Detection',
    )
    expect(measured).toContain('Score: 0.8000 (max_hallucinated_token_score)')
    expect(measured).not.toContain('%')
    const zero = field({ hallucination_score_available: true }, 'Hallucination Detection')
    expect(zero).toContain('Score: 0.0000')
  })

  it.each([false, true])(
    'keeps every captured span, including alongside outcomes (%s)',
    (withOutcomes) => {
      const spans = ['one', 'two', 'three']
      const html = field(
        {
          hallucination_detected: true,
          hallucination_spans: spans,
          outcomes: withOutcomes
            ? [
                {
                  source: 'router',
                  target: 'hallucination:grounding',
                  verdict: 'detected',
                  metadata: { stage: 'response', signal: 'hallucination' },
                },
              ]
            : undefined,
        },
        'Hallucination Detection',
      )
      spans.forEach((value) => expect(html).toContain(`<li>${value}</li>`))
      expect(html).toContain('Inspect 3 unsupported spans')
      expect(html).not.toContain('open=""')
      expect(spans).toEqual(['one', 'two', 'three'])
    },
  )

  it('preserves zero cache similarity instead of showing an absent measurement', () => {
    expect(field({ cache_similarity: 0 }, 'Cache similarity')).toBe('0.000')
    expect(field({}, 'Cache similarity')).toBe('Not recorded')
  })
})
