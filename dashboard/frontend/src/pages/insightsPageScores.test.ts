import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { buildInsightsRecordSections } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

const record = (score: number | null, available?: boolean): InsightsRecord => ({
  id: 'record', timestamp: '2026-09-12T00:00:00Z', turn_index: 0,
  decision_tier: 0, decision_priority: 0, signals: {},
  confidence_score: score, confidence_score_available: available,
  jailbreak_enabled: true, jailbreak_detected: true,
  jailbreak_confidence: score, jailbreak_score_available: available,
})

const fields = (value: InsightsRecord) => buildInsightsRecordSections(value, { isReadonly: true }).flatMap(section => section.fields ?? [])

describe('replay score availability', () => {
  it.each([[null, false], [1, undefined]] as const)('does not invent a confidence for unscored or legacy records', (score, available) => {
    const values = fields(record(score, available))
    expect(values.find(field => field.label === 'Confidence score')?.value).toBe('Score unavailable')
    const guard = values.find(field => field.label === 'Guardrails')?.value
    const html = renderToStaticMarkup(createElement('div', null, guard))
    expect(html).toContain('Score unavailable')
    expect(html).not.toContain('0.0%')
  })

  it('retains an explicitly reported zero score', () => {
    const values = fields(record(0, true))
    expect(values.find(field => field.label === 'Confidence score')?.value).toBe('0.0%')
  })
})
