import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { buildRoutingExplanationSections } from './insightsPageRouting'
import type { InsightsRecord } from './insightsPageTypes'

const recordBase: InsightsRecord = {
  id: 'candidate-record',
  timestamp: '2026-09-14T00:00:00Z',
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  signals: {},
}

describe('session candidate inventory', () => {
  it('shows unscored eligible models without giving them synthetic scores', () => {
    const record: InsightsRecord = {
      ...recordBase,
      session_id: 'session-one',
      conversation_id: 'conversation-two',
      turn_index: 1,
      selected_model: 'current',
      session_policy: {
        mode: 'apply',
        candidate_models: ['current', 'proposal', 'unscored'],
        base_scores: { proposal: 0 },
        final_scores: { current: 0, proposal: 0 },
      },
    }
    const sections = buildRoutingExplanationSections(record)
    const fields = sections.find((section) => section.title === 'Session Routing')?.fields
    expect(fields?.find((field) => field.label === 'Conversation')?.value).toBe('conversation-two')
    expect(fields?.find((field) => field.label === 'Eligible models')?.value).toBe(
      'current, proposal, unscored',
    )
    const scores = sections.find((section) => section.title === 'Protection Candidate Scores')
    const html = renderToStaticMarkup(<>{scores?.fields[0].value}</>)
    expect(html).toContain('<td>unscored</td><td>—</td><td>—</td>')
    expect(html).toContain('<td>proposal</td><td>0.0000</td><td>0.0000</td>')
  })

  it('does not infer eligibility from legacy score maps', () => {
    const sections = buildRoutingExplanationSections({
      ...recordBase,
      turn_index: 0,
      session_policy: { base_scores: { proposal: 1 } },
    })
    const field = sections
      .find((section) => section.title === 'Session Routing')
      ?.fields.find((item) => item.label === 'Eligible models')
    expect(field?.value).toBe('Not recorded')
  })
})
