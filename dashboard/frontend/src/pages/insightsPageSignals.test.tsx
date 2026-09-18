import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import InsightsRecordSection from './InsightsRecordSection'
import { buildInsightsRecordSections, collectSignals } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

describe('recorded signal evidence', () => {
  it('shows protocol and classifier matches in both the summary and details', () => {
    const record: InsightsRecord = {
      id: 'signal-record',
      timestamp: '2026-09-14T00:00:00Z',
      turn_index: 0,
      decision_tier: 0,
      decision_priority: 0,
      signals: {
        conversation: ['active_tool_loop'],
        event: ['service_degraded'],
        metadata: ['customer_tier'],
        classifier: ['sensitive_content'],
        input_modality: ['has_image'],
      },
      signal_values: { 'classifier:sensitive_content': 0, 'embedding:action': 0.42 },
      projection_scores: { action_margin: -0.125 },
    }
    const expected = [
      'Active Tool Loop', 'Service Degraded', 'Customer Tier',
      'Sensitive Content', 'Has Image',
    ]
    expect(collectSignals(record.signals)).toEqual(expect.arrayContaining(expected))
    const html = renderToStaticMarkup(
      <>
        {buildInsightsRecordSections(record, { isReadonly: true }).map((section, index) => (
          <InsightsRecordSection key={section.title} section={section} sectionIndex={index} />
        ))}
      </>,
    )
    expected.forEach(value => expect(html).toContain(value))
    const metadata = buildInsightsRecordSections(record, { isReadonly: true }).find(
      section => section.title === 'Routing Metadata',
    )
    const values = renderToStaticMarkup(<>{metadata?.fields.map(field => field.value)}</>)
    expect(values).toContain('action_margin: -0.125')
    expect(values).toContain('classifier:sensitive_content: 0')
    expect(html).not.toContain('undefined')
  })

  it('does not invent matches for an older replay without these families', () => {
    expect(collectSignals({ domain: ['technical'], conversation: [] })).toEqual(['Technical'])
    expect(collectSignals({})).toEqual([])
  })
})
