import { describe, expect, it } from 'vitest'

import { getInsightsRecordSectionPresentation } from './insightsRecordSectionSupport'

describe('insight record section layout', () => {
  it.each(['Routing Metadata', 'Projection Trace', 'Request / Response'])(
    'keeps %s on its own full-width row',
    (title) => {
      expect(getInsightsRecordSectionPresentation(title)).toMatchObject({
        size: 'wide',
        collapsible: true,
      })
    },
  )

  it('keeps summary sections compact', () => {
    expect(getInsightsRecordSectionPresentation('Lifecycle')).toEqual({ size: 'compact' })
    expect(getInsightsRecordSectionPresentation('Usage & Cost')).toMatchObject({
      size: 'wide',
      noteFields: ['Cost basis'],
      secondaryFields: ['Baseline basis', 'Current pricing'],
    })
    expect(getInsightsRecordSectionPresentation('Signals').size).toBe('half')
    expect(getInsightsRecordSectionPresentation('Plugin Status').size).toBe('half')
  })

  it('opens the primary trace while keeping supporting detail quiet', () => {
    expect(getInsightsRecordSectionPresentation('Projection Trace')).toMatchObject({
      defaultExpanded: true,
    })
    expect(getInsightsRecordSectionPresentation('Routing Metadata')).toMatchObject({
      defaultExpanded: false,
    })
    expect(getInsightsRecordSectionPresentation('Request / Response')).toMatchObject({
      defaultExpanded: false,
    })
  })
})
