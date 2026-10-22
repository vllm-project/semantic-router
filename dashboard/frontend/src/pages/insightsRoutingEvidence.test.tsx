import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { buildProjectionTraceFields } from './insightsPageProjectionTrace'
import { projectionMetric } from './insightsRoutingMetrics'
import { buildRoutingMetadataFields } from './insightsRoutingMetadata'
import { buildInsightsRecordSections } from './insightsPageSupport'
import type { InsightsRecord } from './insightsPageTypes'

const record: InsightsRecord = {
  id: 'evidence-record',
  timestamp: '2026-09-19T00:00:00Z',
  turn_index: 0,
  decision_tier: 0,
  decision_priority: 0,
  signals: {},
  projection_trace: {
    schema_version: '1',
    partitions: [
      {
        group_name: 'intent',
        signal_type: 'embedding',
        winner: 'general',
        default_used: true,
        winner_score: 0,
        margin: 0,
        contenders: [{ name: 'general', raw_score: 0 }],
      },
    ],
    scores: [
      {
        name: 'route_score',
        total: -0.25,
        inputs: [
          {
            type: 'knowledge',
            name: 'coverage',
            kb: 'manual',
            metric: 'similarity',
            value: 0,
            weight: -0.5,
            contribution: 0,
          },
        ],
      },
    ],
    mappings: [
      {
        mapping_name: 'route',
        source_score: 'route_score',
        score_value: -0.25,
        confidence: 0,
        outputs: [{ name: 'specialist', matched: false, boundary_distance: -0.25 }],
      },
    ],
  },
}

const projection = (value: InsightsRecord) =>
  renderToStaticMarkup(<>{buildProjectionTraceFields(value).map((field) => field.value)}</>)
const metadata = (value: InsightsRecord) =>
  renderToStaticMarkup(<>{buildRoutingMetadataFields(value).map((field) => field.value)}</>)

describe('insight routing evidence presentation', () => {
  it('keeps defaults, no-match outputs and zero/negative measurements distinct from missing evidence', () => {
    const html = projection(record)
    expect(html).toContain('Default fallback')
    expect(html).toContain('No output selected')
    expect(html).toContain('No match')
    expect(html).toContain('0.0000')
    expect(html).toContain('-0.2500')
    expect(html).toContain('Not recorded')
    expect(html).toContain('knowledge · manual · similarity')
    expect(html).toContain('href="#projection-evidence-record-score-0"')
    expect(html).toContain('id="projection-evidence-record-score-0"')
    expect(html).not.toContain(' open=""')
    expect(html).not.toContain('NaN')
  })

  it('does not create a source link or stages when the corresponding evidence is absent', () => {
    const html = projection({
      ...record,
      projection_trace: {
        schema_version: '1',
        mappings: [{ mapping_name: 'legacy', source_score: 'uncaptured', score_value: 0 }],
      },
    })
    expect(html).toContain('uncaptured')
    expect(html).not.toContain('href=')
    expect(html).not.toContain('aria-label="Signal groups"')
    expect(projection({ ...record, projection_trace: { schema_version: '1' } })).toContain(
      'No projection stages were recorded',
    )
    expect(buildProjectionTraceFields({ ...record, projection_trace: undefined })).toEqual([])
  })

  it('joins signal values, confidences and error matches by their exact qualified signal name', () => {
    const html = metadata({
      ...record,
      projections: ['general'],
      projection_scores: { margin: -0.25 },
      signal_values: { 'classifier:zero': 0 },
      signal_confidences: { 'classifier:zero': 0, 'embedding:other': 0.4 },
      signal_error_matches: { 'domain:missing': true, 'classifier:zero': false },
    })
    expect(html).toContain('<dt>margin</dt><dd>-0.2500</dd>')
    expect(html).toContain(
      '<th scope="row">classifier:zero</th><td>0.0000</td><td>0.0000</td><td>No</td>',
    )
    expect(html).toContain(
      '<th scope="row">embedding:other</th><td>Not recorded</td><td>0.4000</td>',
    )
    expect(html).toContain(
      '<th scope="row">domain:missing</th><td>Not recorded</td><td>Not recorded</td><td>Yes</td>',
    )
    expect(html).not.toContain('%')
    expect(buildRoutingMetadataFields(record)).toEqual([])
  })

  it('places projection evidence after the route outcome and before lower-level session details', () => {
    const titles = buildInsightsRecordSections(
      { ...record, session_id: 'session', signal_values: { 'classifier:zero': 0 } },
      { isReadonly: true },
    ).map((section) => section.title)
    expect(titles.indexOf('Model Selection')).toBeLessThan(titles.indexOf('Projection Trace'))
    expect(titles.indexOf('Projection Trace')).toBeLessThan(titles.indexOf('Routing Metadata'))
    expect(titles.indexOf('Routing Metadata')).toBeLessThan(titles.indexOf('Session Routing'))
  })

  it('marks nonfinite numbers unavailable without converting absent measurements to zero', () => {
    for (const value of [undefined, null, NaN, Infinity, -Infinity])
      expect(projectionMetric(value)).toBe('Not recorded')
    expect(projectionMetric(0)).toBe('0.0000')
    expect(projectionMetric(-0.125)).toBe('-0.1250')
  })
})
