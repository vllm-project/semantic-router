import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ConfigPageEvaluationDetails, { EvaluationEvidenceCard } from './ConfigPageEvaluationDetails'
import {
  filterEvaluationDetails,
  type EffectiveEvaluationGroup,
  type EffectiveEvaluationRecord,
} from './configPageEffectiveEvaluations'

const record = (index: number): EffectiveEvaluationRecord => ({
  key: `record-${index}`,
  origin: index === 0 ? 'configured' : 'built_in',
  benchmark: {
    id: `benchmark-${index}`,
    display_name: `Benchmark ${index}`,
    domain: 'general',
    default_profile: 'standard',
    profiles: [],
    metrics: [{ id: 'accuracy', unit: 'ratio', direction: 'higher_is_better', range: [0, 1] }],
  },
  evaluation: {
    id: `evidence-${index}`,
    model: 'model-a',
    benchmark: `benchmark-${index}`,
    benchmark_profile: 'standard',
    reasoning_effort: index === 0 ? 'high' : 'default',
    subject: { parameters: { evaluation_harness: 'public-reference' } },
    metrics: { accuracy: 0.82 + index / 100 },
    status: index === 1 ? 'withheld' : 'available',
    evidence: {
      provenance: index === 0 ? 'operator' : 'vendor_claimed',
      verification: 'claimed',
      source: `https://example.com/evidence-${index}`,
      redistributable: false,
    },
  },
  issues: index === 0 ? ['Duplicate available value: accuracy'] : [],
})
const records = Array.from({ length: 7 }, (_, index) => record(index))
const group: EffectiveEvaluationGroup = {
  modelName: 'model-a',
  catalogId: 'model-a',
  model: { name: 'model-a', endpoints: [] },
  records,
  benchmarkCount: 7,
  availableCount: 6,
  builtInCount: 6,
  configuredCount: 1,
}

describe('model benchmark evidence', () => {
  it('reveals score, effort, profile, status, provenance, source and conflicts for a record', () => {
    const html = renderToStaticMarkup(<EvaluationEvidenceCard record={records[0]} />)
    expect(html).toContain('Benchmark 0')
    expect(html).toContain('0.82')
    expect(html).toContain('ratio · 0–1')
    expect(html).toContain('high')
    expect(html).toContain('standard')
    expect(html).toContain('Configured record')
    expect(html).toContain('operator · claimed')
    expect(html).toContain('href="https://example.com/evidence-0"')
    expect(html).toContain('Duplicate available value: accuracy')
    expect(html).toContain('Measurement details')
  })

  it('lists every record of the model without ranking them', () => {
    const html = renderToStaticMarkup(<ConfigPageEvaluationDetails group={group} />)
    expect(html).toContain('benchmark-0')
    expect(html).toContain('benchmark-6')
    expect(html).toContain('withheld')
    expect(html).toContain('Search evidence')
    expect(html).not.toMatch(/rank|leaderboard|index score/i)
  })

  it('searches all records by benchmark, effort and source', () => {
    expect(filterEvaluationDetails(records, 'benchmark-6').map((item) => item.key)).toEqual([
      'record-6',
    ])
    expect(filterEvaluationDetails(records, 'HIGH').map((item) => item.key)).toEqual(['record-0'])
    expect(filterEvaluationDetails(records, 'configured').map((item) => item.key)).toEqual([
      'record-0',
    ])
    expect(filterEvaluationDetails(records, 'evidence-5').map((item) => item.key)).toEqual([
      'record-5',
    ])
    expect(filterEvaluationDetails(records, 'does not exist')).toEqual([])
  })

  it('explains no evidence and missing sources without inventing metrics or links', () => {
    const empty = { ...group, records: [] }
    expect(renderToStaticMarkup(<ConfigPageEvaluationDetails group={empty} />)).toContain(
      'No built-in or configured evaluation evidence',
    )
    const unavailable = record(1)
    unavailable.evaluation.metrics = {}
    unavailable.evaluation.evidence.source = 'javascript:alert(1)'
    const html = renderToStaticMarkup(
      <ConfigPageEvaluationDetails group={{ ...group, records: [unavailable] }} />,
    )
    expect(html).toContain('No metric values recorded')
    expect(html).toContain('javascript:alert(1)')
    expect(html).not.toContain('href="javascript:')
  })
})
