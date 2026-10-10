import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { DataTable } from '../components/DataTable'
import ViewPanel from '../components/ViewPanel'
import ConfigPageEvaluationDetails, { filterEvaluationDetails } from './ConfigPageEvaluationDetails'
import type {
  EffectiveEvaluationGroup,
  EffectiveEvaluationRecord,
} from './configPageEffectiveEvaluations'
import {
  evaluationModelColumns,
  evaluationModelViewSections,
} from './configPageEvaluationRecordsSupport'

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

describe('model-grouped evaluation display', () => {
  it('shows one model row with evidence totals and View instead of benchmark rows', () => {
    const html = renderToStaticMarkup(
      <DataTable
        columns={evaluationModelColumns}
        data={[group]}
        keyExtractor={(row) => row.modelName}
        onView={() => {}}
      />,
    )
    expect(html).toContain('model-a')
    expect(html).toContain('6 built-in · 1 configured')
    expect(html).toContain('View')
    expect(html).not.toContain('benchmark-')
    expect(evaluationModelColumns.map((column) => column.header)).not.toContain('Benchmark')
  })

  it('reveals score, effort, profile, status, provenance, source and conflict without hiding records', () => {
    const html = renderToStaticMarkup(
      <ViewPanel title="Evidence" sections={evaluationModelViewSections(group)} />,
    )
    expect(html).toContain('accuracy: <strong>0.82</strong>')
    expect(html).toContain('ratio (0–1)')
    expect(html).toContain('Measurement details')
    expect(html).toContain('public-reference')
    expect(html).toContain('Effort: high')
    expect(html).toContain('Profile: standard')
    expect(html).toContain('Status: withheld')
    expect(html).toContain('Provenance: operator')
    expect(html).toContain('Verification: claimed')
    expect(html).toContain('href="https://example.com/evidence-0"')
    expect(html).toContain('Duplicate available value: accuracy')
    expect(html).toContain('Configured records do not replace built-in results')
    expect(html).toContain('value="5" selected=""')
    expect(html).toContain('benchmark-4')
    expect(html).not.toContain('benchmark-5')
  })

  it('searches all records beyond the first page by benchmark, effort and source', () => {
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
