import { describe, expect, it } from 'vitest'

import {
  projectionFromDraft,
  removeProjection,
  upsertProjection,
} from './configPageProjectionDraft'
import {
  bandGeometry,
  projectionDecisionConsumers,
  scoreMappings,
} from './configPageProjectionTableSupport'
import type { ConfigData, ProjectionMapping, ProjectionScore } from './configPageSupport'

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

const config = (): ConfigData => ({
  projections: {
    partitions: [
      {
        name: 'practice_areas',
        semantics: 'exclusive',
        temperature: 0.3,
        members: ['business', 'law'],
        default: 'business',
      },
    ],
    scores: [
      {
        name: 'request_difficulty',
        method: 'weighted_sum',
        inputs: [
          {
            type: 'systemone',
            name: 'technical_support',
            weight: 0.18,
            value_source: 'confidence',
            match: 1,
            miss: 0,
          },
          { type: 'complexity', name: 'needs_reasoning:hard', weight: 0.36 },
        ],
      },
    ],
    mappings: [
      {
        name: 'request_band',
        source: 'request_difficulty',
        method: 'threshold_bands',
        calibration: { method: 'sigmoid_distance', slope: 10 },
        outputs: [
          { name: 'support_fast', lte: 0.2 },
          { name: 'support_balanced', gt: 0.2, lt: 0.45 },
          { name: 'support_escalated', gte: 0.45 },
        ],
      },
    ],
  },
  decisions: [
    {
      name: 'support_selection_route',
      description: '',
      priority: 150,
      rules: { operator: 'AND', conditions: [{ type: 'projection', name: 'support_escalated' }] },
      modelRefs: [],
    },
    {
      name: 'law_route',
      description: '',
      priority: 140,
      rules: { operator: 'AND', conditions: [{ type: 'domain', name: 'law' }] },
      modelRefs: [],
    },
  ],
})

describe('projection drafts', () => {
  it('saves untouched projections exactly as written', () => {
    const projections = config().projections!
    expect(projectionFromDraft('partition', clone(projections.partitions![0]))).toEqual(
      projections.partitions![0],
    )
    expect(projectionFromDraft('score', clone(projections.scores![0]))).toEqual(
      projections.scores![0],
    )
    expect(projectionFromDraft('mapping', clone(projections.mappings![0]))).toEqual(
      projections.mappings![0],
    )
  })

  it('validates like the earlier forms', () => {
    expect(() =>
      projectionFromDraft('partition', {
        name: 'p',
        semantics: 'exclusive',
        members: ['a'],
        default: 'b',
      }),
    ).toThrow(/must also appear/)
    expect(() =>
      projectionFromDraft('partition', {
        name: 'p',
        semantics: 'softmax_exclusive',
        members: ['a'],
        default: 'a',
      }),
    ).toThrow(/temperature greater than zero/)
    expect(() =>
      projectionFromDraft('score', { name: 's', method: 'weighted_sum', inputs: [] }),
    ).toThrow(/at least one/)
    expect(() =>
      projectionFromDraft('mapping', {
        name: 'm',
        source: 's',
        method: 'multi_emit',
        outputs: [{ name: 'only', gte: 0 }],
      }),
    ).toThrow(/at least two/)
    expect(() =>
      projectionFromDraft('mapping', {
        name: 'm',
        source: ' ',
        method: 'threshold_bands',
        outputs: [],
      }),
    ).toThrow(/Source/)
  })

  it('writes in place, appends new entries and removes by name', () => {
    const edited = { ...clone(config().projections!.scores![0]), name: 'difficulty' }
    const after = upsertProjection(config(), 'score', edited, 'request_difficulty')
    expect(after.projections?.scores?.map((score) => score.name)).toEqual(['difficulty'])
    const added = upsertProjection(config(), 'mapping', {
      name: 'second',
      source: 'request_difficulty',
      method: 'threshold_bands',
      outputs: [{ name: 'x', gte: 0 }],
    } as ProjectionMapping)
    expect(added.projections?.mappings?.map((mapping) => mapping.name)).toEqual([
      'request_band',
      'second',
    ])
    expect(() =>
      upsertProjection(config(), 'score', {
        name: 'request_difficulty',
        method: 'weighted_sum',
        inputs: [],
      } as ProjectionScore),
    ).toThrow(/already exists/)
    expect(
      removeProjection(config(), 'partition', 'practice_areas').projections?.partitions,
    ).toEqual([])
  })

  it('finds the decisions a projection feeds', () => {
    const projections = config().projections!
    expect(
      projectionDecisionConsumers(config(), 'mapping', projections.mappings![0]).map(
        (row) => row.decision.name,
      ),
    ).toEqual(['support_selection_route'])
    expect(
      projectionDecisionConsumers(config(), 'score', projections.scores![0]).map((row) => row.via),
    ).toEqual(['projection("support_escalated")'])
    expect(
      projectionDecisionConsumers(config(), 'partition', projections.partitions![0]).map(
        (row) => row.decision.name,
      ),
    ).toEqual(['law_route'])
    expect(scoreMappings(config(), 'request_difficulty').map((mapping) => mapping.name)).toEqual([
      'request_band',
    ])
  })

  it('lays bands out on one axis', () => {
    const { low, high, bands } = bandGeometry(config().projections!.mappings![0].outputs)
    expect(low).toBe(0)
    expect(high).toBe(1)
    expect(bands.map((band) => [band.name, band.from, band.to])).toEqual([
      ['support_fast', 0, 0.2],
      ['support_balanced', 0.2, 0.45],
      ['support_escalated', 0.45, 1],
    ])
  })
})
