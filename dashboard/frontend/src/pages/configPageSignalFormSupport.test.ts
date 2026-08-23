import { describe, expect, it } from 'vitest'

import type { RoutingConfig } from './configPageSupport'
import {
  getSignalReferenceCountInRoutingProfile,
  normalizeConditions,
  normalizeStringList,
  normalizeStructureFeature,
  normalizeStructurePredicate,
  normalizeSubjects,
} from './configPageSignalFormSupport'

describe('signal form support', () => {
  it('normalizes typed string lists without accepting empty or duplicate values', () => {
    expect(normalizeStringList([' urgent ', 'billing'], 'Keywords', true)).toEqual([
      'urgent',
      'billing',
    ])
    expect(() => normalizeStringList(['urgent', 'URGENT'], 'Keywords')).toThrow(/duplicate/i)
    expect(() => normalizeStringList([''], 'Keywords')).toThrow(/empty/i)
  })

  it('validates typed conditions and subjects while preserving object shape', () => {
    expect(normalizeConditions([{ type: ' domain ', name: ' finance ' }])).toEqual([
      { type: 'domain', name: 'finance' },
    ])
    expect(normalizeSubjects([{ kind: 'Group', name: ' admins ' }])).toEqual([
      { kind: 'Group', name: 'admins' },
    ])
    expect(() => normalizeConditions([{ type: 'domain', name: '' }])).toThrow(/both/i)
    expect(() => normalizeSubjects([])).toThrow(/at least one/i)
  })

  it('normalizes every supported structure source into the canonical nested shape', () => {
    expect(
      normalizeStructureFeature({
        type: 'density',
        source: { type: 'keyword_set', keywords: [' at least ', 'within'], case_sensitive: false },
      }),
    ).toEqual({
      type: 'density',
      source: { type: 'keyword_set', keywords: ['at least', 'within'], case_sensitive: false },
    })

    const sequence = normalizeStructureFeature({
      type: 'sequence',
      source: { type: 'sequence', sequences: [[' first ', 'then']], case_sensitive: true },
    })
    expect(sequence.source.sequences).toEqual([['first', 'then']])
    expect(normalizeStructurePredicate(sequence, { gte: 2 })).toEqual({ gte: 2 })
    expect(
      normalizeStructurePredicate(
        { type: 'exists', source: { type: 'regex', pattern: 'x' } },
        { gte: 1 },
      ),
    ).toBeUndefined()
  })

  it('counts decision, projection, and composer references before deletion', () => {
    const document: RoutingConfig = {
      decisions: [
        {
          name: 'route-finance',
          description: '',
          priority: 1,
          rules: { operator: 'AND', conditions: [{ type: 'domain', name: 'finance' }] },
        },
      ],
      projections: {
        scores: [
          {
            name: 'risk',
            method: 'weighted_sum',
            inputs: [{ type: 'domain', name: 'finance', weight: 1 }],
          },
        ],
      },
      signals: {
        complexity: [
          {
            name: 'difficulty',
            threshold: 0.1,
            hard: { candidates: ['hard'] },
            easy: { candidates: ['easy'] },
            composer: {
              operator: 'AND',
              conditions: [{ type: 'domain', name: 'finance' }],
            },
          },
        ],
      },
    }

    expect(getSignalReferenceCountInRoutingProfile(document, 'Domain', 'finance')).toBe(3)
    expect(getSignalReferenceCountInRoutingProfile(document, 'Domain', 'legal')).toBe(0)
  })

  it('keeps deletion references local to the selected recipe', () => {
    const alpha: RoutingConfig = { decisions: [] }
    const beta: RoutingConfig = {
      decisions: [
        {
          name: 'beta-route',
          description: '',
          priority: 100,
          rules: {
            operator: 'AND',
            conditions: [{ type: 'metadata', name: 'shared-local-name' }],
          },
        },
      ],
    }

    expect(getSignalReferenceCountInRoutingProfile(alpha, 'Metadata', 'shared-local-name')).toBe(0)
    expect(getSignalReferenceCountInRoutingProfile(beta, 'Metadata', 'shared-local-name')).toBe(1)
  })
})
