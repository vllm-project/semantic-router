import { describe, expect, it } from 'vitest'

import {
  applySignalDraft,
  copyName,
  removeSignalsFromConfig,
  signalDraftErrors,
  signalDraftFromRecord,
  signalSummary,
} from './configPageSignalForm'
import type { ConfigData } from './configPageSupport'

const config = (): ConfigData =>
  ({
    signals: {
      keywords: [
        {
          name: 'code_keywords',
          operator: 'OR',
          method: 'bm25',
          keywords: ['code', 'debug'],
          bm25_threshold: 0.1,
        },
        { name: 'urgent', operator: 'OR', keywords: ['urgent'], case_sensitive: false },
      ],
      pii: [{ name: 'restricted_pii', predicate: { gte: 0.7 }, ignore_labels: ['EMAIL_ADDRESS'] }],
    },
  }) as unknown as ConfigData

describe('signal drafts', () => {
  it('saves an untouched signal exactly as written and in place', () => {
    const before = config()
    const record = (before.signals?.keywords ?? [])[0] as unknown as Record<string, unknown>
    const draft = signalDraftFromRecord('keyword', record)
    const after = applySignalDraft(before, draft, { type: 'keyword', name: 'code_keywords' })
    expect(after.signals).toEqual(before.signals)
  })

  it('edits a field without moving the signal or touching others', () => {
    const draft = signalDraftFromRecord(
      'keyword',
      (config().signals?.keywords ?? [])[0] as unknown as Record<string, unknown>,
    )
    draft.fields.keywords = ['code', 'debug', 'refactor']
    const after = applySignalDraft(config(), draft, { type: 'keyword', name: 'code_keywords' })
    expect(after.signals?.keywords?.map((signal) => signal.name)).toEqual([
      'code_keywords',
      'urgent',
    ])
    expect(after.signals?.keywords?.[0].keywords).toEqual(['code', 'debug', 'refactor'])
    expect(after.signals?.pii).toEqual(config().signals?.pii)
  })

  it('moves a signal whose type changes and appends new signals', () => {
    const moved = applySignalDraft(
      config(),
      { type: 'pii', name: 'urgent', fields: { predicate: { gte: 0.5 } } },
      { type: 'keyword', name: 'urgent' },
    )
    expect(moved.signals?.keywords?.map((signal) => signal.name)).toEqual(['code_keywords'])
    expect(moved.signals?.pii?.map((signal) => signal.name)).toEqual(['restricted_pii', 'urgent'])
    const created = applySignalDraft(config(), {
      type: 'language',
      name: ' es ',
      fields: { threshold: 0.3 },
    })
    expect(created.signals?.language).toEqual([{ name: 'es', threshold: 0.3 }])
  })

  it('reports missing names, required fields and duplicates', () => {
    expect(signalDraftErrors(config(), { type: 'keyword', name: '', fields: {} })).toEqual(
      expect.arrayContaining([
        'Name is required.',
        'Operator is required.',
        'Keywords is required.',
      ]),
    )
    expect(
      signalDraftErrors(config(), {
        type: 'keyword',
        name: 'urgent',
        fields: { operator: 'OR', keywords: ['x'] },
      }),
    ).toEqual(['Keywords signal “urgent” already exists.'])
    expect(() =>
      applySignalDraft(config(), { type: 'keyword', name: 'urgent', fields: {} }),
    ).toThrow()
  })

  it('removes the chosen signals only', () => {
    const after = removeSignalsFromConfig(config(), [{ type: 'keyword', name: 'urgent' }])
    expect(after.signals?.keywords?.map((signal) => signal.name)).toEqual(['code_keywords'])
    expect(after.signals?.pii).toHaveLength(1)
  })

  it('names copies and summarizes rules from their settings', () => {
    expect(copyName(['a', 'a_copy'], 'a')).toBe('a_copy_2')
    expect(signalSummary({ name: 'x', operator: 'OR', method: 'bm25', keywords: ['a', 'b'] })).toBe(
      'OR · bm25 · 2 keywords',
    )
    expect(signalSummary({ name: 'x', description: 'Explicit text' })).toBe('Explicit text')
    expect(signalSummary({ name: 'x', predicate: { gte: 0.7 } })).toBe('score ≥ 0.7')
  })
})
