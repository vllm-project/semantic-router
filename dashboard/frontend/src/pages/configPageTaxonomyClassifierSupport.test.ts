import { describe, expect, it } from 'vitest'

import {
  classifierDraftFromRecord,
  payloadFromDraft,
  type TaxonomyClassifierRecord,
} from './configPageTaxonomyClassifierSupport'
import {
  buildGroupRows,
  buildKnowledgeBaseRows,
  buildLabelRows,
  renameLabelInDraft,
} from './configPageKnowledgeBaseManagerSupport'

const record: TaxonomyClassifierRecord = {
  name: 'privacy',
  type: 'taxonomy',
  builtin: false,
  managed: true,
  editable: true,
  threshold: 0.55,
  description: 'Privacy policy routing',
  source: { path: '/tmp/privacy.yaml' },
  labels: [
    { name: 'private', description: 'Private data', exemplars: ['my account number'] },
    { name: 'public', exemplars: ['press release'] },
  ],
  groups: { sensitive: ['private'] },
  metrics: [],
  signal_references: [],
  bind_options: { labels: ['private', 'public'], groups: ['sensitive'], metrics: [] },
}

describe('taxonomy classifier form support', () => {
  it('round-trips labels, exemplars, and group membership as arrays', () => {
    const draft = classifierDraftFromRecord(record)
    expect(draft.groups[0].labels).toEqual(['private'])
    expect(draft.labels[0].exemplars).toEqual(['my account number'])

    const payload = payloadFromDraft(draft)
    expect(payload.groups).toEqual({ sensitive: ['private'] })
    expect(payload.labels[0].exemplars).toEqual(['my account number'])
  })

  it('rewrites typed group membership when a label is renamed', () => {
    const next = renameLabelInDraft(classifierDraftFromRecord(record), 'private', {
      name: 'restricted',
      description: 'Restricted data',
      exemplars: ['account secret'],
    })
    expect(next.groups[0].labels).toEqual(['restricted'])
    expect(next.labels[0]).toMatchObject({ name: 'restricted', exemplars: ['account secret'] })
  })

  it('rejects empty, duplicate, and unknown structured values before persistence', () => {
    const emptyExemplar = classifierDraftFromRecord(record)
    emptyExemplar.labels[0].exemplars = ['']
    expect(() => payloadFromDraft(emptyExemplar)).toThrow(/empty/i)

    const duplicate = classifierDraftFromRecord(record)
    duplicate.groups[0].labels = ['private', 'PRIVATE']
    expect(() => payloadFromDraft(duplicate)).toThrow(/unique/i)

    const unknown = classifierDraftFromRecord(record)
    unknown.groups[0].labels = ['missing']
    expect(() => payloadFromDraft(unknown)).toThrow(/unknown label/i)
  })

  it('applies consistent search filters across bases, groups, and labels', () => {
    expect(buildKnowledgeBaseRows([record], 'privacy')).toHaveLength(1)
    expect(buildKnowledgeBaseRows([record], 'account number')).toHaveLength(1)
    expect(buildGroupRows(record, 'private')).toHaveLength(1)
    expect(buildLabelRows(record, 'private data')).toHaveLength(1)
    expect(buildLabelRows(record, 'press release')).toHaveLength(1)
    expect(buildLabelRows(record, 'not-present')).toHaveLength(0)
  })
})
