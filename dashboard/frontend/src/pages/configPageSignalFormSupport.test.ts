import { describe, expect, it } from 'vitest'

import type {
  AddSignalFormState,
  ConfigData,
  RecipeRoutingConfig,
} from './configPageSupport'
import {
  buildClassifierSignal,
  getSignalReferenceCount,
  getSignalReferenceCountInRoutingProfile,
  normalizeConditions,
  normalizeConversationFeature,
  normalizeConversationPredicate,
  normalizeStringList,
  normalizeStructureFeature,
  normalizeStructurePredicate,
  normalizeSubjects,
  readConversationFeature,
} from './configPageSignalFormSupport'
import { buildSignalFormFields } from './configPageSignalFormFields'

const classifierForm = (overrides: Partial<AddSignalFormState> = {}): AddSignalFormState => ({
  type: 'Classifier',
  name: '',
  description: '',
  operator: 'AND',
  keywords: [],
  case_sensitive: false,
  threshold: 0.8,
  candidates: [],
  aggregation_method: 'mean',
  mmlu_categories: [],
  ...overrides,
})

describe('signal form support', () => {
  it('builds sequence classifiers without LLM-only fields', () => {
    expect(
      buildClassifierSignal(
        classifierForm({
          classifier_type: 'sequence_classifier',
          classifier_model: 'toxicity-endpoint',
          classifier_labels: ['benign', 'toxic'],
          classifier_instructions: 'must not be persisted',
        }),
        'toxicity',
      ),
    ).toEqual({
      name: 'toxicity',
      description: undefined,
      type: 'sequence_classifier',
      model: 'toxicity-endpoint',
      labels: ['benign', 'toxic'],
    })
    expect(() =>
      buildClassifierSignal(
        classifierForm({
          classifier_type: 'sequence_classifier',
          classifier_model: 'toxicity-endpoint',
          classifier_labels: ['toxic'],
        }),
        'toxicity',
      ),
    ).toThrow(/at least two labels/i)
  })

  it('shows the external model field for sequence classifiers', () => {
    const fields = buildSignalFormFields()
    const backend = fields.find((field) => field.name === 'classifier_type')
    const model = fields.find((field) => field.name === 'classifier_model')
    const instructions = fields.find((field) => field.name === 'classifier_instructions')

    expect(backend?.options).toContain('sequence_classifier')
    expect(model?.shouldHide?.(classifierForm({ classifier_type: 'sequence_classifier' }))).toBe(
      false,
    )
    expect(
      instructions?.shouldHide?.(classifierForm({ classifier_type: 'sequence_classifier' })),
    ).toBe(true)
  })

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
    const config: ConfigData = {
      decisions: [
        {
          name: 'route-finance',
          description: '',
          priority: 1,
          rules: { operator: 'AND', conditions: [{ type: 'domain', name: 'finance' }] },
          modelRefs: [],
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

    expect(getSignalReferenceCount(config, 'Domain', 'finance')).toBe(3)
    expect(getSignalReferenceCount(config, 'Domain', 'legal')).toBe(0)
  })

  it('counts recipe-only references before deleting shared signals', () => {
    const config: ConfigData = {
      recipes: [
        {
          name: 'private',
          routing: {
            decisions: [
              {
                name: 'private-route',
                description: '',
                priority: 1,
                rules: {
                  operator: 'AND',
                  conditions: [{ type: 'metadata', name: 'private-cohort' }],
                },
                modelRefs: [],
              },
            ],
          },
        },
      ],
    }

    expect(getSignalReferenceCount(config, 'Metadata', 'private-cohort')).toBe(1)
  })

  it('keeps deletion references local to the selected recipe', () => {
    const config: ConfigData = {
      recipes: [
        {
          name: 'alpha',
          routing: {
            decisions: [],
          },
        },
        {
          name: 'beta',
          routing: {
            decisions: [
              {
                name: 'beta-route',
                description: '',
                priority: 100,
                rules: {
                  operator: 'AND',
                  conditions: [{ type: 'metadata', name: 'shared-local-name' }],
                },
                modelRefs: [],
              },
            ],
          },
        },
      ],
    }

    expect(
      getSignalReferenceCountInRoutingProfile(
        config.recipes?.[0].routing,
        'Metadata',
        'shared-local-name',
      ),
    ).toBe(0)
    expect(
      getSignalReferenceCountInRoutingProfile(
        config.recipes?.[1].routing,
        'Metadata',
        'shared-local-name',
      ),
    ).toBe(1)
  })

  it('reads a valid default for junk conversation input and round-trips a valid rule', () => {
    expect(readConversationFeature(null)).toEqual({
      type: 'exists',
      source: { type: 'image_content' },
    })
    expect(
      readConversationFeature({ type: 'count', source: { type: 'message', role: 'non_user' } }),
    ).toEqual({ type: 'count', source: { type: 'message', role: 'non_user' } })
  })

  it('normalizes conversation features and enforces the validator rules', () => {
    expect(
      normalizeConversationFeature({
        type: 'count',
        source: { type: 'message', role: 'non_user' },
      }),
    ).toEqual({ type: 'count', source: { type: 'message', role: 'non_user' } })

    expect(() =>
      normalizeConversationFeature({
        type: 'count',
        source: { type: 'tool_definition', role: 'user' },
      }),
    ).toThrow(/only valid when the source type is "message"/)

    expect(() =>
      normalizeConversationFeature({ type: 'exists', source: { type: 'telepathy' } }),
    ).toThrow(/Unsupported conversation source type/)

    expect(() =>
      normalizeConversationFeature({
        type: 'count',
        source: { type: 'message', role: 'wizard' },
      }),
    ).toThrow(/Unsupported conversation role/)
  })

  it('drops the predicate for an exists feature and validates count predicates', () => {
    const exists = normalizeConversationFeature({ type: 'exists', source: { type: 'message' } })
    expect(normalizeConversationPredicate(exists, { gte: 2 })).toBeUndefined()

    const count = normalizeConversationFeature({ type: 'count', source: { type: 'message' } })
    expect(normalizeConversationPredicate(count, { gte: 2 })).toEqual({ gte: 2 })
    expect(() => normalizeConversationPredicate(count, { gt: 1, gte: 2 })).toThrow(
      /both gt and gte/,
    )
  })

  it('counts conversation signal references in a routing profile', () => {
    const routing: RecipeRoutingConfig = {
      decisions: [
        {
          name: 'route-images',
          description: '',
          priority: 1,
          rules: { operator: 'AND', conditions: [{ type: 'conversation', name: 'has_images' }] },
          modelRefs: [],
        },
      ],
    }

    expect(getSignalReferenceCountInRoutingProfile(routing, 'Conversation', 'has_images')).toBe(1)
  })
})
