import { describe, expect, it } from 'vitest'

import { DECISION_SIGNAL_TYPES } from '@/generated/routerConfigContract'

import { SIGNAL_CATALOG } from '../pages/configPageSignalCatalog'
import { projectConfigForRoutingScope, type RoutingScopedConfigLike } from '../utils/routingScopes'
import { decisionRuleSignalDescriptors } from './decisionRuleSignals'

const config: RoutingScopedConfigLike = {
  routing: {
    signals: { keywords: [{ name: 'root-keyword' }], classifiers: [{ name: 'root-intent', labels: ['x'] }] },
    decisions: [],
  },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
  ],
  recipes: [
    {
      name: 'balanced',
      routing: {
        model_bindings: {
          'classifier.bound': { deployment: 'cls', contract: 'label_scores.v1', operating_point: { name: 'default' } },
          'classifier.unbound': { deployment: 'cls', contract: 'label_scores.v1' },
        },
        signals: {
          keywords: [{ name: 'balanced-keyword' }],
          complexity: [{ name: 'reasoning' }],
          classifiers: [
            { name: 'intent', type: 'llm', labels: ['code', 'chat'] },
            { name: 'local-intent', type: 'local', labels: ['code'] },
            { name: 'bound', type: 'sequence_classifier', labels: ['a'] },
            { name: 'unbound', type: 'sequence_classifier', labels: ['a'] },
          ],
          decision: [
            { name: 'route', question: { type: 'choice', choices: [{ key: 'fast' }, { key: 'deep' }] } },
            { name: 'tags', question: { type: 'set', labels: [{ key: 'pii' }, { key: 'code' }] } },
            { name: 'difficulty', question: { type: 'score', levels: ['low', 'mid', 'high'] } },
            { name: 'is_code', question: { type: 'noul' } },
          ],
          safety: [{ name: 'balanced-safety' }],
          conversation: [{ name: 'balanced-conversation' }],
          events: [{ name: 'balanced-event' }],
          input_modality: [{ name: 'balanced-image' }],
          actions: [{ name: 'balanced-action' }],
          hallucination: [{ name: 'not-referenceable' }],
        },
        projections: {
          mappings: [{ name: 'effort', outputs: [{ name: 'effort_low' }, { name: 'effort_high' }] }],
        },
        decisions: [],
      },
    },
    {
      name: 'privacy',
      routing: { signals: { pii: [{ name: 'private-pii' }] }, decisions: [] },
    },
  ],
}

const scoped = (recipe: string) => {
  const projected = projectConfigForRoutingScope(config, recipe)
  return {
    signals: projected.signals,
    projections: projected.projections,
  } as Parameters<typeof decisionRuleSignalDescriptors>[0]
}

const names = (recipe: string) =>
  decisionRuleSignalDescriptors(scoped(recipe)).map((descriptor) => `${descriptor.signalType}:${descriptor.name}`)

describe('decisionRuleSignalDescriptors', () => {
  it('uses the selected recipe signals, never the root or a sibling recipe', () => {
    const balanced = names('balanced')
    expect(balanced).toContain('keyword:balanced-keyword')
    expect(balanced).not.toContain('keyword:root-keyword')
    expect(balanced).not.toContain('classifier:root-intent')
    expect(balanced).not.toContain('pii:private-pii')
    expect(names('privacy')).toEqual(['pii:private-pii'])
  })

  it('covers the catalog types, including safety, conversation, event, input_modality and action', () => {
    expect(names('balanced')).toEqual(
      expect.arrayContaining([
        'safety:balanced-safety',
        'conversation:balanced-conversation',
        'event:balanced-event',
        'input_modality:balanced-image',
        'action:balanced-action',
      ]),
    )
    expect(names('balanced').some((name) => name.startsWith('hallucination:'))).toBe(false)
    const catalogTypes = new Set(SIGNAL_CATALOG.map((entry) => entry.type as string))
    for (const descriptor of decisionRuleSignalDescriptors(scoped('balanced'))) {
      expect(DECISION_SIGNAL_TYPES).toContain(descriptor.signalType)
      if (descriptor.signalType !== 'projection') expect(catalogTypes).toContain(descriptor.signalType)
    }
  })

  it('expands complexity names with every reference suffix', () => {
    expect(names('balanced')).toEqual(
      expect.arrayContaining(['complexity:reasoning:easy', 'complexity:reasoning:medium', 'complexity:reasoning:hard']),
    )
    expect(names('balanced')).not.toContain('complexity:reasoning')
  })

  it('lists projection mapping outputs', () => {
    expect(names('balanced')).toEqual(expect.arrayContaining(['projection:effort_low', 'projection:effort_high']))
  })

  it('offers classifier labels and choice option keys as label suggestions', () => {
    const byName = new Map(decisionRuleSignalDescriptors(scoped('balanced')).map((d) => [`${d.signalType}:${d.name}`, d]))
    expect(byName.get('classifier:intent')).toEqual({ signalType: 'classifier', name: 'intent', labels: ['code', 'chat'] })
    expect(byName.get('decision:route')?.labels).toEqual(['fast', 'deep'])
    expect(byName.get('decision:tags')?.labels).toEqual(['pii', 'code'])
    // Score and yes/no questions are matched without a label.
    expect(byName.get('decision:difficulty')?.labels).toBeUndefined()
    expect(byName.get('decision:is_code')?.labels).toBeUndefined()
  })

  it('offers an authz role binding by both its name and its role', () => {
    const descriptors = decisionRuleSignalDescriptors({
      signals: { role_bindings: [{ name: 'admin-binding', role: 'admin', subjects: [] }] },
    } as Parameters<typeof decisionRuleSignalDescriptors>[0])
    expect(descriptors.map((descriptor) => `${descriptor.signalType}:${descriptor.name}`)).toEqual([
      'authz:admin-binding',
      'authz:admin',
    ])
  })
})
