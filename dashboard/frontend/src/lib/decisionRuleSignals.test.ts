import { describe, expect, it } from 'vitest'

import { DECISION_SIGNAL_TYPES } from '@/generated/routerConfigContract'

import { SIGNAL_CATALOG } from '../pages/configPageSignalCatalog'
import { projectConfigForRoutingScope, type RoutingScopedConfigLike } from '../utils/routingScopes'
import { canonicalToRuleDocument } from './decisionRuleAst'
import { decisionRuleSignalDescriptors, decisionRuleSignalScope } from './decisionRuleSignals'
import { validateRuleDocument } from './decisionRuleValidation'

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
    model_bindings: projected.routing?.model_bindings,
  } as Parameters<typeof decisionRuleSignalDescriptors>[0]
}

const names = (recipe: string) =>
  decisionRuleSignalDescriptors(scoped(recipe)).map((descriptor) => `${descriptor.signalType}:${descriptor.name}`)

describe('decisionRuleSignalScope', () => {
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

  it('describes classifier labels, local classifiers and the label_scores exemption', () => {
    const byName = new Map(decisionRuleSignalDescriptors(scoped('balanced')).map((d) => [`${d.signalType}:${d.name}`, d]))
    expect(byName.get('classifier:intent')).toEqual({
      signalType: 'classifier',
      name: 'intent',
      labels: ['code', 'chat'],
      labelRequired: true,
      localClassifier: false,
    })
    expect(byName.get('classifier:local-intent')?.localClassifier).toBe(true)
    expect(byName.get('classifier:bound')?.predicateRequired).toBe(false)
    expect(byName.get('classifier:unbound')?.predicateRequired).toBeUndefined()
  })

  it('describes decision question option keys like Go OptionKeys', () => {
    const byName = new Map(decisionRuleSignalDescriptors(scoped('balanced')).map((d) => [`${d.signalType}:${d.name}`, d]))
    expect(byName.get('decision:route')).toMatchObject({ labels: ['fast', 'deep'], labelRequired: true })
    expect(byName.get('decision:tags')).toMatchObject({ labels: ['pii', 'code'], labelRequired: true })
    expect(byName.get('decision:difficulty')).toMatchObject({ labels: ['0', '1', '2'], labelForbidden: true })
    expect(byName.get('decision:is_code')).toMatchObject({ labels: ['false', 'true'], labelForbidden: true })
  })

  it('validates a named recipe rule against that recipe only', () => {
    const scope = decisionRuleSignalScope(scoped('balanced'), { strictNames: true })
    expect(scope.strictNames).toBe(true)
    const issues = validateRuleDocument(
      canonicalToRuleDocument({
        operator: 'OR',
        conditions: [
          { type: 'keyword', name: 'balanced-keyword' },
          { type: 'keyword', name: 'root-keyword' },
          { type: 'classifier', name: 'bound', label: 'a', on_error: 'match' },
        ],
      }),
      scope,
    )
    expect(issues.map((issue) => `${issue.severity}: ${issue.message}`)).toEqual([
      'error: signal keyword("root-keyword") is not declared in this recipe',
    ])
    expect(decisionRuleSignalScope(scoped('balanced')).strictNames).toBe(false)
  })
  it('declares an authz role binding by both its name and its role, like collectRoleBindingNames', () => {
    const scope = decisionRuleSignalScope(
      { signals: { role_bindings: [{ name: 'admin-binding', role: 'admin', subjects: [] }] } },
      { strictNames: true },
    )
    expect(scope.descriptors.map((descriptor) => `${descriptor.signalType}:${descriptor.name}`)).toEqual([
      'authz:admin-binding',
      'authz:admin',
    ])
    for (const name of ['admin', 'admin-binding']) {
      expect(validateRuleDocument(canonicalToRuleDocument({ type: 'authz', name }), scope)).toEqual([])
    }
  })

  it('applies a global classifier binding when the recipe has no local one, like EffectiveModelBindings', () => {
    const operatingPoint = { deployment: 'cls', contract: 'label_scores.v1', operating_point: { name: 'default' } }
    const signals = { classifiers: [{ name: 'content-risk', type: 'sequence_classifier', labels: ['unsafe'] }] }
    const globalOnly = decisionRuleSignalDescriptors({
      signals,
      global_model_bindings: { 'classifier.content-risk': operatingPoint },
    } as Parameters<typeof decisionRuleSignalDescriptors>[0])
    expect(globalOnly[0].predicateRequired).toBe(false)

    const localOverride = decisionRuleSignalDescriptors({
      signals,
      model_bindings: { 'classifier.content-risk': { deployment: 'cls', contract: 'label_scores.v1' } },
      global_model_bindings: { 'classifier.content-risk': operatingPoint },
    } as Parameters<typeof decisionRuleSignalDescriptors>[0])
    expect(localOverride[0].predicateRequired).toBeUndefined()
  })
})
