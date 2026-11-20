import { describe, expect, it } from 'vitest'

import { DECISION_ACTION_SCHEMA, DECISION_DECLARATIVE_SCHEMA } from '../lib/decisionSchemas'
import { getAlgorithmFieldSchema } from '../lib/dslAlgorithmSchemas'
import { requiredFieldErrors } from '../lib/schemaFieldValidation'
import { algorithmFields, algorithmType } from './configPageDecisionAlgorithmSupport'
import {
  DecisionDraftError,
  decisionFromDraft,
  decisionModelRefsForSave,
  decisionPluginsForSave,
  decisionRulesForSave,
  upsertDecision,
} from './configPageDecisionFormSupport'
import {
  decisionOutputForEditor,
  decisionOutputFromEditor,
  mergeDecisionForSave,
  type DecisionConfig,
} from './configPageSupport'

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

const validate = {
  actionErrors: (action: Record<string, unknown>) =>
    requiredFieldErrors(DECISION_ACTION_SCHEMA, action),
  algorithmErrors: (algorithm: Record<string, unknown>) =>
    requiredFieldErrors(
      getAlgorithmFieldSchema(algorithmType(algorithm)),
      algorithmFields(algorithm),
    ),
  declarativeErrors: (decision: DecisionConfig) =>
    requiredFieldErrors(DECISION_DECLARATIVE_SCHEMA, {
      candidateIterations: decision.candidateIterations,
      retention: decision.retention,
      output: decision.output ? decisionOutputForEditor(decision.output) : undefined,
      annotations: decision.annotations,
    }),
}

// A decision written with every field the old modal form managed.
const business: DecisionConfig = {
  name: 'static_business_route',
  description: 'Static fallback for standard business traffic.',
  priority: 200,
  tier: 2,
  output_contract: 'Preserve any explicit user-requested output format exactly.',
  annotations: { policy_class: 'business', owner: 'routing-platform' },
  rules: { operator: 'AND', conditions: [{ type: 'domain', name: 'business' }] },
  modelRefs: [
    {
      model: 'qwen3-8b',
      use_reasoning: false,
      reasoning_description: 'Prefer fast, direct business answers.',
    },
    {
      model: 'qwen3-32b',
      lora_name: 'business-adapter',
      use_reasoning: true,
      reasoning_mode: 'enabled',
      reasoning_effort: 'low',
    },
  ],
  candidateIterations: [
    {
      variable: 'candidate',
      source: 'models',
      models: [{ model: 'qwen3-32b', weight: 1, use_reasoning: true }],
      outputs: [{ type: 'model', value: 'candidate' }],
    },
  ],
  algorithm: { type: 'static' },
  plugins: [
    {
      type: 'system_prompt',
      configuration: { enabled: true, mode: 'insert', system_prompt: 'You are a consultant.' },
    },
    {
      type: 'router_replay',
      configuration: { enabled: true, shadow: { model: 'qwen3-32b', sample_rate: 0.05 } },
    },
  ],
  retention: { drop: false, ttl_turns: 4 },
}

// A hand-written decision with fields the old form did not manage, no
// priority and model references without use_reasoning.
const escalate = {
  name: 'typed-escalate',
  rules: {},
  modelRefs: [{ model: 'native-kai' }, { model: 'native-vega' }],
  algorithm: {
    type: 'escalate',
    budget: { deadline: '3s', max_calls: 4 },
    escalate: { stages: [{ name: 'fast', model: 'native-kai' }] },
  },
} as unknown as DecisionConfig

/** The save path of the earlier modal form, for its lossless fields. */
function legacySave(existing: DecisionConfig): DecisionConfig {
  const form = {
    rules: clone(existing.rules),
    modelRefs: existing.modelRefs.map((ref) => ({
      model: ref.model,
      use_reasoning: !!ref.use_reasoning,
      reasoning_description: ref.reasoning_description || '',
      reasoning_mode: (ref.reasoning_mode ||
        '') as DecisionConfig['modelRefs'][number]['reasoning_mode'],
      reasoning_effort: ref.reasoning_effort || '',
      lora_name: ref.lora_name || '',
      weight: typeof ref.weight === 'number' ? ref.weight : undefined,
    })),
    plugins: (existing.plugins ?? []).map((plugin) => ({
      type: plugin.type,
      configuration: { ...plugin.configuration },
    })),
  }
  return clone(
    mergeDecisionForSave(existing, {
      name: existing.name,
      description: existing.description,
      priority: existing.priority ?? 1,
      rules: decisionRulesForSave(form.rules),
      modelRefs: decisionModelRefsForSave(form.modelRefs),
      plugins: decisionPluginsForSave(form.plugins),
      tier: existing.tier,
      output_contract: existing.output_contract?.trim() || undefined,
      output_contract_spec: existing.output_contract_spec,
      action: existing.action,
      algorithm: existing.algorithm,
      adaptations: existing.adaptations,
      reliability: existing.reliability,
      fallback: existing.fallback,
      candidateIterations: existing.candidateIterations,
      retention: existing.retention,
      output: decisionOutputFromEditor(decisionOutputForEditor(existing.output)),
      annotations: existing.annotations,
    }),
  )
}

describe('decision drafts', () => {
  it('saves an untouched decision exactly as written', () => {
    expect(decisionFromDraft(clone(business), validate)).toEqual(business)
    expect(decisionFromDraft(clone(escalate), validate)).toEqual(escalate)
  })

  it('produces what the earlier form produced for a decision it could represent', () => {
    expect(decisionFromDraft(clone(business), validate)).toEqual(legacySave(business))
  })

  it('no longer writes use_reasoning or a priority the decision did not have', () => {
    const saved = decisionFromDraft(clone(escalate), validate)
    expect(saved.modelRefs).toEqual([{ model: 'native-kai' }, { model: 'native-vega' }])
    expect('priority' in saved).toBe(false)
    const legacy = legacySave(escalate)
    expect(legacy.modelRefs[0]).toEqual({ model: 'native-kai', use_reasoning: false })
    expect(legacy.priority).toBe(1)
  })

  it('normalizes edits the way the form did: trimmed text and no empty blocks', () => {
    const draft = clone(business) as DecisionConfig & Record<string, unknown>
    draft.name = '  static_business_route  '
    draft.output_contract = '   '
    draft.reliability = {}
    draft.modelRefs = [
      ...draft.modelRefs,
      { model: '  ', use_reasoning: false },
      { model: ' qwen3-8b ', lora_name: ' ', weight: Number.NaN } as never,
    ]
    const saved = decisionFromDraft(draft, validate)
    expect(saved.name).toBe('static_business_route')
    expect(saved.output_contract).toBeUndefined()
    expect(saved.reliability).toBeUndefined()
    expect(saved.modelRefs[saved.modelRefs.length - 1]).toEqual({ model: 'qwen3-8b' })
    expect(saved.modelRefs).toHaveLength(3)
  })

  it('names the tab of the first problem', () => {
    const tabOf = (draft: DecisionConfig) => {
      try {
        decisionFromDraft(draft, validate)
        return null
      } catch (error) {
        return error instanceof DecisionDraftError ? error.tab : 'unknown'
      }
    }
    expect(tabOf({ ...clone(business), name: ' ' })).toBe('overview')
    expect(tabOf({ ...clone(business), rules: { operator: 'NOT', conditions: [] } })).toBe('rules')
    expect(tabOf({ ...clone(business), action: { type: 'route', destination: '' } })).toBe('models')
    expect(tabOf({ ...clone(business), algorithm: { type: 'remom' } })).toBe('algorithm')
    expect(
      tabOf({ ...clone(business), plugins: [{ type: '', configuration: { enabled: true } }] }),
    ).toBe('plugins')
    expect(
      tabOf({ ...clone(business), candidateIterations: [{ variable: '', source: 'models' }] }),
    ).toBe('advanced')
  })

  it('replaces an edited decision in place and appends a new one', () => {
    const list = [clone(business), clone(escalate)]
    const renamed = { ...clone(business), name: 'business_route', priority: 210 }
    expect(
      upsertDecision(list, renamed, 'static_business_route').map((entry) => entry.name),
    ).toEqual(['business_route', 'typed-escalate'])
    expect(
      upsertDecision(list, { ...clone(escalate), name: 'new_route' }).map((entry) => entry.name),
    ).toEqual(['static_business_route', 'typed-escalate', 'new_route'])
    expect(() =>
      upsertDecision(list, { ...clone(escalate), name: 'static_business_route' }, 'typed-escalate'),
    ).toThrow(/already exists/)
  })
})
