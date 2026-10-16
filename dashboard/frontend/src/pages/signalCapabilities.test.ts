import { describe, expect, it } from 'vitest'
import { getSignalFieldSchema } from '../lib/dslSchemas'
import {
  decisionQuestionTypes,
  signalAvailability,
  signalCapabilitySchema,
} from './signalCapabilities'
import type { DecisionTasks } from './useDecisionTasks'

function catalog(): DecisionTasks {
  const data: DecisionTasks = {
    default_deployment: 'primary',
    tasks: [],
    deployments: [],
    bindings: [],
    default_bindings: {
      domain_classifier: { deployment: 'primary', contract: 'label_distribution.v1' },
    },
  }
  for (const [id, type, binding, optional] of [
    ['domain', 'domain', 'domain_classifier', false],
    ['pii_presence', 'pii', 'pii_classifier', false],
    ['pii_categories', 'pii', 'pii_classifier', false],
    ['pii_spans', 'pii', 'pii_classifier', true],
    ['classifier', 'classifier', 'classifier.{name}', false],
  ] as const)
    data.tasks.push({
      id,
      title: id,
      description: '',
      stage: 'request',
      input: 'text',
      output: 'choice',
      full_input: true,
      template: { state: '', questions: {} },
      consumers: [{ kind: 'signal', type, binding, optional }],
    })
  data.deployments.push({
    deployment: 'primary',
    model: 'provider/judgment',
    ready: false,
    native_question_types: ['choice', 'noul'],
    tasks: data.tasks.map((task) => ({
      task_id: task.id,
      supported: ['pii_presence', 'pii_categories'].includes(task.id),
      reason: 'Unsupported native question',
      quality: 'unevaluated',
    })),
  })
  return data
}

describe('signal capability admission', () => {
  it('uses task capabilities, not readiness or model family', () => {
    const data = catalog()
    expect(signalAvailability(data, { recipe: 'default' }, 'domain').supported).toBe(false)
    expect(signalAvailability(data, { recipe: 'default' }, 'pii').supported).toBe(true)
    expect(signalAvailability(data, { recipe: 'default' }, 'keyword').supported).toBe(true)
  })
  it('keeps optional span detection from disabling a verdict task', () => {
    const data = catalog()
    expect(data.deployments[0].tasks.find((task) => task.task_id === 'pii_spans')?.supported).toBe(
      false,
    )
    expect(signalAvailability(data, { recipe: 'private' }, 'pii').supported).toBe(true)
    data.deployments[0].tasks.find((task) => task.task_id === 'pii_categories')!.supported = false
    expect(signalAvailability(data, { recipe: 'private' }, 'pii').supported).toBe(false)
  })
  it('honors explicit specialist overrides before default capabilities', () => {
    const data = catalog()
    data.global_bindings = {
      domain_classifier: { deployment: 'specialist', contract: 'label_distribution.v1' },
    }
    expect(signalAvailability(data, { recipe: 'private' }, 'domain').supported).toBe(true)
    expect(
      signalAvailability(data, { recipe: 'private', globalBindings: {} }, 'domain').supported,
    ).toBe(false)
    expect(
      signalAvailability(
        data,
        {
          recipe: 'private',
          bindings: { domain_classifier: { deployment: 'primary', contract: 'decision.v1' } },
        },
        'domain',
      ).supported,
    ).toBe(false)
  })
  it('isolates a named recipe and does not borrow another recipe binding', () => {
    const data = catalog()
    data.bindings.push({
      task_id: 'domain',
      recipe: 'other',
      consumer: 'domain_classifier',
      source: 'recipe',
      deployment: 'specialist',
      model: '',
      ready: true,
      editable: true,
      path: [],
      binding: { deployment: 'specialist', contract: 'label_distribution.v1' },
    })
    expect(signalAvailability(data, { recipe: 'private' }, 'domain').supported).toBe(false)
  })
  it('resolves namespaced classifier overrides and explicit legacy specialists', () => {
    const data = catalog()
    const scope = {
      recipe: 'private',
      bindings: {
        'classifier.risk': { deployment: 'specialist', contract: 'label_distribution.v1' },
      },
    }
    expect(signalAvailability(data, scope, 'classifier', 'risk').supported).toBe(true)
    expect(signalAvailability(data, scope, 'classifier', 'other').supported).toBe(false)
    expect(
      signalAvailability(data, scope, 'classifier', 'other', {
        type: 'local',
        model_path: 'models/risk',
      }).supported,
    ).toBe(true)
  })
  it('allows composed Set through Noul but blocks unsupported Score and Span', () => {
    const data = catalog()
    const scope = { recipe: 'default' }
    expect(decisionQuestionTypes(data, scope, {})).toEqual(['choice', 'noul', 'set'])
    expect(
      signalAvailability(data, scope, 'decision', 'needs', { question: { type: 'set' } }).supported,
    ).toBe(true)
    expect(
      signalAvailability(data, scope, 'decision', 'difficulty', { question: { type: 'score' } })
        .supported,
    ).toBe(false)
    const schema = signalCapabilitySchema(
      getSignalFieldSchema('decision'),
      decisionQuestionTypes(data, scope, {}),
    )
    const questionType = schema
      .find((item) => item.key === 'question')
      ?.fields?.find((item) => item.key === 'type')
    expect(questionType?.disabledOptions).toHaveProperty('score')
    expect(questionType?.disabledOptions).toHaveProperty('span')
    expect(questionType?.disabledOptions).not.toHaveProperty('set')
  })
  it('does not guess an unavailable model capability', () => {
    expect(signalAvailability(null, { recipe: 'default' }, 'domain').supported).toBe(false)
    expect(
      signalAvailability(catalog(), { recipe: 'default' }, 'decision', '', {
        deployment: 'missing',
      }).supported,
    ).toBe(false)
  })
})
