import { QUESTION_TYPES } from './systemOnePlayground'
import type { FieldSchema } from '../lib/dslSchemas'
import type { DecisionTasks } from './useDecisionTasks'

export type SignalBindings = Record<string, { deployment?: string; contract?: string }>
export interface SignalCapabilityScope {
  recipe: string
  bindings?: SignalBindings
  globalBindings?: SignalBindings
  defaultDeployment?: string
}
export interface SignalAvailability {
  supported: boolean
  reason?: string
}
const available: SignalAvailability = { supported: true }

function selectedResource(data: DecisionTasks, scope: SignalCapabilityScope, key: string) {
  const explicit =
    scope.bindings?.[key] ??
    (scope.globalBindings === undefined ? data.global_bindings?.[key] : scope.globalBindings[key])
  if (explicit) return explicit
  const inherited = data.default_bindings?.[key]
  return inherited?.deployment === data.default_deployment && scope.defaultDeployment
    ? { ...inherited, deployment: scope.defaultDeployment }
    : inherited
}

export function decisionQuestionTypes(
  data: DecisionTasks | null,
  scope: SignalCapabilityScope,
  fields: Record<string, unknown>,
) {
  if (!data) return []
  const key =
    typeof fields.deployment === 'string' && fields.deployment
      ? fields.deployment
      : scope.defaultDeployment || data.default_deployment
  const native =
    data.deployments.find((item) => item.deployment === key)?.native_question_types ?? []
  return [...new Set([...native, ...(native.includes('noul') ? ['set'] : [])])]
}

/** Capabilities come from the Router's task registry; readiness is deliberately separate. */
export function signalAvailability(
  data: DecisionTasks | null,
  scope: SignalCapabilityScope,
  signalType: string,
  name = '',
  fields: Record<string, unknown> = {},
): SignalAvailability {
  if (!data) return { supported: false, reason: 'Model capabilities are not available yet.' }
  if (signalType === 'decision') {
    const types = decisionQuestionTypes(data, scope, fields)
    const question = fields.question as Record<string, unknown> | undefined
    if (!types.length)
      return {
        supported: false,
        reason:
          'Select a decision deployment with supported question types in System One → Decision Models.',
      }
    if (typeof question?.type === 'string' && !types.includes(question.type))
      return {
        supported: false,
        reason: `The selected decision deployment does not support ${question.type} questions.`,
      }
    return available
  }
  if (
    signalType === 'classifier' &&
    (fields.model_path || (fields.type !== 'local' && fields.model))
  )
    return available
  const tasks = data.tasks.filter((task) =>
    task.consumers?.some(
      (item) =>
        item.kind === 'signal' &&
        item.type === signalType &&
        (!item.optional || Boolean(fields.hazard)),
    ),
  )
  for (const task of tasks) {
    const consumer = task.consumers?.find(
      (item) => item.kind === 'signal' && item.type === signalType,
    )
    const key = consumer?.binding?.split('{name}').join(name) ?? ''
    const binding = key ? selectedResource(data, scope, key) : undefined
    const observed = data.bindings.find(
      (item) =>
        item.recipe === scope.recipe && item.task_id === task.id && (!key || item.consumer === key),
    )
    const resource = binding ?? (!key ? observed?.binding : undefined)
    // Specialist contracts implement their own task; they are not native decision questions.
    const deployment = resource?.deployment || scope.defaultDeployment || data.default_deployment
    const target = data.deployments.find((item) => item.deployment === deployment)
    if (!target && resource?.deployment && resource.contract && resource.contract !== 'decision.v1')
      continue
    const capability = target?.tasks.find((item) => item.task_id === task.id)
    if (!capability?.supported)
      return {
        supported: false,
        reason: `${task.title}: ${capability?.reason || `deployment “${deployment}” has no verified task capability`}. Change the task binding in System One → Decision Models.`,
      }
  }
  return available
}

export function signalCapabilitySchema(schema: FieldSchema[], types: string[]): FieldSchema[] {
  return schema.map((field) =>
    field.key !== 'question'
      ? field
      : {
          ...field,
          fields: field.fields?.map((nested) =>
            nested.key !== 'type'
              ? nested
              : {
                  ...nested,
                  type: 'select',
                  options: QUESTION_TYPES.map((item) => item.type),
                  disabledOptions: Object.fromEntries(
                    QUESTION_TYPES.map((item) => item.type)
                      .filter((type) => !types.includes(type))
                      .map((type) => [type, 'Not supported by the selected decision deployment']),
                  ),
                },
          ),
        },
  )
}

export function signalBindings(value: unknown): SignalBindings | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, item]) => {
      if (!item || typeof item !== 'object' || typeof item.deployment !== 'string') return []
      return [
        [
          key,
          {
            deployment: item.deployment,
            contract: typeof item.contract === 'string' ? item.contract : undefined,
          },
        ],
      ]
    }),
  )
}
