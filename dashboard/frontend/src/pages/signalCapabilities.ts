import { QUESTION_TYPES } from './systemOnePlayground'
import type { FieldSchema } from '../lib/dslSchemas'
import type { DecisionTasks } from './useDecisionTasks'
import { bindingDeployment } from './modelBindings'

interface BoundResource {
  deployment?: string
  explicit?: boolean
}
export type SignalBindings = Record<string, BoundResource>
export interface SignalCapabilityScope {
  recipe: string
  bindings?: Record<string, unknown>
  globalBindings?: Record<string, unknown>
  defaultDeployment?: string
}
export interface SignalAvailability {
  supported: boolean
  reason?: string
}
const available: SignalAvailability = { supported: true }

function boundResource(value: unknown): BoundResource | undefined {
  const deployment = bindingDeployment(value)
  return deployment ? { deployment } : undefined
}

function explicitResource(bindings: Record<string, unknown> | undefined, key: string) {
  const resource = boundResource(bindings?.[key])
  return resource ? { ...resource, explicit: true } : undefined
}

function selectedResource(
  data: DecisionTasks,
  scope: SignalCapabilityScope,
  key: string,
): BoundResource | undefined {
  const explicit =
    explicitResource(scope.bindings, key) ??
    explicitResource(
      scope.globalBindings === undefined ? data.global_bindings : scope.globalBindings,
      key,
    )
  if (explicit) return explicit
  const inherited = boundResource(data.default_bindings?.[key])
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
  fields: Record<string, unknown> = {},
): SignalAvailability {
  if (!data) return { supported: false, reason: 'Model capabilities are not available yet.' }
  if (signalType === 'systemone') {
    const types = decisionQuestionTypes(data, scope, fields)
    const question = fields.question as Record<string, unknown> | undefined
    if (!types.length)
      return {
        supported: false,
        reason:
          'Select a System One deployment with supported question types in System One → Decision Models.',
      }
    if (typeof question?.type === 'string' && !types.includes(question.type))
      return {
        supported: false,
        reason: `The selected System One deployment does not support ${question.type} questions.`,
      }
    return available
  }
  const tasks = data.tasks.filter((task) =>
    task.consumers?.some((item) => item.kind === 'signal' && item.type === signalType),
  )
  for (const task of tasks) {
    const consumer = task.consumers?.find(
      (item) => item.kind === 'signal' && item.type === signalType,
    )
    // Rules that ask a question of their own may name the deployment that answers it.
    const own = bindingDeployment(fields.deployment)
    const resource: BoundResource | undefined = own
      ? { deployment: own, explicit: true }
      : consumer?.binding
        ? selectedResource(data, scope, consumer.binding)
        : undefined
    const deployment = resource?.deployment || scope.defaultDeployment || data.default_deployment
    const target = data.deployments.find((item) => item.deployment === deployment)
    // A deployment the task catalog does not list answers through its own
    // head or API; the Router checks that it can when the config is saved.
    if (!target && resource?.deployment && resource.explicit) continue
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

/** Bindings written as a deployment name or an object, keyed as written. */
export function signalBindings(value: unknown): SignalBindings | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, item]) => {
      const resource = boundResource(item)
      return resource ? [[key, resource]] : []
    }),
  )
}
