import { useDashboardObservation } from '../utils/useDashboardObservation'
import type { ModelBindingValue } from './modelBindings'
import type { SystemOneQuestion } from './systemOnePlayground'

export interface DecisionTask {
  id: string
  consumers?: Array<{
    kind: 'signal' | 'plugin' | 'algorithm'
    type: string
    binding?: string
    optional?: boolean
  }>
  title: string
  description: string
  stage: 'request' | 'response' | 'selection'
  input: 'text' | 'conversation' | 'grounded' | 'pair'
  output: string
  full_input: boolean
  template: {
    state: string | Record<string, string>
    questions: Record<string, SystemOneQuestion & { criteria?: unknown }>
  }
}
export interface DecisionTaskBinding {
  task_id: string
  /** The Router's consumer name: a signal type's (see bindingKey) or a rule's (see ruleTarget). */
  consumer: string
  recipe: string
  deployment: string
  model: string
  source: 'recipe' | 'global' | 'default' | 'module'
  ready: boolean
  editable: boolean
  path: string[]
  binding: ModelBindingValue
}
export interface DecisionTasks {
  default_deployment: string
  global_bindings?: Record<string, ModelBindingValue>
  default_bindings?: Record<string, ModelBindingValue>
  tasks: DecisionTask[]
  deployments: Array<{
    deployment: string
    model: string
    ready: boolean
    native_question_types: string[]
    tasks: Array<{
      task_id: string
      supported: boolean
      implementation?: 'native' | 'trained' | 'composed_noul' | 'trained_noul'
      reason?: string
      quality: 'unevaluated'
    }>
  }>
  models?: Array<{
    model: string
    native_question_types: string[]
    tasks: Array<{ task_id: string; supported: boolean; implementation?: string; reason?: string }>
  }>
  bindings: DecisionTaskBinding[]
}
export function useDecisionTasks(enabled = true) {
  const observation = useDashboardObservation<DecisionTasks>('/api/decision-model/tasks', {
    enabled,
  })
  const data = observation.data
  const invalid =
    data &&
    (!Array.isArray(data.tasks) ||
      !Array.isArray(data.deployments) ||
      !Array.isArray(data.bindings))
  return {
    ...observation,
    data: invalid ? null : data,
    error: invalid ? 'Task discovery returned an invalid response.' : observation.error,
  }
}
