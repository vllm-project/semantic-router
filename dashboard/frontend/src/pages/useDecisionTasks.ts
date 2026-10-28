import { useDashboardObservation } from '../utils/useDashboardObservation'
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
  consumer: string
  recipe: string
  deployment: string
  model: string
  source: 'recipe' | 'global' | 'default' | 'module'
  ready: boolean
  editable: boolean
  path: string[]
  binding: {
    deployment: string
    contract: string
    adapter?: string
    head?: string
    mapping_path?: string
  }
}
export interface DecisionTasks {
  default_deployment: string
  global_bindings?: Record<string, { deployment: string; contract: string }>
  default_bindings?: Record<string, { deployment: string; contract: string }>
  tasks: DecisionTask[]
  deployments: Array<{
    deployment: string
    model: string
    ready: boolean
    native_question_types: string[]
    tasks: Array<{
      task_id: string
      supported: boolean
      implementation?: 'native' | 'composed_noul'
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
