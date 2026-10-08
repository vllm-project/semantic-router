import { useEffect, useState } from 'react'
import { withRequestTimeout } from '../utils/boundedRequest'
import { responseErrorMessage } from './configPageRequestErrors'
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
  const [data, setData] = useState<DecisionTasks | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    if (!enabled) return
    const controller = new AbortController()
    void withRequestTimeout(
      async (signal) => {
        const response = await fetch('/api/decision-model/tasks', { signal })
        if (!response.ok) throw new Error(await responseErrorMessage(response))
        const result = (await response.json()) as DecisionTasks
        if (
          !Array.isArray(result.tasks) ||
          !Array.isArray(result.deployments) ||
          !Array.isArray(result.bindings)
        )
          throw new Error('Task discovery returned an invalid response.')
        if (!controller.signal.aborted) {
          setData(result)
          setError(null)
        }
      },
      controller.signal,
      10000,
    ).catch((cause: unknown) => {
      if (!controller.signal.aborted) {
        setData(null)
        setError(cause instanceof Error ? cause.message : 'Task capabilities are unavailable.')
      }
    })
    return () => controller.abort()
  }, [revision, enabled])
  useEffect(() => {
    if (!enabled) return
    const refresh = () => setRevision((current) => current + 1)
    const refreshVisible = () => {
      if (document.visibilityState !== 'hidden') refresh()
    }
    window.addEventListener('config-deployed', refresh)
    window.addEventListener('instance-deployed', refresh)
    document.addEventListener('visibilitychange', refreshVisible)
    const timer = window.setInterval(refreshVisible, 10000)
    return () => {
      window.removeEventListener('config-deployed', refresh)
      window.removeEventListener('instance-deployed', refresh)
      document.removeEventListener('visibilitychange', refreshVisible)
      window.clearInterval(timer)
    }
  }, [enabled])
  return { data, error, refresh: () => setRevision((current) => current + 1) }
}
