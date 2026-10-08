import { useDashboardObservation } from '../utils/useDashboardObservation'

export type InstanceMode = 'router' | 'engine'
export interface InstanceDeployment {
  ownership: 'managed' | 'external' | 'kubernetes'
  controller_available: boolean
  desired_mode?: InstanceMode
  observed_mode?: InstanceMode | 'unknown'
  active_deployment?: string
  deployment?: string
  model?: string
  can_switch: boolean
  unavailable_reason?: string
  operation?: {
    id: string
    phase: string
    target_mode: InstanceMode
    started_at: string
    finished_at?: string
    error?: string
    rolled_back?: boolean
  } | null
}

export function useInstanceDeployment() {
  const observation = useDashboardObservation<InstanceDeployment>('/api/instance', {
    timeoutMs: 8000,
    pollMs: 3000,
  })
  const status = observation.data
  const busy = Boolean(
    status?.operation &&
      !status.operation.finished_at &&
      !['ready', 'completed', 'failed', 'rolled_back'].includes(status.operation.phase),
  )
  return { ...observation, status, busy }
}
