import { responseErrorMessage } from './configPageRequestErrors'

export interface ModelRuntimeDeployment {
  name: string
  managed: boolean
  process?: string
  served_name: string
  ready: boolean
  state: string
  reason?: string
  restarts?: number
  family?: string
  repo?: string
  revision?: string
  surfaces?: string[]
  heads?: Array<{ name: string; kind: string; labels?: string[] }>
  device?: string
  profile?: string
  engine?: string
  desired_replicas?: number
  ready_replicas?: number
  replicas?: Array<{
    id: string
    managed: boolean
    device?: string
    ready: boolean
    state: string
    reason?: string
    inflight: number
    estimated_work: number
    restarts: number
  }>
}

export interface ModelRuntimeInventory {
  deployments: ModelRuntimeDeployment[]
}

export interface DecisionModelActivation {
  activation_status?: string
  generated_runtime_hash?: string
  active_runtime_hash?: string
  activation?: {
    status?: string
    reasons?: Array<{ code: string; path?: string; message?: string }>
    error?: string
  }
}

export interface DecisionModelApplyResult {
  status: 'success' | 'restart_required' | 'persisted'
  message: string
}

export async function readDecisionModelApplyResult(
  response: Response,
): Promise<DecisionModelApplyResult> {
  if (!response.ok) throw new Error(await responseErrorMessage(response))
  const result = (await response.json()) as { status?: string; message?: string }
  if (result.status === 'restart_required' || result.status === 'persisted') {
    return {
      status: result.status,
      message:
        result.message ||
        (result.status === 'restart_required'
          ? 'Configuration saved. Run vllm-sr serve to activate the selected model.'
          : 'Configuration saved. Roll out Router and Envoy to activate the selected model.'),
    }
  }
  if (response.status === 200 && result.status === 'success') {
    return {
      status: 'success',
      message: 'Configuration applied. Check the observed runtime below for model readiness.',
    }
  }
  throw new Error(
    'The server did not confirm activation. Refresh the saved configuration and runtime status before retrying.',
  )
}

export function decisionRuntimeModelName(deployment: ModelRuntimeDeployment): string {
  return deployment.repo?.split('/').slice(-1)[0] || deployment.served_name || deployment.name
}

export function decisionModelRuntimeState(
  deployment: string,
  inventory: ModelRuntimeInventory | null,
): string {
  const observed = inventory?.deployments.find((entry) => entry.name === deployment)
  if (!observed) return 'Not reported'
  return observed.ready ? (observed.state === 'degraded' ? 'Degraded capacity' : 'Ready') : 'Needs attention'
}

export function decisionActivationLabel(activation: DecisionModelActivation | null): string {
  if (!activation) return 'Not reported'
  if (activation.activation?.reasons?.some((reason) => reason.code === 'restart_required'))
    return 'Restart required'
  if (activation.activation?.status === 'failed' || activation.activation?.status === 'rejected')
    return 'Activation failed'
  if (activation.activation_status === 'active') return 'Active'
  if (activation.activation_status === 'pending' || activation.activation?.status === 'pending')
    return 'Activation pending'
  return activation.activation_status || activation.activation?.status || 'Not reported'
}
