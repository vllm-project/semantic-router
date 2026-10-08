import { getRouterDecisionModelName } from '../components/routerModelPresentation'
import type { RouterModelsInfo } from '../utils/routerRuntime'
import { responseErrorMessage } from './configPageRequestErrors'
import { getDecisionRuntimeSummary } from './dashboardRouterIntelligenceSupport'
import type { DecisionModelName } from './decisionModelSupport'

export interface ModelRuntimeDeployment {
  name: string
  managed: boolean
  process: string
  served_name: string
  ready: boolean
  state: string
  reason?: string
  restarts: number
  family?: string
  repo?: string
  revision?: string
  surfaces?: string[]
  heads?: Array<{ name: string; kind: string; labels?: string[] }>
  device?: string
  profile?: string
  engine?: string
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

export function decisionModelPatch(model: DecisionModelName) {
  return { model_catalog: { system: { decision_model: model } } }
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

export function decisionRuntimeModelName(deployment: ModelRuntimeDeployment): string | undefined {
  return getRouterDecisionModelName({
    name: deployment.name,
    type: 'model_runtime',
    loaded: deployment.ready,
    metadata: { provider: 'model_runtime', deployment: deployment.name },
  })
}

export function decisionModelRuntimeState(
  model: DecisionModelName,
  inventory: ModelRuntimeInventory | null,
  consumers?: RouterModelsInfo | null,
): string {
  if (model === 'Vela-1.0') return 'Specialist models'
  if (inventory) {
    const matching = inventory.deployments.filter(
      (deployment) => decisionRuntimeModelName(deployment) === model,
    )
    if (!matching.length) return 'Not reported'
    return matching.every((deployment) => deployment.ready && deployment.state === 'ready')
      ? 'Matching runtime ready'
      : 'Runtime needs attention'
  }
  const state = getDecisionRuntimeSummary(
    { global: { model_catalog: { system: { decision_model: model } } } },
    consumers,
  ).state
  return state === 'ready'
    ? 'Matching runtime ready'
    : state === 'attention'
      ? 'Runtime needs attention'
      : 'Not reported'
}

export function decisionActivationLabel(activation: DecisionModelActivation | null): string {
  if (!activation) return 'Not reported'
  if (activation.activation?.reasons?.some((reason) => reason.code === 'restart_required'))
    return 'Restart required'
  if (activation.activation?.status === 'failed' || activation.activation?.status === 'rejected')
    return 'Activation failed'
  if (
    activation.activation_status === 'active' &&
    activation.generated_runtime_hash &&
    activation.generated_runtime_hash === activation.active_runtime_hash
  )
    return 'Active'
  if (activation.activation_status === 'pending' || activation.activation?.status === 'pending')
    return 'Activation pending'
  return activation.activation_status || activation.activation?.status || 'Not reported'
}
