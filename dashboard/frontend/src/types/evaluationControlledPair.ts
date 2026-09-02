import type { EvaluationRun } from './evaluationPlane'

export const EVALUATION_CONTROLLED_PAIR_CONTRACT_VERSION = 'evaluation-controlled-pair.v1' as const
export const EVALUATION_CONTROLLED_PAIR_PROTOCOL = 'abba-interleaved.v1' as const

export type EvaluationControlledPairState = 'pending' | 'running' | 'terminal'

export interface EvaluationControlledPairCapabilities {
  can_cancel: boolean
  can_delete: boolean
}

export interface CreateEvaluationControlledPairPayload {
  client_request_id: string
  baseline_source_run_id: string
  candidate_source_run_id: string
  baseline_run_id: string
  candidate_run_id: string
}

export interface EvaluationControlledPairExecution {
  schema_version: 'evaluation.v1'
  contract_version: typeof EVALUATION_CONTROLLED_PAIR_CONTRACT_VERSION
  id: string
  protocol: typeof EVALUATION_CONTROLLED_PAIR_PROTOCOL
  baseline_source_run_id: string
  candidate_source_run_id: string
  baseline_run: EvaluationRun
  candidate_run: EvaluationRun
  state: EvaluationControlledPairState
  capabilities: EvaluationControlledPairCapabilities
}
