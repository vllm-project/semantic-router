import type { EvaluationControlledPairExecution } from '../types/evaluationControlledPair'

export type ControlledPairStatus =
  | 'idle'
  | 'creating'
  | 'recovering'
  | 'running'
  | 'assigning'
  | 'ready'
  | 'error'

export interface ControlledPairState {
  status: ControlledPairStatus
  execution: EvaluationControlledPairExecution | null
  error: string | null
  sourceIDs: { baseline: string; candidate: string } | null
}

export const INITIAL_CONTROLLED_PAIR_STATE: ControlledPairState = {
  status: 'idle',
  execution: null,
  error: null,
  sourceIDs: null,
}

export type EvaluationControlledPairReadyGuard = () => boolean
export type EvaluationControlledPairReadyHandler = (
  execution: EvaluationControlledPairExecution,
  isCurrent: EvaluationControlledPairReadyGuard,
) => void | Promise<void>

export interface EvaluationControlledPairWorkflow {
  activePairID: string | null
  onPairIdentity: (pairID: string | null) => void | Promise<void>
}

export function controlledPairErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback
}

export async function handoffEvaluationControlledPair(
  execution: EvaluationControlledPairExecution,
  onReady: EvaluationControlledPairReadyHandler,
  isCurrent: EvaluationControlledPairReadyGuard = () => true,
): Promise<string | null> {
  if (!isCurrent()) return null
  try {
    await onReady(execution, isCurrent)
    return null
  } catch (error) {
    return controlledPairErrorMessage(
      error,
      'Controlled pair completed, but its fresh runs could not be assigned to the campaign.',
    )
  }
}

export function controlledPairTerminalFailure(
  execution: EvaluationControlledPairExecution,
): string | null {
  if (execution.state !== 'terminal') return null
  for (const [label, run] of [
    ['Baseline', execution.baseline_run],
    ['Candidate', execution.candidate_run],
  ] as const) {
    if (run.status === 'failed') {
      return `${label} controlled run failed: ${run.error || 'no server rationale was returned.'}`
    }
    if (run.status === 'cancelled') return `${label} controlled run was cancelled.`
  }
  return null
}

export function controlledPairIsReady(execution: EvaluationControlledPairExecution): boolean {
  return (
    execution.state === 'terminal' &&
    execution.baseline_run.status === 'completed' &&
    execution.candidate_run.status === 'completed'
  )
}
