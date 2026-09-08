import type { EvaluationControlledPairExecution } from '../../types/evaluationControlledPair'
import type { EvaluationRun, EvaluationRunEvent } from '../../types/evaluationPlane'

export interface EvaluationRunInspectorProps {
  selectedRunID: string | null
  run: EvaluationRun | null
  loading: boolean
  error: string | null
  controlledPairExecution: EvaluationControlledPairExecution | null
  controlledPairLoading: boolean
  controlledPairRefreshing: boolean
  controlledPairError: string | null
  events: EvaluationRunEvent[]
  eventsConnected: boolean
  eventsError: string | null
  canRun: boolean
  canDelete: boolean
  mutationKey: string | null
  onRetry: () => void
  onRetryControlledPair: () => void
  onReconnectEvents: () => void
  onStart: (run: EvaluationRun) => void
  onCancel: (run: EvaluationRun) => void
  onDelete: (run: EvaluationRun) => void
  onOpenReport: (run: EvaluationRun) => void
}

export type LoadedRunInspectorProps = Omit<EvaluationRunInspectorProps, 'run'> & {
  run: EvaluationRun
}
