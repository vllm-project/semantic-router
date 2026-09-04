import type { EvaluationRunEventType, EvaluationTrackId } from '../../types/evaluationPlane'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'

const RUN_EVENT_LABELS = {
  snapshot: 'Run checkpoint',
  progress: 'Progress update',
  track: 'Evaluation area',
  gate: 'Readiness check',
  artifact: 'Report output',
  completed: 'Run completed',
  failed: 'Run failed',
  cancelled: 'Run cancelled',
} satisfies Record<EvaluationRunEventType, string>

export function evaluationRunEventLabel(event: {
  type: string
  track_id?: EvaluationTrackId
}): string {
  if (event.type === 'track' && event.track_id) {
    return TRACK_PRESENTATION[event.track_id].label
  }
  return RUN_EVENT_LABELS[event.type as EvaluationRunEventType] || 'Run update'
}
