import type { EvaluationRunEvent } from '../types/evaluationPlane'

export function appendEvaluationEvent(
  events: EvaluationRunEvent[],
  nextEvent: EvaluationRunEvent,
  limit = 50,
): EvaluationRunEvent[] {
  if (nextEvent.id && events.some((event) => event.id === nextEvent.id)) return events
  return [...events, nextEvent].slice(-Math.max(1, limit))
}
