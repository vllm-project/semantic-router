import { describe, expect, it } from 'vitest'

import type { EvaluationRunEvent } from '../types/evaluationPlane'
import { appendEvaluationEvent } from './evaluationEventSupport'

const event: EvaluationRunEvent = {
  id: 'event-1',
  run_id: 'run-1',
  type: 'progress',
  timestamp: '2026-08-29T00:00:00Z',
  message: 'Routing track started',
}

describe('evaluation event support', () => {
  it('deduplicates replayed SSE events by stable event id', () => {
    const initial = appendEvaluationEvent([], event)
    expect(appendEvaluationEvent(initial, event)).toBe(initial)
  })

  it('retains only the bounded newest event window', () => {
    const events = Array.from({ length: 4 }, (_, index) => ({
      ...event,
      id: `event-${index}`,
    }))
    expect(
      appendEvaluationEvent(events, { ...event, id: 'event-new' }, 3).map(({ id }) => id),
    ).toEqual(['event-2', 'event-3', 'event-new'])
  })
})
