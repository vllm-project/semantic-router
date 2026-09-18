import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationRun, EvaluationRunEvent } from '../../types/evaluationPlane'
import EvaluationRunTimeline from './EvaluationRunTimeline'
import { evaluationRunEventLabel } from './evaluationRunTimelinePresentation'

const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: '11111111-1111-4111-8111-111111111111',
  client_request_id: '11111111-1111-4111-8111-111111111111',
  name: 'Timeline contract',
  description: '',
  status: 'running',
  mode: 'replay',
  evidence_level: 'E0',
  track_evidence_levels: { routing: 'E0' },
  target_id: 'fixture',
  change_profile: 'recipe',
  suite_ids: ['evaluation-smoke'],
  track_ids: ['routing'],
  sample_limit: 4,
  concurrency: 1,
  seed: 42,
  progress: { percent: 50, completed: 0, total: 1 },
  created_at: '2026-08-30T00:00:00Z',
  started_at: '2026-08-30T00:00:01Z',
}

const trackEvent: EvaluationRunEvent = {
  id: '2',
  run_id: run.id,
  type: 'track',
  timestamp: '2026-08-30T00:00:30Z',
  message: 'Evaluation track evidence collected',
  track_id: 'routing',
  progress: {
    percent: 100,
    completed: 1,
    total: 1,
    current_track_id: 'routing',
    message: 'Evaluation track evidence collected',
  },
  payload: { record_count: 1_024 },
}

describe('EvaluationRunTimeline', () => {
  it('shows the typed track result count without exposing worker-only terminal payloads', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationRunTimeline, {
        run,
        events: [trackEvent],
        connected: true,
        error: null,
        onReconnect: () => undefined,
      }),
    )

    expect(markup).toContain('1,024 results')
    expect(markup).toContain('<strong>Routing</strong>')
  })

  it('presents sealing as an active result-finalization phase', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationRunTimeline, {
        run: { ...run, status: 'sealing' },
        events: [],
        connected: false,
        error: null,
        onReconnect: () => undefined,
      }),
    )

    expect(markup).toContain('Connecting')
    expect(markup).toContain('Finalizing results')
    expect(markup).not.toContain('server-sealed')
    expect(markup).not.toContain('durable lifecycle')
  })

  it('maps every lifecycle event type to a product-facing label', () => {
    expect(
      ['snapshot', 'progress', 'track', 'gate', 'artifact', 'completed', 'failed', 'cancelled'].map(
        (type) => evaluationRunEventLabel({ type }),
      ),
    ).toEqual([
      'Run checkpoint',
      'Progress update',
      'Evaluation area',
      'Readiness check',
      'Report output',
      'Run completed',
      'Run failed',
      'Run cancelled',
    ])
  })

  it('renders the product label instead of a non-track event identifier', () => {
    const checkpoint: EvaluationRunEvent = {
      id: '3',
      run_id: run.id,
      type: 'snapshot',
      timestamp: '2026-08-30T00:00:15Z',
      message: 'Run configuration saved',
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationRunTimeline, {
        run,
        events: [checkpoint],
        connected: true,
        error: null,
        onReconnect: () => undefined,
      }),
    )

    expect(markup).toContain('<strong>Run checkpoint</strong>')
    expect(markup).not.toContain('<strong>snapshot</strong>')
  })

  it('uses a stable fallback for an unfamiliar lifecycle event identifier', () => {
    expect(evaluationRunEventLabel({ type: 'worker_stage_started' })).toBe('Run update')
    expect(evaluationRunEventLabel({ type: '' })).toBe('Run update')
  })

  it('keeps adversarial worker and decoder text behind closed technical details', () => {
    const workerMessage = 'worker://seal G9 decoder-stack=opaque-secret'
    const decoderError = 'event-stream decoder trace=private-internal-chain'
    const markup = renderToStaticMarkup(
      createElement(EvaluationRunTimeline, {
        run,
        events: [{ ...trackEvent, message: workerMessage }],
        connected: false,
        error: decoderError,
        onReconnect: () => undefined,
      }),
    )

    expect(markup).toContain('Reconnect to resume new timeline updates')
    expect(markup).toContain('1,024 results recorded for this evaluation area')
    for (const rawText of [workerMessage, decoderError]) {
      const rawIndex = markup.indexOf(rawText)
      const boundaryIndex = markup.lastIndexOf('data-evaluation-technical-details="true"', rawIndex)
      expect(rawIndex).toBeGreaterThan(-1)
      expect(boundaryIndex).toBeGreaterThan(-1)
      expect(markup.slice(boundaryIndex, rawIndex)).not.toContain('</details>')
    }
    expect(markup).not.toContain('<details open')
  })
})
