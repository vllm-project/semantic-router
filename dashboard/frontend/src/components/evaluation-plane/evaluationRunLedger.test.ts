import { describe, expect, it } from 'vitest'

import type { EvaluationRun } from '../../types/evaluationPlane'
import { filterEvaluationRuns } from './evaluationRunLedger'

function run(overrides: Partial<EvaluationRun>): EvaluationRun {
  const id = overrides.id || 'run'
  return {
    schema_version: 'evaluation.v1',
    id,
    client_request_id: id,
    name: 'Routing baseline',
    description: 'Production replay cohort',
    status: 'completed',
    mode: 'replay',
    evidence_level: 'E0',
    track_evidence_levels: { routing: 'E0' },
    target_id: 'fixture',
    change_profile: 'recipe',
    suite_ids: ['suite'],
    track_ids: ['routing'],
    sample_limit: 100,
    concurrency: 4,
    seed: 42,
    progress: { percent: 100, completed: 100, total: 100 },
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

describe('evaluation run ledger filtering', () => {
  const completed = run({})
  const liveAgentic = run({
    id: 'live-agentic',
    name: 'Agent candidate',
    description: 'Tool trajectory',
    status: 'running',
    mode: 'live',
    target_id: 'approved-live',
    change_profile: 'agent_multimodal',
    track_ids: ['agentic'],
  })

  it('searches the full run identity and cohort vocabulary case-insensitively', () => {
    expect(
      filterEvaluationRuns([completed, liveAgentic], {
        query: 'APPROVED-LIVE',
        status: 'all',
        track: 'all',
      }),
    ).toEqual([liveAgentic])
    expect(
      filterEvaluationRuns([completed, liveAgentic], {
        query: 'tool trajectory',
        status: 'all',
        track: 'all',
      }),
    ).toEqual([liveAgentic])
  })

  it('composes status, track, and search constraints without widening the ledger', () => {
    expect(
      filterEvaluationRuns([completed, liveAgentic], {
        query: 'agent',
        status: 'running',
        track: 'agentic',
      }),
    ).toEqual([liveAgentic])
    expect(
      filterEvaluationRuns([completed, liveAgentic], {
        query: 'agent',
        status: 'completed',
        track: 'agentic',
      }),
    ).toEqual([])
  })
})
