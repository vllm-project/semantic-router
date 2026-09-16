import { describe, expect, it } from 'vitest'

import type { EvaluationRun } from '../../types/evaluationPlane'
import {
  changeProfileLabel,
  comparisonRunOptionLabels,
  runCohortTargetLabel,
  runEvaluationTargetLabel,
  runOptionLabels,
  runWorkloadLabel,
} from './evaluationRunPresentation'

function run(id: string): EvaluationRun {
  return {
    schema_version: 'evaluation.v1',
    id,
    client_request_id: id,
    name: 'Candidate live capacity protocol',
    description: '',
    status: 'completed',
    mode: 'live',
    evidence_level: 'E5',
    track_evidence_levels: { capacity: 'E5' },
    target_id: 'target',
    change_profile: 'model_pool',
    suite_ids: ['capacity'],
    track_ids: ['capacity'],
    sample_limit: 1,
    concurrency: 1,
    seed: 42,
    progress: { percent: 100, completed: 1, total: 1 },
    created_at: '2026-09-01T00:00:00Z',
  }
}

describe('evaluation run option labels', () => {
  it('presents cohort identity without exposing an internal target identifier', () => {
    const candidate = run('candidate-target')
    candidate.target_id = 'internal-candidate-deployment'

    expect(changeProfileLabel('schema_adapter')).toBe('API and integration')
    expect(changeProfileLabel('agent_multimodal')).toBe('Agents and multimodal')
    expect(changeProfileLabel('future_server_profile')).toBe('Evaluation change')
    expect(runCohortTargetLabel(candidate)).toBe('Frozen deployment snapshot')
    expect(runEvaluationTargetLabel(candidate)).toBe('Saved evaluation target')
    expect(runWorkloadLabel(candidate)).toBe('1 case · 1 concurrent request')
    expect(
      runCohortTargetLabel({
        ...candidate,
        mixture: { entrypoint_model: 'vllm-sr/auto' } as EvaluationRun['mixture'],
      }),
    ).toBe('vllm-sr/auto')
  })

  it('keeps canonical runs with the same UUID prefix visibly distinct', () => {
    const first = run('00000000-0000-4000-8000-000000000001')
    const second = run('00000000-0000-4000-8000-000000000002')
    const labels = runOptionLabels([first, second])

    expect(labels.get(first.id)).toBe(
      'Candidate live capacity protocol · Option 1 · Model pool · Live · End-to-end validation · 1 case',
    )
    expect(labels.get(first.id)).not.toContain('E5')
    expect(labels.get(second.id)).not.toBe(labels.get(first.id))
  })

  it('disambiguates repeated names without exposing internal run identifiers', () => {
    const first = run('00000000-0000-4000-8000-000100000001')
    const second = run('00000000-0000-4000-8000-000200000001')
    const labels = runOptionLabels([first, second])

    expect(labels.get(first.id)).toContain('Option 1')
    expect(labels.get(second.id)).toContain('Option 2')
    expect([...labels.values()].join(' ')).not.toContain('000100000001')

    const comparisonLabels = comparisonRunOptionLabels([first, second])
    expect(comparisonLabels.get(first.id)).toBe('Candidate live capacity protocol · Option 1')
    expect(comparisonLabels.get(second.id)).toBe('Candidate live capacity protocol · Option 2')
  })
})
