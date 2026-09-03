import { describe, expect, it } from 'vitest'

import type { EvaluationRun } from '../types/evaluationPlane'
import { buildEvaluationRoutingRecipePlan } from '../test/evaluationRoutingRecipeFixture'
import {
  cohortMismatches,
  defaultComparisonPair,
  eligibleComparisonCandidates,
} from './evaluationComparisonCohort'

const MIXTURE_SUBJECT = {
  id: 'mixture',
  entrypoint_model: 'vllm-sr/auto',
  aliases: ['vllm-sr/auto'],
  recipe_name: 'balanced',
  recipe_description: '',
  recipe_digest: `sha256:${'1'.repeat(64)}`,
  pool_digest: `sha256:${'2'.repeat(64)}`,
  selector_policy_digest: `sha256:${'3'.repeat(64)}`,
  selector_digest: `sha256:${'4'.repeat(64)}`,
  adaptation_digest: `sha256:${'5'.repeat(64)}`,
  binding_digest: `sha256:${'6'.repeat(64)}`,
  model_arms: [],
  support_models: [],
  fallback_arm_id: '',
  decisions: [],
}

const MIXTURE: NonNullable<EvaluationRun['mixture']> = {
  ...MIXTURE_SUBJECT,
  routing_recipe_plan: buildEvaluationRoutingRecipePlan(MIXTURE_SUBJECT),
}

function run(overrides: Partial<EvaluationRun>): EvaluationRun {
  const id = overrides.id || 'baseline'
  return {
    schema_version: 'evaluation.v1',
    id,
    client_request_id: id,
    name: 'Baseline',
    description: '',
    status: 'completed',
    mode: 'replay',
    evidence_level: 'E0',
    track_evidence_levels: { routing: 'E0' },
    target_id: 'target',
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

describe('evaluation comparison lineage', () => {
  it('only exposes completed candidates pinned to an exact completed cohort', () => {
    const baseline = run({})
    const candidate = run({ id: 'candidate', name: 'Candidate', baseline_run_id: baseline.id })
    const mismatched = run({
      id: 'mismatch',
      baseline_run_id: baseline.id,
      concurrency: 8,
    })
    const unpinned = run({ id: 'unpinned' })
    expect(eligibleComparisonCandidates([candidate, mismatched, unpinned, baseline])).toEqual([
      candidate,
    ])
    expect(defaultComparisonPair([candidate, baseline])).toEqual({
      baselineID: 'baseline',
      candidateID: 'candidate',
    })
  })

  it('accepts a server-owned controlled pair while rejecting a forged cross-target pair', () => {
    const baseline = run({
      mode: 'live',
      mixture: MIXTURE,
      controlled_pair: { pair_id: 'pair', role: 'baseline' },
    })
    const candidate = run({
      id: 'candidate',
      name: 'Candidate',
      mode: 'live',
      mixture: MIXTURE,
      target_id: 'candidate-target',
      baseline_run_id: baseline.id,
      controlled_pair: { pair_id: 'pair', role: 'candidate' },
    })
    const forged = run({
      id: 'forged',
      mode: 'live',
      mixture: MIXTURE,
      target_id: 'forged-target',
      baseline_run_id: baseline.id,
      controlled_pair: { pair_id: 'other-pair', role: 'candidate' },
    })

    expect(eligibleComparisonCandidates([candidate, forged, baseline])).toEqual([candidate])
    expect(defaultComparisonPair([candidate, baseline])).toEqual({
      baselineID: baseline.id,
      candidateID: candidate.id,
    })
  })

  it.each([
    {
      name: 'same target',
      candidate: { target_id: 'target' },
    },
    {
      name: 'wrong role',
      candidate: { controlled_pair: { pair_id: 'pair', role: 'baseline' as const } },
    },
    {
      name: 'wrong pair identity',
      candidate: { controlled_pair: { pair_id: 'other', role: 'candidate' as const } },
    },
    {
      name: 'missing membership',
      candidate: { controlled_pair: undefined },
    },
    {
      name: 'wrong baseline identity',
      candidate: { baseline_run_id: 'other-baseline' },
    },
    {
      name: 'replay mode',
      candidate: { mode: 'replay' as const },
    },
    {
      name: 'missing Mixture identity',
      candidate: { mixture: undefined },
    },
    {
      name: 'cohort drift',
      candidate: { concurrency: 8 },
    },
  ])('rejects controlled-pair $name', ({ candidate: candidateOverride }) => {
    const baseline = run({
      mode: 'live',
      mixture: MIXTURE,
      controlled_pair: { pair_id: 'pair', role: 'baseline' },
    })
    const candidate = run({
      id: 'candidate',
      mode: 'live',
      mixture: MIXTURE,
      target_id: 'candidate-target',
      baseline_run_id: baseline.id,
      controlled_pair: { pair_id: 'pair', role: 'candidate' },
      ...candidateOverride,
    })

    expect(eligibleComparisonCandidates([candidate, baseline])).toEqual([])
  })

  it('requires server-order-exact suite and track cohorts', () => {
    const baseline = run({ suite_ids: ['a', 'b'], track_ids: ['routing', 'joint'] })
    const candidate = run({
      id: 'candidate',
      baseline_run_id: baseline.id,
      suite_ids: ['b', 'a'],
      track_ids: ['joint', 'routing'],
    })

    expect(cohortMismatches(baseline, candidate)).toEqual([
      'benchmark selection',
      'evaluation areas',
    ])
    expect(eligibleComparisonCandidates([candidate, baseline])).toEqual([])
  })

  it('requires exact capacity contracts', () => {
    const capacitySLO = {
      schema_version: 'evaluation.v1' as const,
      required_concurrency: 8,
      max_latency_p95_ms: 2_000,
      max_error_rate: 0.01,
      min_throughput_rps: 4,
      min_throughput_scaling_efficiency: 0.8,
    }
    const loadProtocol = {
      schema_version: 'evaluation.v1' as const,
      kind: 'closed-loop' as const,
      concurrency_levels: [1, 4, 8],
      warmup_request_multiplier: 1,
      measurement_requests_per_repetition: 32,
      repetitions_per_level: 3,
      minimum_measurement_clusters_per_level: 3 as const,
      confidence_level: 0.95 as const,
      max_error_rate_cluster_range: 0.05 as const,
      max_throughput_cv: 0.1,
      max_latency_p95_cv: 0.1,
    }
    const baseline = run({ capacity_slo: capacitySLO, capacity_load_protocol: loadProtocol })
    const candidate = run({
      id: 'candidate',
      baseline_run_id: baseline.id,
      capacity_slo: { ...capacitySLO, max_error_rate: 0.02 },
      capacity_load_protocol: { ...loadProtocol, concurrency_levels: [1, 8, 4] },
    })

    expect(cohortMismatches(baseline, candidate)).toEqual(['performance goals', 'load pattern'])
    expect(eligibleComparisonCandidates([candidate, baseline])).toEqual([])
  })

  it('names every client-visible cohort mismatch', () => {
    const baseline = run({})
    const candidate = run({
      id: 'candidate',
      target_id: 'other',
      seed: 7,
      suite_ids: ['other-suite'],
    })
    expect(cohortMismatches(baseline, candidate)).toEqual([
      'tested Mixture',
      'repeatability setting',
      'benchmark selection',
    ])
  })
})
