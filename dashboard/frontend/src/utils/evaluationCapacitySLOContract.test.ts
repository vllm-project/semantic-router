import { describe, expect, it } from 'vitest'

import type {
  CreateEvaluationRunPayload,
  EvaluationCapacityLoadProtocol,
  EvaluationCapacitySLO,
  EvaluationCatalog,
  EvaluationRun,
} from '../types/evaluationPlane'
import { buildCreateRunPayload } from './evaluationCreateRunContract'
import {
  decodeEvaluationCapacityLoadProtocol,
  decodeEvaluationCapacitySLO,
  defaultEvaluationCapacityLoadProtocol,
  equalEvaluationCapacityLoadProtocol,
  equalEvaluationCapacitySLO,
} from './evaluationCapacitySLOContract'
import { decodeEvaluationRun } from './evaluationRunContract'
import { buildEvaluationRoutingRecipePlan } from '../test/evaluationRoutingRecipeFixture'

const RUN_ID = '11111111-1111-4111-8111-111111111111'

const capacitySLO: EvaluationCapacitySLO = {
  schema_version: 'evaluation.v1',
  required_concurrency: 4,
  max_latency_p95_ms: 750,
  max_error_rate: 0.01,
  min_throughput_rps: 8,
  min_throughput_scaling_efficiency: 0.65,
}

const loadProtocol: EvaluationCapacityLoadProtocol = defaultEvaluationCapacityLoadProtocol(8)

const mixtureBase = {
  id: 'capacity-mixture',
  entrypoint_model: 'vllm-sr/auto',
  aliases: ['vllm-sr/auto'],
  recipe_name: 'capacity-recipe',
  recipe_description: 'Frozen capacity recipe.',
  recipe_digest: `sha256:${'1'.repeat(64)}`,
  pool_digest: `sha256:${'2'.repeat(64)}`,
  selector_policy_digest: `sha256:${'4'.repeat(64)}`,
  selector_digest: `sha256:${'5'.repeat(64)}`,
  adaptation_digest: `sha256:${'6'.repeat(64)}`,
  binding_digest: `sha256:${'3'.repeat(64)}`,
  model_arms: [
    {
      id: 'primary',
      model: 'models/primary',
      provider_model_id_digest: `sha256:${'4'.repeat(64)}`,
      input_cost_per_million_tokens_usd: 0.2,
      output_cost_per_million_tokens_usd: 0.4,
    },
  ],
  support_models: [],
  fallback_arm_id: 'primary',
  decisions: [{ name: 'default', algorithm: 'confidence', arm_ids: ['primary'] }],
}
const mixture = {
  ...mixtureBase,
  routing_recipe_plan: buildEvaluationRoutingRecipePlan(mixtureBase),
}

const catalog = {
  schema_version: 'evaluation.v1',
  gate_contract_version: 'evaluation-release-gates.v2',
  generated_at: '2026-08-30T00:00:00Z',
  change_profiles: [
    { id: 'runtime_capacity', name: 'Runtime capacity', description: '', campaign_slots: [] },
  ],
  tracks: [
    {
      id: 'capacity',
      name: 'Capacity',
      description: '',
      modes: ['live'],
      metrics: ['capacity.slo_headroom'],
      evidence_levels: ['E5'],
    },
  ],
  suites: [
    {
      id: 'live-capacity',
      executors: { live: 'live-runtime.v1' },
      name: 'Live capacity',
      description: '',
      track_ids: ['capacity'],
      modes: ['live'],
      evidence_level: 'E5',
      revision: 'live-capacity.v1',
      tags: [],
    },
  ],
  targets: [
    {
      id: 'production-target',
      name: 'Production target',
      description: '',
      kind: 'mixture-of-models',
      track_ids: ['capacity'],
      modes: ['live'],
      accepted_executors: { live: ['live-runtime.v1'] },
      healthy: true,
      mixture,
    },
  ],
} as unknown as EvaluationCatalog

const request: CreateEvaluationRunPayload = {
  client_request_id: RUN_ID,
  name: 'Capacity envelope',
  description: 'Validate the frozen service objective.',
  suite_ids: ['live-capacity'],
  track_ids: ['capacity'],
  mode: 'live',
  target_id: 'production-target',
  change_profile: 'runtime_capacity',
  sample_limit: 32,
  concurrency: 8,
  capacity_slo: capacitySLO,
  capacity_load_protocol: loadProtocol,
  seed: 42,
}

const run: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: RUN_ID,
  client_request_id: RUN_ID,
  name: request.name,
  description: request.description,
  status: 'pending',
  mode: request.mode,
  evidence_level: 'E5',
  track_evidence_levels: { capacity: 'E5' },
  target_id: request.target_id,
  mixture,
  change_profile: request.change_profile,
  suite_ids: request.suite_ids,
  track_ids: request.track_ids,
  sample_limit: request.sample_limit,
  concurrency: request.concurrency,
  capacity_slo: capacitySLO,
  capacity_load_protocol: loadProtocol,
  seed: request.seed,
  progress: { percent: 0, completed: 0, total: 1 },
  created_at: '2026-08-30T00:00:00Z',
}

describe('Capacity SLO and repeated load contract', () => {
  it('requires every bounded latency, error, throughput, and scaling objective', () => {
    expect(decodeEvaluationCapacitySLO(capacitySLO)).toEqual(capacitySLO)
    for (const tampered of [
      { ...capacitySLO, required_concurrency: 0 },
      { ...capacitySLO, max_latency_p95_ms: 0 },
      { ...capacitySLO, max_error_rate: 1 },
      { ...capacitySLO, min_throughput_rps: 0 },
      { ...capacitySLO, min_throughput_scaling_efficiency: 1.01 },
      { ...capacitySLO, proxy_passed: true },
    ]) {
      expect(() => decodeEvaluationCapacitySLO(tampered)).toThrow(/bounded concurrency/i)
    }
  })

  it('freezes the platform geometric ladder and repeated measurement protocol', () => {
    expect(decodeEvaluationCapacityLoadProtocol(loadProtocol, 8)).toEqual(loadProtocol)
    expect(loadProtocol).toMatchObject({
      concurrency_levels: [1, 2, 4, 8],
      warmup_request_multiplier: 2,
      measurement_requests_per_repetition: 100,
      repetitions_per_level: 3,
      minimum_measurement_clusters_per_level: 3,
      confidence_level: 0.95,
      max_error_rate_cluster_range: 0.05,
      max_throughput_cv: 0.2,
      max_latency_p95_cv: 0.2,
    })
    for (const tampered of [
      { ...loadProtocol, concurrency_levels: [1, 2, 8] },
      { ...loadProtocol, repetitions_per_level: 2 },
      { ...loadProtocol, measurement_requests_per_repetition: 99 },
      { ...loadProtocol, minimum_measurement_clusters_per_level: 2 },
      { ...loadProtocol, confidence_level: 0.9 },
      { ...loadProtocol, max_error_rate_cluster_range: 0.1 },
      { ...loadProtocol, max_throughput_cv: 0.21 },
    ]) {
      expect(() => decodeEvaluationCapacityLoadProtocol(tampered, 8)).toThrow(/platform geometric/i)
    }
  })

  it('rejects a live capacity create intent without an explicit attainable SLO', () => {
    expect(buildCreateRunPayload(request, catalog)).toMatchObject({
      capacity_slo: capacitySLO,
      capacity_load_protocol: loadProtocol,
    })
    expect(() => buildCreateRunPayload({ ...request, capacity_slo: undefined }, catalog)).toThrow(
      /requires performance goals and a load pattern/i,
    )
    expect(() =>
      buildCreateRunPayload({ ...request, capacity_load_protocol: undefined }, catalog),
    ).toThrow(/requires performance goals and a load pattern/i)
    expect(() =>
      buildCreateRunPayload(
        {
          ...request,
          concurrency: 1,
          capacity_slo: { ...capacitySLO, required_concurrency: 1 },
        },
        catalog,
      ),
    ).toThrow(/at least two parallel requests/i)
    expect(() =>
      buildCreateRunPayload(
        {
          ...request,
          capacity_slo: { ...capacitySLO, required_concurrency: request.concurrency + 1 },
        },
        catalog,
      ),
    ).toThrow(/cannot exceed the run limit/i)
  })

  it('forbids the contract outside live capacity and freezes it in run responses', () => {
    expect(decodeEvaluationRun(run, RUN_ID)).toMatchObject({
      capacity_slo: capacitySLO,
      capacity_load_protocol: loadProtocol,
    })
    expect(() => decodeEvaluationRun({ ...run, capacity_slo: undefined }, RUN_ID)).toThrow(
      /missing its frozen SLO or load protocol/i,
    )
    expect(() =>
      decodeEvaluationRun({ ...run, capacity_load_protocol: undefined }, RUN_ID),
    ).toThrow(/missing its frozen SLO or load protocol/i)
    expect(() =>
      decodeEvaluationRun(
        {
          ...run,
          concurrency: 1,
          capacity_slo: { ...capacitySLO, required_concurrency: 1 },
        },
        RUN_ID,
      ),
    ).toThrow(/concurrency of at least 2/i)
    expect(() =>
      decodeEvaluationRun(
        { ...run, capacity_slo: { ...capacitySLO, required_concurrency: 9 } },
        RUN_ID,
      ),
    ).toThrow(/exceeds its run concurrency/i)

    const replayRequest = {
      ...request,
      mode: 'replay' as const,
      track_ids: ['routing' as const],
    }
    expect(() => buildCreateRunPayload(replayRequest, catalog)).toThrow(
      /only for live performance evaluation/i,
    )
    expect(() =>
      decodeEvaluationRun(
        {
          ...run,
          mode: 'replay',
          mixture: undefined,
          track_ids: ['routing'],
          track_evidence_levels: { routing: 'E5' },
          capacity_slo: capacitySLO,
          capacity_load_protocol: loadProtocol,
        },
        RUN_ID,
      ),
    ).toThrow(/outside live capacity mode/i)
  })

  it('compares every frozen objective without a proxy decision', () => {
    expect(equalEvaluationCapacitySLO(capacitySLO, { ...capacitySLO })).toBe(true)
    expect(
      equalEvaluationCapacitySLO(capacitySLO, {
        ...capacitySLO,
        min_throughput_scaling_efficiency: 0.66,
      }),
    ).toBe(false)
    expect(equalEvaluationCapacityLoadProtocol(loadProtocol, { ...loadProtocol })).toBe(true)
    expect(
      equalEvaluationCapacityLoadProtocol(loadProtocol, {
        ...loadProtocol,
        repetitions_per_level: 4,
      }),
    ).toBe(false)
  })
})
