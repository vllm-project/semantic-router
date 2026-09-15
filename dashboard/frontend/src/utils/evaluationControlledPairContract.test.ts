import { describe, expect, it } from 'vitest'

import type { EvaluationRun } from '../types/evaluationPlane'
import {
  buildCreateEvaluationControlledPairPayload,
  decodeEvaluationControlledPairExecution,
} from './evaluationControlledPairContract'
import { isCanonicalEvaluationRunID } from './evaluationRunContract'
import { buildEvaluationRoutingRecipePlan } from '../test/evaluationRoutingRecipeFixture'

const BASELINE_SOURCE = '11111111-1111-4111-8111-111111111111'
const CANDIDATE_SOURCE = '22222222-2222-4222-8222-222222222222'

function liveRun(
  id: string,
  pairID: string,
  role: 'baseline' | 'candidate',
  baselineRunID?: string,
): EvaluationRun {
  const mixtureBase = {
    id: 'mom-live',
    entrypoint_model: 'quality-router',
    aliases: ['quality-router'],
    recipe_name: 'quality',
    recipe_description: 'Quality routing',
    recipe_digest: `sha256:${'1'.repeat(64)}`,
    pool_digest: `sha256:${'2'.repeat(64)}`,
    selector_policy_digest: `sha256:${'3'.repeat(64)}`,
    selector_digest: `sha256:${'4'.repeat(64)}`,
    adaptation_digest: `sha256:${'5'.repeat(64)}`,
    binding_digest: `sha256:${'6'.repeat(64)}`,
    model_arms: [
      {
        id: 'arm-a',
        model: 'model-a',
        provider_model_id_digest: `sha256:${'7'.repeat(64)}`,
        input_cost_per_million_tokens_usd: 1,
        output_cost_per_million_tokens_usd: 2,
      },
    ],
    support_models: [],
    decisions: [{ name: 'route', algorithm: 'semantic', arm_ids: ['arm-a'] }],
  }
  return {
    schema_version: 'evaluation.v1',
    id,
    client_request_id: id,
    name: baselineRunID ? 'Controlled candidate' : 'Controlled baseline',
    description: 'Server-owned abba-interleaved.v1 execution',
    status: 'running',
    mode: 'live',
    evidence_level: 'E2',
    track_evidence_levels: { routing: 'E2' },
    target_id: 'mom-live',
    mixture: {
      ...mixtureBase,
      routing_recipe_plan: buildEvaluationRoutingRecipePlan(mixtureBase),
    },
    change_profile: 'recipe',
    suite_ids: ['live-mom-core'],
    track_ids: ['routing'],
    sample_limit: 64,
    concurrency: 2,
    seed: 42,
    ...(baselineRunID ? { baseline_run_id: baselineRunID } : {}),
    controlled_pair: { pair_id: pairID, role },
    progress: { percent: 10, completed: 6, total: 64, message: 'AB block 1' },
    created_at: '2026-08-31T00:00:00Z',
    started_at: '2026-08-31T00:00:01Z',
  }
}

describe('controlled pair contract', () => {
  it('builds exactly five distinct canonical UUID fields and no client target claims', () => {
    const request = buildCreateEvaluationControlledPairPayload(BASELINE_SOURCE, CANDIDATE_SOURCE)
    expect(Object.keys(request).sort()).toEqual([
      'baseline_run_id',
      'baseline_source_run_id',
      'candidate_run_id',
      'candidate_source_run_id',
      'client_request_id',
    ])
    expect(Object.values(request).every(isCanonicalEvaluationRunID)).toBe(true)
    expect(new Set(Object.values(request)).size).toBe(5)
    expect(JSON.stringify(request)).not.toMatch(/endpoint|credential|version|label/)
  })

  it('strictly decodes the requested AB/BA execution and rejects extra claims', () => {
    const request = buildCreateEvaluationControlledPairPayload(BASELINE_SOURCE, CANDIDATE_SOURCE)
    const response = {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-controlled-pair.v1',
      id: request.client_request_id,
      protocol: 'abba-interleaved.v1',
      baseline_source_run_id: request.baseline_source_run_id,
      candidate_source_run_id: request.candidate_source_run_id,
      baseline_run: liveRun(request.baseline_run_id, request.client_request_id, 'baseline'),
      candidate_run: liveRun(
        request.candidate_run_id,
        request.client_request_id,
        'candidate',
        request.baseline_run_id,
      ),
      state: 'running',
      capabilities: { can_cancel: true, can_delete: false },
    }
    expect(
      decodeEvaluationControlledPairExecution(response, request.client_request_id, request)
        .candidate_run.id,
    ).toBe(request.candidate_run_id)
    expect(() =>
      decodeEvaluationControlledPairExecution(
        { ...response, endpoint: 'https://client.invalid' },
        request.client_request_id,
        request,
      ),
    ).toThrow('Controlled pair response is incomplete.')
    expect(() =>
      decodeEvaluationControlledPairExecution(
        { ...response, protocol: 'post-hoc-independent.v1' },
        request.client_request_id,
        request,
      ),
    ).toThrow('Controlled pair response is incomplete.')
    expect(() =>
      decodeEvaluationControlledPairExecution(
        {
          ...response,
          candidate_run: liveRun(
            request.candidate_run_id,
            request.client_request_id,
            'candidate',
            CANDIDATE_SOURCE,
          ),
        },
        request.client_request_id,
        request,
      ),
    ).toThrow('Controlled pair response does not match the requested AB/BA execution.')
    expect(() =>
      decodeEvaluationControlledPairExecution(
        { ...response, capabilities: { can_cancel: false, can_delete: true } },
        request.client_request_id,
        request,
      ),
    ).toThrow('Controlled pair response is incomplete.')
    expect(() =>
      decodeEvaluationControlledPairExecution(
        {
          ...response,
          candidate_run: {
            ...response.candidate_run,
            controlled_pair: { pair_id: request.client_request_id, role: 'baseline' },
          },
        },
        request.client_request_id,
        request,
      ),
    ).toThrow(/Controlled-pair baseline member|does not match the requested AB\/BA execution/)

    for (const collision of [
      { ...response, baseline_source_run_id: request.client_request_id },
      { ...response, candidate_source_run_id: request.baseline_source_run_id },
      {
        ...response,
        baseline_run: {
          ...response.baseline_run,
          id: request.baseline_source_run_id,
          client_request_id: request.baseline_source_run_id,
        },
      },
      {
        ...response,
        candidate_run: {
          ...response.candidate_run,
          id: request.baseline_run_id,
          client_request_id: request.baseline_run_id,
        },
      },
    ]) {
      expect(() =>
        decodeEvaluationControlledPairExecution(collision, request.client_request_id),
      ).toThrow(/candidate member|does not match the requested AB\/BA execution/)
    }
  })

  it('accepts a terminal member while the authoritative pair remains running', () => {
    const request = buildCreateEvaluationControlledPairPayload(BASELINE_SOURCE, CANDIDATE_SOURCE)
    const baselineRun = {
      ...liveRun(request.baseline_run_id, request.client_request_id, 'baseline'),
      status: 'completed' as const,
      completed_at: '2026-08-31T00:05:00Z',
    }
    const candidateRun = liveRun(
      request.candidate_run_id,
      request.client_request_id,
      'candidate',
      request.baseline_run_id,
    )
    const response = {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-controlled-pair.v1',
      id: request.client_request_id,
      protocol: 'abba-interleaved.v1',
      baseline_source_run_id: request.baseline_source_run_id,
      candidate_source_run_id: request.candidate_source_run_id,
      baseline_run: baselineRun,
      candidate_run: candidateRun,
      state: 'running',
      capabilities: { can_cancel: true, can_delete: false },
    }

    expect(
      decodeEvaluationControlledPairExecution(response, request.client_request_id).capabilities,
    ).toEqual({ can_cancel: true, can_delete: false })
    expect(
      decodeEvaluationControlledPairExecution(
        {
          ...response,
          candidate_run: {
            ...candidateRun,
            status: 'completed',
            completed_at: '2026-08-31T00:05:01Z',
          },
        },
        request.client_request_id,
      ).state,
    ).toBe('running')
    expect(() =>
      decodeEvaluationControlledPairExecution(
        {
          ...response,
          candidate_run: { ...candidateRun, status: 'pending', started_at: undefined },
        },
        request.client_request_id,
      ),
    ).toThrow('Controlled pair response does not match the requested AB/BA execution.')
    expect(() =>
      decodeEvaluationControlledPairExecution(
        {
          ...response,
          state: 'terminal',
          capabilities: { can_cancel: false, can_delete: true },
        },
        request.client_request_id,
      ),
    ).toThrow('Controlled pair response does not match the requested AB/BA execution.')
  })

  it('accepts conservative false capabilities while rejecting impossible true capabilities', () => {
    const request = buildCreateEvaluationControlledPairPayload(BASELINE_SOURCE, CANDIDATE_SOURCE)
    const baseline = liveRun(request.baseline_run_id, request.client_request_id, 'baseline')
    const candidate = liveRun(
      request.candidate_run_id,
      request.client_request_id,
      'candidate',
      request.baseline_run_id,
    )
    const response = {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-controlled-pair.v1',
      id: request.client_request_id,
      protocol: 'abba-interleaved.v1',
      baseline_source_run_id: request.baseline_source_run_id,
      candidate_source_run_id: request.candidate_source_run_id,
      baseline_run: baseline,
      candidate_run: candidate,
      state: 'running',
      capabilities: { can_cancel: false, can_delete: false },
    }

    expect(
      decodeEvaluationControlledPairExecution(response, request.client_request_id).capabilities,
    ).toEqual({ can_cancel: false, can_delete: false })
    expect(
      decodeEvaluationControlledPairExecution(
        {
          ...response,
          state: 'pending',
          baseline_run: { ...baseline, status: 'pending', started_at: undefined },
          candidate_run: { ...candidate, status: 'pending', started_at: undefined },
        },
        request.client_request_id,
      ).capabilities,
    ).toEqual({ can_cancel: false, can_delete: false })
    expect(
      decodeEvaluationControlledPairExecution(
        {
          ...response,
          state: 'terminal',
          baseline_run: {
            ...baseline,
            status: 'cancelled',
            completed_at: '2026-08-31T00:06:00Z',
          },
          candidate_run: {
            ...candidate,
            status: 'cancelled',
            completed_at: '2026-08-31T00:06:00Z',
          },
        },
        request.client_request_id,
      ).capabilities,
    ).toEqual({ can_cancel: false, can_delete: false })

    for (const invalid of [
      { ...response, state: 'pending', capabilities: { can_cancel: true, can_delete: false } },
      { ...response, capabilities: { can_cancel: false, can_delete: true } },
      { ...response, state: 'terminal', capabilities: { can_cancel: true, can_delete: false } },
    ]) {
      expect(() =>
        decodeEvaluationControlledPairExecution(invalid, request.client_request_id),
      ).toThrow('Controlled pair response is incomplete.')
    }
  })
})
