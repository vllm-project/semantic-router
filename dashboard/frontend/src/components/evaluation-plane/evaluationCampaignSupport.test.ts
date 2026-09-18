import { describe, expect, it } from 'vitest'

import type { EvaluationCampaignReadiness } from '../../types/evaluationCampaign'
import type {
  EvaluationCatalog,
  EvaluationCatalogCampaignSlot,
  EvaluationRun,
} from '../../types/evaluationPlane'
import { decodeEvaluationCampaignReadiness } from '../../utils/evaluationCampaignReadinessContract'
import {
  buildEvaluationCampaignRequest,
  campaignRunOptions,
  controlledPairBaselineSourceOptions,
  controlledPairCandidateSourceOptions,
  fidelityLiveOptions,
  fidelityReferenceOptions,
  type EvaluationCampaignDraft,
  validateEvaluationCampaignDraft,
} from './evaluationCampaignSupport'

const ids = {
  baseline: '10000000-0000-4000-8000-000000000001',
  candidate: '10000000-0000-4000-8000-000000000002',
  run: '10000000-0000-4000-8000-000000000003',
  reference: '10000000-0000-4000-8000-000000000004',
  live: '10000000-0000-4000-8000-000000000005',
  request: '10000000-0000-4000-8000-000000000006',
} as const

function run(id: string, name: string): EvaluationRun {
  return {
    schema_version: 'evaluation.v1',
    id,
    client_request_id: id,
    name,
    description: '',
    status: 'failed',
    mode: 'replay',
    evidence_level: 'E0',
    track_evidence_levels: { routing: 'E0' },
    target_id: 'deliberately-not-inferred-by-browser',
    change_profile: 'recipe',
    suite_ids: ['server-owned-suite'],
    track_ids: ['routing'],
    sample_limit: 1,
    concurrency: 1,
    seed: 1,
    progress: { percent: 0, completed: 0, total: 1 },
    created_at: '2026-09-01T00:00:00Z',
  }
}

const slots: EvaluationCatalogCampaignSlot[] = [
  {
    gate_id: 'G3',
    name: 'Controlled value comparison',
    description: '',
    disposition: 'required',
    binding_kind: 'controlled_pair',
    track_id: 'joint',
    mode: 'live',
    minimum_evidence_level: 'E5',
    accepted_executor_ids: ['server-executor.v9'],
  },
  {
    gate_id: 'G4',
    name: 'Workload-shift robustness',
    description: '',
    disposition: 'required',
    binding_kind: 'run',
    track_id: 'routing',
    mode: 'live',
    minimum_evidence_level: 'E5',
    accepted_executor_ids: ['server-executor.v9'],
  },
  {
    gate_id: 'G5',
    name: 'Live fidelity',
    description: '',
    disposition: 'required',
    binding_kind: 'fidelity_pair',
    track_id: 'joint',
    mode: 'live',
    minimum_evidence_level: 'E5',
    accepted_executor_ids: ['server-executor.v9'],
  },
]

const catalog = {
  change_profiles: [
    { id: 'recipe', name: 'Routing recipe', description: '', campaign_slots: slots },
  ],
} as EvaluationCatalog

const runs = [
  run(ids.baseline, 'Baseline'),
  run(ids.candidate, 'Candidate'),
  run(ids.run, 'Shift run'),
  run(ids.reference, 'Reference'),
  run(ids.live, 'Fresh live'),
]

const readiness: EvaluationCampaignReadiness = {
  schema_version: 'evaluation.v1',
  change_profile: 'recipe',
  total_runs: runs.length,
  slots: [
    {
      gate_id: 'G3',
      binding_kind: 'controlled_pair',
      eligible_run_ids: [],
      controlled_pair_source_run_ids: [ids.baseline],
      controlled_pair_candidate_run_ids: [ids.candidate],
      fidelity_reference_run_ids: [],
      fidelity_live_run_ids: [],
    },
    {
      gate_id: 'G4',
      binding_kind: 'run',
      eligible_run_ids: [ids.run],
      controlled_pair_source_run_ids: [],
      controlled_pair_candidate_run_ids: [],
      fidelity_reference_run_ids: [],
      fidelity_live_run_ids: [],
    },
    {
      gate_id: 'G5',
      binding_kind: 'fidelity_pair',
      eligible_run_ids: [],
      controlled_pair_source_run_ids: [],
      controlled_pair_candidate_run_ids: [],
      fidelity_reference_run_ids: [ids.reference],
      fidelity_live_run_ids: [ids.live],
    },
  ],
}

const draft: EvaluationCampaignDraft = {
  clientRequestID: ids.request,
  name: 'Recipe release decision',
  description: 'Server-approved evidence only.',
  changeProfile: 'recipe',
  gateBindings: {
    g3_controlled_pair: {
      baseline_run_id: ids.baseline,
      candidate_run_id: ids.candidate,
    },
    g4_run_id: ids.run,
    g5_fidelity: { reference_run_id: ids.reference, live_run_id: ids.live },
  },
}

describe('evaluation campaign readiness projection', () => {
  it('accepts only exact server projections for the selected comparison anchors', () => {
    const profile = catalog.change_profiles[0]
    expect(
      decodeEvaluationCampaignReadiness(
        readiness,
        profile,
        {
          controlledPairBaselineRunID: ids.baseline,
          fidelityReferenceRunID: ids.reference,
        },
      ),
    ).toEqual(readiness)
    expect(() =>
      decodeEvaluationCampaignReadiness(
        {
          ...readiness,
          slots: readiness.slots.map((slot) =>
            slot.gate_id === 'G4'
              ? { ...slot, eligible_run_ids: [ids.run, ids.run] }
              : slot,
          ),
        },
        profile,
        {
          controlledPairBaselineRunID: ids.baseline,
          fidelityReferenceRunID: ids.reference,
        },
      ),
    ).toThrow(/readiness response is incomplete/i)
    expect(() => decodeEvaluationCampaignReadiness(readiness, profile, {})).toThrow(
      /readiness response is incomplete/i,
    )
  })

  it('uses only server-returned eligible IDs and pairs for authoring options', () => {
    expect(campaignRunOptions(runs, readiness, slots[1]).map((item) => item.id)).toEqual([
      ids.run,
    ])
    expect(
      controlledPairBaselineSourceOptions(runs, readiness, slots[0]).map((item) => item.id),
    ).toEqual([ids.baseline])
    expect(
      controlledPairCandidateSourceOptions(runs, readiness, slots[0]).map((item) => item.id),
    ).toEqual([ids.candidate])
    expect(fidelityReferenceOptions(runs, readiness, slots[2]).map((item) => item.id)).toEqual([
      ids.reference,
    ])
    expect(
      fidelityLiveOptions(runs, readiness, slots[2]).map((item) => item.id),
    ).toEqual([ids.live])
  })

  it('validates selected IDs against the server projection without replaying admission rules', () => {
    expect(
      validateEvaluationCampaignDraft(
        catalog,
        runs,
        draft,
        readiness,
        false,
        null,
        true,
        true,
        true,
      ),
    ).toBeNull()

    const withoutRun = {
      ...readiness,
      slots: readiness.slots.map((slot) =>
        slot.gate_id === 'G4' ? { ...slot, eligible_run_ids: [] } : slot,
      ),
    }
    expect(
      validateEvaluationCampaignDraft(
        catalog,
        runs,
        draft,
        withoutRun,
        false,
        null,
        true,
        true,
        true,
      ),
    ).toMatch(/no longer meets its release check/i)

    expect(
      validateEvaluationCampaignDraft(
        catalog,
        runs,
        draft,
        null,
        true,
        null,
        true,
        true,
        true,
      ),
    ).toMatch(/Verifying which runs/i)
  })

  it('builds only the current campaign request contract', () => {
    expect(buildEvaluationCampaignRequest(draft)).toEqual({
      client_request_id: ids.request,
      name: draft.name,
      description: draft.description,
      change_profile: 'recipe',
      gate_bindings: draft.gateBindings,
    })
  })
})
