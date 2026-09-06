import { describe, expect, it } from 'vitest'

import type {
  CreateEvaluationCampaignPayload,
  EvaluationCampaignGateBindings,
} from '../types/evaluationCampaign'
import { evaluationCampaignExpectedAnchors } from './evaluationCampaignBindingContract'
import {
  buildCreateEvaluationCampaignPayload,
  decodeEvaluationCampaign,
} from './evaluationCampaignContract'

const IDS = {
  campaign: '10000000-0000-4000-8000-000000000001',
  evidence: '10000000-0000-4000-8000-000000000002',
  baseline: '10000000-0000-4000-8000-000000000003',
  candidate: '10000000-0000-4000-8000-000000000004',
}
const digest = (character: string) => `sha256:${character.repeat(64)}`
const GATES = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9'] as const

function gate(id: (typeof GATES)[number], active: boolean) {
  return {
    id,
    name: `${id} decision`,
    disposition: active ? ('required' as const) : ('not_applicable' as const),
    verdict: active ? ('pass' as const) : ('not_applicable' as const),
    evidence_level: 'E5' as const,
    source: active ? 'server_bound_evidence' : 'campaign_contract',
    evidence_refs: [],
    rationale: active ? 'Server-bound evidence passed.' : 'Not applicable to this profile.',
  }
}

function campaign(bindings: EvaluationCampaignGateBindings, paired = false) {
  const expected = evaluationCampaignExpectedAnchors(bindings)
  const evidence = expected.map((anchor, index) => ({
    ...anchor,
    ...(anchor.slot_id === 'g3' && anchor.binding_role === 'baseline'
      ? {}
      : { candidate_subject_digest: digest('a') }),
    manifest_semantic_digest: digest(String(index + 1)),
    manifest_artifact_digest: digest(String(index + 2)),
    report_digest: digest(String(index + 3)),
    private_receipt_digest: digest(String(index + 5)),
    execution_attestation_digest: digest(String(index + 7)),
  }))
  const baseline = evidence.find(
    (anchor) => anchor.slot_id === 'g3' && anchor.binding_role === 'baseline',
  )
  const candidate = evidence.find(
    (anchor) => anchor.slot_id === 'g3' && anchor.binding_role === 'candidate',
  )
  const fidelityReference = evidence.find(
    (anchor) => anchor.slot_id === 'g5' && anchor.binding_role === 'reference',
  )
  const fidelityLive = evidence.find(
    (anchor) => anchor.slot_id === 'g5' && anchor.binding_role === 'live',
  )
  const gates = GATES.map((id) =>
    gate(
      id,
      id === 'G0' ||
        id === 'G1' ||
        (id === 'G2' && Boolean(bindings.g2_run_id)) ||
        (id === 'G3' && Boolean(bindings.g3_controlled_pair)) ||
        (id === 'G5' && Boolean(bindings.g5_fidelity)),
    ),
  )
  return {
    schema_version: 'evaluation.v1',
    contract_version: 'evaluation-campaign.v2',
    id: IDS.campaign,
    name: 'Campaign v2',
    description: '',
    change_profile: 'recipe',
    status: 'decided',
    gate_bindings: bindings,
    manifest_digest: digest('b'),
    created_at: '2026-08-31T00:00:00Z',
    decision: {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-campaign.v2',
      attestation_revision: 'evaluation-server-attestation.v2',
      campaign_id: IDS.campaign,
      campaign_digest: digest('b'),
      decision_digest: digest('c'),
      verdict: 'pass',
      summary: 'All required server-owned slots passed.',
      gates,
      evidence,
      ...(paired && baseline && candidate
        ? {
            paired_live_evidence: {
              schema_version: 'evaluation.v1',
              contract_version: 'evaluation-campaign-paired-live.v3',
              controlled_pair_session_id: '10000000-0000-4000-8000-000000000005',
              controlled_pair_protocol: 'abba-interleaved.v1',
              baseline_run_id: baseline.run_id,
              candidate_run_id: candidate.run_id,
              candidate_subject_digest: candidate.candidate_subject_digest,
              baseline_target_id: 'baseline--mom-target',
              candidate_target_id: 'candidate--mom-target',
              mixture_id: 'mom-target',
              recipe_name: 'default',
              track_ids: ['routing', 'model_pool', 'joint'],
              workload_snapshot_digest: digest('d'),
              benchmark_revisions: { 'mom-core': 'revision-1' },
              seed: 42,
              bootstrap_samples: 1000,
              confidence_level: 0.95,
              promotion_policy: {
                candidate_normalized_regret_maximum: 0.25,
                paired_normalized_regret_margin: 0.05,
                minimum_no_information_frontier_lift: 0.05,
                minimum_joint_reliability: 0.8,
                maximum_all_arm_failure_rate: 0.2,
                minimum_candidate_arm_reliability: 0.8,
              },
              promotion_statistics: [
                {
                  id: 'campaign.g3.candidate_normalized_regret',
                  direction: 'lower_is_better',
                  estimate: 0.1,
                  confidence_level: 0.95,
                  confidence_interval: [0.08, 0.12],
                  threshold: { operator: '<=', value: 0.25, unit: 'fraction' },
                  sample_count: 20,
                  missing_cases: 0,
                  verdict: 'pass',
                },
                {
                  id: 'campaign.g3.paired_normalized_regret_delta',
                  direction: 'lower_is_better',
                  estimate: -0.01,
                  confidence_level: 0.95,
                  confidence_interval: [-0.02, 0],
                  threshold: { operator: '<=', value: 0.05, unit: 'fraction' },
                  sample_count: 20,
                  missing_cases: 0,
                  verdict: 'pass',
                },
                {
                  id: 'campaign.g3.no_information_frontier_lift',
                  direction: 'higher_is_better',
                  estimate: 0.2,
                  confidence_level: 0.95,
                  confidence_interval: [0.15, 0.25],
                  threshold: { operator: '>=', value: 0.05, unit: 'quality' },
                  sample_count: 20,
                  missing_cases: 0,
                  verdict: 'pass',
                },
                {
                  id: 'campaign.g3.joint_reliability',
                  direction: 'higher_is_better',
                  estimate: 0.9,
                  confidence_level: 0.95,
                  confidence_interval: [0.85, 0.95],
                  threshold: { operator: '>=', value: 0.8, unit: 'fraction' },
                  sample_count: 20,
                  missing_cases: 0,
                  verdict: 'pass',
                },
                {
                  id: 'campaign.g3.all_arm_failure_rate',
                  direction: 'lower_is_better',
                  estimate: 0.1,
                  confidence_level: 0.95,
                  confidence_interval: [0.05, 0.15],
                  threshold: { operator: '<=', value: 0.2, unit: 'fraction' },
                  sample_count: 20,
                  missing_cases: 0,
                  verdict: 'pass',
                },
              ],
              baseline_manifest_digest: baseline.manifest_semantic_digest,
              candidate_manifest_digest: candidate.manifest_semantic_digest,
              baseline_execution_attestation_digest: baseline.execution_attestation_digest,
              candidate_execution_attestation_digest: candidate.execution_attestation_digest,
              baseline_policy_snapshot_digest: digest('e'),
              candidate_policy_snapshot_digest: digest('f'),
              baseline_binding_snapshot_digest: digest('1'),
              candidate_binding_snapshot_digest: digest('1'),
              baseline_pool_snapshot_digest: digest('2'),
              candidate_pool_snapshot_digest: digest('2'),
              baseline_environment_snapshot_digest: digest('3'),
              candidate_environment_snapshot_digest: digest('3'),
              baseline_backend_topology_digest: digest('4'),
              candidate_backend_topology_digest: digest('4'),
              baseline_code_revision: 'baseline-code',
              candidate_code_revision: 'candidate-code',
              statistics: [
                {
                  id: 'campaign.g3.routing.absolute_regret_non_inferiority',
                  gate_id: 'G3',
                  track_id: 'routing',
                  analysis_unit: 'case_absolute_regret',
                  direction: 'higher_is_better',
                  margin: 0.05,
                  baseline_value: 0.4,
                  candidate_value: 0.41,
                  delta: 0.01,
                  confidence_level: 0.95,
                  confidence_interval: [-0.01, 0.02],
                  sample_count: 20,
                  missing_pairs: 0,
                  verdict: 'pass',
                },
              ],
              model_pool_arm_reliability: ['arm-a', 'arm-b'].map((arm_id) => ({
                arm_id,
                cohort: 'paired',
                direction: 'lower_is_better',
                margin: 0.02,
                baseline_failure_rate: 0.1,
                candidate_failure_rate: 0.1,
                delta: 0,
                confidence_level: 0.95,
                confidence_interval: [-0.01, 0.01],
                candidate_confidence_interval: [0.05, 0.15],
                baseline_sample_count: 20,
                candidate_sample_count: 20,
                verdict: 'pass',
              })),
              digest: digest('9'),
            },
          }
        : {}),
      ...(bindings.g5_fidelity && fidelityReference && fidelityLive
        ? {
            fidelity_evidence: {
              schema_version: 'evaluation.v1',
              contract_version: 'evaluation-campaign-fidelity.v2',
              reference_run_id: fidelityReference.run_id,
              live_run_id: fidelityLive.run_id,
              candidate_subject_digest: digest('a'),
              reference_manifest_digest: fidelityReference.manifest_semantic_digest,
              live_manifest_digest: fidelityLive.manifest_semantic_digest,
              live_execution_attestation_digest: fidelityLive.execution_attestation_digest,
              track_id: 'joint',
              suite_ids: ['mom-core'],
              workload_snapshot_digest: digest('d'),
              benchmark_revisions: { 'mom-core': 'revision-1' },
              matched_cases: 59,
              decision_mismatches: 0,
              outcome_mismatches: 0,
              unavailable_cases: 0,
              sample_count: 59,
              point_estimate: 1,
              lower_bound: 0.95,
              confidence_level: 0.95,
              verdict: 'pass',
              digest: digest('8'),
            },
          }
        : {}),
      recommendations: [],
      created_at: '2026-08-31T00:00:00Z',
    },
  }
}

describe('evaluation campaign v2 contract', () => {
  it('decodes gate bindings and slot-owned evidence without fixed roles', () => {
    const decoded = decodeEvaluationCampaign(campaign({ g2_run_id: IDS.evidence }))
    expect(decoded.contract_version).toBe('evaluation-campaign.v2')
    expect(decoded.gate_bindings).toEqual({ g2_run_id: IDS.evidence })
    expect(decoded.decision.evidence[0]).toMatchObject({
      slot_id: 'g2',
      gate_id: 'G2',
      binding_role: 'evidence',
      run_id: IDS.evidence,
    })
    expect(decoded).not.toHaveProperty('runs')
  })

  it('rejects v1 campaigns and retired run-role fields', () => {
    expect(() =>
      decodeEvaluationCampaign({
        ...campaign({ g2_run_id: IDS.evidence }),
        contract_version: 'evaluation-campaign.v1',
      }),
    ).toThrow(/campaign response is incomplete/i)
    expect(() =>
      decodeEvaluationCampaign({
        ...campaign({ g2_run_id: IDS.evidence }),
        runs: { baseline_replay_run_id: IDS.baseline },
      }),
    ).toThrow(/campaign response is incomplete/i)
  })

  it('accepts new paired statistic identities and analysis units in the v2 receipt', () => {
    const decoded = decodeEvaluationCampaign(
      campaign(
        {
          g3_controlled_pair: {
            baseline_run_id: IDS.baseline,
            candidate_run_id: IDS.candidate,
          },
        },
        true,
      ),
    )
    expect(decoded.decision.paired_live_evidence?.statistics[0]).toMatchObject({
      id: 'campaign.g3.routing.absolute_regret_non_inferiority',
      analysis_unit: 'case_absolute_regret',
    })
  })

  it('accepts a server-owned promotion policy and self-describing statistic inventory', () => {
    const value = campaign(
      {
        g3_controlled_pair: {
          baseline_run_id: IDS.baseline,
          candidate_run_id: IDS.candidate,
        },
      },
      true,
    )
    const promotion = value.decision.paired_live_evidence!
    promotion.promotion_policy.candidate_normalized_regret_maximum = 0.3
    promotion.promotion_statistics = [
      {
        id: 'campaign.g3.server_selected_value_check',
        direction: 'lower_is_better',
        estimate: 0.1,
        confidence_level: 0.95,
        confidence_interval: [0.08, 0.12],
        threshold: { operator: '<=', value: 0.2, unit: 'fraction' },
        sample_count: 20,
        missing_cases: 0,
        verdict: 'pass',
      },
    ]

    const decoded = decodeEvaluationCampaign(value)
    expect(decoded.decision.paired_live_evidence?.promotion_policy).toMatchObject({
      candidate_normalized_regret_maximum: 0.3,
    })
    expect(decoded.decision.paired_live_evidence?.promotion_statistics).toMatchObject([
      { id: 'campaign.g3.server_selected_value_check' },
    ])
  })

  it('rejects internally inconsistent self-describing promotion statistics', () => {
    const makeEvidence = () => {
      const value = campaign(
        {
          g3_controlled_pair: {
            baseline_run_id: IDS.baseline,
            candidate_run_id: IDS.candidate,
          },
        },
        true,
      )
      return value
    }

    const duplicate = makeEvidence()
    const first = duplicate.decision.paired_live_evidence!.promotion_statistics[0]
    duplicate.decision.paired_live_evidence!.promotion_statistics = [first, { ...first }]
    expect(() => decodeEvaluationCampaign(duplicate)).toThrow(/statistic cohort is invalid/i)

    const mismatchedCohort = makeEvidence()
    mismatchedCohort.decision.paired_live_evidence!.promotion_statistics[1].sample_count = 21
    expect(() => decodeEvaluationCampaign(mismatchedCohort)).toThrow(/statistic cohort is invalid/i)

    const mismatchedDirection = makeEvidence()
    mismatchedDirection.decision.paired_live_evidence!.promotion_statistics[0].threshold.operator =
      '>='
    expect(() => decodeEvaluationCampaign(mismatchedDirection)).toThrow(/statistic .* is invalid/i)

    const mismatchedVerdict = makeEvidence()
    mismatchedVerdict.decision.paired_live_evidence!.promotion_statistics[0].verdict = 'fail'
    expect(() => decodeEvaluationCampaign(mismatchedVerdict)).toThrow(/statistic .* is invalid/i)
  })

  it('binds a strict G5 fidelity receipt to both slot anchors', () => {
    const decoded = decodeEvaluationCampaign(
      campaign({
        g5_fidelity: {
          reference_run_id: IDS.baseline,
          live_run_id: IDS.candidate,
        },
      }),
    )
    expect(decoded.decision.fidelity_evidence).toMatchObject({
      reference_run_id: IDS.baseline,
      live_run_id: IDS.candidate,
      track_id: 'joint',
      verdict: 'pass',
    })
    const wrongTrack = campaign({
      g5_fidelity: {
        reference_run_id: IDS.baseline,
        live_run_id: IDS.candidate,
      },
    })
    wrongTrack.decision.fidelity_evidence!.track_id = 'multimodal'
    expect(() => decodeEvaluationCampaign(wrongTrack)).toThrow(/fidelity evidence is invalid/i)
  })

  it('rejects anchor identity drift from the requested slot binding', () => {
    const value = campaign({ g2_run_id: IDS.evidence })
    value.decision.evidence[0].binding_role = 'candidate'
    expect(() => decodeEvaluationCampaign(value)).toThrow(/anchor g2 is invalid/i)
  })

  it('rejects a decision verdict that overclaims required gates', () => {
    const value = campaign({ g2_run_id: IDS.evidence })
    ;(value.decision.gates[2] as { verdict: string }).verdict = 'unavailable'
    expect(() => decodeEvaluationCampaign(value)).toThrow(/verdict does not match/i)
  })

  it('rejects retired waived gate and decision states', () => {
    const value = campaign({ g2_run_id: IDS.evidence })
    const waivedGate = {
      ...value,
      decision: {
        ...value.decision,
        gates: value.decision.gates.map((item, index) =>
          index === 0 ? { ...item, disposition: 'waived', verdict: 'waived' } : item,
        ),
      },
    }
    expect(() => decodeEvaluationCampaign(waivedGate)).toThrow(/campaign gate G0 is invalid/i)
    expect(() =>
      decodeEvaluationCampaign({
        ...value,
        decision: { ...value.decision, verdict: 'waived' },
      }),
    ).toThrow(/campaign decision is invalid/i)
  })

  it('builds only canonical v2 gate bindings and trims presentation text', () => {
    const request: CreateEvaluationCampaignPayload = {
      client_request_id: IDS.campaign,
      name: '  Campaign v2  ',
      description: '  Slot-owned evidence.  ',
      change_profile: 'recipe',
      gate_bindings: {
        g3_controlled_pair: {
          baseline_run_id: IDS.baseline,
          candidate_run_id: IDS.candidate,
        },
      },
    }
    expect(buildCreateEvaluationCampaignPayload(request)).toEqual({
      ...request,
      name: 'Campaign v2',
      description: 'Slot-owned evidence.',
    })
    expect(() => buildCreateEvaluationCampaignPayload({ ...request, runs: {} } as never)).toThrow(
      /non-contract fields/i,
    )
  })
})
