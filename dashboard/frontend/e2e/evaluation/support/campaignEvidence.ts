import type {
  EvaluationCampaignEvidenceAnchor,
  EvaluationCampaignFidelityEvidence,
  EvaluationCampaignPairedLiveEvidence,
  EvaluationCampaignPairedStatistic,
} from '../../../src/types/evaluationCampaign'

const digest = (character: string) => `sha256:${character.repeat(64)}`

function pairedStatistic(
  id: string,
  gate_id: 'G3' | 'G8',
  track_id: 'routing' | 'model_pool' | 'joint',
  analysis_unit: string,
  direction: 'higher_is_better' | 'lower_is_better',
  margin: number,
  baseline_value: number,
  candidate_value: number,
  confidence_interval: number[],
  candidate_confidence_interval?: number[],
): EvaluationCampaignPairedStatistic {
  return {
    id,
    gate_id,
    track_id,
    analysis_unit,
    direction,
    margin,
    baseline_value,
    candidate_value,
    delta: candidate_value - baseline_value,
    confidence_level: 0.95,
    confidence_interval,
    ...(candidate_confidence_interval ? { candidate_confidence_interval } : {}),
    sample_count: 20,
    missing_pairs: 0,
    verdict: 'pass',
  }
}

function pairedStatistics(): EvaluationCampaignPairedStatistic[] {
  return (['routing', 'model_pool', 'joint'] as const).flatMap((trackID) => {
    const qualityUnit = trackID === 'model_pool' ? 'case_pool_oracle_quality' : 'case_mean_quality'
    const failureUnit = trackID === 'model_pool' ? 'case_all_arm_failure' : 'case_failure_fraction'
    const statistics = [
      pairedStatistic(
        `campaign.g3.${trackID}.quality_non_inferiority`,
        'G3',
        trackID,
        qualityUnit,
        'higher_is_better',
        0.05,
        0.8,
        0.81,
        [-0.01, 0.03],
      ),
    ]
    if (trackID === 'model_pool') {
      statistics.push(
        pairedStatistic(
          'campaign.g3.model_pool.worst_arm_reliability_non_inferiority',
          'G3',
          trackID,
          'pool_worst_arm_reliability',
          'higher_is_better',
          0.02,
          0.88,
          0.9,
          [-0.01, 0.04],
          [0.85, 0.95],
        ),
      )
    }
    statistics.push(
      pairedStatistic(
        `campaign.g8.${trackID}.failure_risk`,
        'G8',
        trackID,
        failureUnit,
        'lower_is_better',
        0.02,
        0.1,
        0.1,
        [-0.01, 0.01],
      ),
      pairedStatistic(
        `campaign.g8.${trackID}.latency_risk`,
        'G8',
        trackID,
        'case_max_latency_relative_delta',
        'lower_is_better',
        0.05,
        0.2,
        0.21,
        [-0.01, 0.03],
      ),
    )
    return statistics
  })
}

export function evaluationPairedLiveEvidence(
  baseline: EvaluationCampaignEvidenceAnchor,
  candidate: EvaluationCampaignEvidenceAnchor,
): EvaluationCampaignPairedLiveEvidence {
  return {
    schema_version: 'evaluation.v1',
    contract_version: 'evaluation-campaign-paired-live.v3',
    controlled_pair_session_id: '00000000-0000-4000-8000-000000000020',
    controlled_pair_protocol: 'abba-interleaved.v1',
    baseline_run_id: baseline.run_id,
    candidate_run_id: candidate.run_id,
    candidate_subject_digest: candidate.candidate_subject_digest!,
    baseline_target_id:
      'baseline--mom-37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f',
    candidate_target_id:
      'candidate--mom-37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f',
    mixture_id: 'mom-37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f',
    recipe_name: 'default',
    track_ids: ['routing', 'model_pool', 'joint'],
    workload_snapshot_digest: digest('f'),
    benchmark_revisions: { 'live-mom-core': 'mom-campaign-cohort-v1' },
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
    baseline_execution_attestation_digest: baseline.execution_attestation_digest!,
    candidate_execution_attestation_digest: candidate.execution_attestation_digest!,
    baseline_policy_snapshot_digest: digest('1'),
    candidate_policy_snapshot_digest: digest('2'),
    baseline_binding_snapshot_digest: digest('3'),
    candidate_binding_snapshot_digest: digest('4'),
    baseline_pool_snapshot_digest: digest('5'),
    candidate_pool_snapshot_digest: digest('6'),
    baseline_environment_snapshot_digest: digest('7'),
    candidate_environment_snapshot_digest: digest('8'),
    baseline_backend_topology_digest: digest('9'),
    candidate_backend_topology_digest: digest('a'),
    baseline_code_revision: 'baseline-revision',
    candidate_code_revision: 'candidate-revision',
    statistics: pairedStatistics(),
    model_pool_arm_reliability: ['arm-fast', 'arm-strong'].map((arm_id) => ({
      arm_id,
      cohort: 'paired' as const,
      direction: 'lower_is_better' as const,
      margin: 0.02,
      baseline_failure_rate: 0.1,
      candidate_failure_rate: 0.1,
      delta: 0,
      confidence_level: 0.95,
      confidence_interval: [-0.01, 0.01],
      candidate_confidence_interval: [0.05, 0.15],
      baseline_sample_count: 20,
      candidate_sample_count: 20,
      verdict: 'pass' as const,
    })),
    digest: digest('b'),
  }
}

export function evaluationFidelityEvidence(
  reference: EvaluationCampaignEvidenceAnchor,
  live: EvaluationCampaignEvidenceAnchor,
): EvaluationCampaignFidelityEvidence {
  return {
    schema_version: 'evaluation.v1',
    contract_version: 'evaluation-campaign-fidelity.v2',
    reference_run_id: reference.run_id,
    live_run_id: live.run_id,
    candidate_subject_digest: reference.candidate_subject_digest!,
    reference_manifest_digest: reference.manifest_semantic_digest,
    live_manifest_digest: live.manifest_semantic_digest,
    live_execution_attestation_digest: live.execution_attestation_digest!,
    track_id: 'joint',
    suite_ids: ['live-mom-core'],
    workload_snapshot_digest: digest('c'),
    benchmark_revisions: { 'live-mom-core': 'mom-campaign-cohort-v1' },
    matched_cases: 59,
    decision_mismatches: 0,
    outcome_mismatches: 0,
    unavailable_cases: 0,
    sample_count: 59,
    point_estimate: 1,
    lower_bound: 0.95,
    confidence_level: 0.95,
    verdict: 'pass',
    digest: digest('d'),
  }
}
