import type {
  EvaluationAttestationRevision,
  EvaluationCampaignContractVersion,
  EvaluationCampaignFidelityContractVersion,
  EvaluationCampaignGateID,
  EvaluationCampaignBindingKind,
  EvaluationCampaignPairedLiveContractVersion,
  EvaluationCampaignSlotID,
  EvaluationChangeProfileId,
  EvaluationSchemaVersion,
  EvaluationSummaryVerdict,
  EvaluationTrackId,
  EvidenceLevel,
  GateDisposition,
  GateVerdict,
} from './evaluationPlane'

export type EvaluationCampaignStatus = 'decided'
export interface EvaluationCampaignControlledPairBinding {
  baseline_run_id: string
  candidate_run_id: string
}

export interface EvaluationCampaignFidelityBinding {
  reference_run_id: string
  live_run_id: string
}

export interface EvaluationCampaignGateBindings {
  g2_run_id?: string
  g3_controlled_pair?: EvaluationCampaignControlledPairBinding
  g4_run_id?: string
  g5_fidelity?: EvaluationCampaignFidelityBinding
  g6_run_id?: string
  g7_run_id?: string
  g8_run_id?: string
  g9_run_id?: string
}

export interface CreateEvaluationCampaignPayload {
  client_request_id: string
  name: string
  description: string
  change_profile: EvaluationChangeProfileId
  gate_bindings: EvaluationCampaignGateBindings
}

export type EvaluationCampaignEvidenceBindingRole =
  | 'evidence'
  | 'baseline'
  | 'candidate'
  | 'reference'
  | 'live'

export interface EvaluationCampaignEvidenceAnchor {
  slot_id: EvaluationCampaignSlotID
  gate_id: Uppercase<EvaluationCampaignSlotID>
  binding_role: EvaluationCampaignEvidenceBindingRole
  run_id: string
  candidate_subject_digest?: string
  manifest_semantic_digest: string
  manifest_artifact_digest: string
  report_digest: string
  private_receipt_digest: string
  execution_attestation_digest?: string
}

export interface EvaluationCampaignGate {
  id: string
  name: string
  disposition: GateDisposition
  verdict: GateVerdict
  evidence_level: EvidenceLevel
  source: string
  evidence_refs: string[]
  observed?: number
  threshold?: {
    operator: string
    value: number
    unit?: string
  }
  sample_count?: number
  rationale: string
}

export type EvaluationCampaignPairedDirection = 'higher_is_better' | 'lower_is_better'

export interface EvaluationCampaignPairedStatistic {
  id: string
  gate_id: 'G3' | 'G8'
  track_id: Extract<
    EvaluationTrackId,
    'routing' | 'model_pool' | 'joint' | 'multimodal' | 'capacity'
  >
  analysis_unit: string
  direction: EvaluationCampaignPairedDirection
  margin: number
  baseline_value?: number
  candidate_value?: number
  delta?: number
  confidence_level: number
  confidence_interval: number[]
  candidate_confidence_interval?: number[]
  sample_count: number
  missing_pairs: number
  verdict: EvaluationSummaryVerdict
}

export interface EvaluationCampaignArmReliabilityStatistic {
  arm_id: string
  cohort: 'paired' | 'baseline_only' | 'candidate_only'
  direction: 'lower_is_better'
  margin: number
  baseline_failure_rate?: number
  candidate_failure_rate?: number
  delta?: number
  confidence_level: number
  confidence_interval: number[]
  candidate_confidence_interval?: number[]
  baseline_sample_count: number
  candidate_sample_count: number
  verdict: EvaluationSummaryVerdict
}

export interface EvaluationCampaignG3PromotionPolicy {
  candidate_normalized_regret_maximum: number
  paired_normalized_regret_margin: number
  minimum_no_information_frontier_lift: number
  minimum_joint_reliability: number
  maximum_all_arm_failure_rate: number
  minimum_candidate_arm_reliability: number
}

export interface EvaluationCampaignG3PromotionStatistic {
  id: string
  direction: EvaluationCampaignPairedDirection
  estimate: number
  confidence_level: number
  confidence_interval: number[]
  threshold: {
    operator: '>=' | '<='
    value: number
    unit: string
  }
  sample_count: number
  missing_cases: number
  verdict: EvaluationSummaryVerdict
}

export interface EvaluationCampaignPairedLiveEvidence {
  schema_version: EvaluationSchemaVersion
  contract_version: EvaluationCampaignPairedLiveContractVersion
  controlled_pair_session_id: string
  controlled_pair_protocol: 'abba-interleaved.v1'
  baseline_run_id: string
  candidate_run_id: string
  candidate_subject_digest: string
  baseline_target_id: string
  candidate_target_id: string
  mixture_id: string
  recipe_name: string
  track_ids: Array<
    Extract<EvaluationTrackId, 'routing' | 'model_pool' | 'joint' | 'multimodal' | 'capacity'>
  >
  workload_snapshot_digest: string
  benchmark_revisions: Record<string, string>
  seed: number
  bootstrap_samples: number
  confidence_level: number
  promotion_policy: EvaluationCampaignG3PromotionPolicy
  promotion_statistics: EvaluationCampaignG3PromotionStatistic[]
  baseline_manifest_digest: string
  candidate_manifest_digest: string
  baseline_execution_attestation_digest: string
  candidate_execution_attestation_digest: string
  baseline_policy_snapshot_digest: string
  candidate_policy_snapshot_digest: string
  baseline_binding_snapshot_digest: string
  candidate_binding_snapshot_digest: string
  baseline_pool_snapshot_digest: string
  candidate_pool_snapshot_digest: string
  baseline_environment_snapshot_digest: string
  candidate_environment_snapshot_digest: string
  baseline_backend_topology_digest: string
  candidate_backend_topology_digest: string
  baseline_code_revision: string
  candidate_code_revision: string
  statistics: EvaluationCampaignPairedStatistic[]
  model_pool_arm_reliability: EvaluationCampaignArmReliabilityStatistic[]
  digest: string
}

export interface EvaluationCampaignFidelityEvidence {
  schema_version: EvaluationSchemaVersion
  contract_version: EvaluationCampaignFidelityContractVersion
  reference_run_id: string
  live_run_id: string
  candidate_subject_digest: string
  reference_manifest_digest: string
  live_manifest_digest: string
  live_execution_attestation_digest: string
  track_id: EvaluationTrackId
  suite_ids: string[]
  workload_snapshot_digest: string
  benchmark_revisions: Record<string, string>
  matched_cases: number
  decision_mismatches: number
  outcome_mismatches: number
  unavailable_cases: number
  sample_count: number
  point_estimate: number
  lower_bound: number
  confidence_level: number
  verdict: EvaluationSummaryVerdict
  digest: string
}

export interface EvaluationCampaignDecision {
  schema_version: EvaluationSchemaVersion
  contract_version: EvaluationCampaignContractVersion
  attestation_revision: EvaluationAttestationRevision
  campaign_id: string
  campaign_digest: string
  decision_digest: string
  verdict: EvaluationSummaryVerdict
  summary: string
  gates: EvaluationCampaignGate[]
  evidence: EvaluationCampaignEvidenceAnchor[]
  paired_live_evidence?: EvaluationCampaignPairedLiveEvidence
  fidelity_evidence?: EvaluationCampaignFidelityEvidence
  recommendations: string[]
  created_at: string
}

export interface EvaluationCampaign {
  schema_version: EvaluationSchemaVersion
  contract_version: EvaluationCampaignContractVersion
  id: string
  name: string
  description: string
  change_profile: EvaluationChangeProfileId
  status: EvaluationCampaignStatus
  gate_bindings: EvaluationCampaignGateBindings
  manifest_digest: string
  created_at: string
  decision: EvaluationCampaignDecision
}

export interface EvaluationCampaignSlotReadiness {
  gate_id: EvaluationCampaignGateID
  binding_kind: EvaluationCampaignBindingKind
  eligible_run_ids: string[]
  controlled_pair_source_run_ids: string[]
  controlled_pair_candidate_run_ids: string[]
  fidelity_reference_run_ids: string[]
  fidelity_live_run_ids: string[]
}

export interface EvaluationCampaignReadiness {
  schema_version: EvaluationSchemaVersion
  change_profile: EvaluationChangeProfileId
  next_cursor?: string
  total_runs: number
  slots: EvaluationCampaignSlotReadiness[]
}
