export const EVALUATION_TRACK_IDS = [
  'routing',
  'model_pool',
  'joint',
  'agentic',
  'multimodal',
  'preference',
  'safety',
  'capacity',
] as const
export const EVALUATION_GATE_DISPOSITIONS = [
  'required',
  'advisory',
  'not_applicable',
] as const
export const EVALUATION_SUMMARY_VERDICTS = ['pass', 'fail', 'unavailable'] as const
export const EVALUATION_GATE_VERDICTS = [
  ...EVALUATION_SUMMARY_VERDICTS,
  'not_applicable',
] as const

export const EVALUATION_SCHEMA_VERSION = 'evaluation.v1' as const
export const EVALUATION_ATTESTATION_REVISION = 'evaluation-server-attestation.v2' as const
export const EVALUATION_GATE_CONTRACT_VERSION = 'evaluation-release-gates.v2' as const
export const EVALUATION_CAMPAIGN_CONTRACT_VERSION = 'evaluation-campaign.v2' as const
export const EVALUATION_CAMPAIGN_COHORT_SCHEMA_VERSION =
  'evaluation-campaign-cohort.v1' as const
export const EVALUATION_CAMPAIGN_PAIRED_LIVE_CONTRACT_VERSION =
  'evaluation-campaign-paired-live.v3' as const
export const EVALUATION_CAMPAIGN_FIDELITY_CONTRACT_VERSION =
  'evaluation-campaign-fidelity.v2' as const

export type EvaluationSchemaVersion = typeof EVALUATION_SCHEMA_VERSION
export type EvaluationAttestationRevision = typeof EVALUATION_ATTESTATION_REVISION
export type EvaluationGateContractVersion = typeof EVALUATION_GATE_CONTRACT_VERSION
export type EvaluationCampaignContractVersion = typeof EVALUATION_CAMPAIGN_CONTRACT_VERSION
export type EvaluationCampaignCohortSchemaVersion =
  typeof EVALUATION_CAMPAIGN_COHORT_SCHEMA_VERSION
export type EvaluationCampaignPairedLiveContractVersion =
  typeof EVALUATION_CAMPAIGN_PAIRED_LIVE_CONTRACT_VERSION
export type EvaluationCampaignFidelityContractVersion =
  typeof EVALUATION_CAMPAIGN_FIDELITY_CONTRACT_VERSION
export type EvaluationTrackId = (typeof EVALUATION_TRACK_IDS)[number]
// Change profiles are server-owned catalog entries, not a browser-side enum.
export type EvaluationChangeProfileId = string
export type EvaluationMode = 'replay' | 'live'
export type EvidenceLevel = 'E0' | 'E1' | 'E2' | 'E3' | 'E4' | 'E5'
export type EvaluationRunStatus =
  | 'pending'
  | 'running'
  | 'sealing'
  | 'completed'
  | 'failed'
  | 'cancelled'
export type EvaluationTrackStatus = EvaluationRunStatus | 'unavailable' | 'skipped'
export type GateDisposition = (typeof EVALUATION_GATE_DISPOSITIONS)[number]
export type EvaluationSummaryVerdict = (typeof EVALUATION_SUMMARY_VERDICTS)[number]
export type GateVerdict = (typeof EVALUATION_GATE_VERDICTS)[number]
export const EVALUATION_METHOD_EVIDENCE_SOURCE = Object.freeze({
  DIAGNOSTIC_FIXTURE: 'diagnostic_fixture',
  LIVE_RUNTIME: 'live_runtime',
  NORMALIZED_IMPORT: 'normalized_import',
  SERVER_BROKERED_LIVE: 'server_brokered_live',
  LIVE_PRODUCTION: 'live_production',
} as const)
export type EvaluationMethodEvidenceSource =
  (typeof EVALUATION_METHOD_EVIDENCE_SOURCE)[keyof typeof EVALUATION_METHOD_EVIDENCE_SOURCE]
export const EVALUATION_METHOD_EVIDENCE_SOURCES = Object.freeze(
  Object.values(EVALUATION_METHOD_EVIDENCE_SOURCE),
)
const EVALUATION_METHOD_EVIDENCE_SOURCE_SET: ReadonlySet<string> = new Set(
  EVALUATION_METHOD_EVIDENCE_SOURCES,
)
export function isEvaluationMethodEvidenceSource(
  value: unknown,
): value is EvaluationMethodEvidenceSource {
  return typeof value === 'string' && EVALUATION_METHOD_EVIDENCE_SOURCE_SET.has(value)
}
export type EvaluationMethodStatus = 'configured' | 'data_required'

export interface EvaluationCatalogMethod {
  id: string
  track_id: EvaluationTrackId
  qualified_gate_ids: string[]
  evidence_source: EvaluationMethodEvidenceSource
  status: EvaluationMethodStatus
  reason?: string
}

export interface EvaluationCatalogTrack {
  id: EvaluationTrackId
  name: string
  description: string
  modes: EvaluationMode[]
  metrics: string[]
  evidence_levels: EvidenceLevel[]
}

export interface EvaluationCampaignProtocol {
  schema_version: EvaluationCampaignCohortSchemaVersion
  minimum_cases: number
}

export interface EvaluationCatalogSuite {
  id: string
  executors: Partial<Record<EvaluationMode, string>>
  name: string
  description: string
  track_ids: EvaluationTrackId[]
  modes: EvaluationMode[]
  evidence_level: EvidenceLevel
  case_count?: number
  campaign_protocol?: EvaluationCampaignProtocol
  revision: string
  tags: string[]
  methods: EvaluationCatalogMethod[]
}

export interface EvaluationModelArm {
  id: string
  model: string
  provider_model_id_digest: string
  input_cost_per_million_tokens_usd: number
  output_cost_per_million_tokens_usd: number
  capabilities?: string[]
  modalities?: Array<'text' | 'image' | 'document' | 'audio' | 'video'>
  context_window_tokens?: number
  parameter_size?: string
  runtime_revision?: string
  config_digest?: string
}

export interface EvaluationMixtureDecision {
  name: string
  algorithm: string
  arm_ids: string[]
}

export interface EvaluationSupportModel {
  model: string
  provider_model_id_digest: string
  config_digest: string
  runtime_revision?: string
  backend_topology_digest: string
}

export interface EvaluationRoutingRecipeInputSpec {
  id: string
  value_kind: 'numeric' | 'none'
}

export interface EvaluationRoutingRecipeProjectionSpec {
  id: string
  value_kind: 'numeric' | 'probability'
  outcome_binding: 'selected_pool_quality' | 'selected_is_oracle'
}

export interface EvaluationRoutingRecipePlan {
  contract_version: 'routing-recipe-plan.v1'
  plan_digest: string
  target_snapshot_digest: string
  arm_ids: string[]
  fallback_arm_id?: string
  signals: EvaluationRoutingRecipeInputSpec[]
  projections: EvaluationRoutingRecipeProjectionSpec[]
  top_k: number[]
}

/**
 * Browser-safe, immutable view of the exact Mixture-of-Models binding being
 * evaluated. Connectivity and provider identities never cross this boundary.
 */
export interface EvaluationMixture {
  id: string
  entrypoint_model: string
  aliases: string[]
  recipe_name: string
  recipe_description: string
  recipe_digest: string
  pool_digest: string
  selector_policy_digest: string
  selector_digest: string
  adaptation_digest: string
  binding_digest: string
  model_arms: EvaluationModelArm[]
  support_models: EvaluationSupportModel[]
  fallback_arm_id?: string
  decisions: EvaluationMixtureDecision[]
  routing_recipe_plan: EvaluationRoutingRecipePlan
}

export interface EvaluationCatalogTarget {
  id: string
  name: string
  description: string
  kind: string
  track_ids: EvaluationTrackId[]
  modes: EvaluationMode[]
  accepted_executors: Partial<Record<EvaluationMode, string[]>>
  evidence_level?: EvidenceLevel
  healthy?: boolean
  labels?: Record<string, string>
  mixture?: EvaluationMixture
}

export interface EvaluationCatalogChangeProfile {
  id: EvaluationChangeProfileId
  name: string
  description: string
  campaign_slots: EvaluationCatalogCampaignSlot[]
}

export const EVALUATION_CAMPAIGN_GATE_IDS = [
  'G2',
  'G3',
  'G4',
  'G5',
  'G6',
  'G7',
  'G8',
  'G9',
] as const
export const EVALUATION_RELEASE_GATE_IDS = [
  'G0',
  'G1',
  ...EVALUATION_CAMPAIGN_GATE_IDS,
] as const
export type EvaluationCampaignSlotID = 'g2' | 'g3' | 'g4' | 'g5' | 'g6' | 'g7' | 'g8' | 'g9'
export type EvaluationCampaignGateID = (typeof EVALUATION_CAMPAIGN_GATE_IDS)[number]
export type EvaluationCampaignBindingKind = 'run' | 'controlled_pair' | 'fidelity_pair'

export interface EvaluationCatalogCampaignSlot {
  gate_id: EvaluationCampaignGateID
  name: string
  description: string
  disposition: GateDisposition
  binding_kind: EvaluationCampaignBindingKind
  track_id: EvaluationTrackId
  mode: EvaluationMode
  minimum_evidence_level: EvidenceLevel
  accepted_executor_ids: string[]
}

export interface EvaluationCatalog {
  schema_version: EvaluationSchemaVersion
  gate_contract_version: EvaluationGateContractVersion
  generated_at: string
  change_profiles: EvaluationCatalogChangeProfile[]
  tracks: EvaluationCatalogTrack[]
  suites: EvaluationCatalogSuite[]
  targets: EvaluationCatalogTarget[]
}

export interface EvaluationCapacitySLO {
  schema_version: EvaluationSchemaVersion
  required_concurrency: number
  max_latency_p95_ms: number
  max_error_rate: number
  min_throughput_rps: number
  min_throughput_scaling_efficiency: number
}

export interface EvaluationCapacityLoadProtocol {
  schema_version: EvaluationSchemaVersion
  kind: 'closed-loop'
  concurrency_levels: number[]
  warmup_request_multiplier: number
  measurement_requests_per_repetition: number
  repetitions_per_level: number
  minimum_measurement_clusters_per_level: 3
  confidence_level: 0.95
  max_error_rate_cluster_range: 0.05
  max_throughput_cv: number
  max_latency_p95_cv: number
}

export interface CreateEvaluationRunPayload {
  client_request_id: string
  name: string
  description: string
  suite_ids: string[]
  track_ids: EvaluationTrackId[]
  mode: EvaluationMode
  target_id: string
  change_profile: EvaluationChangeProfileId
  sample_limit: number
  concurrency: number
  capacity_slo?: EvaluationCapacitySLO
  capacity_load_protocol?: EvaluationCapacityLoadProtocol
  seed: number
  baseline_run_id?: string
}

export interface EvaluationExperimentIntent extends CreateEvaluationRunPayload {
  autoStart: boolean
}

export interface EvaluationRunProgress {
  percent: number
  completed: number
  total: number
  current_track_id?: EvaluationTrackId
  message?: string
}

export type EvaluationControlledPairRole = 'baseline' | 'candidate'

export interface EvaluationControlledPairMembership {
  pair_id: string
  role: EvaluationControlledPairRole
}

export interface EvaluationRun {
  schema_version: EvaluationSchemaVersion
  id: string
  client_request_id: string
  name: string
  description: string
  status: EvaluationRunStatus
  mode: EvaluationMode
  evidence_level: EvidenceLevel
  track_evidence_levels: Partial<Record<EvaluationTrackId, EvidenceLevel>>
  target_id: string
  mixture?: EvaluationMixture
  change_profile: EvaluationChangeProfileId
  suite_ids: string[]
  track_ids: EvaluationTrackId[]
  sample_limit: number
  concurrency: number
  capacity_slo?: EvaluationCapacitySLO
  capacity_load_protocol?: EvaluationCapacityLoadProtocol
  seed: number
  baseline_run_id?: string
  controlled_pair?: EvaluationControlledPairMembership
  progress: EvaluationRunProgress
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
}

export interface EvaluationRunLedgerWarning {
  code: string
  evidence_id: string
  evidence_file: string
  message: string
}

export interface EvaluationRunLedger {
  schema_version: EvaluationSchemaVersion
  runs: EvaluationRun[]
  next_cursor?: string
  total_runs: number
  ledger_complete: boolean
  warning_count: number
  warnings: EvaluationRunLedgerWarning[]
}

export type EvaluationRunEventType =
  | 'snapshot'
  | 'progress'
  | 'track'
  | 'gate'
  | 'artifact'
  | 'completed'
  | 'failed'
  | 'cancelled'

interface EvaluationRunEventBase {
  id: string
  run_id: string
  timestamp: string
  message: string
  progress?: EvaluationRunProgress
}

export interface EvaluationTrackRunEvent extends EvaluationRunEventBase {
  type: 'track'
  track_id: EvaluationTrackId
  progress: EvaluationRunProgress
  payload: {
    record_count: number
  }
}

export interface EvaluationTerminalRunEvent extends EvaluationRunEventBase {
  type: 'completed' | 'failed' | 'cancelled'
  progress: EvaluationRunProgress
  track_id?: never
  payload?: never
}

export interface EvaluationPayloadlessRunEvent extends EvaluationRunEventBase {
  type: Exclude<EvaluationRunEventType, 'track' | EvaluationTerminalRunEvent['type']>
  track_id?: never
  payload?: never
}

export type EvaluationRunEvent =
  | EvaluationTrackRunEvent
  | EvaluationTerminalRunEvent
  | EvaluationPayloadlessRunEvent
