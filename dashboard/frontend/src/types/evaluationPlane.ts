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

export const EVALUATION_CHANGE_PROFILE_IDS = [
  'schema_adapter',
  'recipe',
  'selector',
  'model_pool',
  'runtime_capacity',
  'agent_multimodal',
  'online_adaptation',
] as const

export const EVALUATION_SCHEMA_VERSION = 'evaluation.v1' as const

export type EvaluationSchemaVersion = typeof EVALUATION_SCHEMA_VERSION
export type EvaluationTrackId = (typeof EVALUATION_TRACK_IDS)[number]
export type EvaluationChangeProfileId = (typeof EVALUATION_CHANGE_PROFILE_IDS)[number]
export type EvaluationMode = 'replay' | 'live'
export type EvidenceLevel = 'E0' | 'E1' | 'E2' | 'E3' | 'E4' | 'E5'
export type EvaluationRunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
export type EvaluationTrackStatus = EvaluationRunStatus | 'unavailable' | 'skipped'
export type GateDisposition = 'required' | 'advisory' | 'not_applicable' | 'waived'
export type GateVerdict = 'pass' | 'fail' | 'unavailable' | 'waived' | 'not_applicable'

export interface EvaluationCatalogTrack {
  id: EvaluationTrackId
  name: string
  description: string
  modes: EvaluationMode[]
  metrics: string[]
  evidence_levels?: EvidenceLevel[]
}

export interface EvaluationCatalogSuite {
  id: string
  name: string
  description: string
  track_ids: EvaluationTrackId[]
  modes: EvaluationMode[]
  evidence_level: EvidenceLevel
  case_count?: number
  revision?: string
  tags?: string[]
}

export interface EvaluationCatalogTarget {
  id: string
  name: string
  description: string
  kind: string
  track_ids: EvaluationTrackId[]
  modes: EvaluationMode[]
  evidence_level?: EvidenceLevel
  healthy?: boolean
  labels?: Record<string, string>
}

export interface EvaluationCatalogChangeProfile {
  id: EvaluationChangeProfileId
  name: string
  description: string
}

export interface EvaluationCatalog {
  schema_version: EvaluationSchemaVersion
  gate_contract_version: string
  generated_at?: string
  change_profiles: EvaluationCatalogChangeProfile[]
  tracks: EvaluationCatalogTrack[]
  suites: EvaluationCatalogSuite[]
  targets: EvaluationCatalogTarget[]
}

export interface CreateEvaluationRunRequest {
  name: string
  description: string
  suite_ids: string[]
  track_ids: EvaluationTrackId[]
  mode: EvaluationMode
  target_id: string
  change_profile: EvaluationChangeProfileId
  sample_limit: number
  concurrency: number
  seed: number
  baseline_run_id?: string
  auto_start: boolean
}

export interface EvaluationRunProgress {
  percent: number
  completed: number
  total: number
  current_track_id?: EvaluationTrackId
  message?: string
}

export interface EvaluationRun {
  schema_version: EvaluationSchemaVersion
  id: string
  name: string
  description: string
  status: EvaluationRunStatus
  mode: EvaluationMode
  evidence_level: EvidenceLevel
  target_id: string
  change_profile: EvaluationChangeProfileId
  suite_ids: string[]
  track_ids: EvaluationTrackId[]
  sample_limit: number
  concurrency: number
  seed: number
  baseline_run_id?: string
  progress: EvaluationRunProgress
  created_at: string
  started_at?: string
  completed_at?: string
  error?: string
}

export interface EvaluationCoverage {
  evaluated: number
  total: number
  fraction: number
  unavailable?: number
  confidence_level?: number
  confidence_interval?: [number, number]
}

export interface EvaluationMetric {
  id: string
  name: string
  track_id?: EvaluationTrackId
  value: number | null
  unit: string
  direction?: 'higher_is_better' | 'lower_is_better' | 'target'
  baseline_value?: number | null
  delta?: number | null
  confidence_interval?: [number, number]
  sample_count?: number
}

export interface EvaluationGate {
  id: string
  name: string
  description?: string
  track_id?: EvaluationTrackId
  disposition: GateDisposition
  verdict: GateVerdict
  change_profile: EvaluationChangeProfileId
  contract_version: string
  evidence_refs: string[]
  evidence_level?: EvidenceLevel
  observed?: number | null
  threshold?: {
    operator: string
    value: number
    unit?: string
  }
  sample_count?: number
  coverage?: EvaluationCoverage
  owner?: string
  evaluated_at?: string
  rationale?: string
}

export interface EvaluationArtifact {
  id: string
  name: string
  kind: string
  uri?: string
  digest?: string
  media_type?: string
  size_bytes?: number
}

export interface EvaluationProvenance {
  schema_version: EvaluationSchemaVersion
  generated_at: string
  code_revision?: string
  benchmark_revisions?: Record<string, string>
  workload_snapshot_digest?: string
  policy_snapshot_digest?: string
  binding_snapshot_digest?: string
  pool_snapshot_digest?: string
  environment_snapshot_digest?: string
  target_id: string
  seed: number
  redaction_policy?: string
}

export interface EvaluationCostAmount {
  amount: number | null
  currency: string
  input_tokens?: number
  output_tokens?: number
  gpu_seconds?: number
  energy_kwh?: number
}

export interface EvaluationCostLedgers {
  runtime: EvaluationCostAmount
  evaluation_overhead: EvaluationCostAmount
  capacity_tco: EvaluationCostAmount
}

export interface EvaluationTrackReport {
  track_id: EvaluationTrackId
  status: EvaluationTrackStatus
  evidence_level: EvidenceLevel
  summary: string
  coverage: EvaluationCoverage
  metrics: EvaluationMetric[]
  gates: EvaluationGate[]
  artifacts?: EvaluationArtifact[]
  error?: string
}

export interface EvaluationReportSummary {
  verdict: GateVerdict
  quality_score: number | null
  latency_p95_ms: number | null
  runtime_cost: number | null
  capacity_tco: number | null
  coverage: EvaluationCoverage
  passed_gates: number
  failed_gates: number
  unavailable_gates: number
}

export interface EvaluationReport {
  schema_version: EvaluationSchemaVersion
  run: EvaluationRun
  summary: EvaluationReportSummary
  tracks: EvaluationTrackReport[]
  metrics: EvaluationMetric[]
  gates: EvaluationGate[]
  costs: EvaluationCostLedgers
  recommendations: string[]
  provenance: EvaluationProvenance
  artifacts: EvaluationArtifact[]
}

export interface EvaluationComparison {
  schema_version: EvaluationSchemaVersion
  baseline_run_id: string
  candidate_run_id: string
  verdict: GateVerdict
  summary: string
  metrics: EvaluationMetric[]
  gates: EvaluationGate[]
  recommendations: string[]
  created_at?: string
}

export interface EvaluationRunEvent {
  id?: string
  run_id: string
  type: string
  timestamp: string
  message: string
  track_id?: EvaluationTrackId
  progress?: EvaluationRunProgress
}

export const TRACK_PRESENTATION: Record<EvaluationTrackId, { label: string; description: string }> =
  {
    routing: {
      label: 'Routing',
      description: 'Recipe decisions, oracle regret, abstention, calibration, and eligibility.',
    },
    model_pool: {
      label: 'Model pool',
      description: 'Arm quality, complementarity, coverage, dominance, and failure isolation.',
    },
    joint: {
      label: 'Routing + pool',
      description: 'End-to-end quality, latency, cost, reliability, and decomposition.',
    },
    agentic: {
      label: 'Agentic',
      description: 'Trajectory success, tool use, state continuity, recovery, and budget.',
    },
    multimodal: {
      label: 'Multimodal',
      description: 'Modality-aware routing, perception, grounding, and cross-modal quality.',
    },
    preference: {
      label: 'Preference',
      description: 'Offline preference, online trials, stability, and feedback adaptation.',
    },
    safety: {
      label: 'Safety',
      description: 'Policy adherence, attack resistance, privacy, and unsafe regressions.',
    },
    capacity: {
      label: 'Capacity',
      description: 'Throughput, saturation, queueing, SLOs, GPU efficiency, and TCO.',
    },
  }
