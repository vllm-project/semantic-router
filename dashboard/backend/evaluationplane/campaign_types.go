package evaluationplane

import "time"

const CampaignContractVersion = "evaluation-campaign.v2"

const CampaignPairedLiveContractVersion = "evaluation-campaign-paired-live.v3"

const CampaignFidelityContractVersion = "evaluation-campaign-fidelity.v2"

type CampaignStatus string

const CampaignStatusDecided CampaignStatus = "decided"

type CampaignControlledPairBinding struct {
	BaselineRunID  string `json:"baseline_run_id"`
	CandidateRunID string `json:"candidate_run_id"`
}

type CampaignFidelityBinding struct {
	ReferenceRunID string `json:"reference_run_id"`
	LiveRunID      string `json:"live_run_id"`
}

// CampaignGateBindings binds each gate to independent, purpose-qualified run
// evidence. A run ID may appear at most once, so one broad report cannot be
// substituted into several release-gate slots.
type CampaignGateBindings struct {
	G2RunID          string                         `json:"g2_run_id,omitempty"`
	G3ControlledPair *CampaignControlledPairBinding `json:"g3_controlled_pair,omitempty"`
	G4RunID          string                         `json:"g4_run_id,omitempty"`
	G5Fidelity       *CampaignFidelityBinding       `json:"g5_fidelity,omitempty"`
	G6RunID          string                         `json:"g6_run_id,omitempty"`
	G7RunID          string                         `json:"g7_run_id,omitempty"`
	G8RunID          string                         `json:"g8_run_id,omitempty"`
	G9RunID          string                         `json:"g9_run_id,omitempty"`
}

type CreateCampaignRequest struct {
	ClientRequestID string               `json:"client_request_id"`
	Name            string               `json:"name"`
	Description     string               `json:"description"`
	ChangeProfile   ChangeProfile        `json:"change_profile"`
	GateBindings    CampaignGateBindings `json:"gate_bindings"`
}

type CampaignEvidenceAnchor struct {
	SlotID                     string `json:"slot_id"`
	GateID                     string `json:"gate_id"`
	BindingRole                string `json:"binding_role"`
	RunID                      string `json:"run_id"`
	CandidateSubjectDigest     string `json:"candidate_subject_digest,omitempty"`
	ManifestSemanticDigest     string `json:"manifest_semantic_digest"`
	ManifestArtifactDigest     string `json:"manifest_artifact_digest"`
	ReportDigest               string `json:"report_digest"`
	PrivateReceiptDigest       string `json:"private_receipt_digest"`
	ExecutionAttestationDigest string `json:"execution_attestation_digest,omitempty"`
}

type CampaignGate struct {
	ID            string          `json:"id"`
	Name          string          `json:"name"`
	Disposition   GateDisposition `json:"disposition"`
	Verdict       GateVerdict     `json:"verdict"`
	EvidenceLevel EvidenceLevel   `json:"evidence_level"`
	Source        string          `json:"source"`
	EvidenceRefs  []string        `json:"evidence_refs"`
	Observed      *float64        `json:"observed,omitempty"`
	Threshold     *GateThreshold  `json:"threshold,omitempty"`
	SampleCount   int             `json:"sample_count,omitempty"`
	Rationale     string          `json:"rationale"`
}

type CampaignPairedStatistic struct {
	ID                          string      `json:"id"`
	GateID                      string      `json:"gate_id"`
	TrackID                     TrackID     `json:"track_id"`
	AnalysisUnit                string      `json:"analysis_unit"`
	Direction                   string      `json:"direction"`
	Margin                      float64     `json:"margin"`
	BaselineValue               *float64    `json:"baseline_value,omitempty"`
	CandidateValue              *float64    `json:"candidate_value,omitempty"`
	Delta                       *float64    `json:"delta,omitempty"`
	ConfidenceLevel             float64     `json:"confidence_level"`
	ConfidenceInterval          []float64   `json:"confidence_interval"`
	CandidateConfidenceInterval []float64   `json:"candidate_confidence_interval,omitempty"`
	SampleCount                 int         `json:"sample_count"`
	MissingPairs                int         `json:"missing_pairs"`
	Verdict                     GateVerdict `json:"verdict"`
}

type CampaignArmReliabilityStatistic struct {
	ArmID                       string      `json:"arm_id"`
	Cohort                      string      `json:"cohort"`
	Direction                   string      `json:"direction"`
	Margin                      float64     `json:"margin"`
	BaselineFailureRate         *float64    `json:"baseline_failure_rate,omitempty"`
	CandidateFailureRate        *float64    `json:"candidate_failure_rate,omitempty"`
	Delta                       *float64    `json:"delta,omitempty"`
	ConfidenceLevel             float64     `json:"confidence_level"`
	ConfidenceInterval          []float64   `json:"confidence_interval"`
	CandidateConfidenceInterval []float64   `json:"candidate_confidence_interval,omitempty"`
	BaselineSampleCount         int         `json:"baseline_sample_count"`
	CandidateSampleCount        int         `json:"candidate_sample_count"`
	Verdict                     GateVerdict `json:"verdict"`
}

// CampaignG3PromotionPolicy is immutable server policy. It is copied into the
// paired-live receipt so a decision can be validated after restart without
// trusting request or browser state.
type CampaignG3PromotionPolicy struct {
	CandidateNormalizedRegretMaximum float64 `json:"candidate_normalized_regret_maximum"`
	PairedNormalizedRegretMargin     float64 `json:"paired_normalized_regret_margin"`
	MinimumNoInformationFrontierLift float64 `json:"minimum_no_information_frontier_lift"`
	MinimumJointReliability          float64 `json:"minimum_joint_reliability"`
	MaximumAllArmFailureRate         float64 `json:"maximum_all_arm_failure_rate"`
	MinimumCandidateArmReliability   float64 `json:"minimum_candidate_arm_reliability"`
}

type CampaignG3PromotionStatistic struct {
	ID                 string        `json:"id"`
	Direction          string        `json:"direction"`
	Estimate           float64       `json:"estimate"`
	ConfidenceLevel    float64       `json:"confidence_level"`
	ConfidenceInterval []float64     `json:"confidence_interval"`
	Threshold          GateThreshold `json:"threshold"`
	SampleCount        int           `json:"sample_count"`
	MissingCases       int           `json:"missing_cases"`
	Verdict            GateVerdict   `json:"verdict"`
}

type CampaignPairedLiveEvidence struct {
	SchemaVersion                       string                            `json:"schema_version"`
	ContractVersion                     string                            `json:"contract_version"`
	ControlledPairSessionID             string                            `json:"controlled_pair_session_id"`
	ControlledPairProtocol              string                            `json:"controlled_pair_protocol"`
	BaselineRunID                       string                            `json:"baseline_run_id"`
	CandidateRunID                      string                            `json:"candidate_run_id"`
	CandidateSubjectDigest              string                            `json:"candidate_subject_digest"`
	BaselineTargetID                    string                            `json:"baseline_target_id"`
	CandidateTargetID                   string                            `json:"candidate_target_id"`
	MixtureID                           string                            `json:"mixture_id"`
	RecipeName                          string                            `json:"recipe_name"`
	TrackIDs                            []TrackID                         `json:"track_ids"`
	WorkloadSnapshotDigest              string                            `json:"workload_snapshot_digest"`
	BenchmarkRevisions                  map[string]string                 `json:"benchmark_revisions"`
	Seed                                int64                             `json:"seed"`
	BootstrapSamples                    int                               `json:"bootstrap_samples"`
	ConfidenceLevel                     float64                           `json:"confidence_level"`
	PromotionPolicy                     CampaignG3PromotionPolicy         `json:"promotion_policy"`
	PromotionStatistics                 []CampaignG3PromotionStatistic    `json:"promotion_statistics"`
	BaselineManifestDigest              string                            `json:"baseline_manifest_digest"`
	CandidateManifestDigest             string                            `json:"candidate_manifest_digest"`
	BaselineExecutionAttestationDigest  string                            `json:"baseline_execution_attestation_digest"`
	CandidateExecutionAttestationDigest string                            `json:"candidate_execution_attestation_digest"`
	BaselinePolicySnapshotDigest        string                            `json:"baseline_policy_snapshot_digest"`
	CandidatePolicySnapshotDigest       string                            `json:"candidate_policy_snapshot_digest"`
	BaselineBindingSnapshotDigest       string                            `json:"baseline_binding_snapshot_digest"`
	CandidateBindingSnapshotDigest      string                            `json:"candidate_binding_snapshot_digest"`
	BaselinePoolSnapshotDigest          string                            `json:"baseline_pool_snapshot_digest"`
	CandidatePoolSnapshotDigest         string                            `json:"candidate_pool_snapshot_digest"`
	BaselineEnvironmentSnapshotDigest   string                            `json:"baseline_environment_snapshot_digest"`
	CandidateEnvironmentSnapshotDigest  string                            `json:"candidate_environment_snapshot_digest"`
	BaselineBackendTopologyDigest       string                            `json:"baseline_backend_topology_digest"`
	CandidateBackendTopologyDigest      string                            `json:"candidate_backend_topology_digest"`
	BaselineCodeRevision                string                            `json:"baseline_code_revision"`
	CandidateCodeRevision               string                            `json:"candidate_code_revision"`
	Statistics                          []CampaignPairedStatistic         `json:"statistics"`
	ModelPoolArmReliability             []CampaignArmReliabilityStatistic `json:"model_pool_arm_reliability"`
	Digest                              string                            `json:"digest"`
}

type CampaignFidelityEvidence struct {
	SchemaVersion                  string            `json:"schema_version"`
	ContractVersion                string            `json:"contract_version"`
	ReferenceRunID                 string            `json:"reference_run_id"`
	LiveRunID                      string            `json:"live_run_id"`
	CandidateSubjectDigest         string            `json:"candidate_subject_digest"`
	ReferenceManifestDigest        string            `json:"reference_manifest_digest"`
	LiveManifestDigest             string            `json:"live_manifest_digest"`
	LiveExecutionAttestationDigest string            `json:"live_execution_attestation_digest"`
	TrackID                        TrackID           `json:"track_id"`
	SuiteIDs                       []string          `json:"suite_ids"`
	WorkloadSnapshotDigest         string            `json:"workload_snapshot_digest"`
	BenchmarkRevisions             map[string]string `json:"benchmark_revisions"`
	MatchedCases                   int               `json:"matched_cases"`
	DecisionMismatches             int               `json:"decision_mismatches"`
	OutcomeMismatches              int               `json:"outcome_mismatches"`
	UnavailableCases               int               `json:"unavailable_cases"`
	SampleCount                    int               `json:"sample_count"`
	PointEstimate                  float64           `json:"point_estimate"`
	LowerBound                     float64           `json:"lower_bound"`
	ConfidenceLevel                float64           `json:"confidence_level"`
	Verdict                        GateVerdict       `json:"verdict"`
	Digest                         string            `json:"digest"`
}

type CampaignDecision struct {
	SchemaVersion       string                      `json:"schema_version"`
	ContractVersion     string                      `json:"contract_version"`
	AttestationRevision string                      `json:"attestation_revision"`
	CampaignID          string                      `json:"campaign_id"`
	CampaignDigest      string                      `json:"campaign_digest"`
	DecisionDigest      string                      `json:"decision_digest"`
	Verdict             DecisionVerdict             `json:"verdict"`
	Summary             string                      `json:"summary"`
	Gates               []CampaignGate              `json:"gates"`
	Evidence            []CampaignEvidenceAnchor    `json:"evidence"`
	PairedLiveEvidence  *CampaignPairedLiveEvidence `json:"paired_live_evidence,omitempty"`
	FidelityEvidence    *CampaignFidelityEvidence   `json:"fidelity_evidence,omitempty"`
	Recommendations     []string                    `json:"recommendations"`
	CreatedAt           time.Time                   `json:"created_at"`
}

type Campaign struct {
	SchemaVersion   string               `json:"schema_version"`
	ContractVersion string               `json:"contract_version"`
	ID              string               `json:"id"`
	Name            string               `json:"name"`
	Description     string               `json:"description"`
	ChangeProfile   ChangeProfile        `json:"change_profile"`
	Status          CampaignStatus       `json:"status"`
	GateBindings    CampaignGateBindings `json:"gate_bindings"`
	ManifestDigest  string               `json:"manifest_digest"`
	CreatedAt       time.Time            `json:"created_at"`
	Decision        CampaignDecision     `json:"decision"`
}
