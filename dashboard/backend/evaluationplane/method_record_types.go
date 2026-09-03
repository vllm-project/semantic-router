package evaluationplane

import "time"

const (
	minimumRecoveryPairCount              = 20
	minimumRecoveryClusterCount           = 20
	minimumRecoveryDistinctSeedCount      = 5
	minimumRecoveryPassRateLowerBound     = 0.8
	minimumProductionAssignmentCount      = 20
	minimumProductionEffectiveSampleSize  = 10.0
	minimumProductionEffectiveSampleRatio = 0.5
	minimumProductionSegmentSampleSize    = 5
	minimumProductionRewardLift           = 0.0
	maximumProductionRiskBudgetRate       = 0.2
)

type robustnessMethodEvidence struct {
	MethodID                   string   `json:"method_id"`
	SuiteID                    *string  `json:"suite_id,omitempty"`
	SuiteRevision              *string  `json:"suite_revision,omitempty"`
	QualificationReceiptDigest *string  `json:"qualification_receipt_digest,omitempty"`
	PerturbationArtifactDigest *string  `json:"perturbation_artifact_digest,omitempty"`
	PairID                     string   `json:"pair_id"`
	SourceCaseID               string   `json:"source_case_id"`
	TargetCaseID               string   `json:"target_case_id"`
	ShiftType                  string   `json:"shift_type"`
	Relation                   string   `json:"relation"`
	SourceActionID             string   `json:"source_action_id"`
	ExpectedActionID           *string  `json:"expected_action_id,omitempty"`
	SliceIDs                   []string `json:"slice_ids"`
	NativePairCount            int      `json:"native_pair_count"`
	SourceRecordDigest         string   `json:"source_record_digest"`
}

type recoveryMethodEvidence struct {
	MethodID                    string    `json:"method_id"`
	LedgerID                    string    `json:"ledger_id"`
	SourceID                    string    `json:"source_id"`
	PolicySnapshotDigest        string    `json:"policy_snapshot_digest"`
	ConfigDigest                string    `json:"config_digest"`
	TargetID                    string    `json:"target_id"`
	BackendTopologyDigest       string    `json:"backend_topology_digest"`
	MixtureSnapshotDigest       string    `json:"mixture_snapshot_digest"`
	LedgerTotalPairCount        int       `json:"ledger_total_pair_count"`
	MinimumPairCount            int       `json:"minimum_pair_count"`
	MinimumClusterCount         int       `json:"minimum_cluster_count"`
	MinimumDistinctSeedCount    int       `json:"minimum_distinct_seed_count"`
	FaultID                     string    `json:"fault_id"`
	CohortPairID                string    `json:"cohort_pair_id"`
	RepetitionID                string    `json:"repetition_id"`
	ConversationID              string    `json:"conversation_id"`
	ClusterID                   string    `json:"cluster_id"`
	Seed                        int64     `json:"seed"`
	Concurrency                 int64     `json:"concurrency"`
	TreatmentSystem             string    `json:"treatment_system"`
	FaultKind                   string    `json:"fault_kind"`
	FaultSequence               int64     `json:"fault_sequence"`
	FailureTurn                 int64     `json:"failure_turn"`
	FaultPlanDigest             string    `json:"fault_plan_digest"`
	FaultInjectionReceiptDigest string    `json:"fault_injection_receipt_digest"`
	BaselineRecordDigest        string    `json:"baseline_record_digest"`
	TreatmentRecordDigest       string    `json:"treatment_record_digest"`
	InjectionObserved           bool      `json:"injection_observed"`
	Recovered                   bool      `json:"recovered"`
	StatePreserved              bool      `json:"state_preserved"`
	BaselineTerminalSuccess     bool      `json:"baseline_terminal_success"`
	TreatmentTerminalSuccess    bool      `json:"treatment_terminal_success"`
	BaselineRecoveryLatencyMS   float64   `json:"baseline_recovery_latency_ms"`
	TreatmentRecoveryLatencyMS  float64   `json:"treatment_recovery_latency_ms"`
	BaselineRetryCount          int64     `json:"baseline_retry_count"`
	TreatmentRetryCount         int64     `json:"treatment_retry_count"`
	MaximumRecoveryLatencyMS    float64   `json:"maximum_recovery_latency_ms"`
	MaximumRetryAmplification   float64   `json:"maximum_retry_amplification"`
	SideEffectScope             string    `json:"side_effect_scope"`
	SideEffectCount             int64     `json:"side_effect_count"`
	DuplicateSideEffectCount    int64     `json:"duplicate_side_effect_count"`
	ObservedAt                  time.Time `json:"observed_at"`
}

type experimentPolicyArmEvidence struct {
	ID                         string  `json:"id"`
	ConfigDigest               string  `json:"config_digest"`
	AssignmentProbability      float64 `json:"assignment_probability"`
	TargetPolicyProbability    float64 `json:"target_policy_probability"`
	ReferencePolicyProbability float64 `json:"reference_policy_probability"`
}

type productionExperimentMethodEvidence struct {
	ContractVersion             string                        `json:"contract_version"`
	ExperimentID                string                        `json:"experiment_id"`
	LedgerID                    string                        `json:"ledger_id"`
	LedgerTotalAssignmentCount  int                           `json:"ledger_total_assignment_count"`
	LedgerTotalOutcomeCount     int                           `json:"ledger_total_outcome_count"`
	SourceID                    string                        `json:"source_id"`
	PolicySnapshotDigest        string                        `json:"policy_snapshot_digest"`
	ConfigDigest                string                        `json:"config_digest"`
	TargetID                    string                        `json:"target_id"`
	BackendTopologyDigest       string                        `json:"backend_topology_digest"`
	MixtureSnapshotDigest       string                        `json:"mixture_snapshot_digest"`
	Environment                 string                        `json:"environment"`
	AssignmentScheme            string                        `json:"assignment_scheme"`
	AssignmentID                string                        `json:"assignment_id"`
	ExposureID                  string                        `json:"exposure_id"`
	ParticipantDigest           string                        `json:"participant_digest"`
	SegmentID                   string                        `json:"segment_id"`
	PolicyArms                  []experimentPolicyArmEvidence `json:"policy_arms"`
	AssignedPolicyArmID         string                        `json:"assigned_policy_arm_id"`
	SelectedModelID             *string                       `json:"selected_model_id,omitempty"`
	AssignmentProbability       float64                       `json:"assignment_probability"`
	ExposureProbability         float64                       `json:"exposure_probability"`
	BehaviorPropensity          float64                       `json:"behavior_propensity"`
	TargetPolicyProbability     float64                       `json:"target_policy_probability"`
	MinimumEffectiveSampleSize  float64                       `json:"minimum_effective_sample_size"`
	MinimumEffectiveSampleRatio float64                       `json:"minimum_effective_sample_ratio"`
	MinimumSegmentSampleSize    int                           `json:"minimum_segment_sample_size"`
	MinimumAssignmentCount      int                           `json:"minimum_assignment_count"`
	MinimumRewardLift           float64                       `json:"minimum_reward_lift"`
	ConfidenceLevel             float64                       `json:"confidence_level"`
	RiskEvent                   bool                          `json:"risk_event"`
	RiskBudgetMaxRate           float64                       `json:"risk_budget_max_rate"`
	StopRuleID                  string                        `json:"stop_rule_id"`
	StopRuleEvaluatedAt         time.Time                     `json:"stop_rule_evaluated_at"`
	StopTriggered               bool                          `json:"stop_triggered"`
	RollbackReceiptID           string                        `json:"rollback_receipt_id"`
	RollbackValidatedAt         time.Time                     `json:"rollback_validated_at"`
	RollbackReady               bool                          `json:"rollback_ready"`
	RollbackExecutedAt          *time.Time                    `json:"rollback_executed_at,omitempty"`
	RollbackSucceeded           *bool                         `json:"rollback_succeeded,omitempty"`
	AssignedAt                  time.Time                     `json:"assigned_at"`
	ExposedAt                   time.Time                     `json:"exposed_at"`
	LedgerSealedAt              time.Time                     `json:"ledger_sealed_at"`
}

type onlinePreferenceOutcomeEvidence struct {
	ContractVersion   string    `json:"contract_version"`
	OutcomeID         string    `json:"outcome_id"`
	AssignmentID      string    `json:"assignment_id"`
	ExposureID        string    `json:"exposure_id"`
	ParticipantDigest string    `json:"participant_digest"`
	SegmentID         string    `json:"segment_id"`
	Reward            float64   `json:"reward"`
	ObservedAt        time.Time `json:"observed_at"`
}

type onlinePreferenceMethodEvidence struct {
	ContractVersion string                             `json:"contract_version"`
	Experiment      productionExperimentMethodEvidence `json:"experiment"`
	Outcome         onlinePreferenceOutcomeEvidence    `json:"outcome"`
}

type hardPolicyEnforcementBindingEvidence struct {
	RuleID           string `json:"rule_id"`
	EnforcementPoint string `json:"enforcement_point"`
}

type hardPolicyStaticProofEvidence struct {
	ContractVersion             string                                 `json:"contract_version"`
	ProofID                     string                                 `json:"proof_id"`
	SourceID                    string                                 `json:"source_id"`
	PolicySnapshotDigest        string                                 `json:"policy_snapshot_digest"`
	ConfigDigest                string                                 `json:"config_digest"`
	TargetID                    string                                 `json:"target_id"`
	BackendTopologyDigest       string                                 `json:"backend_topology_digest"`
	MixtureSnapshotDigest       string                                 `json:"mixture_snapshot_digest"`
	RuntimeInstanceDigest       string                                 `json:"runtime_instance_digest"`
	LedgerTotalObservationCount int                                    `json:"ledger_total_observation_count"`
	RequiredBindings            []hardPolicyEnforcementBindingEvidence `json:"required_bindings"`
	VerifiedAt                  time.Time                              `json:"verified_at"`
}

type hardPolicyMethodEvidence struct {
	ContractVersion   string                        `json:"contract_version"`
	Proof             hardPolicyStaticProofEvidence `json:"proof"`
	ObservationID     string                        `json:"observation_id"`
	AttackID          string                        `json:"attack_id"`
	RuleID            string                        `json:"rule_id"`
	EnforcementPoint  string                        `json:"enforcement_point"`
	DecisionReceiptID string                        `json:"decision_receipt_id"`
	ShouldBlock       bool                          `json:"should_block"`
	Blocked           bool                          `json:"blocked"`
	Violations        int64                         `json:"violations"`
	ObservedAt        time.Time                     `json:"observed_at"`
}
