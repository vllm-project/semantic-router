package evaluationplane

import "time"

const (
	agentTaskLedgerContractVersion      = "evaluation-agent-task-ledger.v1"
	agentTaskAttemptContractVersion     = "evaluation-agent-task-attempt.v1"
	agentTaskMethodID                   = "live-agent-task.v1"
	agentTaskEvidenceKind               = "live-agent-task-ledger.v1"
	agentTaskExecutionSemantics         = "provider-observed-explicit-tool-policy"
	agentTaskBenchmarkParityClaim       = "none"
	minimumAgentTaskDistinctTaskCount   = 20
	minimumAgentTaskAttemptsPerTask     = 2
	maximumAgentTaskToolCallsPerAttempt = 128
)

// agentTaskToolPolicy is part of the immutable repeated-task contract. An
// empty expected_tools array is valid only for an explicitly pure-reasoning
// task; omission decodes to nil and is rejected.
type agentTaskToolPolicy struct {
	RequiresTools bool     `json:"requires_tools"`
	ExpectedTools []string `json:"expected_tools"`
}

// agentTaskToolCallEvidence describes what an external, provider-attested
// agent runtime observed. The evaluation worker never executes a tool. An
// invalid call is rejected before execution and therefore cannot carry a
// result or execution receipt.
type agentTaskToolCallEvidence struct {
	Sequence               int64      `json:"sequence"`
	ToolCallID             string     `json:"tool_call_id"`
	ToolName               string     `json:"tool_name"`
	ArgumentsDigest        string     `json:"arguments_digest"`
	Outcome                string     `json:"outcome"`
	ResultDigest           *string    `json:"result_digest,omitempty"`
	ExecutionReceiptDigest *string    `json:"execution_receipt_digest,omitempty"`
	CostUSD                float64    `json:"cost_usd"`
	StartedAt              *time.Time `json:"started_at,omitempty"`
	CompletedAt            *time.Time `json:"completed_at,omitempty"`
}

// agentTaskMethodEvidence is one repeated task attempt from a sealed provider
// ledger. It intentionally makes no claim that the provider's execution is
// equivalent to an upstream benchmark's native runner.
type agentTaskMethodEvidence struct {
	ContractVersion              string                      `json:"contract_version"`
	MethodID                     string                      `json:"method_id"`
	LedgerID                     string                      `json:"ledger_id"`
	SourceID                     string                      `json:"source_id"`
	SuiteID                      string                      `json:"suite_id"`
	SuiteRevision                string                      `json:"suite_revision"`
	TaskSetDigest                string                      `json:"task_set_digest"`
	BenchmarkParityClaim         string                      `json:"benchmark_parity_claim"`
	ExecutionSemantics           string                      `json:"execution_semantics"`
	PolicySnapshotDigest         string                      `json:"policy_snapshot_digest"`
	ConfigDigest                 string                      `json:"config_digest"`
	TargetID                     string                      `json:"target_id"`
	BackendTopologyDigest        string                      `json:"backend_topology_digest"`
	MixtureSnapshotDigest        string                      `json:"mixture_snapshot_digest"`
	LedgerTotalAttemptCount      int                         `json:"ledger_total_attempt_count"`
	LedgerTotalDistinctTaskCount int                         `json:"ledger_total_distinct_task_count"`
	MinimumDistinctTaskCount     int                         `json:"minimum_distinct_task_count"`
	MinimumAttemptsPerTask       int                         `json:"minimum_attempts_per_task"`
	TaskID                       string                      `json:"task_id"`
	TaskSpecDigest               string                      `json:"task_spec_digest"`
	ToolPolicy                   agentTaskToolPolicy         `json:"tool_policy"`
	AttemptID                    string                      `json:"attempt_id"`
	RepetitionID                 string                      `json:"repetition_id"`
	TrajectoryID                 string                      `json:"trajectory_id"`
	Seed                         int64                       `json:"seed"`
	SelectedArmID                string                      `json:"selected_arm_id"`
	TaskSuccess                  bool                        `json:"task_success"`
	TaskScore                    float64                     `json:"task_score"`
	SuccessThreshold             float64                     `json:"success_threshold"`
	GraderID                     string                      `json:"grader_id"`
	GraderRevisionDigest         string                      `json:"grader_revision_digest"`
	GradingReceiptDigest         string                      `json:"grading_receipt_digest"`
	PrivacyAuditReceiptDigest    string                      `json:"privacy_audit_receipt_digest"`
	ExecutionReceiptDigest       string                      `json:"execution_receipt_digest"`
	TrajectorySteps              int64                       `json:"trajectory_steps"`
	ToolCallCount                int64                       `json:"tool_call_count"`
	InvalidToolCallCount         int64                       `json:"invalid_tool_call_count"`
	PrivacyExposureCount         int64                       `json:"privacy_exposure_count"`
	InputTokens                  int64                       `json:"input_tokens"`
	OutputTokens                 int64                       `json:"output_tokens"`
	ModelCostUSD                 float64                     `json:"model_cost_usd"`
	ToolCostUSD                  float64                     `json:"tool_cost_usd"`
	EvaluationCostUSD            float64                     `json:"evaluation_cost_usd"`
	TotalCostUSD                 float64                     `json:"total_cost_usd"`
	StartedAt                    time.Time                   `json:"started_at"`
	CompletedAt                  time.Time                   `json:"completed_at"`
	GradedAt                     time.Time                   `json:"graded_at"`
	PrivacyAuditedAt             time.Time                   `json:"privacy_audited_at"`
	ToolCalls                    []agentTaskToolCallEvidence `json:"tool_calls"`
}

type agentTaskLedgerPayload struct {
	ContractVersion              string                    `json:"contract_version"`
	LedgerID                     string                    `json:"ledger_id"`
	SourceID                     string                    `json:"source_id"`
	Environment                  string                    `json:"environment"`
	SuiteID                      string                    `json:"suite_id"`
	SuiteRevision                string                    `json:"suite_revision"`
	TaskSetDigest                string                    `json:"task_set_digest"`
	BenchmarkParityClaim         string                    `json:"benchmark_parity_claim"`
	ExecutionSemantics           string                    `json:"execution_semantics"`
	ProviderAttestationDigest    string                    `json:"provider_attestation_digest"`
	PolicySnapshotDigest         string                    `json:"policy_snapshot_digest"`
	ConfigDigest                 string                    `json:"config_digest"`
	TargetID                     string                    `json:"target_id"`
	BackendTopologyDigest        string                    `json:"backend_topology_digest"`
	Mixture                      methodMixtureBinding      `json:"mixture"`
	LedgerTotalAttemptCount      int                       `json:"ledger_total_attempt_count"`
	LedgerTotalDistinctTaskCount int                       `json:"ledger_total_distinct_task_count"`
	MinimumDistinctTaskCount     int                       `json:"minimum_distinct_task_count"`
	MinimumAttemptsPerTask       int                       `json:"minimum_attempts_per_task"`
	WindowStartedAt              time.Time                 `json:"window_started_at"`
	WindowEndedAt                time.Time                 `json:"window_ended_at"`
	SealedAt                     time.Time                 `json:"sealed_at"`
	Attempts                     []agentTaskMethodEvidence `json:"attempts"`
}
