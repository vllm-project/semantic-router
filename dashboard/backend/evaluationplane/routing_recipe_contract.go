package evaluationplane

import "time"

const (
	RoutingRecipePlanContractVersion       = "routing-recipe-plan.v1"
	RoutingDecisionEvidenceContractVersion = "routing-decision-evidence.v1"
	RoutingRecipeEvaluationContractVersion = "routing-recipe-eval.v1"
	routingRecipeMaxItems                  = 128
	routingRecipeMaxArms                   = 64
	routingRecipeMaxCases                  = 100_000
	routingRecipeMaxOutcomes               = 1_000_000
)

// RoutingRecipePlan freezes every input that may be observed at decision time.
// It deliberately carries no grader or execution outcome.
type RoutingRecipePlan struct {
	ContractVersion      string                        `json:"contract_version"`
	PlanDigest           string                        `json:"plan_digest"`
	TargetSnapshotDigest string                        `json:"target_snapshot_digest"`
	ArmIDs               []string                      `json:"arm_ids"`
	FallbackArmID        string                        `json:"fallback_arm_id,omitempty"`
	Signals              []RoutingRecipeInputSpec      `json:"signals"`
	Projections          []RoutingRecipeProjectionSpec `json:"projections"`
	TopK                 []int                         `json:"top_k"`
}

type RoutingRecipeInputSpec struct {
	ID        string `json:"id"`
	ValueKind string `json:"value_kind"` // numeric or none
}

type RoutingRecipeProjectionSpec struct {
	ID             string `json:"id"`
	ValueKind      string `json:"value_kind"` // numeric or probability
	OutcomeBinding string `json:"outcome_binding"`
}

// RoutingRecipeDecisionSnapshot is normalized by a server-owned broker. It is
// intentionally separate from aggregate routing diagnostics and has no outcome
// fields; DecodeRoutingRecipeDecisionSnapshot rejects them as unknown fields.
type RoutingRecipeDecisionSnapshot struct {
	ContractVersion string                       `json:"contract_version"`
	DecisionID      string                       `json:"decision_id"`
	PlanDigest      string                       `json:"plan_digest"`
	CaseID          string                       `json:"case_id"`
	ObservedAt      time.Time                    `json:"observed_at"`
	Signals         []RoutingRecipeObservedInput `json:"signals"`
	Projections     []RoutingRecipeObservedInput `json:"projections"`
	Eligibility     []RoutingRecipeEligibility   `json:"eligibility"`
	RankedArmIDs    []string                     `json:"ranked_arm_ids"`
	SelectedArmID   string                       `json:"selected_arm_id,omitempty"`
	SelectionStatus string                       `json:"selection_status"`
}

type RoutingRecipeObservedInput struct {
	ID        string   `json:"id"`
	State     string   `json:"state"` // present, missing, error, timeout
	Value     *float64 `json:"value,omitempty"`
	LatencyMS *float64 `json:"latency_ms,omitempty"`
	ErrorCode string   `json:"error_code,omitempty"`
}

type RoutingRecipeEligibility struct {
	ArmID      string `json:"arm_id"`
	State      string `json:"state"` // eligible, ineligible, error, timeout, unavailable
	ReasonCode string `json:"reason_code"`
}

// RoutingRecipeOutcome is a server-observed, post-decision candidate outcome.
// It is deliberately a separate input so the reducer can prove ordering.
type RoutingRecipeOutcome struct {
	DecisionID string    `json:"decision_id"`
	CaseID     string    `json:"case_id"`
	ArmID      string    `json:"arm_id"`
	ObservedAt time.Time `json:"observed_at"`
	Quality    float64   `json:"quality"`
}

type RoutingRecipeReductionInput struct {
	Plan            RoutingRecipePlan
	ExpectedCaseIDs []string
	Decisions       []RoutingRecipeDecisionSnapshot
	Outcomes        []RoutingRecipeOutcome
}

type RoutingRecipeLatencyReport struct {
	Available   bool    `json:"available"`
	Reason      string  `json:"reason,omitempty"`
	SampleCount int     `json:"sample_count"`
	P50MS       float64 `json:"p50_ms,omitempty"`
	P95MS       float64 `json:"p95_ms,omitempty"`
}

type RoutingRecipeInputAvailabilityReport struct {
	ID       string                     `json:"id"`
	Expected int                        `json:"expected"`
	Present  int                        `json:"present"`
	Missing  int                        `json:"missing"`
	Error    int                        `json:"error"`
	Timeout  int                        `json:"timeout"`
	Latency  RoutingRecipeLatencyReport `json:"latency"`
}

type RoutingRecipeE1Report struct {
	ExpectedDecisions   int                                    `json:"expected_decisions"`
	ObservedDecisions   int                                    `json:"observed_decisions"`
	Signals             []RoutingRecipeInputAvailabilityReport `json:"signals"`
	Projections         []RoutingRecipeInputAvailabilityReport `json:"projections"`
	EligibilityComplete int                                    `json:"eligibility_complete"`
	SelectedFeasible    int                                    `json:"selected_feasible"`
}

type RoutingRecipeMetricAvailability struct {
	Available   bool    `json:"available"`
	Reason      string  `json:"reason,omitempty"`
	Value       float64 `json:"value,omitempty"`
	SampleCount int     `json:"sample_count"`
}

type RoutingRecipeReliabilityBin struct {
	Lower             float64 `json:"lower"`
	Upper             float64 `json:"upper"`
	Count             int     `json:"count"`
	MeanPrediction    float64 `json:"mean_prediction,omitempty"`
	ObservedFrequency float64 `json:"observed_frequency,omitempty"`
}

type RoutingRecipeProjectionOutcomeReport struct {
	ProjectionID string                          `json:"projection_id"`
	Spearman     RoutingRecipeMetricAvailability `json:"spearman"`
	Brier        RoutingRecipeMetricAvailability `json:"brier"`
	ECE10        RoutingRecipeMetricAvailability `json:"ece_10"`
	Reliability  []RoutingRecipeReliabilityBin   `json:"reliability_bins"`
}

type RoutingRecipeTopKReport struct {
	K      int                             `json:"k"`
	Recall RoutingRecipeMetricAvailability `json:"feasible_oracle_recall"`
}

type RoutingRecipeE2Report struct {
	ProjectionOutcomes []RoutingRecipeProjectionOutcomeReport `json:"projection_outcomes"`
	TopK               []RoutingRecipeTopKReport              `json:"top_k"`
	OracleRegret       RoutingRecipeMetricAvailability        `json:"oracle_regret"`
}

type RoutingRecipeEvaluationReport struct {
	ContractVersion string                `json:"contract_version"`
	PlanDigest      string                `json:"plan_digest"`
	E1              RoutingRecipeE1Report `json:"e1"`
	E2              RoutingRecipeE2Report `json:"e2"`
}
