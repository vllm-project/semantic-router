package extproc

// Meta-routing trigger names are runtime artifacts, not public config fields.
const (
	metaRoutingTriggerLowDecisionMargin        = "low_decision_margin"
	metaRoutingTriggerProjectionBoundary       = "projection_boundary_pressure"
	metaRoutingTriggerPartitionConflict        = "partition_conflict"
	metaRoutingTriggerRequiredFamilyMissing    = "required_family_missing"
	metaRoutingTriggerRequiredFamilyLowConf    = "required_family_low_confidence"
	metaRoutingTriggerSignalFamilyDisagreement = "signal_family_disagreement"
)

const (
	metaRoutingCauseMissingRequiredFamily = "missing_required_family"
	metaRoutingCauseLowConfidenceFamily   = "low_confidence_family"
	metaRoutingCauseProjectionBoundary    = "projection_boundary_pressure"
	metaRoutingCauseDecisionOverlap       = "decision_overlap"
	metaRoutingCausePartitionConflict     = "partition_conflict"
	metaRoutingCauseFamilyDisagreement    = "family_disagreement"
	metaRoutingCauseCompressionLossRisk   = "compression_loss_risk"
)

const (
	metaRoutingPassKindBase       = "base"
	metaRoutingPassKindRefinement = "refinement"
)

// TraceQuality captures the pass-level quality features used for later
// calibration and learned-policy work.
type TraceQuality struct {
	SignalDominance               float64  `json:"signal_dominance,omitempty"`
	AvgSignalConfidence           float64  `json:"avg_signal_confidence,omitempty"`
	DecisionMargin                float64  `json:"decision_margin,omitempty"`
	ProjectionBoundaryMinDistance *float64 `json:"projection_boundary_min_distance,omitempty"`
	Fragile                       bool     `json:"fragile,omitempty"`
}

// RoutingTrace summarizes the request-level meta-routing state across passes.
type RoutingTrace struct {
	Mode                    string                       `json:"mode,omitempty"`
	MaxPasses               int                          `json:"max_passes,omitempty"`
	PolicyProvider          *MetaRoutingPolicyDescriptor `json:"policy_provider,omitempty"`
	PassCount               int                          `json:"pass_count,omitempty"`
	TriggerNames            []string                     `json:"trigger_names,omitempty"`
	RefinedSignalFamilies   []string                     `json:"refined_signal_families,omitempty"`
	OverturnedDecision      bool                         `json:"overturned_decision,omitempty"`
	LatencyDeltaMs          int64                        `json:"latency_delta_ms,omitempty"`
	DecisionMarginDelta     float64                      `json:"decision_margin_delta,omitempty"`
	ProjectionBoundaryDelta *float64                     `json:"projection_boundary_delta,omitempty"`
	FinalDecisionName       string                       `json:"final_decision_name,omitempty"`
	FinalDecisionConfidence float64                      `json:"final_decision_confidence,omitempty"`
	FinalModel              string                       `json:"final_model,omitempty"`
	FinalAssessment         *MetaAssessment              `json:"final_assessment,omitempty"`
	FinalPlan               *RefinementPlan              `json:"final_plan,omitempty"`
	Passes                  []PassTrace                  `json:"passes,omitempty"`
}

// PassTrace is one pass through the request-phase routing pipeline.
type PassTrace struct {
	Index                  int             `json:"index"`
	Kind                   string          `json:"kind,omitempty"`
	LatencyMs              int64           `json:"latency_ms,omitempty"`
	InputCompressed        bool            `json:"input_compressed,omitempty"`
	DecisionName           string          `json:"decision_name,omitempty"`
	DecisionConfidence     float64         `json:"decision_confidence,omitempty"`
	DecisionMargin         float64         `json:"decision_margin,omitempty"`
	DecisionCandidateCount int             `json:"decision_candidate_count,omitempty"`
	DecisionWinnerBasis    string          `json:"decision_winner_basis,omitempty"`
	RunnerUpDecisionName   string          `json:"runner_up_decision_name,omitempty"`
	RunnerUpConfidence     float64         `json:"runner_up_confidence,omitempty"`
	CategoryName           string          `json:"category_name,omitempty"`
	SelectedModel          string          `json:"selected_model,omitempty"`
	SelectionMethod        string          `json:"selection_method,omitempty"`
	MatchedSignalCounts    map[string]int  `json:"matched_signal_counts,omitempty"`
	PartitionConflicts     []string        `json:"partition_conflicts,omitempty"`
	TraceQuality           TraceQuality    `json:"trace_quality"`
	Assessment             *MetaAssessment `json:"assessment,omitempty"`
}

// MetaAssessment is the deterministic output of pass-level assessment.
type MetaAssessment struct {
	NeedsRefine  bool         `json:"needs_refine,omitempty"`
	Triggers     []string     `json:"triggers,omitempty"`
	RootCauses   []string     `json:"root_causes,omitempty"`
	TraceQuality TraceQuality `json:"trace_quality"`
}

// RefinementPlan is a bounded plan produced from assessment plus configured
// allowed actions. Execution stays in later workstream increments.
type RefinementPlan struct {
	MaxPasses    int                    `json:"max_passes,omitempty"`
	TriggerNames []string               `json:"trigger_names,omitempty"`
	RootCauses   []string               `json:"root_causes,omitempty"`
	Actions      []RefinementActionPlan `json:"actions,omitempty"`
}

// RefinementActionPlan is one planned refinement action after assessment.
type RefinementActionPlan struct {
	Type           string   `json:"type"`
	SignalFamilies []string `json:"signal_families,omitempty"`
}

// FeedbackRecord is the durable observation/action/outcome schema reserved for
// later persistence and offline calibration.
type FeedbackRecord struct {
	Mode        string              `json:"mode,omitempty"`
	Observation FeedbackObservation `json:"observation"`
	Action      FeedbackAction      `json:"action"`
	Outcome     FeedbackOutcome     `json:"outcome"`
}

// FeedbackObservation captures route-relevant request state and pass traces.
type FeedbackObservation struct {
	RequestID      string                       `json:"request_id,omitempty"`
	RequestModel   string                       `json:"request_model,omitempty"`
	RequestQuery   string                       `json:"request_query,omitempty"`
	PolicyProvider *MetaRoutingPolicyDescriptor `json:"policy_provider,omitempty"`
	Trace          *RoutingTrace                `json:"trace,omitempty"`
}

// FeedbackAction captures planned or executed refinement behavior.
type FeedbackAction struct {
	Planned                bool            `json:"planned,omitempty"`
	Executed               bool            `json:"executed,omitempty"`
	ExecutedPassCount      int             `json:"executed_pass_count,omitempty"`
	ExecutedActionTypes    []string        `json:"executed_action_types,omitempty"`
	ExecutedSignalFamilies []string        `json:"executed_signal_families,omitempty"`
	Plan                   *RefinementPlan `json:"plan,omitempty"`
}

// FeedbackOutcome captures final route selection and downstream weak labels.
type FeedbackOutcome struct {
	FinalDecisionName         string   `json:"final_decision_name,omitempty"`
	FinalDecisionConfidence   float64  `json:"final_decision_confidence,omitempty"`
	FinalModel                string   `json:"final_model,omitempty"`
	ResponseStatus            int      `json:"response_status,omitempty"`
	Streaming                 bool     `json:"streaming,omitempty"`
	CacheHit                  bool     `json:"cache_hit,omitempty"`
	PIIBlocked                bool     `json:"pii_blocked,omitempty"`
	HallucinationDetected     bool     `json:"hallucination_detected,omitempty"`
	UnverifiedFactualResponse bool     `json:"unverified_factual_response,omitempty"`
	ResponseJailbreakDetected bool     `json:"response_jailbreak_detected,omitempty"`
	RAGBackend                string   `json:"rag_backend,omitempty"`
	RouterReplayID            string   `json:"router_replay_id,omitempty"`
	UserFeedbackSignals       []string `json:"user_feedback_signals,omitempty"`
}
