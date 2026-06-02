package services

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
)

// IntentRequest represents a request for intent classification.
type IntentRequest struct {
	Text     string          `json:"text"`
	Messages []IntentMessage `json:"messages,omitempty"`
	Options  *IntentOptions  `json:"options,omitempty"`
}

// IntentOptions contains options for intent classification.
type IntentOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
	EvaluateAllSignals  bool    `json:"evaluate_all_signals,omitempty"` // Force evaluate all configured signals (for eval scenarios)
	Trace               bool    `json:"trace,omitempty"`                // Return per-decision evaluation trace trees
}

// MatchedSignals represents all matched signals from signal evaluation.
type MatchedSignals struct {
	Keywords     []string `json:"keywords,omitempty"`
	Embeddings   []string `json:"embeddings,omitempty"`
	Domains      []string `json:"domains,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Reask        []string `json:"reask,omitempty"`
	Preferences  []string `json:"preferences,omitempty"`
	Language     []string `json:"language,omitempty"`
	Context      []string `json:"context,omitempty"`
	Structure    []string `json:"structure,omitempty"`
	Complexity   []string `json:"complexity,omitempty"`
	Modality     []string `json:"modality,omitempty"`
	Authz        []string `json:"authz,omitempty"`
	Jailbreak    []string `json:"jailbreak,omitempty"`
	PII          []string `json:"pii,omitempty"`
	KB           []string `json:"kb,omitempty"`
	Conversation []string `json:"conversation,omitempty"`
	Event        []string `json:"event,omitempty"`
	Projection   []string `json:"projection,omitempty"`
}

// DecisionResult represents the result of decision evaluation.
type DecisionResult struct {
	DecisionName string   `json:"decision_name"`
	Confidence   float64  `json:"confidence"`
	MatchedRules []string `json:"matched_rules"`
}

// EvalDecisionResult represents the decision result for eval scenarios (without confidence).
type EvalDecisionResult struct {
	DecisionName     string          `json:"decision_name"`
	UsedSignals      *MatchedSignals `json:"used_signals"`      // Signals used by this decision (from decision rules)
	MatchedSignals   *MatchedSignals `json:"matched_signals"`   // Signals that matched
	UnmatchedSignals *MatchedSignals `json:"unmatched_signals"` // Signals that didn't match
}

// EvalResponse represents the eval classification response with comprehensive signal information.
type EvalResponse struct {
	OriginalText      string                                  `json:"original_text"` // The evaluated user turn or fallback query text
	DecisionResult    *EvalDecisionResult                     `json:"decision_result,omitempty"`
	EvalTrace         []decision.DecisionTrace                `json:"eval_trace,omitempty"`         // Per-decision evaluation trace (when ?trace=true)
	RecommendedModels []string                                `json:"recommended_models,omitempty"` // All models from matched decision's modelRefs
	RoutingDecision   string                                  `json:"routing_decision,omitempty"`
	Metrics           *classification.SignalMetricsCollection `json:"metrics"`                      // Performance and confidence for each signal
	SignalConfidences map[string]float64                      `json:"signal_confidences,omitempty"` // Real ML confidence scores per signal, e.g. "domain:economics" -> 0.81
	SignalValues      map[string]float64                      `json:"signal_values,omitempty"`      // Raw signal values per signal when exposed, e.g. "structure:many_questions" -> 4
}

// IntentResponse represents the response from intent classification.
type IntentResponse struct {
	Classification   Classification     `json:"classification"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	RecommendedModel string             `json:"recommended_model,omitempty"`
	RoutingDecision  string             `json:"routing_decision,omitempty"`

	// Signal-driven fields
	MatchedSignals *MatchedSignals `json:"matched_signals,omitempty"`
	DecisionResult *DecisionResult `json:"decision_result,omitempty"`
}

// Classification represents basic classification result.
type Classification struct {
	Category         string  `json:"category"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}
