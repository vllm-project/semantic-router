package config

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// Settings of the escalate executor.
const (
	// Order of a Chat escalation's candidates: the modelRefs order unless
	// one of these is set.
	EscalateOrderSize = "size"
	EscalateOrderCost = "cost"

	// on_error of an escalation: when no answer is accepted, serve the last
	// one scored, or fail the request.
	EscalateOnErrorLastAttempt = "last_attempt"
	EscalateOnErrorFail        = "fail"

	ConfidenceMethodAverageLogprob    = "avg_logprob"
	ConfidenceMethodMargin            = "margin"
	ConfidenceMethodHybrid            = "hybrid"
	ConfidenceMethodSelfVerify        = "self_verify"
	ConfidenceTokenFilterAll          = "all"
	ConfidenceTokenFilterToolCallArgs = "tool_call_args" //nolint:gosec // This names a token-filter mode, not a credential.

	defaultConfidenceHybridWeight = 0.5
)

// EscalateAlgorithmConfig calls a decision's candidates one at a time until
// an answer is accepted. A Chat decision calls its modelRefs, in Order, and
// accepts an answer by Accept. A System One decision runs its authored
// Stages, each accepting a typed answer by its own rules, under Quality.
type EscalateAlgorithmConfig struct {
	Order   string           `yaml:"order,omitempty" jsonschema:"enum=size,enum=cost"`
	Accept  *EscalateAccept  `yaml:"accept,omitempty"`
	Quality *EscalateQuality `yaml:"quality,omitempty"`
	Stages  []EscalateStage  `yaml:"stages,omitempty"`
}

// EscalateAccept says when an answer is accepted. A Chat escalation names one
// check: the answer's confidence, the decision model's answer_check of it, or
// (refusal: false) that the refusal preset does not flag it. A System One
// stage or quality policy lists native rules, which must all hold.
type EscalateAccept struct {
	Confidence *EscalateConfidence    `yaml:"confidence,omitempty"`
	SystemOne  *EscalateAnswerCheck   `yaml:"systemone,omitempty"`
	Refusal    *bool                  `yaml:"refusal,omitempty"`
	Native     []NativeAcceptanceRule `yaml:"native,omitempty"`
}

// EscalateConfidence accepts an answer whose confidence, normalized to
// [0, 1], is at least GTE: read from its token logprobs (avg_logprob, margin
// or their weighted hybrid) or from the model's rating of its own answer
// (self_verify).
type EscalateConfidence struct {
	Method        string               `yaml:"method,omitempty" jsonschema:"enum=avg_logprob,enum=margin,enum=hybrid,enum=self_verify"`
	GTE           *float64             `yaml:"gte" jsonschema:"required"`
	HybridWeights *HybridWeightsConfig `yaml:"hybrid_weights,omitempty"`
	TokenFilter   string               `yaml:"token_filter,omitempty" jsonschema:"enum=all,enum=tool_call_args"`
}

// EffectiveMethod is the confidence method: avg_logprob unless declared.
func (c *EscalateConfidence) EffectiveMethod() string {
	if c == nil || c.Method == "" {
		return ConfidenceMethodAverageLogprob
	}
	return c.Method
}

// EscalateAnswerCheck accepts an answer when a System One model, asked the
// answer_check preset about the request and the answer, judges it correct
// and complete with probability at least GTE. Without a deployment the
// decision model answers.
type EscalateAnswerCheck struct {
	GTE        *float64 `yaml:"gte" jsonschema:"required"`
	Deployment string   `yaml:"deployment,omitempty"`
	TimeoutMs  int      `yaml:"timeout_ms,omitempty"`
}

// EffectiveTimeout is the check's call deadline.
func (c EscalateAnswerCheck) EffectiveTimeout() time.Duration {
	return systemOneTimeout(c.TimeoutMs)
}

// EscalateOnError is the escalation's on_error: what it serves when no
// answer is accepted. A Chat escalation serves its last scored answer by
// default; a System One escalation always fails, as quality is its contract.
func (a *AlgorithmConfig) EscalateOnError() string {
	if a.OnError != "" {
		return a.OnError
	}
	if a.IsNative() {
		return EscalateOnErrorFail
	}
	return EscalateOnErrorLastAttempt
}

func validateEscalateAlgorithmConfig(decisionName string, refs []ModelRef, algorithm *AlgorithmConfig) error {
	escalate := algorithm.Escalate
	if escalate == nil {
		return fmt.Errorf("decision '%s': algorithm.type=escalate requires algorithm.escalate configuration", decisionName)
	}
	path := fmt.Sprintf("decision '%s', algorithm.escalate", decisionName)
	if len(escalate.Stages) > 0 {
		if escalate.Order != "" || escalate.Accept != nil {
			return fmt.Errorf("%s: order and accept are the Chat settings; a System One stage carries its own accept", path)
		}
		return validateNativeAlgorithmConfig(decisionName, refs, algorithm)
	}
	if escalate.Quality != nil {
		return fmt.Errorf("%s: quality applies to the System One stages; a Chat escalation accepts by accept", path)
	}
	switch escalate.Order {
	case "", EscalateOrderSize, EscalateOrderCost:
	default:
		return fmt.Errorf("%s.order must be %q or %q, or omitted for the modelRefs order, got %q", path, EscalateOrderSize, EscalateOrderCost, escalate.Order)
	}
	switch algorithm.OnError {
	case "", EscalateOnErrorLastAttempt, EscalateOnErrorFail:
	default:
		return fmt.Errorf("decision '%s': algorithm.on_error of escalate must be %q or %q, got %q", decisionName, EscalateOnErrorLastAttempt, EscalateOnErrorFail, algorithm.OnError)
	}
	if err := algorithm.Budget.Validate(); err != nil {
		return fmt.Errorf("decision '%s': %w", decisionName, err)
	}
	if err := validateChatEscalateAccept(escalate.Accept); err != nil {
		return fmt.Errorf("%s.accept: %w", path, err)
	}
	return nil
}

func validateChatEscalateAccept(accept *EscalateAccept) error {
	if accept == nil {
		return fmt.Errorf("one of confidence, systemone or refusal is required")
	}
	if len(accept.Native) > 0 {
		return fmt.Errorf("native rules accept System One answers; a Chat answer is accepted by confidence, systemone or refusal")
	}
	checks := 0
	for _, set := range []bool{accept.Confidence != nil, accept.SystemOne != nil, accept.Refusal != nil} {
		if set {
			checks++
		}
	}
	if checks != 1 {
		return fmt.Errorf("name exactly one of confidence, systemone or refusal")
	}
	switch {
	case accept.Confidence != nil:
		return validateEscalateConfidence(accept.Confidence)
	case accept.SystemOne != nil:
		return validateEscalateAnswerCheck(accept.SystemOne)
	default:
		if *accept.Refusal {
			return fmt.Errorf("refusal must be false: an answer is accepted when the refusal preset does not flag it")
		}
		return nil
	}
}

func validateEscalateConfidence(cfg *EscalateConfidence) error {
	method := cfg.EffectiveMethod()
	switch method {
	case ConfidenceMethodAverageLogprob, ConfidenceMethodMargin, ConfidenceMethodHybrid, ConfidenceMethodSelfVerify:
	default:
		return fmt.Errorf("confidence.method must be one of %q, %q, %q or %q, got %q",
			ConfidenceMethodAverageLogprob, ConfidenceMethodMargin, ConfidenceMethodHybrid, ConfidenceMethodSelfVerify, cfg.Method)
	}
	if err := validateUnitIntervalBound("confidence.gte", cfg.GTE); err != nil {
		return err
	}
	switch cfg.TokenFilter {
	case "", ConfidenceTokenFilterAll, ConfidenceTokenFilterToolCallArgs:
	default:
		return fmt.Errorf("confidence.token_filter must be %q or %q, got %q", ConfidenceTokenFilterAll, ConfidenceTokenFilterToolCallArgs, cfg.TokenFilter)
	}
	if cfg.TokenFilter != "" && method == ConfidenceMethodSelfVerify {
		return fmt.Errorf("confidence.token_filter applies to the logprob methods, not %q", ConfidenceMethodSelfVerify)
	}
	return validateConfidenceHybridWeights(method, cfg.HybridWeights)
}

func validateConfidenceHybridWeights(method string, weights *HybridWeightsConfig) error {
	if weights == nil {
		return nil
	}
	if method != ConfidenceMethodHybrid {
		return fmt.Errorf("confidence.hybrid_weights applies only to method %q", ConfidenceMethodHybrid)
	}
	for name, weight := range map[string]float64{"logprob_weight": weights.LogprobWeight, "margin_weight": weights.MarginWeight} {
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 || weight > 1 {
			return fmt.Errorf("confidence.hybrid_weights.%s must be within [0, 1], got %v", name, weight)
		}
	}
	logprobWeight := effectiveConfidenceHybridWeight(weights.LogprobWeight)
	marginWeight := effectiveConfidenceHybridWeight(weights.MarginWeight)
	if math.Abs(logprobWeight+marginWeight-1) > 1e-9 {
		return fmt.Errorf("confidence.hybrid_weights must sum to 1, got logprob_weight=%g and margin_weight=%g", logprobWeight, marginWeight)
	}
	return nil
}

func effectiveConfidenceHybridWeight(weight float64) float64 {
	if weight == 0 {
		return defaultConfidenceHybridWeight
	}
	return weight
}

func validateEscalateAnswerCheck(cfg *EscalateAnswerCheck) error {
	if err := validateUnitIntervalBound("systemone.gte", cfg.GTE); err != nil {
		return err
	}
	if strings.TrimSpace(cfg.Deployment) != cfg.Deployment {
		return fmt.Errorf("systemone.deployment must name a model_runtime or systemone deployment; omit it to ask the decision model")
	}
	if cfg.TimeoutMs < 0 || cfg.TimeoutMs > MaxSystemOneTimeoutMs {
		return fmt.Errorf("systemone.timeout_ms must be within [1, %d] when set", MaxSystemOneTimeoutMs)
	}
	return nil
}

func validateUnitIntervalBound(name string, bound *float64) error {
	if bound == nil {
		return fmt.Errorf("%s is required", name)
	}
	if math.IsNaN(*bound) || math.IsInf(*bound, 0) || *bound < 0 || *bound > 1 {
		return fmt.Errorf("%s must be within [0, 1], got %v", name, *bound)
	}
	return nil
}
