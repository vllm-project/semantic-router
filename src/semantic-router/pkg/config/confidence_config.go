package config

import (
	"fmt"
	"math"
	"net/url"
	"strings"
)

const (
	ConfidenceMethodAverageLogprob      = "avg_logprob"
	ConfidenceMethodMargin              = "margin"
	ConfidenceMethodHybrid              = "hybrid"
	ConfidenceMethodSelfVerify          = "self_verify"
	ConfidenceMethodAutoMixEntailment   = "automix_entailment"
	ConfidenceOnErrorSkip               = "skip"
	ConfidenceOnErrorFail               = "fail"
	ConfidenceEscalationOrderSize       = "size"
	ConfidenceEscalationOrderSmallFirst = "small_to_large"
	ConfidenceEscalationOrderDeclared   = "declared"
	ConfidenceEscalationOrderCost       = "cost"
	ConfidenceEscalationOrderAutoMix    = "automix"
	ConfidenceTokenFilterAll            = "all"
	ConfidenceTokenFilterToolCallArgs   = "tool_call_args" //nolint:gosec // This names a token-filter mode, not a credential.

	defaultConfidenceHybridWeight = 0.5
)

// ValidateConfidenceAlgorithmConfig validates the request-time confidence
// cascade contract. Numeric scalar fields in ConfidenceAlgorithmConfig are
// intentionally not pointers, so zero is the canonical "unset" sentinel and
// cannot express a distinct explicit-zero value. Validation mirrors the
// runtime defaults instead of pretending YAML omission can be distinguished
// from an explicit zero.
func ValidateConfidenceAlgorithmConfig(cfg *ConfidenceAlgorithmConfig) error {
	if cfg == nil {
		return nil
	}

	method, err := validateConfidenceMethod(cfg.ConfidenceMethod)
	if err != nil {
		return err
	}
	if err := validateConfidenceUnitInterval("threshold", cfg.Threshold); err != nil {
		return err
	}
	if err := validateConfidenceOnError(cfg.OnError); err != nil {
		return err
	}
	if err := validateConfidenceEscalationOrder(cfg.EscalationOrder); err != nil {
		return err
	}
	if err := validateConfidenceUnitInterval("cost_quality_tradeoff", cfg.CostQualityTradeoff); err != nil {
		return err
	}
	if err := validateConfidenceTokenFilter(cfg.TokenFilter); err != nil {
		return err
	}
	if err := validateConfidenceHybridWeights(method, cfg.HybridWeights); err != nil {
		return err
	}
	return validateConfidenceVerifier(method, cfg.VerifierServerURL, cfg.VerifierTimeoutSeconds, cfg.MaxResponseBytes)
}

func validateConfidenceMethod(method string) (string, error) {
	if method == "" {
		return ConfidenceMethodAverageLogprob, nil
	}
	switch method {
	case ConfidenceMethodAverageLogprob,
		ConfidenceMethodMargin,
		ConfidenceMethodHybrid,
		ConfidenceMethodSelfVerify,
		ConfidenceMethodAutoMixEntailment:
		return method, nil
	default:
		return "", fmt.Errorf(
			"confidence_method must be one of %q, %q, %q, %q, or %q, got %q",
			ConfidenceMethodAverageLogprob,
			ConfidenceMethodMargin,
			ConfidenceMethodHybrid,
			ConfidenceMethodSelfVerify,
			ConfidenceMethodAutoMixEntailment,
			method,
		)
	}
}

func validateConfidenceOnError(onError string) error {
	switch onError {
	case "", ConfidenceOnErrorSkip, ConfidenceOnErrorFail:
		return nil
	default:
		return fmt.Errorf(
			"on_error must be one of %q or %q, got %q",
			ConfidenceOnErrorSkip,
			ConfidenceOnErrorFail,
			onError,
		)
	}
}

func validateConfidenceEscalationOrder(order string) error {
	switch order {
	case "",
		ConfidenceEscalationOrderSize,
		ConfidenceEscalationOrderSmallFirst,
		ConfidenceEscalationOrderDeclared,
		ConfidenceEscalationOrderCost,
		ConfidenceEscalationOrderAutoMix:
		return nil
	default:
		return fmt.Errorf(
			"escalation_order must be one of %q, %q, %q, %q, or %q, got %q",
			ConfidenceEscalationOrderSize,
			ConfidenceEscalationOrderSmallFirst,
			ConfidenceEscalationOrderDeclared,
			ConfidenceEscalationOrderCost,
			ConfidenceEscalationOrderAutoMix,
			order,
		)
	}
}

func validateConfidenceTokenFilter(filter string) error {
	switch filter {
	case "", ConfidenceTokenFilterAll, ConfidenceTokenFilterToolCallArgs:
		return nil
	default:
		return fmt.Errorf(
			"token_filter must be one of %q or %q, got %q",
			ConfidenceTokenFilterAll,
			ConfidenceTokenFilterToolCallArgs,
			filter,
		)
	}
}

func validateConfidenceHybridWeights(method string, weights *HybridWeightsConfig) error {
	if weights == nil {
		return nil
	}
	if method != ConfidenceMethodHybrid {
		return fmt.Errorf("hybrid_weights is only supported when confidence_method=%q", ConfidenceMethodHybrid)
	}
	if err := validateConfidenceUnitInterval("hybrid_weights.logprob_weight", weights.LogprobWeight); err != nil {
		return err
	}
	if err := validateConfidenceUnitInterval("hybrid_weights.margin_weight", weights.MarginWeight); err != nil {
		return err
	}

	logprobWeight := effectiveConfidenceHybridWeight(weights.LogprobWeight)
	marginWeight := effectiveConfidenceHybridWeight(weights.MarginWeight)
	if math.Abs(logprobWeight+marginWeight-1) > 1e-9 {
		return fmt.Errorf(
			"hybrid_weights effective values must sum to 1, got logprob_weight=%g and margin_weight=%g",
			logprobWeight,
			marginWeight,
		)
	}
	return nil
}

func effectiveConfidenceHybridWeight(weight float64) float64 {
	if weight == 0 {
		return defaultConfidenceHybridWeight
	}
	return weight
}

func validateConfidenceUnitInterval(name string, value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return fmt.Errorf("%s must be finite, got %v", name, value)
	}
	if value < 0 || value > 1 {
		return fmt.Errorf("%s must be between 0 and 1, got %g", name, value)
	}
	return nil
}

func validateConfidenceVerifier(method string, rawURL string, timeoutSeconds int, maxResponseBytes int64) error {
	if timeoutSeconds < 0 {
		return fmt.Errorf("verifier_timeout_seconds must be >= 1 when set")
	}
	if maxResponseBytes < 0 {
		return fmt.Errorf("max_response_bytes must be positive when set")
	}
	if method != ConfidenceMethodAutoMixEntailment {
		if rawURL != "" || timeoutSeconds != 0 || maxResponseBytes != 0 {
			return fmt.Errorf(
				"verifier settings are only supported when confidence_method=%q",
				ConfidenceMethodAutoMixEntailment,
			)
		}
		return nil
	}
	if rawURL == "" {
		return fmt.Errorf("verifier_server_url is required when confidence_method=%q", ConfidenceMethodAutoMixEntailment)
	}
	return validateConfidenceVerifierURL(rawURL)
}

func validateConfidenceVerifierURL(rawURL string) error {
	if strings.TrimSpace(rawURL) != rawURL {
		return fmt.Errorf("verifier_server_url must not contain leading or trailing whitespace")
	}
	parsed, err := url.ParseRequestURI(rawURL)
	if err != nil {
		return fmt.Errorf("verifier_server_url must be a valid absolute HTTP(S) URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("verifier_server_url scheme must be http or https, got %q", parsed.Scheme)
	}
	if parsed.Host == "" {
		return fmt.Errorf("verifier_server_url must include a host")
	}
	if parsed.User != nil {
		return fmt.Errorf("verifier_server_url must not include credentials")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("verifier_server_url must not include a query or fragment")
	}
	return nil
}
