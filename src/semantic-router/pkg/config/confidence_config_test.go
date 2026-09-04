package config

import (
	"math"
	"strings"
	"testing"
)

func TestValidateConfidenceAlgorithmConfigAcceptsCanonicalValues(t *testing.T) {
	tests := []struct {
		name string
		cfg  *ConfidenceAlgorithmConfig
	}{
		{name: "nil config", cfg: nil},
		{name: "all defaults", cfg: &ConfidenceAlgorithmConfig{}},
		{
			name: "AMD declared avg logprob cascade",
			cfg: &ConfidenceAlgorithmConfig{
				ConfidenceMethod: ConfidenceMethodAverageLogprob,
				Threshold:        0.72,
				EscalationOrder:  ConfidenceEscalationOrderDeclared,
				OnError:          ConfidenceOnErrorSkip,
			},
		},
		{
			name: "hybrid",
			cfg: &ConfidenceAlgorithmConfig{
				ConfidenceMethod: ConfidenceMethodHybrid,
				Threshold:        1,
				HybridWeights: &HybridWeightsConfig{
					LogprobWeight: 0.65,
					MarginWeight:  0.35,
				},
				EscalationOrder:     ConfidenceEscalationOrderAutoMix,
				CostQualityTradeoff: 1,
				TokenFilter:         ConfidenceTokenFilterToolCallArgs,
				OnError:             ConfidenceOnErrorFail,
			},
		},
		{
			name: "automix entailment",
			cfg: &ConfidenceAlgorithmConfig{
				ConfidenceMethod:       ConfidenceMethodAutoMixEntailment,
				Threshold:              0.7,
				VerifierServerURL:      "https://verifier.example.com/api",
				VerifierTimeoutSeconds: 30,
				MaxResponseBytes:       1234,
			},
		},
		{
			name: "zero sentinels use runtime defaults",
			cfg: &ConfidenceAlgorithmConfig{
				ConfidenceMethod:       ConfidenceMethodHybrid,
				Threshold:              0,
				CostQualityTradeoff:    0,
				VerifierTimeoutSeconds: 0,
				HybridWeights:          &HybridWeightsConfig{},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateConfidenceAlgorithmConfig(tc.cfg); err != nil {
				t.Fatalf("ValidateConfidenceAlgorithmConfig() error = %v", err)
			}
		})
	}
}

func TestValidateConfidenceAlgorithmConfigAcceptsAllEnums(t *testing.T) {
	for _, method := range []string{
		"",
		ConfidenceMethodAverageLogprob,
		ConfidenceMethodMargin,
		ConfidenceMethodHybrid,
		ConfidenceMethodSelfVerify,
		ConfidenceMethodAutoMixEntailment,
	} {
		t.Run("method_"+method, func(t *testing.T) {
			cfg := &ConfidenceAlgorithmConfig{ConfidenceMethod: method}
			if method == ConfidenceMethodHybrid {
				cfg.HybridWeights = &HybridWeightsConfig{}
			}
			if method == ConfidenceMethodAutoMixEntailment {
				cfg.VerifierServerURL = "http://127.0.0.1:8080"
			}
			if err := ValidateConfidenceAlgorithmConfig(cfg); err != nil {
				t.Fatalf("method %q: %v", method, err)
			}
		})
	}

	for _, order := range []string{
		"",
		ConfidenceEscalationOrderSize,
		ConfidenceEscalationOrderSmallFirst,
		ConfidenceEscalationOrderDeclared,
		ConfidenceEscalationOrderCost,
		ConfidenceEscalationOrderAutoMix,
	} {
		t.Run("order_"+order, func(t *testing.T) {
			if err := ValidateConfidenceAlgorithmConfig(&ConfidenceAlgorithmConfig{EscalationOrder: order}); err != nil {
				t.Fatalf("order %q: %v", order, err)
			}
		})
	}

	for _, filter := range []string{"", ConfidenceTokenFilterAll, ConfidenceTokenFilterToolCallArgs} {
		t.Run("filter_"+filter, func(t *testing.T) {
			if err := ValidateConfidenceAlgorithmConfig(&ConfidenceAlgorithmConfig{TokenFilter: filter}); err != nil {
				t.Fatalf("filter %q: %v", filter, err)
			}
		})
	}
}

func TestValidateConfidenceAlgorithmConfigRejectsUnknownEnums(t *testing.T) {
	tests := []struct {
		name       string
		cfg        *ConfidenceAlgorithmConfig
		wantErrSub string
	}{
		{
			name:       "method",
			cfg:        &ConfidenceAlgorithmConfig{ConfidenceMethod: "logprob"},
			wantErrSub: "confidence_method must be one of",
		},
		{
			name:       "method whitespace is not canonical",
			cfg:        &ConfidenceAlgorithmConfig{ConfidenceMethod: " avg_logprob"},
			wantErrSub: "confidence_method must be one of",
		},
		{
			name:       "on error",
			cfg:        &ConfidenceAlgorithmConfig{OnError: "fallback"},
			wantErrSub: "on_error must be one of",
		},
		{
			name:       "escalation order",
			cfg:        &ConfidenceAlgorithmConfig{EscalationOrder: "asc"},
			wantErrSub: "escalation_order must be one of",
		},
		{
			name:       "token filter",
			cfg:        &ConfidenceAlgorithmConfig{TokenFilter: "alpha_numeric"},
			wantErrSub: "token_filter must be one of",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateConfidenceAlgorithmConfig(tc.cfg)
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error = %v, want substring %q", err, tc.wantErrSub)
			}
		})
	}
}

func TestValidateConfidenceAlgorithmConfigRejectsInvalidScalars(t *testing.T) {
	tests := []struct {
		name       string
		cfg        *ConfidenceAlgorithmConfig
		wantErrSub string
	}{
		{name: "negative threshold", cfg: &ConfidenceAlgorithmConfig{Threshold: -0.1}, wantErrSub: "threshold must be between 0 and 1"},
		{name: "threshold above one", cfg: &ConfidenceAlgorithmConfig{Threshold: 1.1}, wantErrSub: "threshold must be between 0 and 1"},
		{name: "NaN threshold", cfg: &ConfidenceAlgorithmConfig{Threshold: math.NaN()}, wantErrSub: "threshold must be finite"},
		{name: "infinite threshold", cfg: &ConfidenceAlgorithmConfig{Threshold: math.Inf(1)}, wantErrSub: "threshold must be finite"},
		{name: "negative cost tradeoff", cfg: &ConfidenceAlgorithmConfig{CostQualityTradeoff: -0.1}, wantErrSub: "cost_quality_tradeoff must be between 0 and 1"},
		{name: "cost tradeoff above one", cfg: &ConfidenceAlgorithmConfig{CostQualityTradeoff: 1.1}, wantErrSub: "cost_quality_tradeoff must be between 0 and 1"},
		{name: "NaN cost tradeoff", cfg: &ConfidenceAlgorithmConfig{CostQualityTradeoff: math.NaN()}, wantErrSub: "cost_quality_tradeoff must be finite"},
		{name: "negative timeout", cfg: &ConfidenceAlgorithmConfig{VerifierTimeoutSeconds: -1}, wantErrSub: "verifier_timeout_seconds must be >= 1 when set"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateConfidenceAlgorithmConfig(tc.cfg)
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error = %v, want substring %q", err, tc.wantErrSub)
			}
		})
	}
}

func TestValidateConfidenceAlgorithmConfigRejectsInvalidHybridWeights(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		weights    *HybridWeightsConfig
		wantErrSub string
	}{
		{
			name:       "weights on non-hybrid method",
			method:     ConfidenceMethodMargin,
			weights:    &HybridWeightsConfig{LogprobWeight: 0.5, MarginWeight: 0.5},
			wantErrSub: "hybrid_weights is only supported",
		},
		{
			name:       "negative logprob weight",
			method:     ConfidenceMethodHybrid,
			weights:    &HybridWeightsConfig{LogprobWeight: -0.1, MarginWeight: 0.5},
			wantErrSub: "logprob_weight must be between 0 and 1",
		},
		{
			name:       "margin weight above one",
			method:     ConfidenceMethodHybrid,
			weights:    &HybridWeightsConfig{LogprobWeight: 0.5, MarginWeight: 1.1},
			wantErrSub: "margin_weight must be between 0 and 1",
		},
		{
			name:       "NaN weight",
			method:     ConfidenceMethodHybrid,
			weights:    &HybridWeightsConfig{LogprobWeight: math.NaN(), MarginWeight: 0.5},
			wantErrSub: "logprob_weight must be finite",
		},
		{
			name:       "effective weights do not sum to one",
			method:     ConfidenceMethodHybrid,
			weights:    &HybridWeightsConfig{LogprobWeight: 0.7, MarginWeight: 0},
			wantErrSub: "effective values must sum to 1",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateConfidenceAlgorithmConfig(&ConfidenceAlgorithmConfig{
				ConfidenceMethod: tc.method,
				HybridWeights:    tc.weights,
			})
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error = %v, want substring %q", err, tc.wantErrSub)
			}
		})
	}
}

func TestValidateConfidenceAlgorithmConfigRejectsInvalidVerifier(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		url        string
		timeout    int
		maxBytes   int64
		wantErrSub string
	}{
		{name: "missing URL", method: ConfidenceMethodAutoMixEntailment, wantErrSub: "verifier_server_url is required"},
		{name: "relative URL", method: ConfidenceMethodAutoMixEntailment, url: "/verifier", wantErrSub: "scheme must be http or https"},
		{name: "unsupported scheme", method: ConfidenceMethodAutoMixEntailment, url: "ftp://verifier.example.com", wantErrSub: "scheme must be http or https"},
		{name: "missing host", method: ConfidenceMethodAutoMixEntailment, url: "http:///verifier", wantErrSub: "must include a host"},
		{name: "credentials", method: ConfidenceMethodAutoMixEntailment, url: "https://user:secret@verifier.example.com", wantErrSub: "must not include credentials"},
		{name: "query", method: ConfidenceMethodAutoMixEntailment, url: "https://verifier.example.com?token=secret", wantErrSub: "must not include a query or fragment"},
		{name: "whitespace", method: ConfidenceMethodAutoMixEntailment, url: " https://verifier.example.com", wantErrSub: "must not contain leading or trailing whitespace"},
		{name: "verifier config on another method", method: ConfidenceMethodSelfVerify, url: "https://verifier.example.com", timeout: 30, wantErrSub: "only supported when confidence_method"},
		{name: "negative response limit", method: ConfidenceMethodAutoMixEntailment, url: "https://verifier.example.com", maxBytes: -1, wantErrSub: "max_response_bytes must be positive"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateConfidenceAlgorithmConfig(&ConfidenceAlgorithmConfig{
				ConfidenceMethod:       tc.method,
				VerifierServerURL:      tc.url,
				VerifierTimeoutSeconds: tc.timeout,
				MaxResponseBytes:       tc.maxBytes,
			})
			if err == nil || !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error = %v, want substring %q", err, tc.wantErrSub)
			}
		})
	}
}

func TestValidateDecisionAlgorithmConfigRunsConfidenceValidation(t *testing.T) {
	err := validateDecisionAlgorithmConfig("invalid-confidence", nil, &AlgorithmConfig{
		Type: "confidence",
		Confidence: &ConfidenceAlgorithmConfig{
			ConfidenceMethod: "unknown",
		},
	})
	if err == nil {
		t.Fatal("expected specialized confidence validation error")
	}
	if got := err.Error(); !strings.Contains(got, "decision 'invalid-confidence', algorithm.confidence") ||
		!strings.Contains(got, "confidence_method must be one of") {
		t.Fatalf("unexpected error: %v", err)
	}
}
