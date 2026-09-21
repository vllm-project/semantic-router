package config

import (
	"strings"
	"testing"
)

var contextDedupValidationCases = []struct {
	name    string
	payload map[string]interface{}
	wantErr string
}{
	{
		name:    "disabled_defaults",
		payload: map[string]interface{}{"enabled": false},
	},
	{
		name:    "enabled_defaults",
		payload: map[string]interface{}{"enabled": true},
	},
	{
		name: "complete_policy",
		payload: map[string]interface{}{
			"enabled":       true,
			"normalization": "whitespace",
			"failure_mode":  "fail_closed",
			"limits": map[string]interface{}{
				"max_history_turns": 64,
				"max_history_bytes": 65536,
				"max_segment_turns": 16,
				"timeout_ms":        25,
			},
		},
	},
	{
		name: "unsupported_normalization",
		payload: map[string]interface{}{
			"enabled":       false,
			"normalization": "case_insensitive",
		},
		wantErr: "normalization must be exact or whitespace",
	},
	{
		name: "unsupported_failure_mode",
		payload: map[string]interface{}{
			"enabled":      false,
			"failure_mode": "fail_fast",
		},
		wantErr: "failure_mode must be",
	},
	{
		name: "negative_limit",
		payload: map[string]interface{}{
			"enabled": false,
			"limits":  map[string]interface{}{"max_history_turns": -1},
		},
		wantErr: "must be positive",
	},
	{
		name: "excessive_limit",
		payload: map[string]interface{}{
			"enabled": false,
			"limits":  map[string]interface{}{"timeout_ms": 60000},
		},
		wantErr: "cannot exceed 5000",
	},
	{
		name: "segment_bound_beyond_history_bound",
		payload: map[string]interface{}{
			"enabled": false,
			"limits":  map[string]interface{}{"max_history_turns": 8, "max_segment_turns": 9},
		},
		wantErr: "max_segment_turns cannot exceed limits.max_history_turns (8)",
	},
	{
		name: "small_history_bound_derives_the_segment_bound",
		payload: map[string]interface{}{
			"enabled": true,
			"limits":  map[string]interface{}{"max_history_turns": 8},
		},
	},
	{
		name: "unknown_field",
		payload: map[string]interface{}{
			"enabled":  false,
			"recovery": map[string]interface{}{"enabled": true},
		},
		wantErr: "recovery",
	},
}

func TestValidateContextDedupPluginContract(t *testing.T) {
	for _, test := range contextDedupValidationCases {
		t.Run(test.name, func(t *testing.T) {
			err := validateDecisionPluginPayload(
				"route",
				0,
				DecisionPlugin{
					Type:          DecisionPluginContextDedup,
					Configuration: MustStructuredPayload(test.payload),
				},
			)
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("expected the policy to validate, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected an error containing %q", test.wantErr)
			}
			if !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("expected an error containing %q, got %v", test.wantErr, err)
			}
		})
	}
	if err := ValidateContextDedupPluginConfig(nil); err == nil {
		t.Fatal("a nil payload must be rejected")
	}
}

func TestContextDedupEffectiveDefaults(t *testing.T) {
	var absent *ContextDedupPluginConfig
	if absent.IsEnabled() {
		t.Fatal("an absent policy must not be enabled")
	}
	if absent.EffectiveNormalization() != ContextDedupNormalizationExact {
		t.Fatalf("unexpected default normalization %q", absent.EffectiveNormalization())
	}
	if absent.EffectiveFailureMode() != ContextDedupFailureOpen {
		t.Fatalf("unexpected default failure mode %q", absent.EffectiveFailureMode())
	}
	limits := absent.EffectiveLimits()
	if limits.MaxHistoryTurns != DefaultContextDedupMaxHistoryTurns ||
		limits.MaxHistoryBytes != DefaultContextDedupMaxHistoryBytes ||
		limits.MaxSegmentTurns != DefaultContextDedupMaxSegmentTurns ||
		limits.TimeoutMs != DefaultContextDedupTimeoutMs {
		t.Fatalf("unexpected default limits %+v", limits)
	}
	policy := &ContextDedupPluginConfig{Limits: &ContextDedupLimitsConfig{MaxSegmentTurns: 8}}
	limits = policy.EffectiveLimits()
	if limits.MaxSegmentTurns != 8 || limits.MaxHistoryTurns != DefaultContextDedupMaxHistoryTurns {
		t.Fatalf("configured bounds must override only themselves: %+v", limits)
	}
	lowered := &ContextDedupPluginConfig{Limits: &ContextDedupLimitsConfig{MaxHistoryTurns: 8}}
	if limits = lowered.EffectiveLimits(); limits.MaxSegmentTurns != 8 {
		t.Fatalf("an omitted segment bound must follow a lowered history bound: %+v", limits)
	}
}

func TestGetContextDedupConfigReadsTheDecisionPlugin(t *testing.T) {
	decision := &Decision{
		Plugins: []DecisionPlugin{{
			Type: DecisionPluginContextDedup,
			Configuration: MustStructuredPayload(map[string]interface{}{
				"enabled":       true,
				"normalization": " whitespace ",
				"failure_mode":  "fail_closed",
				"limits":        map[string]interface{}{"max_segment_turns": 4},
			}),
		}},
	}
	policy := decision.GetContextDedupConfig()
	if !policy.IsEnabled() {
		t.Fatal("expected an enabled policy")
	}
	if policy.EffectiveNormalization() != ContextDedupNormalizationWhitespace {
		t.Fatalf("unexpected normalization %q", policy.EffectiveNormalization())
	}
	if policy.EffectiveFailureMode() != ContextDedupFailureClosed {
		t.Fatalf("unexpected failure mode %q", policy.EffectiveFailureMode())
	}
	if policy.EffectiveLimits().MaxSegmentTurns != 4 {
		t.Fatalf("unexpected limits %+v", policy.EffectiveLimits())
	}
	if (&Decision{}).GetContextDedupConfig() != nil {
		t.Fatal("a decision without the plugin must report no policy")
	}
	if !DecisionPluginEnabled(policy) {
		t.Fatal("introspection must report the policy as enabled")
	}
}
