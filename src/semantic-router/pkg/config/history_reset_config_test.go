package config

import (
	"strings"
	"testing"
)

var historyResetValidationCases = []struct {
	name    string
	payload map[string]interface{}
	wantErr string
}{
	{
		name:    "disabled_defaults",
		payload: map[string]interface{}{"enabled": false},
	},
	{
		name: "disabled_policy_may_reference_a_future_signal",
		payload: map[string]interface{}{
			"enabled": false,
			"trigger": map[string]interface{}{
				"signal":         "topic_boundary",
				"min_confidence": 0.9,
			},
			"scope":        "eligible_history",
			"failure_mode": "fail_closed",
			"limits": map[string]interface{}{
				"max_history_turns": 64,
				"max_history_bytes": 65536,
				"timeout_ms":        25,
			},
			"recovery": map[string]interface{}{
				"enabled":               true,
				"store":                 "response_cache",
				"ttl_seconds":           900,
				"max_bytes_per_request": 1024,
				"max_total_bytes":       4096,
				"max_retrievals":        4,
			},
		},
	},
	{
		name: "enabled_is_rejected_until_the_trigger_family_exists",
		payload: map[string]interface{}{
			"enabled": true,
			"trigger": map[string]interface{}{
				"signal":            "topic_boundary",
				"min_confidence":    0.9,
				"accepted_versions": []string{"v1"},
			},
		},
		wantErr: HistoryResetTriggerUnavailable,
	},
	{
		name: "enabled_fail_closed_is_rejected_the_same_way",
		payload: map[string]interface{}{
			"enabled":      true,
			"failure_mode": "fail_closed",
			"trigger": map[string]interface{}{
				"signal":            "topic_boundary",
				"min_confidence":    0.75,
				"accepted_versions": []string{"v1"},
			},
		},
		wantErr: HistoryResetTriggerUnavailable,
	},
	{
		name: "enabled_requires_accepted_versions",
		payload: map[string]interface{}{
			"enabled": true,
			"trigger": map[string]interface{}{
				"signal":         "topic_boundary",
				"min_confidence": 0.9,
			},
		},
		wantErr: "trigger.accepted_versions is required",
	},
	{
		name: "accepted_versions_cannot_be_blank",
		payload: map[string]interface{}{
			"enabled": true,
			"trigger": map[string]interface{}{
				"signal":            "topic_boundary",
				"min_confidence":    0.9,
				"accepted_versions": []string{" "},
			},
		},
		wantErr: "cannot be empty",
	},
	{
		name:    "enabled_requires_a_trigger",
		payload: map[string]interface{}{"enabled": true},
		wantErr: "trigger is required",
	},
	{
		name: "enabled_requires_an_explicit_threshold",
		payload: map[string]interface{}{
			"enabled": true,
			"trigger": map[string]interface{}{"signal": "topic_boundary"},
		},
		wantErr: "trigger.min_confidence is required",
	},
	{
		name: "trigger_requires_a_signal",
		payload: map[string]interface{}{
			"enabled": false,
			"trigger": map[string]interface{}{"min_confidence": 0.9},
		},
		wantErr: "trigger.signal is required",
	},
	{
		name: "confidence_above_one",
		payload: map[string]interface{}{
			"enabled": false,
			"trigger": map[string]interface{}{
				"signal":         "topic_boundary",
				"min_confidence": 1.5,
			},
		},
		wantErr: "trigger.min_confidence",
	},
	{
		name: "confidence_of_zero",
		payload: map[string]interface{}{
			"enabled": false,
			"trigger": map[string]interface{}{
				"signal":         "topic_boundary",
				"min_confidence": 0,
			},
		},
		wantErr: "trigger.min_confidence",
	},
	{
		name: "unsupported_scope",
		payload: map[string]interface{}{
			"enabled": false,
			"scope":   "all_history",
		},
		wantErr: "scope must be eligible_history",
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
		wantErr: "cannot exceed",
	},
	{
		name: "recovery_requires_a_shared_store",
		payload: map[string]interface{}{
			"enabled":  false,
			"recovery": map[string]interface{}{"enabled": true},
		},
		wantErr: "recovery.store",
	},
	{
		name: "negative_recovery_limit",
		payload: map[string]interface{}{
			"enabled": false,
			"recovery": map[string]interface{}{
				"enabled":     true,
				"store":       "redis",
				"ttl_seconds": -1,
			},
		},
		wantErr: "recovery limits cannot be negative",
	},
	{
		name: "unknown_field",
		payload: map[string]interface{}{
			"enabled": false,
			"reset":   "everything",
		},
		wantErr: "unknown field",
	},
	{
		name: "flat_trigger_field_rejected",
		payload: map[string]interface{}{
			"enabled": false,
			"signal":  "topic_boundary",
		},
		wantErr: "unknown field",
	},
}

func TestValidateHistoryResetPluginContract(t *testing.T) {
	for _, test := range historyResetValidationCases {
		t.Run(test.name, func(t *testing.T) {
			err := validateDecisionPluginPayload(
				"route",
				0,
				DecisionPlugin{
					Type:          DecisionPluginHistoryReset,
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
}

func TestHistoryResetEffectiveDefaults(t *testing.T) {
	var absent *HistoryResetPluginConfig
	if absent.IsEnabled() {
		t.Fatal("an absent policy must not be enabled")
	}
	if absent.EffectiveScope() != HistoryResetScopeEligibleHistory {
		t.Fatalf("unexpected default scope %q", absent.EffectiveScope())
	}
	if absent.EffectiveFailureMode() != HistoryResetFailureOpen {
		t.Fatalf("unexpected default failure mode %q", absent.EffectiveFailureMode())
	}
	if absent.RequiresRecovery() {
		t.Fatal("an absent policy must not require recovery")
	}
	if _, present := absent.EffectiveMinConfidence(); present {
		t.Fatal("an absent policy must not report a confidence threshold")
	}

	limits := absent.EffectiveLimits()
	if limits.MaxHistoryTurns != DefaultHistoryResetMaxHistoryTurns ||
		limits.MaxHistoryBytes != DefaultHistoryResetMaxHistoryBytes ||
		limits.TimeoutMs != DefaultHistoryResetTimeoutMs {
		t.Fatalf("unexpected default limits %+v", limits)
	}
}

func TestHistoryResetEffectiveLimitsOverrideOnlyConfiguredBounds(t *testing.T) {
	policy := &HistoryResetPluginConfig{
		Limits: &HistoryResetLimitsConfig{MaxHistoryTurns: 8},
	}
	limits := policy.EffectiveLimits()
	if limits.MaxHistoryTurns != 8 {
		t.Fatalf("expected the configured turn bound, got %d", limits.MaxHistoryTurns)
	}
	if limits.MaxHistoryBytes != DefaultHistoryResetMaxHistoryBytes ||
		limits.TimeoutMs != DefaultHistoryResetTimeoutMs {
		t.Fatalf("omitted bounds must keep their defaults, got %+v", limits)
	}
}

// The enablement gate must follow the signal catalog rather than a constant, so
// it opens on its own once the topic-continuity family is registered.
func TestHistoryResetTriggerFamilyFollowsTheSignalCatalog(t *testing.T) {
	if HistoryResetTriggerFamilyRegistered() {
		t.Skip("the topic-continuity family is registered; enablement is no longer gated on it")
	}
	original := signalCatalog
	t.Cleanup(func() { signalCatalog = original })

	signalCatalog = append(append([]SignalCatalogEntry(nil), original...), SignalCatalogEntry{
		Type:                  HistoryResetTriggerSignalType,
		DisplayName:           "Topic Continuity",
		Collection:            "topic_continuity",
		DecisionReferenceable: true,
	})
	if !HistoryResetTriggerFamilyRegistered() {
		t.Fatal("expected the registered family to satisfy the trigger gate")
	}

	confidence := 0.9
	policy := &HistoryResetPluginConfig{
		Enabled: true,
		Trigger: &HistoryResetTriggerConfig{
			Signal:           "topic_boundary",
			MinConfidence:    &confidence,
			AcceptedVersions: []string{"v1"},
		},
	}
	if err := ValidateHistoryResetPluginConfig(policy); err != nil {
		t.Fatalf("expected an enabled policy to validate once the family exists, got %v", err)
	}
}

func TestGetHistoryResetConfigReadsTheDecisionPlugin(t *testing.T) {
	decision := &Decision{
		Plugins: []DecisionPlugin{{
			Type: DecisionPluginHistoryReset,
			Configuration: MustStructuredPayload(map[string]interface{}{
				"enabled":      false,
				"failure_mode": "fail_closed",
				"trigger": map[string]interface{}{
					"signal":         "topic_boundary",
					"min_confidence": 0.8,
				},
			}),
		}},
	}
	policy := decision.GetHistoryResetConfig()
	if policy == nil {
		t.Fatal("expected the configured policy to decode")
	}
	if policy.IsEnabled() {
		t.Fatal("expected the decoded policy to stay disabled")
	}
	if policy.EffectiveFailureMode() != HistoryResetFailureClosed {
		t.Fatalf("unexpected failure mode %q", policy.EffectiveFailureMode())
	}
	confidence, present := policy.EffectiveMinConfidence()
	if !present || confidence != 0.8 {
		t.Fatalf("unexpected confidence %v (present=%v)", confidence, present)
	}

	empty := &Decision{}
	if empty.GetHistoryResetConfig() != nil {
		t.Fatal("a decision without the plugin must report no policy")
	}
}

// One request-level recovery store serves every context action, so a decision
// that asks for two different stores is rejected instead of silently losing
// one action's content.
func TestDecisionRejectsDisagreeingContextRecoveryStores(t *testing.T) {
	decision := &Decision{
		Name: "context",
		Plugins: []DecisionPlugin{
			{
				Type: DecisionPluginContextCompression,
				Configuration: MustStructuredPayload(map[string]interface{}{
					"enabled":  true,
					"recovery": map[string]interface{}{"enabled": true, "store": "redis"},
				}),
			},
			{
				Type: DecisionPluginHistoryReset,
				Configuration: MustStructuredPayload(map[string]interface{}{
					"enabled":  false,
					"recovery": map[string]interface{}{"enabled": true, "store": "valkey"},
				}),
			},
		},
	}
	err := validateDecisionContextRecoveryAgreement(decision)
	if err == nil {
		t.Fatal("disagreeing recovery stores were accepted")
	}
	if !strings.Contains(err.Error(), "must match") {
		t.Fatalf("unexpected error %v", err)
	}

	matching := decision.Plugins[1].Configuration
	decision.Plugins[1].Configuration = MustStructuredPayload(map[string]interface{}{
		"enabled":  false,
		"recovery": map[string]interface{}{"enabled": true, "store": "redis"},
	})
	if err = validateDecisionContextRecoveryAgreement(decision); err != nil {
		t.Fatalf("matching stores were rejected: %v", err)
	}
	decision.Plugins[1].Configuration = matching
}

// Registering the topic-continuity family globally must not be enough on its
// own: an enabled trigger has to name a signal the recipe actually declares.
func TestHistoryResetTriggerReferencesResolveWithinTheRecipe(t *testing.T) {
	enabled := func(signal string) Decision {
		return Decision{
			Name: "reset",
			Plugins: []DecisionPlugin{{
				Type: DecisionPluginHistoryReset,
				Configuration: MustStructuredPayload(map[string]interface{}{
					"enabled": true,
					"trigger": map[string]interface{}{
						"signal":            signal,
						"min_confidence":    0.9,
						"accepted_versions": []string{"v1"},
					},
				}),
			}},
		}
	}

	// No topic-continuity signal can be declared yet, so every enabled
	// reference is unresolvable inside a recipe scope.
	scoped := &RouterConfig{RoutingScope: "recipe-a"}
	scoped.Decisions = []Decision{enabled("topic_boundary")}
	err := validateHistoryResetTriggerReferences(scoped)
	if err == nil {
		t.Fatal("an undeclared trigger signal was accepted")
	}
	if !strings.Contains(err.Error(), "not declared in this recipe") {
		t.Fatalf("unexpected error %v", err)
	}

	// A name that resolves to some other signal family is equally invalid: the
	// reference is checked against the topic-continuity family, not any match.
	scoped.KeywordRules = []KeywordRule{{Name: "topic_boundary", Operator: "OR", Keywords: []string{"x"}}}
	if err = validateHistoryResetTriggerReferences(scoped); err == nil {
		t.Fatal("a keyword signal was accepted as a topic-continuity trigger")
	}

	// Unscoped configuration defers to the per-recipe pass.
	unscoped := &RouterConfig{}
	unscoped.Decisions = []Decision{enabled("topic_boundary")}
	if err = validateHistoryResetTriggerReferences(unscoped); err != nil {
		t.Fatalf("unscoped configuration must defer the reference check: %v", err)
	}

	// Once a topic-continuity signal can be declared, a reference to it
	// resolves and a reference to any other family still does not. Supplying
	// the declaration map directly exercises both today.
	withTopicSignal := map[string]map[string]struct{}{
		HistoryResetTriggerSignalType: {"topic_boundary": {}},
		SignalTypeKeyword:             {"unrelated": {}},
	}
	if err = resolveHistoryResetTriggerReferences(scoped, withTopicSignal); err != nil {
		t.Fatalf("a declared topic-continuity signal was rejected: %v", err)
	}
	scoped.Decisions = []Decision{enabled("unrelated")}
	if err = resolveHistoryResetTriggerReferences(scoped, withTopicSignal); err == nil {
		t.Fatal("a signal from another family was accepted as a trigger")
	}
	scoped.Decisions = []Decision{enabled("declared_elsewhere")}
	if err = resolveHistoryResetTriggerReferences(scoped, withTopicSignal); err == nil {
		t.Fatal("a name declared in no recipe was accepted")
	}

	// A disabled policy never resolves its future trigger.
	disabled := &RouterConfig{RoutingScope: "recipe-a"}
	disabled.Decisions = []Decision{{
		Name: "reset",
		Plugins: []DecisionPlugin{{
			Type: DecisionPluginHistoryReset,
			Configuration: MustStructuredPayload(map[string]interface{}{
				"enabled": false,
				"trigger": map[string]interface{}{
					"signal":            "topic_boundary",
					"min_confidence":    0.9,
					"accepted_versions": []string{"v1"},
				},
			}),
		}},
	}}
	if err = validateHistoryResetTriggerReferences(disabled); err != nil {
		t.Fatalf("a disabled policy must not require a declared signal: %v", err)
	}
}
