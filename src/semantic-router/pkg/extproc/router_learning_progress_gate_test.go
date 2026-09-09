package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func gateBoolPtr(v bool) *bool        { return &v }
func gateIntPtr(v int) *int           { return &v }
func gateFloatPtr(v float64) *float64 { return &v }

func TestProgressGateConfigDefaultsWhenSectionOmitted(t *testing.T) {
	cfg := progressGateConfig(config.RouterLearningProtectionTuning{})
	want := selection.DefaultProgressGateConfig()
	if cfg != want {
		t.Fatalf("omitted section = %+v, want packaged defaults %+v", cfg, want)
	}
	if cfg.Enabled {
		t.Fatalf("gate must stay disabled when unconfigured")
	}
}

func TestProgressGateConfigOverridesOnlySuppliedFields(t *testing.T) {
	cfg := progressGateConfig(config.RouterLearningProtectionTuning{
		ProgressGate: &config.ProgressGateTuning{
			Enabled:                   gateBoolPtr(true),
			Mode:                      selection.GateModeEnforce,
			MinConsecutiveRegressions: gateIntPtr(3),
			CooldownSeconds:           gateFloatPtr(45),
		},
	})
	if !cfg.Enabled || cfg.Mode != selection.GateModeEnforce {
		t.Fatalf("supplied fields not applied: %+v", cfg)
	}
	if cfg.MinConsecutiveRegressions != 3 || cfg.CooldownSeconds != 45 {
		t.Fatalf("threshold overrides not applied: %+v", cfg)
	}
	// Untouched fields keep the packaged defaults.
	defaults := selection.DefaultProgressGateConfig()
	if cfg.MinWindowOutcomes != defaults.MinWindowOutcomes ||
		cfg.MaxSwitchesPerWindow != defaults.MaxSwitchesPerWindow {
		t.Fatalf("omitted fields must inherit defaults: %+v", cfg)
	}
}

func TestTurnOutcomeFactsConversion(t *testing.T) {
	now := time.Now()
	window := []sessiontelemetry.TurnOutcome{
		{
			TurnIndex:         2,
			Timestamp:         now.UnixMilli(),
			Model:             "model-a",
			Category:          sessiontelemetry.TurnProviderError,
			ModelAttributable: false,
			OutputTokens:      12,
			LatencyMs:         900,
		},
	}
	facts := turnOutcomeFacts(window)
	if len(facts) != 1 {
		t.Fatalf("facts = %+v, want one", facts)
	}
	got := facts[0]
	if got.Category != string(sessiontelemetry.TurnProviderError) || got.ModelAttributable {
		t.Fatalf("attribution and category must survive conversion: %+v", got)
	}
	if got.TurnIndex != 2 || got.Model != "model-a" || got.OutputTokens != 12 || got.LatencyMs != 900 {
		t.Fatalf("scalar fields lost in conversion: %+v", got)
	}
	if !got.Timestamp.Equal(time.UnixMilli(now.UnixMilli())) {
		t.Fatalf("timestamp = %v, want the stored event time", got.Timestamp)
	}
	if turnOutcomeFacts(nil) != nil {
		t.Fatalf("empty window must convert to nil")
	}
}

func TestSwitchGateTraceCarriesEvidenceAndVerdict(t *testing.T) {
	evidence := selection.ProgressEvidence{
		Version:           selection.ProgressEvidenceVersion,
		Trend:             -0.5,
		RegressionStreak:  2,
		RecoveryStreak:    0,
		AttributableCount: 3,
		MissingCount:      1,
	}
	in := selection.SwitchGateInput{
		Evidence:               evidence,
		SwitchesInWindow:       1,
		SecondsSinceLastSwitch: 30,
		LastSwitchKnown:        true,
	}
	decision := selection.SwitchGateDecision{
		Version:  selection.ProgressEvidenceVersion,
		Mode:     selection.GateModeEnforce,
		Decision: selection.GateDecisionSuppress,
		Reason:   selection.GateReasonCooldown,
		Origin:   selection.SwitchOriginEscalation,
		Enforced: true,
	}

	trace := switchGateTrace(decision, evidence, in, 4)
	if trace.EvidenceVersion != selection.ProgressEvidenceVersion ||
		trace.Decision != selection.GateDecisionSuppress ||
		trace.Reason != selection.GateReasonCooldown {
		t.Fatalf("verdict fields missing: %+v", trace)
	}
	if trace.RegressionStreak != 2 || trace.Trend != -0.5 || trace.AttributableCount != 3 {
		t.Fatalf("evidence fields missing: %+v", trace)
	}
	if trace.WindowCount != 4 || trace.SwitchesInWindow != 1 || !trace.LastSwitchKnown {
		t.Fatalf("window/state fields missing: %+v", trace)
	}
}

func TestAttachSwitchGateTraceCreatesPolicyWhenAbsent(t *testing.T) {
	result := &selection.SelectionResult{SelectedModel: "model-b"}
	trace := &selection.SessionSwitchGateTrace{Decision: selection.GateDecisionSwitch}

	attachSwitchGateTrace(result, trace)
	if result.SessionPolicy == nil || result.SessionPolicy.SwitchGate != trace {
		t.Fatalf("trace not attached: %+v", result.SessionPolicy)
	}

	// Nil inputs must be no-ops rather than panics.
	attachSwitchGateTrace(nil, trace)
	attachSwitchGateTrace(result, nil)
}

func TestSwitchGateTraceReachesReplayMap(t *testing.T) {
	policy := &selection.SessionPolicyTrace{
		SwitchGate: &selection.SessionSwitchGateTrace{
			EvidenceVersion: selection.ProgressEvidenceVersion,
			Mode:            selection.GateModeObserve,
			Decision:        selection.GateDecisionSuppress,
			Reason:          selection.GateReasonInsufficientEvidence,
			Origin:          selection.SwitchOriginEscalation,
		},
	}
	out := policy.ToMap()
	gate, ok := out["switch_gate"].(map[string]interface{})
	if !ok {
		t.Fatalf("switch_gate missing from replay map: %+v", out)
	}
	if gate["suppression_reason"] != selection.GateReasonInsufficientEvidence {
		t.Fatalf("suppression_reason = %v", gate["suppression_reason"])
	}
	if gate["evidence_version"] != selection.ProgressEvidenceVersion {
		t.Fatalf("evidence_version = %v", gate["evidence_version"])
	}

	// A trace-free policy must not emit the section at all.
	if _, exists := (&selection.SessionPolicyTrace{}).ToMap()["switch_gate"]; exists {
		t.Fatalf("switch_gate must be omitted when the gate did not run")
	}
}

func TestSwitchGateVerdictSkipsWhenDisabledOrNoSwitch(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{SessionID: "gate-skip"}

	// Disabled gate.
	if _, _, ran := router.switchGateVerdict(
		config.RouterLearningProtectionConfig{},
		ctx, nil, "model-a", "model-b", false,
	); ran {
		t.Fatalf("disabled gate must not run")
	}

	// Enabled but the proposal keeps the current model.
	enabled := config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			ProgressGate: &config.ProgressGateTuning{Enabled: gateBoolPtr(true)},
		},
	}
	if _, _, ran := router.switchGateVerdict(
		enabled, ctx, nil, "model-a", "model-a", false,
	); ran {
		t.Fatalf("a non-switch must not be gated")
	}

	// Enabled but the session cannot be keyed.
	if _, _, ran := router.switchGateVerdict(
		enabled, &RequestContext{}, nil, "model-a", "model-b", false,
	); ran {
		t.Fatalf("session-less request must not be gated")
	}
}

func TestSwitchGateVerdictColdStartSuppressesInEnforceMode(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{SessionID: "gate-cold"}
	cfg := config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			ProgressGate: &config.ProgressGateTuning{
				Enabled: gateBoolPtr(true),
				Mode:    selection.GateModeEnforce,
			},
		},
	}

	decision, trace, ran := router.switchGateVerdict(
		cfg, ctx, nil, "model-a", "model-b", false,
	)
	if !ran {
		t.Fatalf("gate should have evaluated")
	}
	if !decision.Suppressed() || decision.Reason != selection.GateReasonColdStart {
		t.Fatalf("empty window must suppress as cold start: %+v", decision)
	}
	if trace == nil || !trace.ColdStart || trace.WindowCount != 0 {
		t.Fatalf("trace must record the cold start: %+v", trace)
	}
}

func TestSwitchGateVerdictAllowsOnSustainedRegression(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{SessionID: "gate-regression"}
	sessionKey := routingSessionStateKey(ctx)

	now := time.Now()
	for i, category := range []sessiontelemetry.TurnOutcomeCategory{
		sessiontelemetry.TurnProgress,
		sessiontelemetry.TurnNoProgress,
		sessiontelemetry.TurnNoProgress,
	} {
		sessiontelemetry.RecordTurnOutcome(sessionKey, sessiontelemetry.TurnOutcome{
			TurnIndex: i,
			Model:     "model-a",
			Category:  category,
		}, now.Add(time.Duration(i)*time.Second))
	}

	cfg := config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			ProgressGate: &config.ProgressGateTuning{
				Enabled: gateBoolPtr(true),
				Mode:    selection.GateModeEnforce,
			},
		},
	}
	decision, trace, ran := router.switchGateVerdict(
		cfg, ctx, nil, "model-a", "model-b", false,
	)
	if !ran {
		t.Fatalf("gate should have evaluated")
	}
	if decision.Suppressed() {
		t.Fatalf("two consecutive attributable regressions must clear the gate: %+v", decision)
	}
	if trace.RegressionStreak != 2 || trace.WindowCount != 3 {
		t.Fatalf("trace = %+v, want streak 2 over a 3-entry window", trace)
	}
}

func TestSwitchGateVerdictHardConstraintShortCircuits(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{SessionID: "gate-hardlock"}
	sessionKey := routingSessionStateKey(ctx)

	now := time.Now()
	for i := 0; i < 3; i++ {
		sessiontelemetry.RecordTurnOutcome(sessionKey, sessiontelemetry.TurnOutcome{
			TurnIndex: i,
			Model:     "model-a",
			Category:  sessiontelemetry.TurnNoProgress,
		}, now.Add(time.Duration(i)*time.Second))
	}

	cfg := config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			ProgressGate: &config.ProgressGateTuning{
				Enabled: gateBoolPtr(true),
				Mode:    selection.GateModeEnforce,
			},
		},
	}
	learningCtx := &selection.SelectionContext{
		AgenticSession: &selection.AgenticSessionContext{ActiveToolLoop: true},
	}

	decision, _, ran := router.switchGateVerdict(
		cfg, ctx, learningCtx, "model-a", "model-b", false,
	)
	if !ran {
		t.Fatalf("gate should have evaluated")
	}
	// Evidence is sufficient, but the tool loop is authoritative.
	if !decision.Suppressed() || decision.Reason != selection.GateReasonHardConstraint {
		t.Fatalf("hard lock must win over sufficient evidence: %+v", decision)
	}
}

func TestSwitchGateVerdictDowngradeOrigin(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{SessionID: "gate-downgrade"}
	cfg := config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			ProgressGate: &config.ProgressGateTuning{
				Enabled: gateBoolPtr(true),
				Mode:    selection.GateModeObserve,
			},
		},
	}

	decision, trace, ran := router.switchGateVerdict(
		cfg, ctx, nil, "frontier", "cheap", true,
	)
	if !ran {
		t.Fatal("gate should have evaluated")
	}
	if decision.Origin != selection.SwitchOriginDowngrade ||
		trace.Origin != selection.SwitchOriginDowngrade {
		t.Fatalf("downgrade origin lost: decision=%+v trace=%+v", decision, trace)
	}
}

func TestSwitchGateVerdictObserveModeRecordsWithoutSuppressing(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	ctx := &RequestContext{SessionID: "gate-observe"}
	cfg := config.RouterLearningProtectionConfig{
		Tuning: config.RouterLearningProtectionTuning{
			ProgressGate: &config.ProgressGateTuning{
				Enabled: gateBoolPtr(true),
				Mode:    selection.GateModeObserve,
			},
		},
	}

	decision, trace, ran := router.switchGateVerdict(
		cfg, ctx, nil, "model-a", "model-b", false,
	)
	if !ran {
		t.Fatalf("gate should have evaluated")
	}
	if decision.Decision != selection.GateDecisionSuppress || decision.Reason == "" {
		t.Fatalf("observe mode must still reason: %+v", decision)
	}
	if decision.Suppressed() {
		t.Fatalf("observe mode must not intercept the switch: %+v", decision)
	}
	if trace.Mode != selection.GateModeObserve || trace.Enforced {
		t.Fatalf("trace must record observe mode: %+v", trace)
	}
}
