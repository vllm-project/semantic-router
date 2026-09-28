package selection

import (
	"testing"
	"time"
)

var evidenceBase = time.Date(2026, 9, 7, 12, 0, 0, 0, time.UTC)

func fact(offsetMinutes int, turn int, category string, attributable bool) TurnOutcomeFact {
	return TurnOutcomeFact{
		TurnIndex:         turn,
		Timestamp:         evidenceBase.Add(time.Duration(offsetMinutes) * time.Minute),
		Model:             "model-a",
		Category:          category,
		ModelAttributable: attributable,
		Confidence:        0.5,
		OutputTokens:      100,
		LatencyMs:         1000,
	}
}

func TestEvaluateProgressEvidenceColdStart(t *testing.T) {
	if got := EvaluateProgressEvidence(nil); !got.ColdStart || got.TotalCount != 0 {
		t.Fatalf("empty window = %+v, want cold start", got)
	}

	// Only non-attributable noise is still a cold start for the gate.
	window := []TurnOutcomeFact{
		fact(0, 1, "provider_error", false),
		fact(1, 2, "tool_error", false),
	}
	got := EvaluateProgressEvidence(window)
	if !got.ColdStart || got.AttributableCount != 0 {
		t.Fatalf("noise-only window = %+v, want cold start", got)
	}
	if got.Version != ProgressEvidenceVersion {
		t.Fatalf("version = %q", got.Version)
	}
}

func TestEvaluateProgressEvidenceSustainedRegression(t *testing.T) {
	window := []TurnOutcomeFact{
		fact(0, 1, OutcomeProgress, true),
		fact(2, 2, OutcomeNoProgress, true),
		fact(4, 3, "provider_error", false), // noise must not break the streak
		fact(6, 4, OutcomeNoProgress, true),
	}
	got := EvaluateProgressEvidence(window)
	if got.RegressionStreak != 2 || got.StreakCategory != OutcomeNoProgress {
		t.Fatalf("streak = %d/%q, want 2/no_progress", got.RegressionStreak, got.StreakCategory)
	}
	if got.Trend >= 0 {
		t.Fatalf("trend = %v, want negative", got.Trend)
	}
	if got.AttributableCount != 3 {
		t.Fatalf("attributable = %d, want 3", got.AttributableCount)
	}
}

func TestEvaluateProgressEvidenceSustainedImprovement(t *testing.T) {
	window := []TurnOutcomeFact{
		fact(0, 1, OutcomeNoProgress, true),
		fact(2, 2, OutcomeProgress, true),
		fact(4, 3, OutcomeProgress, true),
	}
	got := EvaluateProgressEvidence(window)
	if got.RecoveryStreak != 2 {
		t.Fatalf("recovery streak = %d, want 2", got.RecoveryStreak)
	}
	if got.RegressionStreak != 0 {
		t.Fatalf("regression streak = %d, want 0", got.RegressionStreak)
	}
	if got.Trend <= 0 {
		t.Fatalf("trend = %v, want positive", got.Trend)
	}
}

func TestEvaluateProgressEvidenceOscillation(t *testing.T) {
	window := []TurnOutcomeFact{
		fact(0, 1, OutcomeNoProgress, true),
		fact(2, 2, OutcomeProgress, true),
		fact(4, 3, OutcomeNoProgress, true),
		fact(6, 4, OutcomeProgress, true),
	}
	got := EvaluateProgressEvidence(window)
	if got.RegressionStreak != 0 {
		t.Fatalf("oscillation must not build a regression streak: %+v", got)
	}
	if got.RecoveryStreak != 1 {
		t.Fatalf("recovery streak = %d, want 1", got.RecoveryStreak)
	}
}

func TestEvaluateProgressEvidenceMixedStreakCategoryBreaks(t *testing.T) {
	window := []TurnOutcomeFact{
		fact(0, 1, OutcomeRegression, true),
		fact(2, 2, OutcomeNoProgress, true),
	}
	// Different regression categories do not accumulate into one streak.
	if got := EvaluateProgressEvidence(window); got.RegressionStreak != 1 {
		t.Fatalf("streak = %d, want 1 (category change breaks it)", got.RegressionStreak)
	}
}

func TestEvaluateProgressEvidenceMissingCounted(t *testing.T) {
	window := []TurnOutcomeFact{
		fact(0, 1, OutcomeNoProgress, true),
		fact(2, 2, "missing", false),
		fact(4, 3, OutcomeNoProgress, true),
	}
	got := EvaluateProgressEvidence(window)
	if got.MissingCount != 1 {
		t.Fatalf("missing = %d, want 1", got.MissingCount)
	}
	if got.RegressionStreak != 2 {
		t.Fatalf("missing must not break the streak: %+v", got)
	}
}

// --- switch gate ---

func enforceConfig() ProgressGateConfig {
	cfg := DefaultProgressGateConfig()
	cfg.Enabled = true
	cfg.Mode = GateModeEnforce
	return cfg
}

func regressionEvidence(streak, attributable int) ProgressEvidence {
	return ProgressEvidence{
		Version:           ProgressEvidenceVersion,
		Trend:             -0.4,
		RegressionStreak:  streak,
		StreakCategory:    OutcomeNoProgress,
		AttributableCount: attributable,
		TotalCount:        attributable,
	}
}

func TestEvaluateSwitchGateAllowsOnSufficientEvidence(t *testing.T) {
	got := EvaluateSwitchGate(enforceConfig(), SwitchGateInput{
		Evidence: regressionEvidence(2, 3),
	})
	if got.Decision != GateDecisionSwitch || got.Suppressed() {
		t.Fatalf("decision = %+v, want switch", got)
	}
	if got.Origin != SwitchOriginEscalation || got.Version != ProgressEvidenceVersion {
		t.Fatalf("trace fields = %+v", got)
	}
}

func TestEvaluateSwitchGateSuppressionReasons(t *testing.T) {
	cfg := enforceConfig()
	cases := []struct {
		name  string
		input SwitchGateInput
		want  string
	}{
		{
			name:  "cold start",
			input: SwitchGateInput{Evidence: ProgressEvidence{Version: ProgressEvidenceVersion, ColdStart: true}},
			want:  GateReasonColdStart,
		},
		{
			name:  "insufficient window",
			input: SwitchGateInput{Evidence: regressionEvidence(2, 1)},
			want:  GateReasonInsufficientEvidence,
		},
		{
			name:  "streak below threshold",
			input: SwitchGateInput{Evidence: regressionEvidence(1, 3)},
			want:  GateReasonInsufficientEvidence,
		},
		{
			name: "cooldown",
			input: SwitchGateInput{
				Evidence:               regressionEvidence(2, 3),
				LastSwitchKnown:        true,
				SecondsSinceLastSwitch: 30,
			},
			want: GateReasonCooldown,
		},
		{
			name: "oscillation guard",
			input: SwitchGateInput{
				Evidence:               regressionEvidence(2, 3),
				LastSwitchKnown:        true,
				SecondsSinceLastSwitch: 600,
				SwitchesInWindow:       2,
			},
			want: GateReasonOscillationGuard,
		},
		{
			name: "hard constraint wins over evidence",
			input: SwitchGateInput{
				Evidence:               regressionEvidence(2, 3),
				HardConstraintConflict: true,
			},
			want: GateReasonHardConstraint,
		},
	}

	for _, tc := range cases {
		got := EvaluateSwitchGate(cfg, tc.input)
		if got.Decision != GateDecisionSuppress {
			t.Fatalf("%s: decision = %q, want suppress", tc.name, got.Decision)
		}
		if got.Reason != tc.want {
			t.Fatalf("%s: reason = %q, want %q", tc.name, got.Reason, tc.want)
		}
		if !got.Suppressed() {
			t.Fatalf("%s: enforce mode must suppress", tc.name)
		}
	}
}

func TestEvaluateSwitchGateHardConstraintReasonPassthrough(t *testing.T) {
	got := EvaluateSwitchGate(enforceConfig(), SwitchGateInput{
		Evidence:               regressionEvidence(2, 3),
		HardConstraintConflict: true,
		HardConstraintReason:   "active_tool_loop",
	})
	if got.Reason != "active_tool_loop" {
		t.Fatalf("reason = %q, want the specific hard-lock reason", got.Reason)
	}
}

func TestEvaluateSwitchGateHysteresis(t *testing.T) {
	cfg := enforceConfig()
	cfg.MinConsecutiveRegressions = 3
	cfg.MinConsecutiveRecoveries = 2

	// Escalation needs the higher threshold.
	if got := EvaluateSwitchGate(cfg, SwitchGateInput{Evidence: regressionEvidence(2, 3)}); got.Decision != GateDecisionSuppress {
		t.Fatalf("escalation at streak 2 should be suppressed with threshold 3: %+v", got)
	}

	// Downgrade uses the lower recovery threshold.
	downgrade := SwitchGateInput{
		Downgrade: true,
		Evidence: ProgressEvidence{
			Version:           ProgressEvidenceVersion,
			Trend:             0.5,
			RecoveryStreak:    2,
			AttributableCount: 3,
		},
	}
	got := EvaluateSwitchGate(cfg, downgrade)
	if got.Decision != GateDecisionSwitch || got.Origin != SwitchOriginDowngrade {
		t.Fatalf("downgrade = %+v, want switch with downgrade origin", got)
	}
}

func TestEvaluateSwitchGateObserveModeDoesNotEnforce(t *testing.T) {
	cfg := DefaultProgressGateConfig()
	cfg.Enabled = true // observe by default

	got := EvaluateSwitchGate(cfg, SwitchGateInput{Evidence: regressionEvidence(1, 3)})
	if got.Decision != GateDecisionSuppress || got.Reason != GateReasonInsufficientEvidence {
		t.Fatalf("observe mode must still record reasoning: %+v", got)
	}
	if got.Suppressed() {
		t.Fatalf("observe mode must not intercept: %+v", got)
	}
}

func TestEvaluateSwitchGateDisabledIsTransparent(t *testing.T) {
	got := EvaluateSwitchGate(ProgressGateConfig{}, SwitchGateInput{
		Evidence: ProgressEvidence{ColdStart: true},
	})
	if got.Decision != GateDecisionSwitch || got.Suppressed() {
		t.Fatalf("disabled gate must be transparent: %+v", got)
	}
	if got.Version != ProgressEvidenceVersion {
		t.Fatalf("version must still be stamped: %+v", got)
	}
}

func TestSecondsSince(t *testing.T) {
	from := evidenceBase
	to := evidenceBase.Add(90 * time.Second)
	if secs, ok := SecondsSince(from, to); !ok || secs != 90 {
		t.Fatalf("SecondsSince = %v/%v, want 90/true", secs, ok)
	}
	if _, ok := SecondsSince(time.Time{}, to); ok {
		t.Fatalf("zero from must report unknown")
	}
	if secs, _ := SecondsSince(to, from); secs != 0 {
		t.Fatalf("negative delta = %v, want clamped to 0", secs)
	}
}
