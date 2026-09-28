package config

import "testing"

func intPtr(v int) *int { return &v }

func TestValidateProgressGateTuningAcceptsOmittedSection(t *testing.T) {
	if err := validateProgressGateTuning("p", nil); err != nil {
		t.Fatalf("omitted section must stay valid: %v", err)
	}
}

func TestValidateProgressGateTuningMode(t *testing.T) {
	for _, mode := range []string{"", "observe", "enforce"} {
		if err := validateProgressGateTuning("p", &ProgressGateTuning{Mode: mode}); err != nil {
			t.Fatalf("mode %q must be valid: %v", mode, err)
		}
	}
	if err := validateProgressGateTuning("p", &ProgressGateTuning{Mode: "apply"}); err == nil {
		t.Fatalf("unknown mode must be rejected")
	}
}

func TestValidateProgressGateTuningWindowRelationship(t *testing.T) {
	cfg := &ProgressGateTuning{
		WindowSize:        intPtr(4),
		MinWindowOutcomes: intPtr(5),
	}
	if err := validateProgressGateTuning("p", cfg); err == nil {
		t.Fatalf("min_window_outcomes above window_size must be rejected")
	}

	cfg.MinWindowOutcomes = intPtr(4)
	if err := validateProgressGateTuning("p", cfg); err != nil {
		t.Fatalf("min_window_outcomes equal to window_size must be valid: %v", err)
	}
}

func TestValidateProgressGateTuningHysteresisRelationship(t *testing.T) {
	cfg := &ProgressGateTuning{
		MinConsecutiveRegressions: intPtr(1),
		MinConsecutiveRecoveries:  intPtr(2),
	}
	if err := validateProgressGateTuning("p", cfg); err == nil {
		t.Fatalf("escalation threshold below downgrade threshold must be rejected")
	}

	cfg.MinConsecutiveRegressions = intPtr(2)
	if err := validateProgressGateTuning("p", cfg); err != nil {
		t.Fatalf("equal thresholds must be valid: %v", err)
	}
}

func TestValidateProtectionTuningIncludesProgressGate(t *testing.T) {
	tuning := RouterLearningProtectionTuning{
		ProgressGate: &ProgressGateTuning{Mode: "bogus"},
	}
	if err := validateProtectionTuning("global.router.learning.protection.tuning", tuning); err == nil {
		t.Fatalf("protection tuning must validate the progress gate section")
	}
}
