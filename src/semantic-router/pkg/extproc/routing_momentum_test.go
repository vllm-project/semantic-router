package extproc

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func approxEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestComputeRoutingMomentum_EmptySignals(t *testing.T) {
	momentum := ComputeRoutingMomentum(nil, 0.2, 0.95)
	if momentum != 0.5 {
		t.Errorf("Expected 0.5 for empty signals, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_SingleComplexMessage(t *testing.T) {
	// Single complex message should escalate quickly (attack path)
	momentum := ComputeRoutingMomentum([]float64{0.95}, 0.2, 0.95)
	// 0.95 > 0.5 → attack: 0.2*0.5 + 0.8*0.95 = 0.86
	if !approxEqual(momentum, 0.86, 0.01) {
		t.Errorf("Expected ~0.86, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_SingleTrivialMessage(t *testing.T) {
	// Single trivial message should de-escalate slowly (release path)
	momentum := ComputeRoutingMomentum([]float64{0.05}, 0.2, 0.95)
	// 0.05 < 0.5 → release: 0.95*0.5 + 0.05*0.05 = 0.4775
	if !approxEqual(momentum, 0.4775, 0.01) {
		t.Errorf("Expected ~0.4775, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_GoodMorningThenKernel(t *testing.T) {
	// The scenario that broke symmetric momentum:
	// "good morning" → trivial, then "develop kernel" → complex
	// Attack should escalate quickly despite trivial history.
	signals := []float64{0.05, 0.95}
	momentum := ComputeRoutingMomentum(signals, 0.2, 0.95)

	// Turn 1: 0.05 < 0.5 → release: 0.95*0.5 + 0.05*0.05 = 0.4775
	// Turn 2: 0.95 > 0.4775 → attack: 0.2*0.4775 + 0.8*0.95 = 0.856
	if !approxEqual(momentum, 0.856, 0.01) {
		t.Errorf("Expected ~0.856 (fast escalation), got %f", momentum)
	}
	if momentum <= 0.5 {
		t.Errorf("Momentum should be above threshold 0.5 after complex message, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_ComplexThenCommitIt(t *testing.T) {
	// "implement LRU cache" → complex, then "commit it" → trivial
	// Release should keep momentum high.
	signals := []float64{0.92, 0.08}
	momentum := ComputeRoutingMomentum(signals, 0.2, 0.95)

	// Turn 1: 0.92 > 0.5 → attack: 0.2*0.5 + 0.8*0.92 = 0.836
	// Turn 2: 0.08 < 0.836 → release: 0.95*0.836 + 0.05*0.08 = 0.798
	if !approxEqual(momentum, 0.798, 0.01) {
		t.Errorf("Expected ~0.798 (slow release), got %f", momentum)
	}
	if momentum <= 0.5 {
		t.Errorf("Momentum should remain above 0.5 after 'commit it', got %f", momentum)
	}
}

func TestComputeRoutingMomentum_DecayOverManyTrivialTurns(t *testing.T) {
	// Complex question followed by many trivial follow-ups.
	// Momentum should eventually decay below threshold.
	// With release=0.95, it takes ~12 trivial turns to drop below 0.5.
	signals := []float64{0.95} // complex start
	for i := 0; i < 15; i++ {
		signals = append(signals, 0.05) // 15 trivial follow-ups
	}

	momentum := ComputeRoutingMomentum(signals, 0.2, 0.95)

	// After 15 trivial turns, momentum should drop below 0.5
	if momentum >= 0.5 {
		t.Errorf("Expected momentum below 0.5 after 15 trivial turns, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_SurvivesDevWorkflow(t *testing.T) {
	// Real dev workflow: complex task → 5 trivial follow-ups.
	// Momentum should stay above 0.5 throughout.
	signals := []float64{0.85, 0.1, 0.1, 0.1, 0.1, 0.1}
	momentum := ComputeRoutingMomentum(signals, 0.2, 0.95)

	if momentum <= 0.5 {
		t.Errorf("Momentum should survive 5 trivial follow-ups in dev workflow, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_AsymmetricBehavior(t *testing.T) {
	// Verify attack is faster than release.
	// Going up from 0.5 to 0.9 signal should move more than going down.
	upSignal := ComputeRoutingMomentum([]float64{0.9}, 0.2, 0.95)
	downSignal := ComputeRoutingMomentum([]float64{0.1}, 0.2, 0.95)

	upDelta := upSignal - 0.5     // how much it moved up
	downDelta := 0.5 - downSignal // how much it moved down

	if upDelta <= downDelta {
		t.Errorf("Attack (up delta=%.3f) should be faster than release (down delta=%.3f)",
			upDelta, downDelta)
	}
}

func TestComputeRoutingMomentum_ClampsInputs(t *testing.T) {
	// Signals outside [0,1] should be clamped
	momentum := ComputeRoutingMomentum([]float64{-0.5, 1.5}, 0.2, 0.95)
	if momentum < 0 || momentum > 1 {
		t.Errorf("Expected momentum in [0,1], got %f", momentum)
	}
}

// Backward compatibility tests — verify that when CRM is disabled (default),
// the routing behavior is identical to the current per-message routing.

func TestCRM_DisabledByDefault(t *testing.T) {
	// RoutingMomentumConfig zero value should have Enabled=false
	cfg := config.RoutingMomentumConfig{}
	if cfg.Enabled {
		t.Error("CRM should be disabled by default (zero value of bool)")
	}
}

func TestCRM_DisabledDoesNotAffectSignals(t *testing.T) {
	// When CRM is disabled, each message is evaluated independently.
	complexSignals := []float64{0.92}
	simpleSignals := []float64{0.08}

	complexMomentum := ComputeRoutingMomentum(complexSignals, 0.2, 0.95)
	simpleMomentum := ComputeRoutingMomentum(simpleSignals, 0.2, 0.95)

	if complexMomentum <= 0.5 {
		t.Errorf("Single complex message momentum should be > 0.5, got %f", complexMomentum)
	}
	if simpleMomentum >= 0.5 {
		t.Errorf("Single simple message momentum should be < 0.5, got %f", simpleMomentum)
	}
}

func TestCRM_EnabledFixesRouteBouncing(t *testing.T) {
	// With CRM enabled: complex → simple should maintain high momentum
	signals := []float64{0.92, 0.08}
	momentum := ComputeRoutingMomentum(signals, 0.2, 0.95)

	if momentum <= 0.5 {
		t.Errorf("CRM should keep momentum > 0.5 after complex→simple, got %f", momentum)
	}

	// Without CRM: simple message alone would route to cheap model
	simpleAlone := ComputeRoutingMomentum([]float64{0.08}, 0.2, 0.95)
	if simpleAlone >= 0.5 {
		t.Errorf("Without history, simple message should be < 0.5, got %f", simpleAlone)
	}
}

func TestCRM_SimpleThenComplex(t *testing.T) {
	// Simple → complex: CRM should escalate quickly (attack path)
	signals := []float64{0.08, 0.95}
	momentum := ComputeRoutingMomentum(signals, 0.2, 0.95)

	if momentum <= 0.5 {
		t.Errorf("CRM should escalate above 0.5 for simple→complex, got %f", momentum)
	}
}

func TestCRM_ConfigDefaults(t *testing.T) {
	cfg := config.RoutingMomentumConfig{}
	if cfg.GetAttack() != 0.2 {
		t.Errorf("Default attack should be 0.2, got %f", cfg.GetAttack())
	}
	if cfg.GetRelease() != 0.95 {
		t.Errorf("Default release should be 0.95, got %f", cfg.GetRelease())
	}
	if cfg.GetThreshold() != 0.5 {
		t.Errorf("Default threshold should be 0.5, got %f", cfg.GetThreshold())
	}
}

func TestFindEnabledMomentumConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *config.RouterConfig
		wantNil bool
	}{
		{name: "nil config", cfg: nil, wantNil: true},
		{name: "no decisions", cfg: &config.RouterConfig{}, wantNil: true},
		{
			name: "decision without momentum",
			cfg: &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{{
						Name:      "simple",
						Algorithm: &config.AlgorithmConfig{Type: "confidence"},
					}},
				},
			},
			wantNil: true,
		},
		{
			name: "decision with momentum enabled",
			cfg: &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{{
						Name: "complex",
						Algorithm: &config.AlgorithmConfig{
							Type:     "confidence",
							Momentum: &config.RoutingMomentumConfig{Enabled: true, Attack: 0.2},
						},
					}},
				},
			},
			wantNil: false,
		},
		{
			name: "decision with momentum disabled",
			cfg: &config.RouterConfig{
				IntelligentRouting: config.IntelligentRouting{
					Decisions: []config.Decision{{
						Name: "complex",
						Algorithm: &config.AlgorithmConfig{
							Type:     "confidence",
							Momentum: &config.RoutingMomentumConfig{Enabled: false},
						},
					}},
				},
			},
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findEnabledMomentumConfig(tt.cfg)
			if tt.wantNil && got != nil {
				t.Error("expected nil")
			}
			if !tt.wantNil && got == nil {
				t.Error("expected non-nil")
			}
		})
	}
}

func TestDecision_GetMomentumConfig(t *testing.T) {
	d := config.Decision{Name: "test"}
	if d.GetMomentumConfig() != nil {
		t.Error("expected nil when no algorithm")
	}

	d.Algorithm = &config.AlgorithmConfig{Type: "confidence"}
	if d.GetMomentumConfig() != nil {
		t.Error("expected nil when no momentum")
	}

	m := &config.RoutingMomentumConfig{Enabled: true}
	d.Algorithm.Momentum = m
	if d.GetMomentumConfig() != m {
		t.Error("expected momentum config")
	}
}

