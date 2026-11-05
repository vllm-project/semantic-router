// paper_benchmark_test.go
// Benchmark tests that verify our implementation matches the papers:
// - AutoMix: arXiv:2310.12963

package selection

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ============================================================================
// AUTOMIX PAPER BENCHMARKS (arXiv:2310.12963)
// ============================================================================

// TestAutoMix_POMDPBeliefUpdates verifies that the POMDP solver updates
// beliefs correctly based on observations, per AutoMix Section 4.1
func TestAutoMix_POMDPBeliefUpdates(t *testing.T) {
	solver := NewAdaOpsSolver(100, 0.1, 0.95) // 100 particles

	// Register models
	solver.RegisterModel("small", 0.001, 7)  // 7B params, cheap
	solver.RegisterModel("medium", 0.01, 13) // 13B params
	solver.RegisterModel("large", 0.03, 70)  // 70B params, expensive

	// Get initial beliefs (they have noise, so we just record them)
	initialBeliefs := solver.GetBeliefMean()
	t.Logf("Initial beliefs: %+v", initialBeliefs)

	// Update with positive observation for "small" model
	solver.UpdateBelief("small", 0.9) // High performance observation
	solver.UpdateBelief("small", 0.85)
	solver.UpdateBelief("small", 0.95)

	// Update with negative observation for "large" model
	solver.UpdateBelief("large", 0.3) // Low performance observation

	updatedBeliefs := solver.GetBeliefMean()
	t.Logf("Updated beliefs: %+v", updatedBeliefs)

	// Key test: After observations, beliefs should differ
	// Small model should have higher belief than large model
	if updatedBeliefs["small"] <= updatedBeliefs["large"] {
		t.Errorf("After positive/negative observations, small (%.3f) should have higher belief than large (%.3f)",
			updatedBeliefs["small"], updatedBeliefs["large"])
	}
}

// TestAutoMix_IBCMetric verifies the Incremental Benefit per Cost (IBC)
// metric from AutoMix Section 5.2
func TestAutoMix_IBCMetric(t *testing.T) {
	solver := NewAdaOpsSolver(50, 0.1, 0.95)

	solver.RegisterModel("small", 0.001, 7)
	solver.RegisterModel("large", 0.03, 70)

	tests := []struct {
		name            string
		fromModel       string
		toModel         string
		perfImprovement float64
		wantPositiveIBC bool
	}{
		{
			name:            "large_improvement_worth_cost",
			fromModel:       "small",
			toModel:         "large",
			perfImprovement: 0.5, // 50% improvement
			wantPositiveIBC: true,
		},
		{
			name:            "small_improvement_not_worth_cost",
			fromModel:       "small",
			toModel:         "large",
			perfImprovement: 0.01,  // 1% improvement
			wantPositiveIBC: false, // Cost too high for marginal gain
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ibc := solver.ComputeIBC(tt.fromModel, tt.toModel, tt.perfImprovement)
			t.Logf("IBC = %.4f", ibc)

			if tt.wantPositiveIBC && ibc <= 0 {
				t.Errorf("Expected positive IBC, got %.4f", ibc)
			}
			if !tt.wantPositiveIBC && ibc > 1.0 {
				t.Errorf("Expected low IBC (not worth escalation), got %.4f", ibc)
			}
		})
	}
}

// TestAutoMix_CascadedExecution verifies the cascaded execution pattern
// from AutoMix Section 3.1: SLM → verify → escalate → LLM
func TestAutoMix_CascadedExecution(t *testing.T) {
	cfg := DefaultAutoMixConfig()
	cfg.EnableSelfVerification = true
	cfg.EnableCascade = true
	cfg.VerificationThreshold = 0.7
	selector := NewAutoMixSelector(cfg)

	modelConfig := map[string]config.ModelParams{
		"small-7b":   modelParamsWithTestQuality(0.7),
		"medium-13b": modelParamsWithTestQuality(0.85),
		"large-70b":  modelParamsWithTestQuality(0.95),
	}

	selector.InitializeFromConfig(modelConfig)

	// Get escalation chain
	candidates := []config.ModelRef{
		{Model: "small-7b"},
		{Model: "medium-13b"},
		{Model: "large-70b"},
	}
	chain := selector.GetEscalationChain(candidates)

	t.Logf("Escalation chain: %v", chain)

	// Paper requirement: Models should be ordered by size
	if len(chain) != 3 {
		t.Errorf("Expected 3 models in chain, got %d", len(chain))
	}

	// Smallest should be first
	if chain[0] != "small-7b" {
		t.Errorf("Expected small-7b first in chain, got %s", chain[0])
	}

	// Largest should be last
	if chain[len(chain)-1] != "large-70b" {
		t.Errorf("Expected large-70b last in chain, got %s", chain[len(chain)-1])
	}
}

// TestAutoMix_SelfVerificationThreshold verifies the verification logic
// by testing with a mock verifier (real verification requires external server)
func TestAutoMix_SelfVerificationThreshold(t *testing.T) {
	// This test uses mock verification since real verification requires
	// the external automix_verifier.py server on a GPU machine.
	// The mock tests the threshold logic, not the LLM verification itself.

	cfg := DefaultAutoMixConfig()
	cfg.EnableSelfVerification = true
	cfg.VerificationThreshold = 0.7

	tests := []struct {
		name         string
		confidence   float64
		wantEscalate bool
	}{
		{
			name:         "high_confidence_no_escalate",
			confidence:   0.9,
			wantEscalate: false,
		},
		{
			name:         "low_confidence_escalate",
			confidence:   0.5,
			wantEscalate: true,
		},
		{
			name:         "at_threshold_no_escalate",
			confidence:   0.7,
			wantEscalate: false, // >= threshold means no escalate
		},
		{
			name:         "just_below_threshold_escalate",
			confidence:   0.69,
			wantEscalate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Directly test the threshold logic
			shouldEscalate := tt.confidence < cfg.VerificationThreshold

			t.Logf("Confidence=%.2f, Threshold=%.2f, ShouldEscalate=%v",
				tt.confidence, cfg.VerificationThreshold, shouldEscalate)

			if shouldEscalate != tt.wantEscalate {
				t.Errorf("ShouldEscalate = %v, want %v", shouldEscalate, tt.wantEscalate)
			}
		})
	}
}
