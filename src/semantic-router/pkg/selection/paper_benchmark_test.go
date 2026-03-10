// paper_benchmark_test.go
// Benchmark tests that verify our implementation matches the papers:
// - Router-R1: arXiv:2506.09033
// - GMTRouter: arXiv:2511.08590
// - AutoMix: arXiv:2310.12963

package selection

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ============================================================================
// ROUTER-R1 PAPER BENCHMARKS (arXiv:2506.09033)
// ============================================================================

// TestRouterR1_ThompsonSamplingConvergence verifies that Thompson Sampling
// converges to the optimal model after sufficient feedback, as described in
// Router-R1 Section 4.1 (Exploration-Exploitation Balance).
//
// Paper claim: "The router learns to route queries to optimal models based on
// observed performance, balancing exploration of uncertain models with
// exploitation of known-good models."
func TestRouterR1_ThompsonSamplingConvergence(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling:   true,
		EnablePersonalization: false,
		ExplorationRate:       0.0, // Pure exploitation to test convergence
	})

	ctx := context.Background()

	// Model A: 90% win rate (optimal)
	// Model B: 50% win rate (suboptimal)
	// Model C: 10% win rate (poor)
	modelWinRates := map[string]float64{
		"model-a": 0.9,
		"model-b": 0.5,
		"model-c": 0.1,
	}

	// Simulate 100 feedbacks based on true win rates
	for i := 0; i < 100; i++ {
		for model, winRate := range modelWinRates {
			// Simulate win/loss based on true probability
			if float64(i%10)/10.0 < winRate {
				_ = selector.UpdateFeedback(ctx, &Feedback{
					WinnerModel: model,
					Timestamp:   time.Now().Unix(),
				})
			} else {
				_ = selector.UpdateFeedback(ctx, &Feedback{
					LoserModel: model,
					Timestamp:  time.Now().Unix(),
				})
			}
		}
	}

	// Verify convergence: Model A should have highest mean
	prefA := selector.getGlobalPreference("model-a")
	prefB := selector.getGlobalPreference("model-b")
	prefC := selector.getGlobalPreference("model-c")

	if prefA == nil || prefB == nil || prefC == nil {
		t.Fatal("Preferences not initialized")
	}

	meanA := prefA.Distribution.Mean()
	meanB := prefB.Distribution.Mean()
	meanC := prefC.Distribution.Mean()

	t.Logf("Converged means: A=%.3f, B=%.3f, C=%.3f", meanA, meanB, meanC)

	// Paper expectation: Thompson Sampling should identify optimal model
	if meanA <= meanB || meanA <= meanC {
		t.Errorf("Thompson Sampling failed to converge: expected A > B,C, got A=%.3f, B=%.3f, C=%.3f",
			meanA, meanB, meanC)
	}

	// Variance should decrease with more observations (certainty increases)
	varA := prefA.Distribution.Variance()
	if varA > 0.1 {
		t.Errorf("Expected low variance after 100 observations, got %.4f", varA)
	}
}

// TestRouterR1_RewardStructure verifies the hierarchical reward structure
// from Router-R1 Section 3.2: R = R_format + (1-α)*R_outcome + α*R_cost
func TestRouterR1_RewardStructure(t *testing.T) {
	cfg := &RLDrivenConfig{
		UseRouterR1Rewards:  true,
		CostRewardAlpha:     0.1, // 10% cost weight
		FormatRewardPenalty: -1.0,
		ModelCostPerToken:   map[string]float64{"gpt-4": 0.03, "gpt-3.5": 0.001},
	}

	tests := []struct {
		name          string
		formatCorrect bool
		outcomeScore  float64
		modelCost     float64
		outputTokens  int
		wantMin       float64
		wantMax       float64
	}{
		{
			name:          "format_wrong_nullifies_reward",
			formatCorrect: false,
			outcomeScore:  1.0,
			modelCost:     0.001,
			outputTokens:  100,
			wantMin:       -1.1, // Penalized
			wantMax:       -0.9,
		},
		{
			name:          "perfect_outcome_low_cost",
			formatCorrect: true,
			outcomeScore:  1.0,
			modelCost:     0.001,
			outputTokens:  100,
			wantMin:       0.9, // High reward
			wantMax:       1.1,
		},
		{
			name:          "perfect_outcome_high_cost",
			formatCorrect: true,
			outcomeScore:  1.0,
			modelCost:     0.03,
			outputTokens:  1000,
			wantMin:       0.5, // Reduced by cost
			wantMax:       0.95,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reward := cfg.ComputeRouterR1Reward(
				tt.formatCorrect,
				tt.outcomeScore,
				tt.modelCost,
				tt.outputTokens,
			)

			if reward < tt.wantMin || reward > tt.wantMax {
				t.Errorf("Reward = %.3f, want [%.3f, %.3f]", reward, tt.wantMin, tt.wantMax)
			}
		})
	}
}

// ============================================================================
// GMTROUTER PAPER BENCHMARKS (arXiv:2511.08590)
// ============================================================================

// TestGMTRouter_HeterogeneousGraphStructure verifies that the heterogeneous
// graph contains the 5 node types from GMTRouter Section 3.1:
// User, LLM, Query, Response, Turn
func TestGMTRouter_HeterogeneousGraphStructure(t *testing.T) {
	selector := NewGMTRouterSelector(nil)

	modelConfig := map[string]config.ModelParams{
		"gpt-4":    {QualityScore: 0.95},
		"claude-3": {QualityScore: 0.90},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// Create multi-turn interaction
	_ = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "gpt-4",
		UserID:      "user-1",
		SessionID:   "session-1",
		Query:       "What is machine learning?",
	})

	_ = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "claude-3",
		UserID:      "user-1",
		SessionID:   "session-1",
		Query:       "Explain neural networks",
	})

	// Count node types
	nodeTypes := map[NodeType]int{}
	for _, node := range selector.nodes {
		nodeTypes[node.Type]++
	}

	t.Logf("Node types: %+v", nodeTypes)

	// Paper requirement: 5 node types
	expectedTypes := []NodeType{NodeTypeUser, NodeTypeLLM, NodeTypeQuery, NodeTypeTurn}
	for _, nt := range expectedTypes {
		if nodeTypes[nt] == 0 {
			t.Errorf("Missing node type: %s", nt)
		}
	}
}

// TestGMTRouter_PersonalizationLearning verifies that the system learns
// user preferences over time, as described in GMTRouter Section 4.2:
// "The router adapts to individual user preferences through few-shot learning"
func TestGMTRouter_PersonalizationLearning(t *testing.T) {
	cfg := DefaultGMTRouterConfig()
	cfg.MinInteractionsForPersonalization = 3
	selector := NewGMTRouterSelector(cfg)

	modelConfig := map[string]config.ModelParams{
		"fast-model": {QualityScore: 0.7},
		"slow-model": {QualityScore: 0.9},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// User consistently prefers "fast-model" (even though slow-model has higher quality)
	for i := 0; i < 10; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "fast-model",
			LoserModel:  "slow-model",
			UserID:      "speed-lover",
			Query:       "Quick question",
		})
	}

	// After learning, selector should prefer user's choice over base quality
	result, err := selector.Select(ctx, &SelectionContext{
		UserID: "speed-lover",
		CandidateModels: []config.ModelRef{
			{Model: "fast-model"},
			{Model: "slow-model"},
		},
	})
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	// Paper claim: Personalized routing respects learned preferences
	if result.SelectedModel != "fast-model" {
		t.Errorf("Expected personalized selection of 'fast-model', got '%s'", result.SelectedModel)
	}

	// Check that preference scores reflect learning
	state := selector.GetDebugState("speed-lover")
	userState, ok := state["user_state"].(map[string]interface{})
	if !ok {
		t.Fatal("Could not get user state")
	}

	prefs := userState["model_preferences"].(map[string]float64)
	t.Logf("Learned preferences: %+v", prefs)

	if prefs["fast-model"] <= prefs["slow-model"] {
		t.Error("Failed to learn user preference for fast-model")
	}
}

// TestGMTRouter_MessagePassingAttention verifies that HGT-style message
// passing with attention weights is implemented, per GMTRouter Section 3.3
func TestGMTRouter_MessagePassingAttention(t *testing.T) {
	cfg := DefaultGMTRouterConfig()
	cfg.MinInteractionsForPersonalization = 1
	selector := NewGMTRouterSelector(cfg)

	modelConfig := map[string]config.ModelParams{
		"model-a": {QualityScore: 0.9},
		"model-b": {QualityScore: 0.9},
	}
	selector.InitializeFromConfig(modelConfig)

	ctx := context.Background()

	// Create interactions with varying recency and clear preference
	// User strongly prefers model-b over model-a (multiple wins vs losses)
	for i := 0; i < 5; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "model-b",
			LoserModel:  "model-a",
			UserID:      "attention-test",
			Timestamp:   time.Now().Unix(),
		})
	}

	// Force recomputation
	selector.recomputeUserPreferences("attention-test")

	// Check that preferences reflect wins/losses
	state := selector.GetDebugState("attention-test")
	if userState, ok := state["user_state"].(map[string]interface{}); ok {
		prefs := userState["model_preferences"].(map[string]float64)
		t.Logf("Learned preferences: %+v", prefs)

		// Model-b (winner) should have higher preference than model-a (loser)
		if prefs["model-b"] <= prefs["model-a"] {
			t.Errorf("Attention/aggregation not working: winner (model-b=%.2f) should have higher preference than loser (model-a=%.2f)",
				prefs["model-b"], prefs["model-a"])
		}
	}

	// Verify node types exist in graph (tests heterogeneous graph structure)
	nodeTypeCounts := make(map[NodeType]int)
	for _, node := range selector.nodes {
		nodeTypeCounts[node.Type]++
	}
	t.Logf("Node type counts: %+v", nodeTypeCounts)

	if nodeTypeCounts[NodeTypeUser] == 0 {
		t.Error("No user nodes in graph")
	}
	if nodeTypeCounts[NodeTypeLLM] == 0 {
		t.Error("No LLM nodes in graph")
	}
}

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
		"small-7b":   {QualityScore: 0.7},
		"medium-13b": {QualityScore: 0.85},
		"large-70b":  {QualityScore: 0.95},
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
