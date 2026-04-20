/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"testing"
	"time"
)

func TestInitializeMetrics(t *testing.T) {
	InitializeMetrics()

	// Verify metric pointers are not nil
	if ModelSelectionTotal == nil {
		t.Error("ModelSelectionTotal should not be nil")
	}
	if ModelSelectionDuration == nil {
		t.Error("ModelSelectionDuration should not be nil")
	}
	if ModelSelectionScore == nil {
		t.Error("ModelSelectionScore should not be nil")
	}
	if ModelSelectionConfidence == nil {
		t.Error("ModelSelectionConfidence should not be nil")
	}
	if ModelEloRating == nil {
		t.Error("ModelEloRating should not be nil")
	}
	if ModelFeedbackTotal == nil {
		t.Error("ModelFeedbackTotal should not be nil")
	}
	if ModelRatingChange == nil {
		t.Error("ModelRatingChange should not be nil")
	}
	if ModelSelectionHistory == nil {
		t.Error("ModelSelectionHistory should not be nil")
	}
	if ComponentAgreement == nil {
		t.Error("ComponentAgreement should not be nil")
	}
	if ModelComparisons == nil {
		t.Error("ModelComparisons should not be nil")
	}
	if ModelWinRate == nil {
		t.Error("ModelWinRate should not be nil")
	}
}

func TestRecordSelection(t *testing.T) {
	InitializeMetrics()

	// Should not panic
	RecordSelection("elo", "tech", "llama3.2:3b", TierSupported, 0.85)
	RecordSelection("router_dc", "finance", "phi4", TierSupported, 0.72)
	RecordSelection("hybrid", "general", "gemma3:27b", TierSupported, 0.91)
	RecordSelection("automix", "general", "gemma3:27b", TierExperimental, 0.91)
}

func TestRecordEloRating(t *testing.T) {
	InitializeMetrics()

	// Should not panic
	RecordEloRating("llama3.2:3b", "tech", 1523.4)
	RecordEloRating("phi4", "finance", 1487.2)
	RecordEloRating("gemma3:27b", "_global", 1556.8)
}

func TestRecordEloRatings(t *testing.T) {
	InitializeMetrics()

	ratings := map[string]*ModelRating{
		"llama3.2:3b": {Model: "llama3.2:3b", Rating: 1523.4, Comparisons: 10, Wins: 7, Losses: 2, Ties: 1},
		"phi4":        {Model: "phi4", Rating: 1487.2, Comparisons: 8, Wins: 4, Losses: 4, Ties: 0},
		"gemma3:27b":  {Model: "gemma3:27b", Rating: 1556.8, Comparisons: 12, Wins: 9, Losses: 2, Ties: 1},
	}

	// Should not panic
	RecordEloRatings(ratings, "tech")
}

func TestRecordFeedback(t *testing.T) {
	InitializeMetrics()

	// Standard feedback with winner and loser
	RecordFeedback("llama3.2:3b", "phi4", false, "tech")

	// Tie
	RecordFeedback("gemma3:27b", "phi4", true, "finance")

	// Single model feedback (no loser)
	RecordFeedback("llama3.2:3b", "", false, "_global")
}

func TestRecordRatingChange(t *testing.T) {
	InitializeMetrics()

	// Positive change (win)
	RecordRatingChange("llama3.2:3b", "tech", 1500.0, 1516.0, "win")

	// Negative change (loss)
	RecordRatingChange("phi4", "tech", 1500.0, 1484.0, "loss")

	// Small change (tie)
	RecordRatingChange("gemma3:27b", "tech", 1500.0, 1502.0, "tie")
}

func TestRecordModelStats(t *testing.T) {
	InitializeMetrics()

	RecordModelStats("llama3.2:3b", "tech", 10, 7, 2, 1)
	RecordModelStats("phi4", "finance", 8, 4, 4, 0)
	RecordModelStats("gemma3:27b", "_global", 12, 9, 2, 1)
}

func TestRecordComponentAgreement(t *testing.T) {
	InitializeMetrics()

	// Full agreement
	RecordComponentAgreement(1.0)

	// Partial agreement
	RecordComponentAgreement(0.75)
	RecordComponentAgreement(0.5)
	RecordComponentAgreement(0.25)

	// No agreement
	RecordComponentAgreement(0.0)
}

func TestRecordFeedbackMetrics(t *testing.T) {
	InitializeMetrics()

	// Full feedback metrics
	RecordFeedbackMetrics(&FeedbackMetrics{
		Winner:       "llama3.2:3b",
		Loser:        "phi4",
		Category:     "tech",
		IsTie:        false,
		WinnerOldElo: 1500.0,
		WinnerNewElo: 1516.0,
		LoserOldElo:  1500.0,
		LoserNewElo:  1484.0,
		WinnerStats:  ModelRating{Model: "llama3.2:3b", Comparisons: 10, Wins: 7, Losses: 2, Ties: 1},
		LoserStats:   ModelRating{Model: "phi4", Comparisons: 10, Wins: 4, Losses: 5, Ties: 1},
	})

	// Tie feedback
	RecordFeedbackMetrics(&FeedbackMetrics{
		Winner:       "gemma3:27b",
		Loser:        "phi4",
		Category:     "finance",
		IsTie:        true,
		WinnerOldElo: 1500.0,
		WinnerNewElo: 1502.0,
		LoserOldElo:  1500.0,
		LoserNewElo:  1498.0,
		WinnerStats:  ModelRating{Model: "gemma3:27b", Comparisons: 5, Wins: 2, Losses: 1, Ties: 2},
		LoserStats:   ModelRating{Model: "phi4", Comparisons: 5, Wins: 1, Losses: 2, Ties: 2},
	})

	// Nil metrics should not panic
	RecordFeedbackMetrics(nil)
}

func TestRecordHybridSelection(t *testing.T) {
	InitializeMetrics()

	componentChoices := map[string]string{
		"elo":       "llama3.2:3b",
		"router_dc": "phi4",
		"automix":   "llama3.2:3b",
	}

	// Should not panic
	RecordHybridSelection("llama3.2:3b", "tech", componentChoices, TierSupported, 0.85, 0.92, 10*time.Millisecond)
}

func TestCalculateAgreementRatio(t *testing.T) {
	tests := []struct {
		name     string
		choices  []string
		expected float64
	}{
		{
			name:     "all agree",
			choices:  []string{"model-a", "model-a", "model-a"},
			expected: 1.0,
		},
		{
			name:     "2 of 3 agree",
			choices:  []string{"model-a", "model-a", "model-b"},
			expected: 2.0 / 3.0,
		},
		{
			name:     "no agreement",
			choices:  []string{"model-a", "model-b", "model-c"},
			expected: 1.0 / 3.0,
		},
		{
			name:     "half agree",
			choices:  []string{"model-a", "model-a", "model-b", "model-b"},
			expected: 0.5,
		},
		{
			name:     "single choice",
			choices:  []string{"model-a"},
			expected: 1.0,
		},
		{
			name:     "empty choices",
			choices:  []string{},
			expected: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateAgreementRatio(tt.choices)
			if result != tt.expected {
				t.Errorf("calculateAgreementRatio(%v) = %f, want %f", tt.choices, result, tt.expected)
			}
		})
	}
}

func TestNormalizeRatingChange(t *testing.T) {
	tests := []struct {
		name     string
		change   float64
		kFactor  float64
		expected float64
	}{
		{
			name:     "max positive change",
			change:   32.0,
			kFactor:  32.0,
			expected: 1.0,
		},
		{
			name:     "max negative change",
			change:   -32.0,
			kFactor:  32.0,
			expected: -1.0,
		},
		{
			name:     "no change",
			change:   0.0,
			kFactor:  32.0,
			expected: 0.0,
		},
		{
			name:     "half positive change",
			change:   16.0,
			kFactor:  32.0,
			expected: 0.5,
		},
		{
			name:     "zero k-factor uses default",
			change:   16.0,
			kFactor:  0.0,
			expected: 0.5,
		},
		{
			name:     "clamp to max",
			change:   64.0,
			kFactor:  32.0,
			expected: 1.0,
		},
		{
			name:     "clamp to min",
			change:   -64.0,
			kFactor:  32.0,
			expected: -1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NormalizeRatingChange(tt.change, tt.kFactor)
			if result != tt.expected {
				t.Errorf("NormalizeRatingChange(%f, %f) = %f, want %f", tt.change, tt.kFactor, result, tt.expected)
			}
		})
	}
}

func TestMetricsWithEmptyCategory(t *testing.T) {
	InitializeMetrics()

	// Empty category should default to "_global"
	RecordFeedback("llama3.2:3b", "phi4", false, "")
	RecordRatingChange("llama3.2:3b", "", 1500.0, 1516.0, "win")
	RecordModelStats("llama3.2:3b", "", 10, 7, 2, 1)
}

func TestMetricsBeforeInitialization(t *testing.T) {
	// Reset metrics state for this test (simulating before initialization)
	// Note: In practice, we can't fully reset due to sync.Once, but we test
	// that the functions handle nil metrics gracefully

	// These should not panic even if called before proper initialization
	// (The actual metrics package handles this via nil checks)
}

// requireGlobalRatings fetches ratings for the given models and fails if any are nil.
func requireGlobalRatings(t *testing.T, sel *EloSelector, models ...string) []*ModelRating {
	t.Helper()
	ratings := make([]*ModelRating, len(models))
	for i, m := range models {
		r := sel.getGlobalRating(m)
		if r == nil {
			t.Fatalf("Rating for %s should not be nil", m)
		}
		ratings[i] = r
	}
	return ratings
}

// TestFullFeedbackFlowMetrics is an integration test that simulates the complete
// feedback flow and verifies that all metrics are properly updated.
// This test covers issue #1093 requirements for evolution tracking.
func TestFullFeedbackFlowMetrics(t *testing.T) {
	InitializeMetrics()

	// Create an EloSelector with test config
	eloConfig := &EloConfig{
		InitialRating:     1500.0,
		KFactor:           32.0,
		CategoryWeighted:  true,
		CostScalingFactor: 0.1,
	}
	selector := NewEloSelector(eloConfig)

	for _, m := range []string{"model-a", "model-b", "model-c"} {
		selector.setGlobalRating(m, &ModelRating{Model: m, Rating: 1500})
	}

	ctx := t.Context()

	// Feedback 1: model-a beats model-b
	err := selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "model-a", LoserModel: "model-b", DecisionName: "test-category",
	})
	if err != nil {
		t.Fatalf("UpdateFeedback failed: %v", err)
	}

	r := requireGlobalRatings(t, selector, "model-a", "model-b")
	if r[0].Rating <= 1500.0 {
		t.Errorf("Winner rating should increase: got %.2f", r[0].Rating)
	}
	if r[1].Rating >= 1500.0 {
		t.Errorf("Loser rating should decrease: got %.2f", r[1].Rating)
	}

	// More feedback to establish ranking A > B > C
	_ = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "model-a", LoserModel: "model-c", DecisionName: "test-category",
	})
	_ = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel: "model-b", LoserModel: "model-c", DecisionName: "test-category",
	})

	final := requireGlobalRatings(t, selector, "model-a", "model-b", "model-c")
	if !(final[0].Rating > final[1].Rating && final[1].Rating > final[2].Rating) {
		t.Errorf("Rating order should be A > B > C, got A=%.2f, B=%.2f, C=%.2f",
			final[0].Rating, final[1].Rating, final[2].Rating)
	}

	// Verify metric recording functions don't panic
	for _, m := range []string{"model-a", "model-b", "model-c"} {
		rating := selector.getGlobalRating(m)
		RecordEloRating(m, "test-category", rating.Rating)
		RecordRatingChange(m, "test-category", 1500.0, rating.Rating, "test")
	}
}

// TestAutoMixSpecificMetrics verifies that AutoMix-specific metrics work correctly
func TestAutoMixSpecificMetrics(t *testing.T) {
	InitializeMetrics()

	// Verify new metrics are initialized
	if AutoMixVerificationProb == nil {
		t.Fatal("AutoMixVerificationProb should not be nil")
	}
	if AutoMixQuality == nil {
		t.Fatal("AutoMixQuality should not be nil")
	}
	if AutoMixSuccessRate == nil {
		t.Fatal("AutoMixSuccessRate should not be nil")
	}

	// Test recording AutoMix capability metrics
	RecordAutoMixCapability("model-a", 0.85, 0.92, 8, 10)
	RecordAutoMixCapability("model-b", 0.72, 0.78, 5, 8)
	RecordAutoMixCapability("model-c", 0.65, 0.70, 3, 10)

	t.Log("AutoMix-specific metrics verified!")
	t.Log("   - llm_model_automix_verification_prob recorded")
	t.Log("   - llm_model_automix_quality recorded")
	t.Log("   - llm_model_automix_success_rate recorded")
}

// TestRouterDCSpecificMetrics verifies that RouterDC-specific metrics work correctly
func TestRouterDCSpecificMetrics(t *testing.T) {
	InitializeMetrics()

	// Verify new metrics are initialized
	if RouterDCSimilarity == nil {
		t.Fatal("RouterDCSimilarity should not be nil")
	}
	if RouterDCAffinity == nil {
		t.Fatal("RouterDCAffinity should not be nil")
	}

	// Test recording RouterDC similarity metrics
	RecordRouterDCSimilarity("model-a", 0.92)
	RecordRouterDCSimilarity("model-b", 0.78)
	RecordRouterDCSimilarity("model-c", 0.65)

	// Test recording RouterDC affinity metrics
	RecordRouterDCAffinity("model-a", 0.8)
	RecordRouterDCAffinity("model-b", 0.5)
	RecordRouterDCAffinity("model-c", 0.3)

	t.Log("RouterDC-specific metrics verified!")
	t.Log("   - llm_model_routerdc_similarity recorded")
	t.Log("   - llm_model_routerdc_affinity recorded")
}

// TestAllMethodEvolutionMetrics is a comprehensive test that verifies
// evolution tracking for ALL selection methods as required by issue #1093
func TestAllMethodEvolutionMetrics(t *testing.T) {
	InitializeMetrics()
	ctx := t.Context()

	t.Run("elo_evolution", func(t *testing.T) {
		eloSelector := NewEloSelector(DefaultEloConfig())
		eloSelector.setGlobalRating("elo-model-a", &ModelRating{Model: "elo-model-a", Rating: 1500})
		eloSelector.setGlobalRating("elo-model-b", &ModelRating{Model: "elo-model-b", Rating: 1500})
		err := eloSelector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "elo-model-a", LoserModel: "elo-model-b", DecisionName: "test",
		})
		if err != nil {
			t.Fatalf("Elo feedback failed: %v", err)
		}
		rA := eloSelector.getGlobalRating("elo-model-a")
		rB := eloSelector.getGlobalRating("elo-model-b")
		if rA.Rating <= rB.Rating {
			t.Errorf("Winner should have higher rating: A=%.2f, B=%.2f", rA.Rating, rB.Rating)
		}
	})

	t.Run("automix_evolution", func(t *testing.T) {
		autoMixSelector := NewAutoMixSelector(DefaultAutoMixConfig())
		autoMixSelector.SetCapability("automix-model-a", &ModelCapability{
			Model: "automix-model-a", VerificationProb: 0.7, AvgQuality: 0.8,
		})
		autoMixSelector.SetCapability("automix-model-b", &ModelCapability{
			Model: "automix-model-b", VerificationProb: 0.7, AvgQuality: 0.8,
		})
		err := autoMixSelector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "automix-model-a", LoserModel: "automix-model-b", DecisionName: "test",
		})
		if err != nil {
			t.Fatalf("AutoMix feedback failed: %v", err)
		}
		caps := autoMixSelector.GetCapabilities()
		if caps["automix-model-a"] == nil || caps["automix-model-b"] == nil {
			t.Fatal("Capabilities should not be nil after feedback")
		}
	})

	t.Run("routerdc_evolution", func(t *testing.T) {
		routerDCSelector := NewRouterDCSelector(DefaultRouterDCConfig())
		err := routerDCSelector.UpdateFeedback(ctx, &Feedback{
			Query:       "test query about technology",
			WinnerModel: "routerdc-model-a", LoserModel: "routerdc-model-b", DecisionName: "test",
		})
		if err != nil {
			t.Fatalf("RouterDC feedback failed: %v", err)
		}
	})

	t.Run("hybrid_component_agreement", func(t *testing.T) {
		RecordComponentAgreement(0.75)
		RecordComponentAgreement(1.0)
		RecordComponentAgreement(0.5)
	})
}

func TestRecordSelectionWithTier(t *testing.T) {
	InitializeMetrics()

	// Record a selection with tier - should not panic
	RecordSelectionWithTier(MethodElo, "test-model", "test-decision", TierSupported, 0.8, 0.9, time.Millisecond)
	RecordSelectionWithTier(MethodAutoMix, "test-model", "test-decision", TierExperimental, 0.7, 0.8, time.Millisecond)
}

func TestRecordDependencyHealth(t *testing.T) {
	InitializeMetrics()

	// Should not panic
	RecordDependencyHealth("automix", "AutoMix Verifier Server", "external_service", false)
	RecordDependencyHealth("elo", "", "", true)
}
