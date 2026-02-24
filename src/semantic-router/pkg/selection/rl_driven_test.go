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
	"context"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBetaDistribution_Sample(t *testing.T) {
	tests := []struct {
		name     string
		alpha    float64
		beta     float64
		wantMean float64
		epsilon  float64
	}{
		{
			name:     "uniform prior (1,1)",
			alpha:    1.0,
			beta:     1.0,
			wantMean: 0.5,
			epsilon:  0.1,
		},
		{
			name:     "strong preference (10,1)",
			alpha:    10.0,
			beta:     1.0,
			wantMean: 10.0 / 11.0,
			epsilon:  0.1,
		},
		{
			name:     "weak preference (1,10)",
			alpha:    1.0,
			beta:     10.0,
			wantMean: 1.0 / 11.0,
			epsilon:  0.1,
		},
		{
			name:     "balanced (5,5)",
			alpha:    5.0,
			beta:     5.0,
			wantMean: 0.5,
			epsilon:  0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dist := &BetaDistribution{Alpha: tt.alpha, Beta: tt.beta}
			rng := rand.New(rand.NewSource(42))

			// Sample many times and compute average
			numSamples := 10000
			sum := 0.0
			for i := 0; i < numSamples; i++ {
				sum += dist.Sample(rng)
			}
			avg := sum / float64(numSamples)

			if math.Abs(avg-tt.wantMean) > tt.epsilon {
				t.Errorf("Sample() average = %v, want mean close to %v (epsilon=%v)", avg, tt.wantMean, tt.epsilon)
			}
		})
	}
}

func TestBetaDistribution_Mean(t *testing.T) {
	tests := []struct {
		name  string
		alpha float64
		beta  float64
		want  float64
	}{
		{"uniform", 1.0, 1.0, 0.5},
		{"strong alpha", 10.0, 2.0, 10.0 / 12.0},
		{"strong beta", 2.0, 10.0, 2.0 / 12.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dist := &BetaDistribution{Alpha: tt.alpha, Beta: tt.beta}
			got := dist.Mean()
			if math.Abs(got-tt.want) > 0.001 {
				t.Errorf("Mean() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewRLDrivenSelector(t *testing.T) {
	// Test with default config
	selector := NewRLDrivenSelector(nil)
	if selector == nil {
		t.Fatal("NewRLDrivenSelector returned nil")
	}
	if selector.Method() != MethodRLDriven {
		t.Errorf("Method() = %v, want %v", selector.Method(), MethodRLDriven)
	}

	// Test with custom config
	cfg := &RLDrivenConfig{
		ExplorationRate:       0.5,
		UseThompsonSampling:   true,
		EnablePersonalization: true,
		PersonalizationBlend:  0.8,
	}
	selector = NewRLDrivenSelector(cfg)
	if selector.config.ExplorationRate != 0.5 {
		t.Errorf("ExplorationRate = %v, want 0.5", selector.config.ExplorationRate)
	}
}

func TestRLDrivenSelector_Select_BasicFunctionality(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling:   true,
		EnablePersonalization: false,
	})

	candidates := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
		{Model: "model-c"},
	}

	ctx := context.Background()
	selCtx := &SelectionContext{
		Query:           "test query",
		DecisionName:    "test-decision",
		CandidateModels: candidates,
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}

	if result == nil {
		t.Fatal("Select() returned nil result")
	}

	if result.SelectedModel == "" {
		t.Error("SelectedModel is empty")
	}

	if result.Method != MethodRLDriven {
		t.Errorf("Method = %v, want %v", result.Method, MethodRLDriven)
	}

	// Verify selected model is one of the candidates
	found := false
	for _, c := range candidates {
		if c.Model == result.SelectedModel {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("SelectedModel %s not in candidates", result.SelectedModel)
	}
}

func TestRLDrivenSelector_UpdateFeedback(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling:   true,
		EnablePersonalization: true,
	})

	ctx := context.Background()

	// Test winner-only feedback (positive self-feedback)
	err := selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel:  "model-a",
		DecisionName: "test",
		Timestamp:    time.Now().Unix(),
	})
	if err != nil {
		t.Fatalf("UpdateFeedback() error = %v", err)
	}

	// Verify preference was updated
	pref := selector.getGlobalPreference("model-a")
	if pref == nil {
		t.Fatal("Global preference not created for model-a")
	}
	if pref.Distribution.Alpha <= 1.0 {
		t.Error("Alpha should have increased after positive feedback")
	}

	// Test comparison feedback
	err = selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel:  "model-a",
		LoserModel:   "model-b",
		DecisionName: "test",
		Timestamp:    time.Now().Unix(),
	})
	if err != nil {
		t.Fatalf("UpdateFeedback() error = %v", err)
	}

	// Verify both models were updated
	prefA := selector.getGlobalPreference("model-a")
	prefB := selector.getGlobalPreference("model-b")

	if prefA == nil || prefB == nil {
		t.Fatal("Preferences not created for both models")
	}

	// Winner should have higher alpha, loser should have higher beta
	if prefB.Distribution.Beta <= 1.0 {
		t.Error("Beta should have increased for loser")
	}
}

func TestRLDrivenSelector_Personalization(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling:   true,
		EnablePersonalization: true,
		PersonalizationBlend:  0.7,
	})

	ctx := context.Background()

	// Add user-specific feedback
	err := selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel:  "model-a",
		LoserModel:   "model-b",
		UserID:       "user-123",
		DecisionName: "test",
		Timestamp:    time.Now().Unix(),
	})
	if err != nil {
		t.Fatalf("UpdateFeedback() error = %v", err)
	}

	// Verify user preference was created
	userPref := selector.getUserPreference("user-123", "model-a")
	if userPref == nil {
		t.Fatal("User preference not created for user-123/model-a")
	}

	// Add many feedbacks to build strong preference for user
	for i := 0; i < 10; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel:  "model-a",
			LoserModel:   "model-b",
			UserID:       "user-123",
			DecisionName: "test",
			Timestamp:    time.Now().Unix(),
		})
	}

	// Selection with user context should reflect personalization
	candidates := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
	}

	selCtx := &SelectionContext{
		Query:           "test query",
		DecisionName:    "test",
		UserID:          "user-123",
		CandidateModels: candidates,
	}

	// Run multiple selections to check preference
	modelACounts := 0
	for i := 0; i < 100; i++ {
		result, err := selector.Select(ctx, selCtx)
		if err != nil {
			t.Fatalf("Select() error = %v", err)
		}
		if result.SelectedModel == "model-a" {
			modelACounts++
		}
	}

	// Model-a should be selected more often due to positive feedback
	if modelACounts < 60 {
		t.Errorf("Expected model-a to be selected more often (got %d/100), personalization may not be working", modelACounts)
	}
}

func TestRLDrivenSelector_ImplicitFeedback(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling:    true,
		EnablePersonalization:  true,
		ImplicitFeedbackWeight: 0.5,
	})

	ctx := context.Background()

	// Add implicit feedback (simulating auto-detected satisfaction)
	err := selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel:  "model-a",
		DecisionName: "test",
		FeedbackType: "satisfied",
		Confidence:   0.8,
		Timestamp:    time.Now().Unix(),
	})
	if err != nil {
		t.Fatalf("UpdateFeedback() error = %v", err)
	}

	// Verify implicit feedback count increased
	pref := selector.getGlobalPreference("model-a")
	if pref == nil {
		t.Fatal("Preference not created for model-a")
	}
	if pref.ImplicitFeedbackCount != 1 {
		t.Errorf("ImplicitFeedbackCount = %d, want 1", pref.ImplicitFeedbackCount)
	}

	// The update should be weighted (0.5 * 0.8 = 0.4)
	// So alpha should increase by ~0.4 instead of 1.0
	expectedAlphaIncrease := 0.5 * 0.8
	actualAlpha := pref.Distribution.Alpha - 1.0 // Subtract prior
	if math.Abs(actualAlpha-expectedAlphaIncrease) > 0.01 {
		t.Errorf("Alpha increase = %v, want ~%v", actualAlpha, expectedAlphaIncrease)
	}
}

func TestRLDrivenSelector_SessionContext(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling:  true,
		SessionContextWeight: 0.3,
	})

	ctx := context.Background()

	// Add session-specific feedback
	err := selector.UpdateFeedback(ctx, &Feedback{
		WinnerModel:  "model-a",
		LoserModel:   "model-b",
		SessionID:    "session-abc",
		DecisionName: "test",
		Timestamp:    time.Now().Unix(),
	})
	if err != nil {
		t.Fatalf("UpdateFeedback() error = %v", err)
	}

	// Verify session context was updated
	selector.sessionMu.RLock()
	sessionStats := selector.sessionContext["session-abc"]
	selector.sessionMu.RUnlock()

	if sessionStats == nil {
		t.Fatal("Session context not created")
	}

	if _, ok := sessionStats["model-a"]; !ok {
		t.Error("Model-a not found in session context")
	}

	if sessionStats["model-a"].Successes != 1 {
		t.Errorf("Expected 1 success for model-a in session, got %d", sessionStats["model-a"].Successes)
	}
}

func TestRLDrivenSelector_ExplorationDecay(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		ExplorationRate:  0.5,
		ExplorationDecay: 0.9,
		MinExploration:   0.1,
	})

	// At selection 0, rate should be 0.5
	rate0 := selector.getCurrentExplorationRate(0)
	if math.Abs(rate0-0.5) > 0.01 {
		t.Errorf("Rate at 0 = %v, want 0.5", rate0)
	}

	// At selection 100, rate should be 0.5 * 0.9 = 0.45
	rate100 := selector.getCurrentExplorationRate(100)
	expectedRate100 := 0.5 * 0.9 // 0.9^1 = 0.9
	if math.Abs(rate100-expectedRate100) > 0.01 {
		t.Errorf("Rate at 100 = %v, want ~%v", rate100, expectedRate100)
	}

	// At very high selection count, should hit minimum
	rateHigh := selector.getCurrentExplorationRate(10000)
	if rateHigh < 0.1 {
		t.Errorf("Rate at 10000 = %v, should not go below min %v", rateHigh, 0.1)
	}
}

func TestRLDrivenSelector_CostAwareness(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling: true,
		CostAwareness:       true,
		CostWeight:          0.3,
	})

	// Set different costs for models
	selector.SetModelCost("cheap-model", 0.5)
	selector.SetModelCost("expensive-model", 30.0)

	// Test cost bonus calculation
	baseScore := 0.5

	cheapScore := selector.applyCostBonus("cheap-model", baseScore)
	expensiveScore := selector.applyCostBonus("expensive-model", baseScore)

	// Cheap model should get a bonus
	if cheapScore <= baseScore {
		t.Error("Cheap model should get cost bonus")
	}

	// Expensive model should get less/no bonus
	if cheapScore <= expensiveScore {
		t.Error("Cheap model score should be higher than expensive model after cost adjustment")
	}
}

func TestRLDrivenSelector_GetLeaderboard(t *testing.T) {
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling: true,
	})

	ctx := context.Background()

	// Add feedback to create preferences
	for i := 0; i < 10; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "model-a",
			LoserModel:  "model-b",
			Timestamp:   time.Now().Unix(),
		})
	}

	for i := 0; i < 3; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "model-b",
			LoserModel:  "model-a",
			Timestamp:   time.Now().Unix(),
		})
	}

	// Get global leaderboard
	leaderboard := selector.GetLeaderboard("")

	if len(leaderboard) < 2 {
		t.Fatalf("Expected at least 2 models in leaderboard, got %d", len(leaderboard))
	}

	// First model should have highest mean
	if leaderboard[0].Model != "model-a" {
		t.Errorf("Expected model-a to be #1 in leaderboard, got %s", leaderboard[0].Model)
	}

	// Verify ordering is by mean descending
	for i := 1; i < len(leaderboard); i++ {
		if leaderboard[i].Distribution.Mean() > leaderboard[i-1].Distribution.Mean() {
			t.Error("Leaderboard not sorted by mean descending")
		}
	}
}

func TestRLDrivenSelector_EmptyCandidates(t *testing.T) {
	selector := NewRLDrivenSelector(nil)

	ctx := context.Background()
	selCtx := &SelectionContext{
		Query:           "test",
		CandidateModels: []config.ModelRef{},
	}

	_, err := selector.Select(ctx, selCtx)
	if err == nil {
		t.Error("Expected error for empty candidates")
	}
}

func TestRLDrivenSelector_InvalidFeedback(t *testing.T) {
	selector := NewRLDrivenSelector(nil)

	ctx := context.Background()

	// Both winner and loser empty should fail
	err := selector.UpdateFeedback(ctx, &Feedback{
		DecisionName: "test",
		Timestamp:    time.Now().Unix(),
	})
	if err == nil {
		t.Error("Expected error for feedback with no winner/loser")
	}
}

func TestRLDrivenSelector_EpsilonGreedy(t *testing.T) {
	// Test epsilon-greedy mode (non-Thompson Sampling)
	selector := NewRLDrivenSelector(&RLDrivenConfig{
		UseThompsonSampling: false,
		ExplorationRate:     0.3,
		MinExploration:      0.1,
	})

	ctx := context.Background()

	// Build up strong preference for model-a
	for i := 0; i < 20; i++ {
		_ = selector.UpdateFeedback(ctx, &Feedback{
			WinnerModel: "model-a",
			LoserModel:  "model-b",
			Timestamp:   time.Now().Unix(),
		})
	}

	candidates := []config.ModelRef{
		{Model: "model-a"},
		{Model: "model-b"},
	}

	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: candidates,
	}

	// Run many selections
	modelACounts := 0
	for i := 0; i < 100; i++ {
		result, err := selector.Select(ctx, selCtx)
		if err != nil {
			t.Fatalf("Select() error = %v", err)
		}
		if result.SelectedModel == "model-a" {
			modelACounts++
		}
	}

	// With epsilon-greedy, should exploit model-a most of the time
	// but still explore sometimes (about 30% exploration rate initially,
	// but decays, so expect model-a to be selected more than 60% of the time)
	if modelACounts < 50 {
		t.Errorf("Expected model-a to be selected more often with epsilon-greedy, got %d/100", modelACounts)
	}
}

func TestRLDrivenSelector_BlendPreferences(t *testing.T) {
	selector := NewRLDrivenSelector(nil)

	pref1 := &ModelPreference{
		Model: "test",
		Distribution: BetaDistribution{
			Alpha: 10.0,
			Beta:  2.0,
		},
	}

	pref2 := &ModelPreference{
		Model: "test",
		Distribution: BetaDistribution{
			Alpha: 2.0,
			Beta:  10.0,
		},
	}

	// Blend at 0.5
	blended := selector.blendPreferences(pref1, pref2, 0.5)

	expectedAlpha := 0.5*10.0 + 0.5*2.0 // 6.0
	expectedBeta := 0.5*2.0 + 0.5*10.0  // 6.0

	if math.Abs(blended.Distribution.Alpha-expectedAlpha) > 0.01 {
		t.Errorf("Blended Alpha = %v, want %v", blended.Distribution.Alpha, expectedAlpha)
	}

	if math.Abs(blended.Distribution.Beta-expectedBeta) > 0.01 {
		t.Errorf("Blended Beta = %v, want %v", blended.Distribution.Beta, expectedBeta)
	}
}

func TestRLDrivenSelector_InitializeFromConfig(t *testing.T) {
	selector := NewRLDrivenSelector(nil)

	modelConfig := map[string]config.ModelParams{
		"model-a": {
			Pricing: config.ModelPricing{
				PromptPer1M: 1.0,
			},
		},
		"model-b": {
			Pricing: config.ModelPricing{
				PromptPer1M: 10.0,
			},
		},
	}

	categories := []config.Category{
		{
			CategoryMetadata: config.CategoryMetadata{
				Name: "test-category",
			},
			ModelScores: []config.ModelScore{
				{Model: "model-a", Score: 0.8},
				{Model: "model-b", Score: 0.6},
			},
		},
	}

	selector.InitializeFromConfig(modelConfig, categories)

	// Verify global preferences were initialized
	prefA := selector.getGlobalPreference("model-a")
	prefB := selector.getGlobalPreference("model-b")

	if prefA == nil || prefB == nil {
		t.Error("Global preferences not initialized from config")
	}

	// Verify costs were set
	selector.costMu.RLock()
	costA := selector.modelCosts["model-a"]
	costB := selector.modelCosts["model-b"]
	selector.costMu.RUnlock()

	if costA != 1.0 {
		t.Errorf("Cost for model-a = %v, want 1.0", costA)
	}
	if costB != 10.0 {
		t.Errorf("Cost for model-b = %v, want 10.0", costB)
	}

	// Verify category preferences were initialized
	catPrefA := selector.getCategoryPreference("test-category", "model-a")
	if catPrefA == nil {
		t.Error("Category preference not initialized from config")
	}

	// Higher score should result in higher alpha
	catPrefB := selector.getCategoryPreference("test-category", "model-b")
	if catPrefA != nil && catPrefB != nil {
		if catPrefA.Distribution.Alpha <= catPrefB.Distribution.Alpha {
			t.Error("Model-a should have higher alpha due to higher score")
		}
	}
}
