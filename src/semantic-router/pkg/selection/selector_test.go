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
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Test helper to create candidate models
func createCandidateModels(names ...string) []config.ModelRef {
	models := make([]config.ModelRef, len(names))
	for i, name := range names {
		models[i] = config.ModelRef{Model: name}
	}
	return models
}

func TestEloSelector_Select(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name          string
		candidates    []config.ModelRef
		setupRatings  map[string]float64
		expectedModel string
		expectError   bool
	}{
		{
			name:          "select highest rated model",
			candidates:    createCandidateModels("model-a", "model-b", "model-c"),
			setupRatings:  map[string]float64{"model-a": 1400, "model-b": 1600, "model-c": 1500},
			expectedModel: "model-b",
			expectError:   false,
		},
		{
			name:          "fallback to default rating",
			candidates:    createCandidateModels("new-model-1", "new-model-2"),
			setupRatings:  map[string]float64{},
			expectedModel: "new-model-1", // First model when equal ratings
			expectError:   false,
		},
		{
			name:          "no candidates",
			candidates:    []config.ModelRef{},
			setupRatings:  map[string]float64{},
			expectedModel: "",
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selector := NewEloSelector(DefaultEloConfig())

			// Setup ratings
			for model, rating := range tt.setupRatings {
				selector.setGlobalRating(model, &ModelRating{Model: model, Rating: rating})
			}

			selCtx := &SelectionContext{
				Query:           "test query",
				CandidateModels: tt.candidates,
			}

			result, err := selector.Select(ctx, selCtx)

			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			if result.SelectedModel != tt.expectedModel {
				t.Errorf("expected model %s, got %s", tt.expectedModel, result.SelectedModel)
			}

			if result.Method != MethodElo {
				t.Errorf("expected method %s, got %s", MethodElo, result.Method)
			}
		})
	}
}

func TestEloSelector_UpdateFeedback(t *testing.T) {
	ctx := context.Background()
	selector := NewEloSelector(DefaultEloConfig())

	// Initialize ratings
	selector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1500})
	selector.setGlobalRating("model-b", &ModelRating{Model: "model-b", Rating: 1500})

	// Submit feedback: model-a wins against model-b
	feedback := &Feedback{
		Query:       "test query",
		WinnerModel: "model-a",
		LoserModel:  "model-b",
		Tie:         false,
	}

	err := selector.UpdateFeedback(ctx, feedback)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check ratings updated
	ratingA := selector.getGlobalRating("model-a")
	ratingB := selector.getGlobalRating("model-b")

	if ratingA == nil {
		t.Fatal("rating A should not be nil")
		return // Explicit return after t.Fatal for staticcheck
	}
	if ratingB == nil {
		t.Fatal("rating B should not be nil")
		return // Explicit return after t.Fatal for staticcheck
	}

	if ratingA.Rating <= 1500 {
		t.Errorf("winner rating should increase, got %f", ratingA.Rating)
	}

	if ratingB.Rating >= 1500 {
		t.Errorf("loser rating should decrease, got %f", ratingB.Rating)
	}

	if ratingA.Wins != 1 {
		t.Errorf("winner wins should be 1, got %d", ratingA.Wins)
	}

	if ratingB.Losses != 1 {
		t.Errorf("loser losses should be 1, got %d", ratingB.Losses)
	}
}

func TestRouterDCSelector_Select(t *testing.T) {
	ctx := context.Background()

	selector := NewRouterDCSelector(DefaultRouterDCConfig())

	// Set up embedding function (mock)
	selector.SetEmbeddingFunc(func(text string) ([]float32, error) {
		// Return a simple embedding based on text length
		embedding := make([]float32, 768)
		for i := range embedding {
			embedding[i] = float32(len(text)%10) / 10.0
		}
		return embedding, nil
	})

	// Set model embeddings
	modelAEmb := make([]float32, 768)
	modelBEmb := make([]float32, 768)
	for i := range modelAEmb {
		modelAEmb[i] = 0.5
		modelBEmb[i] = 0.3
	}
	selector.SetModelEmbedding("model-a", modelAEmb)
	selector.SetModelEmbedding("model-b", modelBEmb)

	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("model-a", "model-b"),
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodRouterDC {
		t.Errorf("expected method %s, got %s", MethodRouterDC, result.Method)
	}

	if result.Score <= 0 {
		t.Errorf("expected positive score, got %f", result.Score)
	}
}

func TestAutoMixSelector_Select(t *testing.T) {
	ctx := context.Background()

	selector := NewAutoMixSelector(DefaultAutoMixConfig())

	// Initialize capabilities
	modelConfig := map[string]config.ModelParams{
		"small-model": {Pricing: config.ModelPricing{PromptPer1M: 0.5}},
		"large-model": {Pricing: config.ModelPricing{PromptPer1M: 5.0}},
	}
	selector.InitializeFromConfig(modelConfig)

	// Set verification probabilities
	selector.SetCapability("small-model", &ModelCapability{
		Model:            "small-model",
		Cost:             0.5,
		AvgQuality:       0.7,
		VerificationProb: 0.8,
		ParamSize:        7.0,
	})
	selector.SetCapability("large-model", &ModelCapability{
		Model:            "large-model",
		Cost:             5.0,
		AvgQuality:       0.95,
		VerificationProb: 0.95,
		ParamSize:        70.0,
	})

	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("small-model", "large-model"),
		CostWeight:      0.5,
		QualityWeight:   0.5,
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodAutoMix {
		t.Errorf("expected method %s, got %s", MethodAutoMix, result.Method)
	}

	// With cost awareness, cheaper model might be selected
	if result.SelectedModel == "" {
		t.Error("expected a selected model")
	}
}

func TestHybridSelector_Select(t *testing.T) {
	ctx := context.Background()

	cfg := DefaultHybridConfig()
	cfg.EloWeight = 0.5
	cfg.RouterDCWeight = 0.0 // Disable RouterDC (no embeddings)
	cfg.AutoMixWeight = 0.5
	cfg.CostWeight = 0.0

	selector := NewHybridSelector(cfg)

	// Initialize Elo component
	selector.eloSelector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1600})
	selector.eloSelector.setGlobalRating("model-b", &ModelRating{Model: "model-b", Rating: 1400})

	// Initialize AutoMix component
	selector.autoMixSelector.SetCapability("model-a", &ModelCapability{
		Model:            "model-a",
		AvgQuality:       0.9,
		VerificationProb: 0.9,
		ParamSize:        70.0,
	})
	selector.autoMixSelector.SetCapability("model-b", &ModelCapability{
		Model:            "model-b",
		AvgQuality:       0.7,
		VerificationProb: 0.8,
		ParamSize:        7.0,
	})

	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("model-a", "model-b"),
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodHybrid {
		t.Errorf("expected method %s, got %s", MethodHybrid, result.Method)
	}

	// Model-a should win (higher Elo and quality)
	if result.SelectedModel != "model-a" {
		t.Errorf("expected model-a, got %s", result.SelectedModel)
	}
}

func TestStaticSelector_Select(t *testing.T) {
	ctx := context.Background()

	selector := NewStaticSelector(DefaultStaticConfig())

	// Set up category scores
	selector.SetCategoryScore("coding", "code-model", 0.9)
	selector.SetCategoryScore("coding", "general-model", 0.5)

	selCtx := &SelectionContext{
		Query:           "write python code",
		DecisionName:    "coding",
		CandidateModels: createCandidateModels("code-model", "general-model"),
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Method != MethodStatic {
		t.Errorf("expected method %s, got %s", MethodStatic, result.Method)
	}

	if result.SelectedModel != "code-model" {
		t.Errorf("expected code-model, got %s", result.SelectedModel)
	}

	if result.Score != 0.9 {
		t.Errorf("expected score 0.9, got %f", result.Score)
	}
}

func TestRegistry(t *testing.T) {
	registry := NewRegistry()

	// Register selectors
	registry.Register(MethodElo, NewEloSelector(nil))
	registry.Register(MethodStatic, NewStaticSelector(nil))

	// Get registered selectors
	eloSelector, ok := registry.Get(MethodElo)
	if !ok || eloSelector == nil {
		t.Error("expected Elo selector to be registered")
	}

	staticSelector, ok := registry.Get(MethodStatic)
	if !ok || staticSelector == nil {
		t.Error("expected Static selector to be registered")
	}

	// Get unregistered selector
	_, ok = registry.Get(MethodRouterDC)
	if ok {
		t.Error("expected RouterDC to not be registered")
	}
}

func TestFactory_Create(t *testing.T) {
	tests := []struct {
		name           string
		method         string
		expectedMethod SelectionMethod
	}{
		{
			name:           "create elo selector",
			method:         "elo",
			expectedMethod: MethodElo,
		},
		{
			name:           "create router_dc selector",
			method:         "router_dc",
			expectedMethod: MethodRouterDC,
		},
		{
			name:           "create automix selector",
			method:         "automix",
			expectedMethod: MethodAutoMix,
		},
		{
			name:           "create hybrid selector",
			method:         "hybrid",
			expectedMethod: MethodHybrid,
		},
		{
			name:           "create static selector (default)",
			method:         "static",
			expectedMethod: MethodStatic,
		},
		{
			name:           "unknown method defaults to static",
			method:         "unknown",
			expectedMethod: MethodStatic,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &ModelSelectionConfig{Method: tt.method}
			factory := NewFactory(cfg)
			selector := factory.Create()

			if selector.Method() != tt.expectedMethod {
				t.Errorf("expected method %s, got %s", tt.expectedMethod, selector.Method())
			}
		})
	}
}

func TestEloSelector_CategoryRatings(t *testing.T) {
	ctx := context.Background()

	cfg := DefaultEloConfig()
	cfg.CategoryWeighted = true
	selector := NewEloSelector(cfg)

	// Set different ratings for different categories
	selector.setCategoryRating("coding", "model-a", &ModelRating{Model: "model-a", Rating: 1700})
	selector.setCategoryRating("coding", "model-b", &ModelRating{Model: "model-b", Rating: 1300})
	selector.setCategoryRating("writing", "model-a", &ModelRating{Model: "model-a", Rating: 1300})
	selector.setCategoryRating("writing", "model-b", &ModelRating{Model: "model-b", Rating: 1700})

	candidates := createCandidateModels("model-a", "model-b")

	// Test coding category
	codingCtx := &SelectionContext{
		Query:           "write code",
		DecisionName:    "coding",
		CandidateModels: candidates,
	}
	result, err := selector.Select(ctx, codingCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SelectedModel != "model-a" {
		t.Errorf("expected model-a for coding, got %s", result.SelectedModel)
	}

	// Test writing category
	writingCtx := &SelectionContext{
		Query:           "write essay",
		DecisionName:    "writing",
		CandidateModels: candidates,
	}
	result, err = selector.Select(ctx, writingCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.SelectedModel != "model-b" {
		t.Errorf("expected model-b for writing, got %s", result.SelectedModel)
	}
}

func TestEloSelector_GetLeaderboard(t *testing.T) {
	selector := NewEloSelector(DefaultEloConfig())

	// Set up ratings
	selector.setGlobalRating("model-c", &ModelRating{Model: "model-c", Rating: 1400})
	selector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1600})
	selector.setGlobalRating("model-b", &ModelRating{Model: "model-b", Rating: 1500})

	leaderboard := selector.GetLeaderboard("")

	if len(leaderboard) != 3 {
		t.Errorf("expected 3 models in leaderboard, got %d", len(leaderboard))
	}

	// Should be sorted by rating descending
	if leaderboard[0].Model != "model-a" {
		t.Errorf("expected model-a first, got %s", leaderboard[0].Model)
	}
	if leaderboard[1].Model != "model-b" {
		t.Errorf("expected model-b second, got %s", leaderboard[1].Model)
	}
	if leaderboard[2].Model != "model-c" {
		t.Errorf("expected model-c third, got %s", leaderboard[2].Model)
	}
}

// TestEloSelector_MultiTurnEvolution tests that Elo ratings evolve correctly
// over multiple feedback rounds, demonstrating convergence and ranking stability.
func TestEloSelector_MultiTurnEvolution(t *testing.T) {
	ctx := context.Background()
	selector := NewEloSelector(DefaultEloConfig())

	// Initialize three models with same starting rating
	models := []string{"weak-model", "medium-model", "strong-model"}
	for _, m := range models {
		selector.setGlobalRating(m, &ModelRating{Model: m, Rating: DefaultEloRating})
	}

	// Simulate 10 rounds of feedback where strong > medium > weak
	for round := 0; round < 10; round++ {
		// Strong beats medium
		_ = selector.UpdateFeedback(ctx, &Feedback{
			Query:       "test",
			WinnerModel: "strong-model",
			LoserModel:  "medium-model",
		})

		// Medium beats weak
		_ = selector.UpdateFeedback(ctx, &Feedback{
			Query:       "test",
			WinnerModel: "medium-model",
			LoserModel:  "weak-model",
		})

		// Strong beats weak
		_ = selector.UpdateFeedback(ctx, &Feedback{
			Query:       "test",
			WinnerModel: "strong-model",
			LoserModel:  "weak-model",
		})
	}

	// Verify final rankings
	strongRating := selector.getGlobalRating("strong-model")
	mediumRating := selector.getGlobalRating("medium-model")
	weakRating := selector.getGlobalRating("weak-model")

	if strongRating == nil || mediumRating == nil || weakRating == nil {
		t.Fatal("ratings should not be nil")
		return
	}

	// Strong should have highest rating
	if strongRating.Rating <= mediumRating.Rating {
		t.Errorf("strong (%f) should beat medium (%f)", strongRating.Rating, mediumRating.Rating)
	}

	// Medium should beat weak
	if mediumRating.Rating <= weakRating.Rating {
		t.Errorf("medium (%f) should beat weak (%f)", mediumRating.Rating, weakRating.Rating)
	}

	// Win/loss records should reflect the matches
	if strongRating.Wins != 20 { // 10 vs medium + 10 vs weak
		t.Errorf("strong should have 20 wins, got %d", strongRating.Wins)
	}
	if weakRating.Losses != 20 { // 10 vs medium + 10 vs strong
		t.Errorf("weak should have 20 losses, got %d", weakRating.Losses)
	}

	// Verify leaderboard order
	leaderboard := selector.GetLeaderboard("")
	if len(leaderboard) < 3 {
		t.Fatalf("expected at least 3 models, got %d", len(leaderboard))
	}
	if leaderboard[0].Model != "strong-model" {
		t.Errorf("strong-model should be first, got %s", leaderboard[0].Model)
	}
	if leaderboard[1].Model != "medium-model" {
		t.Errorf("medium-model should be second, got %s", leaderboard[1].Model)
	}
	if leaderboard[2].Model != "weak-model" {
		t.Errorf("weak-model should be third, got %s", leaderboard[2].Model)
	}
}

// TestEloSelector_TieHandling tests that ties are handled correctly
func TestEloSelector_TieHandling(t *testing.T) {
	ctx := context.Background()
	selector := NewEloSelector(DefaultEloConfig())

	selector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1500})
	selector.setGlobalRating("model-b", &ModelRating{Model: "model-b", Rating: 1500})

	// Submit a tie
	err := selector.UpdateFeedback(ctx, &Feedback{
		Query:       "test",
		WinnerModel: "model-a",
		LoserModel:  "model-b",
		Tie:         true,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	ratingA := selector.getGlobalRating("model-a")
	ratingB := selector.getGlobalRating("model-b")

	if ratingA == nil || ratingB == nil {
		t.Fatal("ratings should not be nil")
		return
	}

	// Both should have a tie recorded
	if ratingA.Ties != 1 {
		t.Errorf("model-a should have 1 tie, got %d", ratingA.Ties)
	}
	if ratingB.Ties != 1 {
		t.Errorf("model-b should have 1 tie, got %d", ratingB.Ties)
	}

	// Ratings should remain close (tie moves both toward each other)
	ratingDiff := ratingA.Rating - ratingB.Rating
	if ratingDiff < -1 || ratingDiff > 1 {
		t.Errorf("ratings should be nearly equal after tie, got diff %f", ratingDiff)
	}
}

// TestEloSelector_SelectionFollowsRatings verifies that Select() respects Elo ratings
func TestEloSelector_SelectionFollowsRatings(t *testing.T) {
	ctx := context.Background()
	selector := NewEloSelector(DefaultEloConfig())

	// Set up ratings with clear winner
	selector.setGlobalRating("low-rated", &ModelRating{Model: "low-rated", Rating: 1300})
	selector.setGlobalRating("high-rated", &ModelRating{Model: "high-rated", Rating: 1700})

	selCtx := &SelectionContext{
		Query:           "test query",
		DecisionName:    "test",
		CandidateModels: createCandidateModels("low-rated", "high-rated"),
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// High-rated should be selected
	if result.SelectedModel != "high-rated" {
		t.Errorf("expected high-rated, got %s", result.SelectedModel)
	}
}
