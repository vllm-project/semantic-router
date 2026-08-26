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
)

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

	simulateEloRankingFeedback(t, ctx, selector)

	// Verify final rankings
	strongRating := selector.getGlobalRating("strong-model")
	mediumRating := selector.getGlobalRating("medium-model")
	weakRating := selector.getGlobalRating("weak-model")
	assertEloRankingEvolution(t, selector, strongRating, mediumRating, weakRating)
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
