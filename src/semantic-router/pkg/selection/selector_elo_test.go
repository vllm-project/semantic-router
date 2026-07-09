package selection

import (
	"context"
	"testing"
)

func simulateEloRankingFeedback(t *testing.T, ctx context.Context, selector *EloSelector) {
	t.Helper()
	for round := 0; round < 10; round++ {
		updateEloFeedback(t, ctx, selector, "strong-model", "medium-model")
		updateEloFeedback(t, ctx, selector, "medium-model", "weak-model")
		updateEloFeedback(t, ctx, selector, "strong-model", "weak-model")
	}
}

func updateEloFeedback(t *testing.T, ctx context.Context, selector *EloSelector, winner string, loser string) {
	t.Helper()
	if err := selector.UpdateFeedback(ctx, &Feedback{
		Query:       "test",
		WinnerModel: winner,
		LoserModel:  loser,
	}); err != nil {
		t.Fatalf("unexpected feedback error: %v", err)
	}
}

func assertEloRankingEvolution(
	t *testing.T,
	selector *EloSelector,
	strongRating *ModelRating,
	mediumRating *ModelRating,
	weakRating *ModelRating,
) {
	t.Helper()
	if strongRating == nil || mediumRating == nil || weakRating == nil {
		t.Fatal("ratings should not be nil")
	}

	if strongRating.Rating <= mediumRating.Rating {
		t.Errorf("strong (%f) should beat medium (%f)", strongRating.Rating, mediumRating.Rating)
	}
	if mediumRating.Rating <= weakRating.Rating {
		t.Errorf("medium (%f) should beat weak (%f)", mediumRating.Rating, weakRating.Rating)
	}
	if strongRating.Wins != 20 {
		t.Errorf("strong should have 20 wins, got %d", strongRating.Wins)
	}
	if weakRating.Losses != 20 {
		t.Errorf("weak should have 20 losses, got %d", weakRating.Losses)
	}

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
