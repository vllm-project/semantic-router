package auth

import (
	"context"
	"errors"
	"testing"
	"time"
)

func newPlaygroundFeedbackSession(t *testing.T, svc *Service, email string) string {
	t.Helper()
	user := newTestUser(t, svc, email, RoleRead, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	claims, err := svc.ParseToken(token)
	if err != nil {
		t.Fatalf("ParseToken() error = %v", err)
	}
	return claims.ID
}

func TestPlaygroundFeedbackReplayLifecycleIsSessionBound(t *testing.T) {
	svc := newTestAuthService(t)
	ctx := context.Background()
	sessionID := newPlaygroundFeedbackSession(t, svc, "feedback-owner@example.com")
	otherSessionID := newPlaygroundFeedbackSession(t, svc, "feedback-other@example.com")

	if err := svc.BindPlaygroundReplay(ctx, sessionID, "replay-1", "model-a"); err != nil {
		t.Fatalf("BindPlaygroundReplay() error = %v", err)
	}
	if err := svc.ValidatePlaygroundReplay(ctx, sessionID, "replay-1", "model-a"); !errors.Is(err, ErrPlaygroundReplayInProgress) {
		t.Fatalf("in-progress validation error = %v", err)
	}
	if err := svc.CompletePlaygroundReplay(ctx, sessionID, "replay-1"); err != nil {
		t.Fatalf("CompletePlaygroundReplay() error = %v", err)
	}
	if err := svc.ValidatePlaygroundReplay(ctx, otherSessionID, "replay-1", "model-a"); !errors.Is(err, ErrPlaygroundReplayNotOwned) {
		t.Fatalf("foreign-session validation error = %v", err)
	}
	if err := svc.ValidatePlaygroundReplay(ctx, sessionID, "replay-1", "model-b"); !errors.Is(err, ErrPlaygroundReplayModelMismatch) {
		t.Fatalf("model-mismatch validation error = %v", err)
	}
	if err := svc.ClaimPlaygroundReplay(ctx, sessionID, "replay-1", "model-a", 60, time.Minute); err != nil {
		t.Fatalf("ClaimPlaygroundReplay() error = %v", err)
	}
	if err := svc.FinishPlaygroundReplay(ctx, sessionID, "replay-1", true); err != nil {
		t.Fatalf("FinishPlaygroundReplay() error = %v", err)
	}
	if err := svc.ValidatePlaygroundReplay(ctx, sessionID, "replay-1", "model-a"); !errors.Is(err, ErrPlaygroundReplayDuplicate) {
		t.Fatalf("duplicate validation error = %v", err)
	}
}

func TestPlaygroundFeedbackFailedSubmissionCanRetry(t *testing.T) {
	svc := newTestAuthService(t)
	ctx := context.Background()
	sessionID := newPlaygroundFeedbackSession(t, svc, "feedback-retry@example.com")

	if err := svc.BindPlaygroundReplay(ctx, sessionID, "replay-retry", "model-a"); err != nil {
		t.Fatal(err)
	}
	if err := svc.CompletePlaygroundReplay(ctx, sessionID, "replay-retry"); err != nil {
		t.Fatal(err)
	}
	if err := svc.ClaimPlaygroundReplay(ctx, sessionID, "replay-retry", "model-a", 60, time.Minute); err != nil {
		t.Fatal(err)
	}
	if err := svc.FinishPlaygroundReplay(ctx, sessionID, "replay-retry", false); err != nil {
		t.Fatal(err)
	}
	if err := svc.ValidatePlaygroundReplay(ctx, sessionID, "replay-retry", "model-a"); err != nil {
		t.Fatalf("released replay validation error = %v", err)
	}
}

func TestPlaygroundFeedbackRateLimitUsesSession(t *testing.T) {
	svc := newTestAuthService(t)
	ctx := context.Background()
	sessionID := newPlaygroundFeedbackSession(t, svc, "feedback-limit@example.com")

	for _, replayID := range []string{"replay-first", "replay-second"} {
		if err := svc.BindPlaygroundReplay(ctx, sessionID, replayID, "model-a"); err != nil {
			t.Fatal(err)
		}
		if err := svc.CompletePlaygroundReplay(ctx, sessionID, replayID); err != nil {
			t.Fatal(err)
		}
	}
	if err := svc.ClaimPlaygroundReplay(ctx, sessionID, "replay-first", "model-a", 1, time.Minute); err != nil {
		t.Fatal(err)
	}
	if err := svc.ClaimPlaygroundReplay(ctx, sessionID, "replay-second", "model-a", 1, time.Minute); !errors.Is(err, ErrPlaygroundFeedbackRateLimited) {
		t.Fatalf("second claim error = %v", err)
	}
}

func TestPlaygroundFeedbackRejectsExpiredReplay(t *testing.T) {
	svc := newTestAuthService(t)
	ctx := context.Background()
	sessionID := newPlaygroundFeedbackSession(t, svc, "feedback-expired@example.com")

	if err := svc.BindPlaygroundReplay(ctx, sessionID, "replay-expired", "model-a"); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.store.db.ExecContext(ctx, `UPDATE playground_feedback_replays SET expires_at = ? WHERE replay_id = ?`, time.Now().Add(-time.Minute).Unix(), "replay-expired"); err != nil {
		t.Fatal(err)
	}
	if err := svc.ValidatePlaygroundReplay(ctx, sessionID, "replay-expired", "model-a"); !errors.Is(err, ErrPlaygroundReplayExpired) {
		t.Fatalf("expired validation error = %v", err)
	}
}
