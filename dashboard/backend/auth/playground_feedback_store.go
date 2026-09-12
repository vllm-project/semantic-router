package auth

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	playgroundReplayStateInProgress = "in_progress"
	playgroundReplayStateCompleted  = "completed"
	playgroundReplayStateSubmitting = "submitting"
	playgroundReplayStateConsumed   = "consumed"
)

var (
	ErrPlaygroundReplayNotOwned      = errors.New("replay does not belong to the current login session")
	ErrPlaygroundReplayExpired       = errors.New("replay feedback window has expired")
	ErrPlaygroundReplayInProgress    = errors.New("replay is still in progress")
	ErrPlaygroundReplayModelMismatch = errors.New("feedback model does not match the routed response")
	ErrPlaygroundReplayDuplicate     = errors.New("feedback was already submitted for this replay")
	ErrPlaygroundFeedbackRateLimited = errors.New("playground feedback rate limit exceeded")
)

// BindPlaygroundReplay associates a router-produced replay with the active
// Dashboard login session. The auth session's expiry is the feedback expiry.
func (s *Service) BindPlaygroundReplay(ctx context.Context, sessionID, replayID, targetRef string) error {
	sessionID = strings.TrimSpace(sessionID)
	replayID = strings.TrimSpace(replayID)
	targetRef = strings.TrimSpace(targetRef)
	if s == nil || s.store == nil || sessionID == "" || replayID == "" || targetRef == "" {
		return ErrPlaygroundReplayNotOwned
	}
	now := time.Now().Unix()
	result, err := s.store.db.ExecContext(ctx, `
		INSERT INTO playground_feedback_replays(
			replay_id, session_id, target_ref, state, created_at, expires_at
		)
		SELECT ?, id, ?, ?, ?, expires_at
		FROM auth_sessions
		WHERE id = ? AND revoked_at IS NULL AND expires_at > ?
		ON CONFLICT(replay_id) DO NOTHING`,
		replayID, targetRef, playgroundReplayStateInProgress, now, sessionID, now,
	)
	if err != nil {
		return fmt.Errorf("bind playground replay: %w", err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("read playground replay bind result: %w", err)
	}
	if rows == 0 {
		return ErrPlaygroundReplayNotOwned
	}
	return nil
}

// CompletePlaygroundReplay marks the proxied response body as fully delivered.
func (s *Service) CompletePlaygroundReplay(ctx context.Context, sessionID, replayID string) error {
	if s == nil || s.store == nil {
		return ErrPlaygroundReplayNotOwned
	}
	_, err := s.store.db.ExecContext(ctx, `
		UPDATE playground_feedback_replays
		SET state = ?, completed_at = ?
		WHERE replay_id = ? AND session_id = ? AND state = ?`,
		playgroundReplayStateCompleted, time.Now().Unix(), strings.TrimSpace(replayID),
		strings.TrimSpace(sessionID), playgroundReplayStateInProgress,
	)
	if err != nil {
		return fmt.Errorf("complete playground replay: %w", err)
	}
	return nil
}

// ValidatePlaygroundReplay verifies session ownership before the Dashboard
// asks the Router for lifecycle state. ClaimPlaygroundReplay repeats these
// checks transactionally before submission.
func (s *Service) ValidatePlaygroundReplay(ctx context.Context, sessionID, replayID, targetRef string) error {
	if s == nil || s.store == nil {
		return ErrPlaygroundReplayNotOwned
	}
	return validatePlaygroundReplayRow(
		s.store.db.QueryRowContext(ctx, `
			SELECT target_ref, state, expires_at
			FROM playground_feedback_replays
			WHERE replay_id = ? AND session_id = ?`, strings.TrimSpace(replayID), strings.TrimSpace(sessionID)),
		strings.TrimSpace(targetRef), time.Now().Unix(),
	)
}

// ClaimPlaygroundReplay atomically reserves one completed replay for feedback
// and enforces the rate limit against the login session rather than the user.
func (s *Service) ClaimPlaygroundReplay(
	ctx context.Context,
	sessionID, replayID, targetRef string,
	limit int,
	window time.Duration,
) error {
	if s == nil || s.store == nil {
		return ErrPlaygroundReplayNotOwned
	}
	tx, err := s.store.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin playground feedback claim: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	now := time.Now().Unix()
	if err := validatePlaygroundReplayRow(
		tx.QueryRowContext(ctx, `
			SELECT target_ref, state, expires_at
			FROM playground_feedback_replays
			WHERE replay_id = ? AND session_id = ?`, strings.TrimSpace(replayID), strings.TrimSpace(sessionID)),
		strings.TrimSpace(targetRef), now,
	); err != nil {
		return err
	}

	if limit > 0 && window > 0 {
		var attempts int
		if err := tx.QueryRowContext(ctx, `
			SELECT COUNT(*) FROM playground_feedback_replays
			WHERE session_id = ? AND claimed_at >= ?`,
			strings.TrimSpace(sessionID), now-int64(window.Seconds()),
		).Scan(&attempts); err != nil {
			return fmt.Errorf("count playground feedback attempts: %w", err)
		}
		if attempts >= limit {
			return ErrPlaygroundFeedbackRateLimited
		}
	}

	result, err := tx.ExecContext(ctx, `
		UPDATE playground_feedback_replays
		SET state = ?, claimed_at = ?
		WHERE replay_id = ? AND session_id = ? AND state = ?`,
		playgroundReplayStateSubmitting, now, strings.TrimSpace(replayID),
		strings.TrimSpace(sessionID), playgroundReplayStateCompleted,
	)
	if err != nil {
		return fmt.Errorf("claim playground replay: %w", err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("read playground replay claim result: %w", err)
	}
	if rows != 1 {
		return ErrPlaygroundReplayDuplicate
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit playground feedback claim: %w", err)
	}
	return nil
}

// FinishPlaygroundReplay commits a successful submission or releases a failed
// upstream attempt for retry. claimed_at remains as rate-limit evidence.
func (s *Service) FinishPlaygroundReplay(ctx context.Context, sessionID, replayID string, submitted bool) error {
	if s == nil || s.store == nil {
		return ErrPlaygroundReplayNotOwned
	}
	nextState := playgroundReplayStateCompleted
	if submitted {
		nextState = playgroundReplayStateConsumed
	}
	_, err := s.store.db.ExecContext(ctx, `
		UPDATE playground_feedback_replays
		SET state = ?
		WHERE replay_id = ? AND session_id = ? AND state = ?`,
		nextState, strings.TrimSpace(replayID), strings.TrimSpace(sessionID), playgroundReplayStateSubmitting,
	)
	if err != nil {
		return fmt.Errorf("finish playground replay: %w", err)
	}
	return nil
}

type playgroundReplayRow interface {
	Scan(dest ...any) error
}

func validatePlaygroundReplayRow(row playgroundReplayRow, targetRef string, now int64) error {
	var storedTarget, state string
	var expiresAt int64
	if err := row.Scan(&storedTarget, &state, &expiresAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return ErrPlaygroundReplayNotOwned
		}
		return fmt.Errorf("read playground replay: %w", err)
	}
	if expiresAt <= now {
		return ErrPlaygroundReplayExpired
	}
	if storedTarget != targetRef {
		return ErrPlaygroundReplayModelMismatch
	}
	switch state {
	case playgroundReplayStateCompleted:
		return nil
	case playgroundReplayStateInProgress:
		return ErrPlaygroundReplayInProgress
	case playgroundReplayStateSubmitting, playgroundReplayStateConsumed:
		return ErrPlaygroundReplayDuplicate
	default:
		return ErrPlaygroundReplayNotOwned
	}
}
