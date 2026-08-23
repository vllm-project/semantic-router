package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

func (s *Store) RotateTokenID(
	ctx context.Context,
	sessionID string,
	expectedTokenID string,
	newTokenID string,
) (managementauth.LiveSession, error) {
	if _, err := uuid.Parse(sessionID); err != nil {
		return managementauth.LiveSession{}, managementauth.ErrSessionNotFound
	}
	if expectedTokenID == "" || newTokenID == "" || expectedTokenID == newTokenID {
		return managementauth.LiveSession{}, managementauth.ErrSessionConflict
	}
	return inTransaction(ctx, s, sql.LevelReadCommitted, func(tx *sql.Tx) (managementauth.LiveSession, error) {
		state, rotateTokenIDErr := lockSession(ctx, tx, sessionID)
		if rotateTokenIDErr != nil {
			return managementauth.LiveSession{}, rotateTokenIDErr
		}
		if state.tokenID != expectedTokenID {
			return managementauth.LiveSession{}, managementauth.ErrSessionConflict
		}
		if state.status != managementauth.SessionActive || !state.now.Before(state.expiresAt) {
			return managementauth.LiveSession{}, managementauth.ErrSessionInactive
		}
		result, rotateTokenIDErr := tx.ExecContext(ctx, rotateSessionTokenIDQuery, sessionID, expectedTokenID, newTokenID)
		if rotateTokenIDErr != nil {
			if isUniqueViolation(rotateTokenIDErr) {
				return managementauth.LiveSession{}, managementauth.ErrSessionConflict
			}
			return managementauth.LiveSession{}, fmt.Errorf("refresh management session token: %w", rotateTokenIDErr)
		}
		if err := requireOneRow(result); err != nil {
			return managementauth.LiveSession{}, managementauth.ErrSessionConflict
		}
		live, rotateTokenIDErr := getWith(ctx, tx, sessionID)
		if rotateTokenIDErr != nil {
			return managementauth.LiveSession{}, rotateTokenIDErr
		}
		if live.TokenID != newTokenID {
			return managementauth.LiveSession{}, managementauth.ErrSessionConflict
		}
		return live, nil
	})
}

func (s *Store) Revoke(
	ctx context.Context,
	sessionID string,
	expectedTokenID string,
) (managementauth.SessionMutation, error) {
	if _, err := uuid.Parse(sessionID); err != nil {
		return managementauth.SessionMutation{}, managementauth.ErrSessionNotFound
	}
	if expectedTokenID == "" {
		return managementauth.SessionMutation{}, managementauth.ErrSessionConflict
	}
	return inTransaction(ctx, s, sql.LevelReadCommitted, func(tx *sql.Tx) (managementauth.SessionMutation, error) {
		state, err := lockSession(ctx, tx, sessionID)
		if err != nil {
			return managementauth.SessionMutation{}, err
		}
		if state.tokenID != expectedTokenID {
			return managementauth.SessionMutation{}, managementauth.ErrSessionConflict
		}
		if state.status == managementauth.SessionRevoked {
			if state.revokedAt == nil {
				return managementauth.SessionMutation{}, fmt.Errorf("revoked management session is missing revoked_at")
			}
			return managementauth.SessionMutation{
				SessionID: sessionID,
				TokenID:   expectedTokenID,
				Changed:   false,
				ChangedAt: *state.revokedAt,
			}, nil
		}
		if state.status != managementauth.SessionActive || !state.now.Before(state.expiresAt) {
			return managementauth.SessionMutation{}, managementauth.ErrSessionInactive
		}
		var revokedAt time.Time
		if err := tx.QueryRowContext(ctx, revokeSessionQuery, sessionID, expectedTokenID).Scan(&revokedAt); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementauth.SessionMutation{}, managementauth.ErrSessionConflict
			}
			return managementauth.SessionMutation{}, fmt.Errorf("revoke management session: %w", err)
		}
		return managementauth.SessionMutation{
			SessionID: sessionID,
			TokenID:   expectedTokenID,
			Changed:   true,
			ChangedAt: revokedAt.UTC(),
		}, nil
	})
}

type lockedSession struct {
	tokenID   string
	status    managementauth.SessionStatus
	expiresAt time.Time
	revokedAt *time.Time
	now       time.Time
}

func lockSession(ctx context.Context, tx *sql.Tx, sessionID string) (lockedSession, error) {
	var (
		state     lockedSession
		status    string
		revokedAt sql.NullTime
	)
	if err := tx.QueryRowContext(ctx, lockSessionQuery, sessionID).Scan(
		&state.tokenID,
		&status,
		&state.expiresAt,
		&revokedAt,
		&state.now,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return lockedSession{}, managementauth.ErrSessionNotFound
		}
		return lockedSession{}, fmt.Errorf("lock management session: %w", err)
	}
	state.status = managementauth.SessionStatus(status)
	state.expiresAt = state.expiresAt.UTC()
	state.now = state.now.UTC()
	if revokedAt.Valid {
		value := revokedAt.Time.UTC()
		state.revokedAt = &value
	}
	return state, nil
}

func isUniqueViolation(err error) bool {
	var databaseError *pq.Error
	return errors.As(err, &databaseError) && databaseError.Code == "23505"
}

func requireOneRow(result sql.Result) error {
	rows, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rows != 1 {
		return fmt.Errorf("expected one affected row, got %d", rows)
	}
	return nil
}
