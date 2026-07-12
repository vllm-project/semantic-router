package auth

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"time"
)

const inactiveAuthSessionRetention = 7 * 24 * time.Hour

func (s *Store) PruneInactiveSessions(ctx context.Context, now time.Time) error {
	if now.IsZero() {
		now = time.Now()
	}
	cutoff := now.Add(-inactiveAuthSessionRetention).Unix()
	_, err := s.db.ExecContext(
		ctx,
		`DELETE FROM auth_sessions
		WHERE expires_at <= ?
		   OR (revoked_at IS NOT NULL AND revoked_at <= ?)`,
		cutoff,
		cutoff,
	)
	return err
}

func (s *Store) CreateSession(ctx context.Context, sessionID, userID string, issuedAt, expiresAt int64) error {
	sessionID = strings.TrimSpace(sessionID)
	userID = strings.TrimSpace(userID)
	if sessionID == "" || userID == "" {
		return errors.New("session id and user id are required")
	}

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO auth_sessions(id, user_id, issued_at, expires_at)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			user_id = excluded.user_id,
			issued_at = excluded.issued_at,
			expires_at = excluded.expires_at,
			revoked_at = NULL`,
		sessionID,
		userID,
		issuedAt,
		expiresAt,
	)
	return err
}

func (s *Store) SessionActive(ctx context.Context, sessionID, userID string, atUnix int64) (bool, error) {
	sessionID = strings.TrimSpace(sessionID)
	userID = strings.TrimSpace(userID)
	if sessionID == "" || userID == "" {
		return false, nil
	}

	var expiresAt int64
	var revokedAt sql.NullInt64
	err := s.db.QueryRowContext(
		ctx,
		`SELECT expires_at, revoked_at FROM auth_sessions WHERE id = ? AND user_id = ?`,
		sessionID,
		userID,
	).Scan(&expiresAt, &revokedAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, err
	}
	if revokedAt.Valid {
		return false, nil
	}
	return expiresAt > atUnix, nil
}

func (s *Store) RevokeSession(ctx context.Context, sessionID string) error {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}

	_, err := s.db.ExecContext(
		ctx,
		`UPDATE auth_sessions SET revoked_at = COALESCE(revoked_at, ?) WHERE id = ?`,
		nowUnix(),
		sessionID,
	)
	return err
}
