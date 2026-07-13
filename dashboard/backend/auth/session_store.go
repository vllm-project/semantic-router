package auth

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"time"
)

const inactiveAuthSessionRetention = 7 * 24 * time.Hour

const (
	// A valid credential must not be able to grow one user's live session set
	// without bound. Older live sessions are revoked when a new one is issued;
	// recent inactive rows remain briefly for forensic correlation.
	maxActiveAuthSessionsPerUser   = 16
	maxRetainedAuthSessionsPerUser = 64
)

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

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	_, err = tx.ExecContext(ctx, `
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
	if err != nil {
		return err
	}
	if err := enforceUserSessionBoundsTx(ctx, tx, userID, nowUnix()); err != nil {
		return err
	}
	return tx.Commit()
}

func insertIssuedSessionTx(ctx context.Context, tx *sql.Tx, userID string, issued *issuedToken) error {
	_, err := tx.ExecContext(
		ctx,
		`INSERT INTO auth_sessions(id, user_id, issued_at, expires_at)
		 VALUES (?, ?, ?, ?)`,
		issued.sessionID,
		userID,
		issued.issuedAt.Unix(),
		issued.expiresAt.Unix(),
	)
	if err != nil {
		return err
	}
	return enforceUserSessionBoundsTx(ctx, tx, userID, nowUnix())
}

func enforceUserSessionBoundsTx(ctx context.Context, tx *sql.Tx, userID string, now int64) error {
	cutoff := time.Unix(now, 0).Add(-inactiveAuthSessionRetention).Unix()
	if _, err := tx.ExecContext(
		ctx,
		`DELETE FROM auth_sessions
		 WHERE user_id = ?
		   AND (expires_at <= ? OR (revoked_at IS NOT NULL AND revoked_at <= ?))`,
		userID,
		cutoff,
		cutoff,
	); err != nil {
		return err
	}

	// Revoke, rather than immediately delete, the oldest excess live sessions.
	// This preserves a bounded recent record while making eviction fail closed.
	if _, err := tx.ExecContext(
		ctx,
		`UPDATE auth_sessions
		 SET revoked_at = COALESCE(revoked_at, ?)
		 WHERE user_id = ?
		   AND revoked_at IS NULL
		   AND expires_at > ?
		   AND id NOT IN (
		     SELECT id FROM auth_sessions
		     WHERE user_id = ? AND revoked_at IS NULL AND expires_at > ?
		     ORDER BY issued_at DESC, rowid DESC
		     LIMIT ?
		   )`,
		now,
		userID,
		now,
		userID,
		now,
		maxActiveAuthSessionsPerUser,
	); err != nil {
		return err
	}

	_, err := tx.ExecContext(
		ctx,
		`DELETE FROM auth_sessions
		 WHERE user_id = ?
		   AND id NOT IN (
		     SELECT id FROM auth_sessions
		     WHERE user_id = ?
		     ORDER BY
		       CASE WHEN revoked_at IS NULL AND expires_at > ? THEN 0 ELSE 1 END,
		       issued_at DESC,
		       rowid DESC
		     LIMIT ?
		   )`,
		userID,
		userID,
		now,
		maxRetainedAuthSessionsPerUser,
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
