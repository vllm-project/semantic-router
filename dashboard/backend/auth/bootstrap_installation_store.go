package auth

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"
)

const bootstrapUserStatusProvisioning = "provisioning"

func (s *Store) CanPrepareBootstrapAdmin(ctx context.Context) (bool, error) {
	var pending int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM dashboard_bootstrap_installation WHERE singleton=1`).Scan(&pending); err != nil {
		return false, err
	}
	if pending == 1 {
		return true, nil
	}
	count, err := s.CountUsers(ctx)
	return count == 0, err
}

func (s *Store) PrepareBootstrapAdmin(
	ctx context.Context,
	email string,
	name string,
	passwordHash string,
	now time.Time,
	sourceExpiresAt time.Time,
) (pendingBootstrapAdmin, string, bool, error) {
	now = now.UTC()
	sourceExpiresAt = sourceExpiresAt.UTC()
	if now.IsZero() || !now.Before(sourceExpiresAt) {
		return pendingBootstrapAdmin{}, "", false, errors.New("dashboard bootstrap source expiry is invalid")
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return pendingBootstrapAdmin{}, "", false, err
	}
	defer func() { _ = tx.Rollback() }()

	pending, storedHash, err := loadPendingBootstrapAdmin(ctx, tx)
	if err == nil {
		if pending.User.Email != strings.ToLower(email) || pending.User.Name != name {
			return pendingBootstrapAdmin{}, "", false, ErrBootstrapClosed
		}
		if !now.Before(pending.SessionExpiresAt) {
			pending.SessionID = uuid.NewString()
			pending.SessionIssuedAt = now
			pending.SessionExpiresAt = sourceExpiresAt
			if _, updateErr := tx.ExecContext(ctx, `UPDATE dashboard_bootstrap_installation
SET session_id=?,source_issued_at=?,source_expires_at=?,updated_at=? WHERE singleton=1 AND user_id=?`,
				pending.SessionID, now.Unix(), sourceExpiresAt.Unix(), now.Unix(), pending.User.ID); updateErr != nil {
				return pendingBootstrapAdmin{}, "", false, updateErr
			}
		}
		if commitErr := tx.Commit(); commitErr != nil {
			return pendingBootstrapAdmin{}, "", false, commitErr
		}
		return pending, storedHash, false, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return pendingBootstrapAdmin{}, "", false, err
	}

	var count int
	if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM users`).Scan(&count); err != nil {
		return pendingBootstrapAdmin{}, "", false, err
	}
	if count != 0 {
		return pendingBootstrapAdmin{}, "", false, ErrBootstrapClosed
	}

	userID, sessionID := uuid.NewString(), uuid.NewString()
	normalizedEmail := strings.ToLower(email)
	if _, err := tx.ExecContext(ctx, `INSERT INTO users
  (id,email,name,password_hash,role,status,created_at,updated_at)
VALUES (?,?,?,?,?,?,?,?)`, userID, normalizedEmail, name, passwordHash, RoleAdmin,
		bootstrapUserStatusProvisioning, now.Unix(), now.Unix()); err != nil {
		return pendingBootstrapAdmin{}, "", false, err
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO dashboard_bootstrap_installation
  (singleton,user_id,session_id,source_issued_at,source_expires_at,created_at,updated_at)
VALUES (1,?,?,?,?,?,?)`,
		userID, sessionID, now.Unix(), sourceExpiresAt.Unix(), now.Unix(), now.Unix()); err != nil {
		return pendingBootstrapAdmin{}, "", false, err
	}
	if err := tx.Commit(); err != nil {
		return pendingBootstrapAdmin{}, "", false, err
	}
	return pendingBootstrapAdmin{User: &User{
		ID: userID, Email: normalizedEmail, Name: name, Role: RoleAdmin,
		Status: bootstrapUserStatusProvisioning, CreatedAt: now.Unix(), UpdatedAt: now.Unix(),
	}, SessionID: sessionID, SessionIssuedAt: now, SessionExpiresAt: sourceExpiresAt}, passwordHash, true, nil
}

func loadPendingBootstrapAdmin(ctx context.Context, tx *sql.Tx) (pendingBootstrapAdmin, string, error) {
	row := tx.QueryRowContext(ctx, `SELECT u.id,u.email,u.name,u.role,u.status,
       u.created_at,u.updated_at,u.last_login_at,u.password_hash,i.session_id,
       i.source_issued_at,i.source_expires_at
FROM dashboard_bootstrap_installation i
JOIN users u ON u.id=i.user_id
WHERE i.singleton=1`)
	user := &User{}
	var lastLogin sql.NullInt64
	var passwordHash, sessionID string
	var sourceIssuedAt, sourceExpiresAt int64
	if err := row.Scan(
		&user.ID, &user.Email, &user.Name, &user.Role, &user.Status,
		&user.CreatedAt, &user.UpdatedAt, &lastLogin, &passwordHash, &sessionID,
		&sourceIssuedAt, &sourceExpiresAt,
	); err != nil {
		return pendingBootstrapAdmin{}, "", err
	}
	if user.Status != bootstrapUserStatusProvisioning || !uuidValid(user.ID) || !uuidValid(sessionID) ||
		sourceIssuedAt <= 0 || sourceExpiresAt <= sourceIssuedAt {
		return pendingBootstrapAdmin{}, "", errors.New("dashboard bootstrap installation state is invalid")
	}
	if lastLogin.Valid {
		value := lastLogin.Int64
		user.LastLoginAt = &value
	}
	return pendingBootstrapAdmin{
		User: user, SessionID: sessionID,
		SessionIssuedAt:  time.Unix(sourceIssuedAt, 0).UTC(),
		SessionExpiresAt: time.Unix(sourceExpiresAt, 0).UTC(),
	}, passwordHash, nil
}

func (s *Store) CompleteBootstrapAdmin(
	ctx context.Context,
	userID string,
	session localSessionDraft,
	updatedAt time.Time,
) (*User, error) {
	updatedAt = updatedAt.UTC()
	if !uuidValid(session.ID) || session.IssuedAt <= 0 || session.ExpiresAt <= session.IssuedAt ||
		updatedAt.IsZero() {
		return nil, ErrBootstrapClosed
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()
	result, err := tx.ExecContext(ctx, `UPDATE users SET status=?,updated_at=?
WHERE id=? AND status=? AND EXISTS (
  SELECT 1 FROM dashboard_bootstrap_installation
  WHERE singleton=1 AND user_id=? AND session_id=?
    AND source_issued_at=? AND source_expires_at=?
)`, defaultUserStatusActive, updatedAt.Unix(), userID, bootstrapUserStatusProvisioning,
		userID, session.ID, session.IssuedAt, session.ExpiresAt)
	if err != nil {
		return nil, err
	}
	affected, err := result.RowsAffected()
	if err != nil || affected != 1 {
		return nil, ErrBootstrapClosed
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO auth_sessions(id,user_id,issued_at,expires_at)
VALUES(?,?,?,?)`, session.ID, userID, session.IssuedAt, session.ExpiresAt); err != nil {
		return nil, err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM dashboard_bootstrap_installation
WHERE singleton=1 AND user_id=? AND session_id=?
  AND source_issued_at=? AND source_expires_at=?`,
		userID, session.ID, session.IssuedAt, session.ExpiresAt); err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return s.GetUserByID(ctx, userID)
}

func uuidValid(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}
