package auth

import (
	"context"
	"database/sql"
	"errors"
	"strings"
)

type userPasswordState struct {
	user           *User
	hash           string
	authGeneration int64
}

func (s *Store) GetUserWithPasswordHash(ctx context.Context, userID string) (*User, string, error) {
	state, err := s.getUserPasswordStateByID(ctx, userID)
	if err != nil {
		return nil, "", err
	}
	return state.user, state.hash, nil
}

func (s *Store) getUserPasswordStateByID(
	ctx context.Context,
	userID string,
) (*userPasswordState, error) {
	row := s.db.QueryRowContext(
		ctx,
		`SELECT id, email, name, role, status, created_at, updated_at,
		        last_login_at, password_hash, auth_generation
		 FROM users WHERE id = ?`,
		userID,
	)
	return scanUserPasswordState(row)
}

func (s *Store) getUserPasswordStateByEmail(
	ctx context.Context,
	email string,
) (*userPasswordState, error) {
	row := s.db.QueryRowContext(
		ctx,
		`SELECT id, email, name, role, status, created_at, updated_at,
		        last_login_at, password_hash, auth_generation
		 FROM users WHERE email = ?`,
		strings.ToLower(strings.TrimSpace(email)),
	)
	return scanUserPasswordState(row)
}

func scanUserPasswordState(row *sql.Row) (*userPasswordState, error) {
	user := &User{}
	var (
		lastLogin      sql.NullInt64
		hash           string
		authGeneration int64
	)
	if err := row.Scan(
		&user.ID,
		&user.Email,
		&user.Name,
		&user.Role,
		&user.Status,
		&user.CreatedAt,
		&user.UpdatedAt,
		&lastLogin,
		&hash,
		&authGeneration,
	); err != nil {
		return nil, err
	}
	user.Role = canonicalRole(user.Role)
	if lastLogin.Valid {
		value := lastLogin.Int64
		user.LastLoginAt = &value
	}
	return &userPasswordState{
		user:           user,
		hash:           hash,
		authGeneration: authGeneration,
	}, nil
}

// ChangePasswordAndReplaceSessions performs a self-service password change as
// one SQLite transaction. The expected hash and monotonic auth generation
// close concurrent password and active/inactive ABA races; every prior session
// is revoked before the fresh session is inserted.
func (s *Store) ChangePasswordAndReplaceSessions(
	ctx context.Context,
	userID string,
	expectedHash string,
	expectedAuthGeneration int64,
	newHash string,
	issued *issuedToken,
) error {
	if issued == nil {
		return errors.New("issued session is required")
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()

	updatedAt := nowUnix()
	result, err := tx.ExecContext(
		ctx,
		`UPDATE users
		 SET password_hash = ?, auth_generation = auth_generation + 1, updated_at = ?
		 WHERE id = ? AND password_hash = ? AND auth_generation = ? AND status = ?`,
		newHash,
		updatedAt,
		userID,
		expectedHash,
		expectedAuthGeneration,
		defaultUserStatusActive,
	)
	if err != nil {
		return err
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if affected == 0 {
		return ErrPasswordChanged
	}
	if err := revokeUserSessionsTx(ctx, tx, userID, updatedAt); err != nil {
		return err
	}
	if err := insertIssuedSessionTx(ctx, tx, userID, issued); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *Store) UpdatePasswordAndRevokeSessions(
	ctx context.Context,
	userID string,
	newHash string,
) error {
	return s.updatePasswordAndRevokeSessions(ctx, nil, userID, newHash)
}

func (s *Store) updatePasswordAndRevokeSessionsAuthorized(
	ctx context.Context,
	authorization *mutationAuthorization,
	userID string,
	newHash string,
) error {
	return s.updatePasswordAndRevokeSessions(ctx, authorization, userID, newHash)
}

func (s *Store) updatePasswordAndRevokeSessions(
	ctx context.Context,
	authorization *mutationAuthorization,
	userID string,
	newHash string,
) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireMutationAuthorizationTx(ctx, tx, authorization); err != nil {
		return err
	}

	updatedAt := nowUnix()
	result, err := tx.ExecContext(
		ctx,
		`UPDATE users
		 SET password_hash = ?, auth_generation = auth_generation + 1, updated_at = ?
		 WHERE id = ?`,
		newHash,
		updatedAt,
		userID,
	)
	if err != nil {
		return err
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if affected == 0 {
		return sql.ErrNoRows
	}
	if err := revokeUserSessionsTx(ctx, tx, userID, updatedAt); err != nil {
		return err
	}
	return tx.Commit()
}

func revokeUserSessionsTx(ctx context.Context, tx *sql.Tx, userID string, revokedAt int64) error {
	_, err := tx.ExecContext(
		ctx,
		`UPDATE auth_sessions
		 SET revoked_at = COALESCE(revoked_at, ?)
		 WHERE user_id = ?`,
		revokedAt,
		userID,
	)
	return err
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
	return err
}
