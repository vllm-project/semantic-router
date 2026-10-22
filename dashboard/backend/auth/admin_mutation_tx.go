package auth

import (
	"context"
	"database/sql"
	"errors"
	"net/http"
	"time"
)

var errAdminSessionInvalid = errors.New("admin session is no longer active")

func optionalAdminActor(actors []AuthContext) *AuthContext {
	if len(actors) == 0 {
		return nil
	}
	return &actors[0]
}

// withAdminMutation serializes the actor check and the account write on the
// same SQLite connection. A revocation committed before the check is observed;
// a concurrent database writer cannot make this transaction commit with a
// stale authorization snapshot.
// Direct in-process callers without a Dashboard session retain the existing
// store API behavior.
func (s *Store) withAdminMutation(ctx context.Context, actor *AuthContext, mutate func(*sql.Tx) error) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if actor != nil && actor.SessionID != "" {
		if err := checkAdminActorInTransaction(ctx, tx, *actor); err != nil {
			return err
		}
	}
	if err := mutate(tx); err != nil {
		return err
	}
	return tx.Commit()
}

func checkAdminActorInTransaction(ctx context.Context, tx *sql.Tx, actor AuthContext) error {
	var role, status string
	if err := tx.QueryRowContext(ctx, `SELECT role, status FROM users WHERE id = ?`, actor.UserID).Scan(&role, &status); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return errAdminSessionInvalid
		}
		return err
	}
	if status != defaultUserStatusActive {
		return errAdminSessionInvalid
	}
	var expiresAt int64
	var revokedAt sql.NullInt64
	if err := tx.QueryRowContext(ctx, `SELECT expires_at, revoked_at FROM auth_sessions WHERE id = ? AND user_id = ?`, actor.SessionID, actor.UserID).Scan(&expiresAt, &revokedAt); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return errAdminSessionInvalid
		}
		return err
	}
	if revokedAt.Valid || expiresAt <= time.Now().Unix() {
		return errAdminSessionInvalid
	}
	var allowed bool
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(
		(SELECT allowed FROM user_permissions WHERE user_id = ? AND permission_key = ?),
		(SELECT allowed FROM role_permissions WHERE role = ? AND permission_key = ?), 0)`,
		actor.UserID, PermUsersManage, role, PermUsersManage).Scan(&allowed); err != nil {
		return err
	}
	if !allowed {
		return ErrPermissionDenied
	}
	return nil
}

func writeAdminMutationAuthorizationError(w http.ResponseWriter, err error) bool {
	switch {
	case errors.Is(err, ErrPermissionDenied):
		http.Error(w, "Forbidden", http.StatusForbidden)
	case errors.Is(err, errAdminSessionInvalid):
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
	default:
		return false
	}
	return true
}
