package auth

import (
	"context"
	"database/sql"
	"errors"
)

var (
	ErrUserStateChanged      = errors.New("user state changed concurrently")
	ErrLastActiveUserManager = errors.New("cannot remove the last active user manager")
)

type userStateExpectation struct {
	role   string
	status string
}

// UpdateUserRoleOrStatus applies identity state, the last-manager invariant,
// and credential invalidation as one transaction. Blank values retain the
// corresponding current field.
func (s *Store) UpdateUserRoleOrStatus(
	ctx context.Context,
	userID string,
	role string,
	status string,
) (*User, error) {
	if role == "" && status == "" {
		return s.GetUserByID(ctx, userID)
	}
	return s.updateUserRoleOrStatus(ctx, userID, role, status, nil, nil)
}

// UpdateUserRoleOrStatusIfCurrent rejects a stale handler snapshot instead of
// restoring role or status values changed by a concurrent request.
func (s *Store) UpdateUserRoleOrStatusIfCurrent(
	ctx context.Context,
	userID string,
	expectedRole string,
	expectedStatus string,
	role string,
	status string,
) (*User, error) {
	return s.updateUserRoleOrStatus(ctx, userID, role, status, &userStateExpectation{
		role:   canonicalRole(expectedRole),
		status: expectedStatus,
	}, nil)
}

func (s *Store) updateUserRoleOrStatusAuthorizedIfCurrent(
	ctx context.Context,
	authorization *mutationAuthorization,
	userID string,
	expectedRole string,
	expectedStatus string,
	role string,
	status string,
) (*User, error) {
	return s.updateUserRoleOrStatus(ctx, userID, role, status, &userStateExpectation{
		role:   canonicalRole(expectedRole),
		status: expectedStatus,
	}, authorization)
}

func (s *Store) updateUserRoleOrStatus(
	ctx context.Context,
	userID string,
	role string,
	status string,
	expected *userStateExpectation,
	authorization *mutationAuthorization,
) (*User, error) {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireMutationAuthorizationTx(ctx, tx, authorization); err != nil {
		return nil, err
	}

	current, err := getUserForMutationTx(ctx, tx, userID)
	if err != nil {
		return nil, userMutationReadError(err, expected != nil)
	}
	if expected != nil &&
		(current.Role != expected.role || current.Status != expected.status) {
		return nil, ErrUserStateChanged
	}

	nextRole, nextStatus, err := normalizeStoredUserMutation(current, role, status)
	if err != nil {
		return nil, err
	}
	if err := requireRemainingUserManagerTx(
		ctx,
		tx,
		current,
		nextRole,
		nextStatus,
	); err != nil {
		return nil, err
	}

	updatedAt := nowUnix()
	result, err := tx.ExecContext(
		ctx,
		`UPDATE users
		 SET role = ?,
		     status = ?,
		     auth_generation = auth_generation + CASE WHEN status <> ? THEN 1 ELSE 0 END,
		     updated_at = ?
		 WHERE id = ? AND role = ? AND status = ?`,
		nextRole,
		nextStatus,
		nextStatus,
		updatedAt,
		userID,
		current.Role,
		current.Status,
	)
	if err != nil {
		return nil, err
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return nil, err
	}
	if affected != 1 {
		return nil, ErrUserStateChanged
	}

	// Deactivation is a credential-lifecycle boundary, not a temporary access
	// mask. A concurrent login either commits first and is revoked here or loses
	// its active-status compare-and-swap.
	if nextStatus != defaultUserStatusActive {
		if err := revokeUserSessionsTx(ctx, tx, userID, updatedAt); err != nil {
			return nil, err
		}
	}

	updated, err := getUserForMutationTx(ctx, tx, userID)
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return updated, nil
}

// DeleteUser removes a user while preserving the last active user manager.
func (s *Store) DeleteUser(ctx context.Context, userID string) error {
	return s.deleteUser(ctx, userID, nil, nil)
}

// DeleteUserIfCurrent rejects deletion when role or status changed after the
// handler loaded its authorization snapshot.
func (s *Store) DeleteUserIfCurrent(
	ctx context.Context,
	userID string,
	expectedRole string,
	expectedStatus string,
) error {
	return s.deleteUser(ctx, userID, &userStateExpectation{
		role:   canonicalRole(expectedRole),
		status: expectedStatus,
	}, nil)
}

func (s *Store) deleteUserAuthorizedIfCurrent(
	ctx context.Context,
	authorization *mutationAuthorization,
	userID string,
	expectedRole string,
	expectedStatus string,
) error {
	return s.deleteUser(ctx, userID, &userStateExpectation{
		role:   canonicalRole(expectedRole),
		status: expectedStatus,
	}, authorization)
}

func (s *Store) deleteUser(
	ctx context.Context,
	userID string,
	expected *userStateExpectation,
	authorization *mutationAuthorization,
) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := requireMutationAuthorizationTx(ctx, tx, authorization); err != nil {
		return err
	}

	current, err := getUserForMutationTx(ctx, tx, userID)
	if err != nil {
		return userMutationReadError(err, expected != nil)
	}
	if expected != nil &&
		(current.Role != expected.role || current.Status != expected.status) {
		return ErrUserStateChanged
	}
	if err := requireRemainingUserManagerTx(
		ctx,
		tx,
		current,
		current.Role,
		"inactive",
	); err != nil {
		return err
	}

	result, err := tx.ExecContext(
		ctx,
		`DELETE FROM users WHERE id = ? AND role = ? AND status = ?`,
		userID,
		current.Role,
		current.Status,
	)
	if err != nil {
		return err
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if affected != 1 {
		return ErrUserStateChanged
	}
	return tx.Commit()
}

func getUserForMutationTx(
	ctx context.Context,
	tx *sql.Tx,
	userID string,
) (*User, error) {
	return scanUser(tx.QueryRowContext(
		ctx,
		`SELECT id, email, name, role, status, created_at, updated_at, last_login_at
		 FROM users WHERE id = ?`,
		userID,
	))
}

func normalizeStoredUserMutation(
	current *User,
	role string,
	status string,
) (string, string, error) {
	nextRole := current.Role
	if role != "" {
		normalizedRole, err := normalizeRole(role)
		if err != nil {
			return "", "", err
		}
		nextRole = normalizedRole
	}

	nextStatus := current.Status
	if status != "" {
		if status != defaultUserStatusActive && status != "inactive" {
			return "", "", errors.New("status must be active or inactive")
		}
		nextStatus = status
	}
	return nextRole, nextStatus, nil
}

func requireRemainingUserManagerTx(
	ctx context.Context,
	tx *sql.Tx,
	current *User,
	nextRole string,
	nextStatus string,
) error {
	currentlyManages, err := userHasPermissionTx(
		ctx,
		tx,
		current.ID,
		current.Role,
		current.Status,
		PermUsersManage,
	)
	if err != nil || !currentlyManages {
		return err
	}
	wouldManage, err := userHasPermissionTx(
		ctx,
		tx,
		current.ID,
		nextRole,
		nextStatus,
		PermUsersManage,
	)
	if err != nil || wouldManage {
		return err
	}

	remaining, err := countActiveUsersWithPermissionTx(
		ctx,
		tx,
		PermUsersManage,
		current.ID,
	)
	if err != nil {
		return err
	}
	if remaining == 0 {
		return ErrLastActiveUserManager
	}
	return nil
}

func userHasPermissionTx(
	ctx context.Context,
	tx *sql.Tx,
	userID string,
	role string,
	status string,
	permission string,
) (bool, error) {
	if status != defaultUserStatusActive {
		return false, nil
	}
	var allowed int
	err := tx.QueryRowContext(
		ctx,
		`SELECT CASE WHEN
			EXISTS (
				SELECT 1 FROM role_permissions
				WHERE role = ? AND permission_key = ? AND allowed = 1
			)
			OR EXISTS (
				SELECT 1 FROM user_permissions
				WHERE user_id = ? AND permission_key = ? AND allowed = 1
			)
		 THEN 1 ELSE 0 END`,
		role,
		permission,
		userID,
		permission,
	).Scan(&allowed)
	return allowed == 1, err
}

func countActiveUsersWithPermissionTx(
	ctx context.Context,
	tx *sql.Tx,
	permission string,
	excludeUserID string,
) (int, error) {
	var count int
	err := tx.QueryRowContext(
		ctx,
		`SELECT COUNT(*)
		 FROM users AS u
		 WHERE u.status = ? AND u.id <> ? AND (
			EXISTS (
				SELECT 1 FROM role_permissions AS rp
				WHERE rp.role = u.role
				  AND rp.permission_key = ?
				  AND rp.allowed = 1
			)
			OR EXISTS (
				SELECT 1 FROM user_permissions AS up
				WHERE up.user_id = u.id
				  AND up.permission_key = ?
				  AND up.allowed = 1
			)
		 )`,
		defaultUserStatusActive,
		excludeUserID,
		permission,
		permission,
	).Scan(&count)
	return count, err
}

func userMutationReadError(err error, staleIsConflict bool) error {
	if staleIsConflict && errors.Is(err, sql.ErrNoRows) {
		return ErrUserStateChanged
	}
	return err
}
