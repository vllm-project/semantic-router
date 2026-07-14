package auth

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"time"
)

var ErrAuthorizationChanged = errors.New("authorization state changed")

type mutationAuthorization struct {
	userID     string
	sessionID  string
	permission string
}

func usersManageAuthorization(ac AuthContext) *mutationAuthorization {
	return &mutationAuthorization{
		userID:     ac.UserID,
		sessionID:  ac.SessionID,
		permission: PermUsersManage,
	}
}

func requireMutationAuthorizationTx(
	ctx context.Context,
	tx *sql.Tx,
	authorization *mutationAuthorization,
) error {
	if authorization == nil {
		return nil
	}
	userID := strings.TrimSpace(authorization.userID)
	sessionID := strings.TrimSpace(authorization.sessionID)
	permission := strings.TrimSpace(authorization.permission)
	if userID == "" || sessionID == "" || permission == "" {
		return ErrAuthorizationChanged
	}

	var allowed int
	err := tx.QueryRowContext(
		ctx,
		`SELECT CASE WHEN EXISTS (
			SELECT 1
			FROM users AS u
			JOIN auth_sessions AS s ON s.user_id = u.id
			WHERE u.id = ?
			  AND u.status = ?
			  AND s.id = ?
			  AND s.revoked_at IS NULL
			  AND s.expires_at > ?
			  AND (
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
			  )
		) THEN 1 ELSE 0 END`,
		userID,
		defaultUserStatusActive,
		sessionID,
		time.Now().Unix(),
		permission,
		permission,
	).Scan(&allowed)
	if err != nil {
		return err
	}
	if allowed != 1 {
		return ErrAuthorizationChanged
	}
	return nil
}
