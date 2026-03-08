package console

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// SaveUser creates or updates a console user.
func (s *SQLiteStore) SaveUser(ctx context.Context, user *User) error {
	if user == nil {
		return fmt.Errorf("user is required")
	}

	ensureID(&user.ID)
	if user.Status == "" {
		user.Status = UserStatusActive
	}
	ensureCreatedUpdated(&user.CreatedAt, &user.UpdatedAt)

	metadata, err := metadataJSON(user.Metadata)
	if err != nil {
		return fmt.Errorf("failed to encode user metadata: %w", err)
	}

	return withWriteLock(ctx, s, func(ctx context.Context) error {
		query := `
			INSERT INTO console_users (
				id, email, display_name, auth_provider, external_subject, status,
				last_login_at, metadata_json, created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				email = excluded.email,
				display_name = excluded.display_name,
				auth_provider = excluded.auth_provider,
				external_subject = excluded.external_subject,
				status = excluded.status,
				last_login_at = excluded.last_login_at,
				metadata_json = excluded.metadata_json,
				updated_at = excluded.updated_at
		`

		_, err := s.db.ExecContext(
			ctx,
			query,
			user.ID,
			user.Email,
			user.DisplayName,
			user.AuthProvider,
			user.ExternalSubject,
			user.Status,
			nullTime(user.LastLoginAt),
			metadata,
			user.CreatedAt,
			user.UpdatedAt,
		)
		return err
	})
}

// GetUser fetches a console user by ID.
func (s *SQLiteStore) GetUser(ctx context.Context, id string) (*User, error) {
	if id == "" {
		return nil, fmt.Errorf("user id is required")
	}

	return withReadLock(ctx, s, func(ctx context.Context) (*User, error) {
		query := `
			SELECT id, email, display_name, auth_provider, external_subject, status,
			       last_login_at, metadata_json, created_at, updated_at
			FROM console_users
			WHERE id = ?
		`

		var user User
		var lastLoginAt sql.NullTime
		var metadata string

		err := s.db.QueryRowContext(ctx, query, id).Scan(
			&user.ID,
			&user.Email,
			&user.DisplayName,
			&user.AuthProvider,
			&user.ExternalSubject,
			&user.Status,
			&lastLoginAt,
			&metadata,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}

		user.LastLoginAt = scanOptionalTime(lastLoginAt)
		user.Metadata, err = decodeMetadata(metadata)
		if err != nil {
			return nil, fmt.Errorf("failed to decode user metadata: %w", err)
		}

		return &user, nil
	})
}

// SaveRoleBinding creates or updates a role binding.
func (s *SQLiteStore) SaveRoleBinding(ctx context.Context, binding *RoleBinding) error {
	if binding == nil {
		return fmt.Errorf("role binding is required")
	}
	if binding.PrincipalID == "" {
		return fmt.Errorf("principal_id is required")
	}
	if binding.Role == "" {
		return fmt.Errorf("role is required")
	}

	ensureID(&binding.ID)
	if binding.PrincipalType == "" {
		binding.PrincipalType = PrincipalTypeUser
	}
	if binding.ScopeType == "" {
		binding.ScopeType = ScopeTypeGlobal
	}
	ensureCreatedUpdated(&binding.CreatedAt, &binding.UpdatedAt)

	metadata, err := metadataJSON(binding.Metadata)
	if err != nil {
		return fmt.Errorf("failed to encode role binding metadata: %w", err)
	}

	return withWriteLock(ctx, s, func(ctx context.Context) error {
		query := `
			INSERT INTO console_role_bindings (
				id, principal_type, principal_id, role, scope_type, scope_id,
				granted_by, metadata_json, created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				principal_type = excluded.principal_type,
				principal_id = excluded.principal_id,
				role = excluded.role,
				scope_type = excluded.scope_type,
				scope_id = excluded.scope_id,
				granted_by = excluded.granted_by,
				metadata_json = excluded.metadata_json,
				updated_at = excluded.updated_at
		`

		_, err := s.db.ExecContext(
			ctx,
			query,
			binding.ID,
			binding.PrincipalType,
			binding.PrincipalID,
			binding.Role,
			binding.ScopeType,
			binding.ScopeID,
			binding.GrantedBy,
			metadata,
			binding.CreatedAt,
			binding.UpdatedAt,
		)
		return err
	})
}

// ListRoleBindings returns role bindings that match the supplied filter.
func (s *SQLiteStore) ListRoleBindings(ctx context.Context, filter RoleBindingFilter) ([]RoleBinding, error) {
	return withReadLock(ctx, s, func(ctx context.Context) ([]RoleBinding, error) {
		query, args := buildRoleBindingListQuery(filter)
		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = rows.Close()
		}()

		var bindings []RoleBinding
		for rows.Next() {
			var binding RoleBinding
			var scopeID sql.NullString
			var grantedBy sql.NullString
			var metadata string
			scanErr := rows.Scan(
				&binding.ID,
				&binding.PrincipalType,
				&binding.PrincipalID,
				&binding.Role,
				&binding.ScopeType,
				&scopeID,
				&grantedBy,
				&metadata,
				&binding.CreatedAt,
				&binding.UpdatedAt,
			)
			if scanErr != nil {
				return nil, scanErr
			}

			binding.ScopeID = scanOptionalString(scopeID)
			binding.GrantedBy = scanOptionalString(grantedBy)
			binding.Metadata, err = decodeMetadata(metadata)
			if err != nil {
				return nil, fmt.Errorf("failed to decode role binding metadata: %w", err)
			}
			bindings = append(bindings, binding)
		}

		if err := rows.Err(); err != nil {
			return nil, err
		}
		return bindings, nil
	})
}

// SaveSession creates or updates a console session record.
func (s *SQLiteStore) SaveSession(ctx context.Context, session *Session) error {
	if session == nil {
		return fmt.Errorf("session is required")
	}
	if session.UserID == "" {
		return fmt.Errorf("user_id is required")
	}

	ensureID(&session.ID)
	if session.Status == "" {
		session.Status = SessionStatusActive
	}
	ensureCreatedUpdated(&session.CreatedAt, &session.UpdatedAt)

	metadata, err := metadataJSON(session.Metadata)
	if err != nil {
		return fmt.Errorf("failed to encode session metadata: %w", err)
	}

	return withWriteLock(ctx, s, func(ctx context.Context) error {
		query := `
			INSERT INTO console_sessions (
				id, user_id, auth_provider, external_subject, status,
				expires_at, revoked_at, metadata_json, created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				user_id = excluded.user_id,
				auth_provider = excluded.auth_provider,
				external_subject = excluded.external_subject,
				status = excluded.status,
				expires_at = excluded.expires_at,
				revoked_at = excluded.revoked_at,
				metadata_json = excluded.metadata_json,
				updated_at = excluded.updated_at
		`

		_, err := s.db.ExecContext(
			ctx,
			query,
			session.ID,
			session.UserID,
			session.AuthProvider,
			session.ExternalSubject,
			session.Status,
			nullTime(session.ExpiresAt),
			nullTime(session.RevokedAt),
			metadata,
			session.CreatedAt,
			session.UpdatedAt,
		)
		return err
	})
}

// GetSession fetches a session by ID.
func (s *SQLiteStore) GetSession(ctx context.Context, id string) (*Session, error) {
	if id == "" {
		return nil, fmt.Errorf("session id is required")
	}

	return withReadLock(ctx, s, func(ctx context.Context) (*Session, error) {
		query := `
			SELECT id, user_id, auth_provider, external_subject, status,
			       expires_at, revoked_at, metadata_json, created_at, updated_at
			FROM console_sessions
			WHERE id = ?
		`

		var session Session
		var authProvider sql.NullString
		var externalSubject sql.NullString
		var expiresAt sql.NullTime
		var revokedAt sql.NullTime
		var metadata string

		err := s.db.QueryRowContext(ctx, query, id).Scan(
			&session.ID,
			&session.UserID,
			&authProvider,
			&externalSubject,
			&session.Status,
			&expiresAt,
			&revokedAt,
			&metadata,
			&session.CreatedAt,
			&session.UpdatedAt,
		)
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}

		session.AuthProvider = scanOptionalString(authProvider)
		session.ExternalSubject = scanOptionalString(externalSubject)
		session.ExpiresAt = scanOptionalTime(expiresAt)
		session.RevokedAt = scanOptionalTime(revokedAt)
		session.Metadata, err = decodeMetadata(metadata)
		if err != nil {
			return nil, fmt.Errorf("failed to decode session metadata: %w", err)
		}

		return &session, nil
	})
}

// ListSessions returns sessions that match the supplied filter.
func (s *SQLiteStore) ListSessions(ctx context.Context, filter SessionFilter) ([]Session, error) {
	return withReadLock(ctx, s, func(ctx context.Context) ([]Session, error) {
		query := `
			SELECT id, user_id, auth_provider, external_subject, status,
			       expires_at, revoked_at, metadata_json, created_at, updated_at
			FROM console_sessions
		`

		var clauses []string
		var args []interface{}
		if filter.UserID != "" {
			clauses = append(clauses, "user_id = ?")
			args = append(args, filter.UserID)
		}
		if filter.Status != "" {
			clauses = append(clauses, "status = ?")
			args = append(args, filter.Status)
		}
		if len(clauses) > 0 {
			query += " WHERE " + strings.Join(clauses, " AND ")
		}
		query += " ORDER BY created_at DESC LIMIT ?"
		args = append(args, normalizedLimit(filter.Limit))

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = rows.Close()
		}()

		var sessions []Session
		for rows.Next() {
			var session Session
			var authProvider sql.NullString
			var externalSubject sql.NullString
			var expiresAt sql.NullTime
			var revokedAt sql.NullTime
			var metadata string

			scanErr := rows.Scan(
				&session.ID,
				&session.UserID,
				&authProvider,
				&externalSubject,
				&session.Status,
				&expiresAt,
				&revokedAt,
				&metadata,
				&session.CreatedAt,
				&session.UpdatedAt,
			)
			if scanErr != nil {
				return nil, scanErr
			}

			session.AuthProvider = scanOptionalString(authProvider)
			session.ExternalSubject = scanOptionalString(externalSubject)
			session.ExpiresAt = scanOptionalTime(expiresAt)
			session.RevokedAt = scanOptionalTime(revokedAt)
			session.Metadata, err = decodeMetadata(metadata)
			if err != nil {
				return nil, fmt.Errorf("failed to decode session metadata: %w", err)
			}
			sessions = append(sessions, session)
		}

		if err := rows.Err(); err != nil {
			return nil, err
		}
		return sessions, nil
	})
}

func buildRoleBindingListQuery(filter RoleBindingFilter) (string, []interface{}) {
	query := `
		SELECT id, principal_type, principal_id, role, scope_type, scope_id,
		       granted_by, metadata_json, created_at, updated_at
		FROM console_role_bindings
	`

	var clauses []string
	var args []interface{}
	clauses, args = appendOptionalClause(clauses, args, "principal_type = ?", string(filter.PrincipalType))
	clauses, args = appendOptionalClause(clauses, args, "principal_id = ?", filter.PrincipalID)
	clauses, args = appendOptionalClause(clauses, args, "role = ?", string(filter.Role))
	clauses, args = appendOptionalClause(clauses, args, "scope_type = ?", string(filter.ScopeType))
	clauses, args = appendOptionalClause(clauses, args, "scope_id = ?", filter.ScopeID)
	if len(clauses) > 0 {
		query += " WHERE " + strings.Join(clauses, " AND ")
	}
	query += " ORDER BY created_at DESC LIMIT ?"
	args = append(args, normalizedLimit(filter.Limit))
	return query, args
}

func appendOptionalClause(clauses []string, args []interface{}, clause string, value string) ([]string, []interface{}) {
	if value == "" {
		return clauses, args
	}
	return append(clauses, clause), append(args, value)
}
