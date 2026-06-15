package auth

import (
	"context"
	"strings"
)

func (s *Store) normalizeStoredRoles() error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer func() {
		_ = tx.Rollback()
	}()

	rows, err := tx.Query(`SELECT id, role FROM users`)
	if err != nil {
		return err
	}
	defer func() {
		_ = rows.Close()
	}()

	type roleUpdate struct {
		userID string
		role   string
	}

	var updates []roleUpdate
	for rows.Next() {
		var userID string
		var rawRole string
		if err := rows.Scan(&userID, &rawRole); err != nil {
			return err
		}
		normalized, err := normalizeRole(rawRole)
		if err != nil || normalized == "" || normalized == rawRole {
			continue
		}
		updates = append(updates, roleUpdate{userID: userID, role: normalized})
	}
	if err := rows.Err(); err != nil {
		return err
	}

	for _, update := range updates {
		if _, err := tx.Exec(`UPDATE users SET role = ?, updated_at = ? WHERE id = ?`, update.role, nowUnix(), update.userID); err != nil {
			return err
		}
	}

	return tx.Commit()
}

func (s *Store) syncDefaultRolePermissions() error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer func() {
		_ = tx.Rollback()
	}()

	for legacyRole, targetRole := range legacyRoleAliases {
		if _, err := tx.Exec(
			`INSERT INTO role_permissions(role, permission_key, allowed)
SELECT ?, permission_key, allowed FROM role_permissions WHERE role = ?
ON CONFLICT(role, permission_key) DO UPDATE SET allowed = excluded.allowed`,
			targetRole,
			legacyRole,
		); err != nil {
			return err
		}

		if _, err := tx.Exec(`DELETE FROM role_permissions WHERE role = ?`, legacyRole); err != nil {
			return err
		}
	}

	for role, perms := range DefaultRolePermissions {
		if _, err := tx.Exec(`DELETE FROM role_permissions WHERE role = ?`, role); err != nil {
			return err
		}

		for _, perm := range perms {
			if _, err := tx.Exec(`INSERT INTO role_permissions(role, permission_key, allowed) VALUES(?,?,1)
ON CONFLICT(role, permission_key) DO UPDATE SET allowed = 1`, role, strings.TrimSpace(perm)); err != nil {
				return err
			}
		}
	}
	return tx.Commit()
}

func (s *Store) GetEffectivePermissions(ctx context.Context, role string, userID string) (map[string]bool, error) {
	permMap := map[string]bool{}
	rows, err := s.db.QueryContext(ctx, `SELECT permission_key FROM role_permissions WHERE role = ? AND allowed = 1`, role)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rows.Close()
	}()
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err == nil {
			permMap[p] = true
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if userID != "" {
		uRows, err := s.db.QueryContext(ctx, `SELECT permission_key FROM user_permissions WHERE user_id = ? AND allowed = 1`, userID)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = uRows.Close()
		}()
		for uRows.Next() {
			var p string
			if err := uRows.Scan(&p); err == nil {
				permMap[p] = true
			}
		}
		if err := uRows.Err(); err != nil {
			return nil, err
		}
	}
	return permMap, nil
}

func (s *Store) CountActiveUsersWithPermission(ctx context.Context, permission string, excludeUserID string) (int, error) {
	query := `SELECT id, role FROM users WHERE status = ?`
	args := []interface{}{defaultUserStatusActive}
	if excludeUserID != "" {
		query += ` AND id <> ?`
		args = append(args, excludeUserID)
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return 0, err
	}

	type activeUser struct {
		id   string
		role string
	}

	var users []activeUser
	for rows.Next() {
		var user activeUser
		if err := rows.Scan(&user.id, &user.role); err != nil {
			_ = rows.Close()
			return 0, err
		}
		users = append(users, user)
	}
	if err := rows.Err(); err != nil {
		_ = rows.Close()
		return 0, err
	}
	if err := rows.Close(); err != nil {
		return 0, err
	}

	count := 0
	for _, user := range users {
		perms, err := s.GetEffectivePermissions(ctx, user.role, user.id)
		if err != nil {
			return 0, err
		}
		if perms[permission] {
			count++
		}
	}
	return count, nil
}

func (s *Store) ListRolePermissions(ctx context.Context) (map[string][]string, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT role, permission_key FROM role_permissions WHERE allowed = 1 ORDER BY role, permission_key`)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rows.Close()
	}()

	out := map[string][]string{}
	for rows.Next() {
		var role, perm string
		if err := rows.Scan(&role, &perm); err != nil {
			return nil, err
		}
		out[role] = append(out[role], perm)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
