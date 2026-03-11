package auth

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
)

const (
	defaultUserStatusActive = "active"
	defaultPageSize         = 100
)

type Store struct {
	db *sql.DB
}

type AuditLog struct {
	ID         int64  `json:"id"`
	UserID     string `json:"userId"`
	Action     string `json:"action"`
	Resource   string `json:"resource"`
	Method     string `json:"method"`
	Path       string `json:"path"`
	IP         string `json:"ip"`
	UserAgent  string `json:"userAgent"`
	StatusCode int    `json:"statusCode"`
	CreatedAt  int64  `json:"createdAt"`
	ExtraJSON  string `json:"extraJson,omitempty"`
}

func NewStore(path string) (*Store, error) {
	dir := filepath.Dir(path)
	if dir != "." && dir != "" && dir != "/" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("create auth db directory: %w", err)
		}
	}
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, fmt.Errorf("open auth db: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetConnMaxLifetime(time.Minute)

	if _, err := db.ExecContext(context.Background(), createUsersSchema); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("migrate schema: %w", err)
	}

	store := &Store{db: db}
	if err := store.normalizeStoredRoles(); err != nil {
		_ = db.Close()
		return nil, err
	}
	if err := store.syncDefaultRolePermissions(); err != nil {
		_ = db.Close()
		return nil, err
	}

	return store, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

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

func (s *Store) CountUsers(ctx context.Context) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM users`).Scan(&n); err != nil {
		return 0, err
	}
	return n, nil
}

func (s *Store) CreateUser(ctx context.Context, email, name, hash, role, status string) (*User, error) {
	if status == "" {
		status = defaultUserStatusActive
	}
	if role == "" {
		role = RoleRead
	}
	normalizedRole, err := normalizeRole(role)
	if err != nil {
		return nil, err
	}
	if normalizedRole != "" {
		role = normalizedRole
	}
	id := uuid.NewString()
	createdAt := nowUnix()
	_, err = s.db.ExecContext(ctx, `INSERT INTO users(id, email, name, password_hash, role, status, created_at, updated_at)
		VALUES(?,?,?,?,?,?,?,?)`, id, strings.ToLower(email), name, hash, role, status, createdAt, createdAt)
	if err != nil {
		return nil, err
	}
	return &User{ID: id, Email: strings.ToLower(email), Name: name, Role: role, Status: status, CreatedAt: createdAt, UpdatedAt: createdAt}, nil
}

func (s *Store) GetUserByEmail(ctx context.Context, email string) (id, emailOut, name, role, status string, createdAt, updatedAt int64, lastLogin *int64, hash string, err error) {
	row := s.db.QueryRowContext(
		ctx,
		`SELECT id, email, name, role, status, created_at, updated_at, last_login_at, password_hash FROM users WHERE email = ?`,
		strings.ToLower(email),
	)
	var last sql.NullInt64
	err = row.Scan(&id, &emailOut, &name, &role, &status, &createdAt, &updatedAt, &last, &hash)
	if err != nil {
		return id, emailOut, name, role, status, createdAt, updatedAt, lastLogin, hash, err
	}
	role = canonicalRole(role)
	if last.Valid {
		t := last.Int64
		lastLogin = &t
	}
	return id, emailOut, name, role, status, createdAt, updatedAt, lastLogin, hash, err
}

func (s *Store) GetUserByID(ctx context.Context, userID string) (*User, error) {
	row := s.db.QueryRowContext(ctx, `SELECT id, email, name, role, status, created_at, updated_at, last_login_at FROM users WHERE id = ?`, userID)
	return scanUser(row)
}

func (s *Store) ListUsers(ctx context.Context, statusFilter string, limit, offset int) ([]*User, error) {
	if limit <= 0 || limit > 200 {
		limit = defaultPageSize
	}
	q := `SELECT id, email, name, role, status, created_at, updated_at, last_login_at FROM users`
	args := []interface{}{}
	if statusFilter != "" {
		q += ` WHERE status = ?`
		args = append(args, statusFilter)
	}
	q += ` ORDER BY created_at DESC LIMIT ? OFFSET ?`
	args = append(args, limit, offset)

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rows.Close()
	}()

	var out []*User
	for rows.Next() {
		u, err := scanUserRows(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, u)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *Store) UpdateUserRoleOrStatus(ctx context.Context, userID, role, status string) (*User, error) {
	if role == "" && status == "" {
		return s.GetUserByID(ctx, userID)
	}

	updatedAt := nowUnix()
	var (
		res sql.Result
		err error
	)

	switch {
	case role != "" && status != "":
		normalizedRole, normalizeErr := normalizeRole(role)
		if normalizeErr != nil {
			return nil, normalizeErr
		}
		res, err = s.db.ExecContext(
			ctx,
			`UPDATE users SET role = ?, status = ?, updated_at = ? WHERE id = ?`,
			normalizedRole,
			status,
			updatedAt,
			userID,
		)
	case role != "":
		normalizedRole, normalizeErr := normalizeRole(role)
		if normalizeErr != nil {
			return nil, normalizeErr
		}
		res, err = s.db.ExecContext(
			ctx,
			`UPDATE users SET role = ?, updated_at = ? WHERE id = ?`,
			normalizedRole,
			updatedAt,
			userID,
		)
	default:
		res, err = s.db.ExecContext(
			ctx,
			`UPDATE users SET status = ?, updated_at = ? WHERE id = ?`,
			status,
			updatedAt,
			userID,
		)
	}

	if err != nil {
		return nil, err
	}
	affected, _ := res.RowsAffected()
	if affected == 0 {
		return nil, sql.ErrNoRows
	}
	return s.GetUserByID(ctx, userID)
}

func (s *Store) DeleteUser(ctx context.Context, userID string) error {
	res, err := s.db.ExecContext(ctx, `DELETE FROM users WHERE id = ?`, userID)
	if err != nil {
		return err
	}
	affected, _ := res.RowsAffected()
	if affected == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *Store) UpdatePassword(ctx context.Context, userID, passwordHash string) error {
	res, err := s.db.ExecContext(ctx, `UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?`, passwordHash, nowUnix(), userID)
	if err != nil {
		return err
	}
	affected, _ := res.RowsAffected()
	if affected == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *Store) UpdateLoginTime(ctx context.Context, userID string) error {
	_, err := s.db.ExecContext(ctx, `UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?`, nowUnix(), nowUnix(), userID)
	return err
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

func (s *Store) AddAuditLog(ctx context.Context, logRow AuditLog) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO user_audit_logs(user_id, action, resource, method, path, ip, user_agent, status_code, created_at, extra_json)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		nilOrString(logRow.UserID), logRow.Action, logRow.Resource, logRow.Method, logRow.Path, logRow.IP, logRow.UserAgent, logRow.StatusCode, nowUnix(), logRow.ExtraJSON)
	return err
}

func (s *Store) ListAuditLogs(ctx context.Context, userID, action, resource string, limit, offset int) ([]AuditLog, error) {
	if limit <= 0 || limit > 200 {
		limit = defaultPageSize
	}
	q := `SELECT id, user_id, action, resource, method, path, ip, user_agent, status_code, created_at, extra_json FROM user_audit_logs`
	args := []interface{}{}
	predicates := []string{}
	if userID != "" {
		predicates = append(predicates, "user_id = ?")
		args = append(args, userID)
	}
	if action != "" {
		predicates = append(predicates, "action = ?")
		args = append(args, action)
	}
	if resource != "" {
		predicates = append(predicates, "resource = ?")
		args = append(args, resource)
	}
	if len(predicates) > 0 {
		q += " WHERE " + strings.Join(predicates, " AND ")
	}
	q += " ORDER BY id DESC LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rows.Close()
	}()

	var out []AuditLog
	for rows.Next() {
		var row AuditLog
		var uid sql.NullString
		if err := rows.Scan(&row.ID, &uid, &row.Action, &row.Resource, &row.Method, &row.Path, &row.IP, &row.UserAgent, &row.StatusCode, &row.CreatedAt, &row.ExtraJSON); err != nil {
			return nil, err
		}
		if uid.Valid {
			row.UserID = uid.String
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func nowUnix() int64 { return time.Now().Unix() }

func nilOrString(v string) interface{} {
	if v == "" {
		return nil
	}
	return v
}
