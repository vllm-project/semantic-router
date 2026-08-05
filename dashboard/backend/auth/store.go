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
	if err := store.PruneInactiveSessions(context.Background(), time.Now()); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("prune auth sessions: %w", err)
	}

	return store, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
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

func nowUnix() int64 { return time.Now().Unix() }
