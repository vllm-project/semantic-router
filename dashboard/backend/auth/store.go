package auth

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
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
	db             *sql.DB
	filesystemPath string
}

func NewStore(path string) (*Store, error) {
	filesystemPath, err := prepareAuthDatabaseStorage(path)
	if err != nil {
		return nil, err
	}
	dsn, err := authSQLiteDSN(path)
	if err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("open auth db: %w", err)
	}
	db.SetMaxOpenConns(1)
	db.SetConnMaxLifetime(time.Minute)

	if _, err := db.ExecContext(context.Background(), createUsersSchema); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("migrate schema: %w", err)
	}
	if err := ensureUserAuthGenerationColumn(context.Background(), db); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("migrate auth generation: %w", err)
	}

	store := &Store{db: db, filesystemPath: filesystemPath}
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
	if err := secureExistingAuthDatabaseFiles(filesystemPath); err != nil {
		_ = db.Close()
		return nil, err
	}

	return store, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	dbErr := s.db.Close()
	modeErr := secureExistingAuthDatabaseFiles(s.filesystemPath)
	return errors.Join(dbErr, modeErr)
}

func (s *Store) CountUsers(ctx context.Context) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM users`).Scan(&n); err != nil {
		return 0, err
	}
	return n, nil
}

func (s *Store) CreateUser(ctx context.Context, email, name, hash, role, status string) (*User, error) {
	user, err := prepareNewUser(email, name, role, status)
	if err != nil {
		return nil, err
	}
	if err := insertPreparedUser(ctx, s.db, user, hash); err != nil {
		return nil, err
	}
	return user, nil
}

type userInsertExecutor interface {
	ExecContext(context.Context, string, ...any) (sql.Result, error)
}

func prepareNewUser(email, name, role, status string) (*User, error) {
	email = strings.ToLower(strings.TrimSpace(email))
	if status == "" {
		status = defaultUserStatusActive
	}
	status = strings.ToLower(strings.TrimSpace(status))
	if status != defaultUserStatusActive && status != "inactive" {
		return nil, errors.New("status must be active or inactive")
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
	return &User{
		ID:        id,
		Email:     email,
		Name:      name,
		Role:      role,
		Status:    status,
		CreatedAt: createdAt,
		UpdatedAt: createdAt,
	}, nil
}

func insertPreparedUser(
	ctx context.Context,
	executor userInsertExecutor,
	user *User,
	hash string,
) error {
	_, err := executor.ExecContext(
		ctx,
		`INSERT INTO users(
		 id, email, name, password_hash, role, status, created_at, updated_at
		) VALUES(?,?,?,?,?,?,?,?)`,
		user.ID,
		user.Email,
		user.Name,
		hash,
		user.Role,
		user.Status,
		user.CreatedAt,
		user.UpdatedAt,
	)
	return err
}

func (s *Store) GetUserByEmail(ctx context.Context, email string) (id, emailOut, name, role, status string, createdAt, updatedAt int64, lastLogin *int64, hash string, err error) {
	row := s.db.QueryRowContext(
		ctx,
		`SELECT id, email, name, role, status, created_at, updated_at, last_login_at, password_hash FROM users WHERE email = ?`,
		strings.ToLower(strings.TrimSpace(email)),
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

func nowUnix() int64 { return time.Now().Unix() }
