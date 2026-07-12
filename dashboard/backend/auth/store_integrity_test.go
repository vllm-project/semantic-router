package auth

import (
	"context"
	"database/sql"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAuthSQLiteDSNEnforcesForeignKeysAndPreservesParameters(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name string
		dsn  string
		base string
	}{
		{name: "plain memory", dsn: ":memory:", base: ":memory:"},
		{name: "uri memory", dsn: "file::memory:?cache=shared&_fk=0", base: "file::memory:"},
		{name: "file", dsn: "file:auth.db?_busy_timeout=4321&_foreign_keys=0", base: "file:auth.db"},
	} {
		t.Run(test.name, func(t *testing.T) {
			dsn, err := authSQLiteDSN(test.dsn)
			if err != nil {
				t.Fatalf("authSQLiteDSN() error = %v", err)
			}
			parts := strings.SplitN(dsn, "?", 2)
			if len(parts) != 2 || parts[0] != test.base {
				t.Fatalf("authSQLiteDSN() = %q, want base %q and query", dsn, test.base)
			}
			values, err := url.ParseQuery(parts[1])
			if err != nil {
				t.Fatalf("ParseQuery(%q): %v", parts[1], err)
			}
			if values.Get("_foreign_keys") != "1" {
				t.Fatalf("_foreign_keys = %q, want 1", values.Get("_foreign_keys"))
			}
			if _, found := values["_fk"]; found {
				t.Fatal("caller-controlled _fk alias survived")
			}
			if strings.Contains(test.dsn, "cache=shared") && values.Get("cache") != "shared" {
				t.Fatalf("cache = %q, want shared", values.Get("cache"))
			}
			if strings.Contains(test.dsn, "_busy_timeout=4321") && values.Get("_busy_timeout") != "4321" {
				t.Fatalf("_busy_timeout = %q, want 4321", values.Get("_busy_timeout"))
			}
		})
	}
}

func TestNewStoreEnforcesForeignKeysOnReplacementConnections(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "auth.db")
	store, err := NewStore("file:" + path + "?_busy_timeout=4321&_fk=0&_foreign_keys=0")
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	assertForeignKeysEnabled(t, store.db)
	var busyTimeout int
	if err := store.db.QueryRowContext(context.Background(), `PRAGMA busy_timeout`).Scan(&busyTimeout); err != nil {
		t.Fatalf("read busy_timeout: %v", err)
	}
	if busyTimeout != 4321 {
		t.Fatalf("busy_timeout = %d, want 4321", busyTimeout)
	}

	// Force database/sql to discard its idle connection. The next query opens
	// a new driver connection, where DSN-level enforcement must still apply.
	store.db.SetMaxIdleConns(0)
	assertForeignKeysEnabled(t, store.db)
}

func TestNewStoreMigratesLegacyAuthGenerationWithoutLosingState(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "legacy-auth.db")
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		t.Fatalf("open legacy database: %v", err)
	}
	legacySchema := `
CREATE TABLE users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  last_login_at INTEGER
);
CREATE TABLE auth_sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  issued_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);`
	if _, err := db.Exec(legacySchema); err != nil {
		_ = db.Close()
		t.Fatalf("create legacy schema: %v", err)
	}
	hash, err := hashVersionedPassword("legacy generation password")
	if err != nil {
		_ = db.Close()
		t.Fatalf("hash legacy password: %v", err)
	}
	now := time.Now().Unix()
	if _, err := db.Exec(
		`INSERT INTO users(
		 id, email, name, password_hash, role, status, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"legacy-user",
		"legacy-generation@example.com",
		"Legacy User",
		hash,
		RoleRead,
		defaultUserStatusActive,
		now,
		now,
	); err != nil {
		_ = db.Close()
		t.Fatalf("insert legacy user: %v", err)
	}
	if _, err := db.Exec(
		`INSERT INTO auth_sessions(id, user_id, issued_at, expires_at)
		 VALUES (?, ?, ?, ?)`,
		"legacy-session",
		"legacy-user",
		now,
		now+3600,
	); err != nil {
		_ = db.Close()
		t.Fatalf("insert legacy session: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close legacy database: %v", err)
	}

	store, err := NewStore(path)
	if err != nil {
		t.Fatalf("NewStore() migration error = %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	state, err := store.getUserPasswordStateByID(context.Background(), "legacy-user")
	if err != nil {
		t.Fatalf("get migrated user state: %v", err)
	}
	if state.authGeneration != 0 {
		t.Fatalf("migrated auth generation = %d, want 0", state.authGeneration)
	}
	active, err := store.SessionActive(
		context.Background(),
		"legacy-session",
		"legacy-user",
		now,
	)
	if err != nil {
		t.Fatalf("read migrated session: %v", err)
	}
	if !active {
		t.Fatal("legacy active session was lost during additive migration")
	}

	svc, err := NewService(store, testJWTSecret, 1)
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	if _, _, err := svc.Login(
		context.Background(),
		"legacy-generation@example.com",
		"legacy generation password",
	); err != nil {
		t.Fatalf("login after additive migration: %v", err)
	}
}

func TestAuthStoreDeleteAppliesForeignKeyActions(t *testing.T) {
	t.Parallel()

	store := mustNewStore(t, filepath.Join(t.TempDir(), "auth.db"))
	svc, err := NewService(store, testJWTSecret, 1)
	if err != nil {
		t.Fatalf("NewService() error = %v", err)
	}
	user := newTestUser(t, svc, "foreign-key-user@example.com", RoleRead, defaultUserStatusActive)
	ctx := context.Background()
	if _, err := store.db.ExecContext(
		ctx,
		`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES (?, ?, 1)`,
		user.ID,
		PermConfigRead,
	); err != nil {
		t.Fatalf("insert user permission: %v", err)
	}
	if err := store.CreateSession(ctx, "foreign-key-session", user.ID, time.Now().Unix(), time.Now().Add(time.Hour).Unix()); err != nil {
		t.Fatalf("CreateSession() error = %v", err)
	}
	if err := store.AddAuditLog(ctx, AuditLog{
		UserID:   user.ID,
		Action:   "user.test",
		Resource: "users",
	}); err != nil {
		t.Fatalf("AddAuditLog() error = %v", err)
	}

	if err := store.DeleteUser(ctx, user.ID); err != nil {
		t.Fatalf("DeleteUser() error = %v", err)
	}
	for table, column := range map[string]string{
		"auth_sessions":    "user_id",
		"user_permissions": "user_id",
	} {
		var count int
		query := `SELECT COUNT(*) FROM ` + table + ` WHERE ` + column + ` = ?`
		if err := store.db.QueryRowContext(ctx, query, user.ID).Scan(&count); err != nil {
			t.Fatalf("count %s: %v", table, err)
		}
		if count != 0 {
			t.Fatalf("%s rows for deleted user = %d, want 0", table, count)
		}
	}
	var auditUserID sql.NullString
	if err := store.db.QueryRowContext(
		ctx,
		`SELECT user_id FROM user_audit_logs WHERE action = 'user.test'`,
	).Scan(&auditUserID); err != nil {
		t.Fatalf("read audit row: %v", err)
	}
	if auditUserID.Valid {
		t.Fatalf("audit user id = %q, want NULL after user deletion", auditUserID.String)
	}
	if err := store.CreateSession(
		ctx,
		"orphan-session",
		"missing-user",
		time.Now().Unix(),
		time.Now().Add(time.Hour).Unix(),
	); err == nil {
		t.Fatal("CreateSession() accepted a nonexistent user")
	}
}

func assertForeignKeysEnabled(t *testing.T, db *sql.DB) {
	t.Helper()
	var enabled int
	if err := db.QueryRowContext(context.Background(), `PRAGMA foreign_keys`).Scan(&enabled); err != nil {
		t.Fatalf("read foreign_keys: %v", err)
	}
	if enabled != 1 {
		t.Fatalf("foreign_keys = %d, want 1", enabled)
	}
}
