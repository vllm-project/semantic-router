package console

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"
)

// SQLiteStore implements the console Store contract on SQLite.
type SQLiteStore struct {
	db *sql.DB
	mu sync.RWMutex
}

// NewSQLiteStore opens a SQLite-backed console store and initializes its schema.
func NewSQLiteStore(dbPath string) (*SQLiteStore, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create console database directory: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=on")
	if err != nil {
		return nil, fmt.Errorf("failed to open console database: %w", err)
	}

	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to ping console database: %w", err)
	}

	store := &SQLiteStore{db: db}
	if err := store.initSchema(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to initialize console schema: %w", err)
	}

	return store, nil
}

// Close closes the underlying database handle.
func (s *SQLiteStore) Close() error {
	return s.db.Close()
}

func (s *SQLiteStore) initSchema() error {
	for _, statement := range consoleSchemaStatements() {
		if _, err := s.db.Exec(statement); err != nil {
			return err
		}
	}
	return nil
}

func ensureID(id *string) {
	if *id == "" {
		*id = uuid.NewString()
	}
}

func ensureCreatedUpdated(createdAt *time.Time, updatedAt *time.Time) {
	now := time.Now().UTC()
	if createdAt.IsZero() {
		*createdAt = now
	}
	*updatedAt = now
}

func defaultOccurredAt(occurredAt *time.Time) {
	if occurredAt.IsZero() {
		*occurredAt = time.Now().UTC()
	}
}

func metadataJSON(metadata map[string]interface{}) (string, error) {
	if len(metadata) == 0 {
		return "{}", nil
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func decodeMetadata(raw string) (map[string]interface{}, error) {
	if raw == "" {
		return map[string]interface{}{}, nil
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		return nil, err
	}
	if decoded == nil {
		return map[string]interface{}{}, nil
	}
	return decoded, nil
}

func nullTime(t *time.Time) interface{} {
	if t == nil || t.IsZero() {
		return nil
	}
	return *t
}

func scanOptionalTime(value sql.NullTime) *time.Time {
	if !value.Valid {
		return nil
	}
	t := value.Time
	return &t
}

func scanOptionalString(value sql.NullString) string {
	if !value.Valid {
		return ""
	}
	return value.String
}

func withReadLock[T any](ctx context.Context, s *SQLiteStore, fn func(context.Context) (T, error)) (T, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return fn(ctx)
}

func withWriteLock(ctx context.Context, s *SQLiteStore, fn func(context.Context) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return fn(ctx)
}

func consoleSchemaStatements() []string {
	return []string{
		identitySchema(),
		revisionSchema(),
		deploymentSchema(),
		secretSchema(),
		auditSchema(),
	}
}

func identitySchema() string {
	return `
	CREATE TABLE IF NOT EXISTS console_users (
		id TEXT PRIMARY KEY,
		email TEXT,
		display_name TEXT,
		auth_provider TEXT,
		external_subject TEXT,
		status TEXT NOT NULL,
		last_login_at TIMESTAMP,
		metadata_json TEXT NOT NULL,
		created_at TIMESTAMP NOT NULL,
		updated_at TIMESTAMP NOT NULL
	);
	CREATE UNIQUE INDEX IF NOT EXISTS idx_console_users_provider_subject
		ON console_users(auth_provider, external_subject);

	CREATE TABLE IF NOT EXISTS console_role_bindings (
		id TEXT PRIMARY KEY,
		principal_type TEXT NOT NULL,
		principal_id TEXT NOT NULL,
		role TEXT NOT NULL,
		scope_type TEXT NOT NULL,
		scope_id TEXT,
		granted_by TEXT,
		metadata_json TEXT NOT NULL,
		created_at TIMESTAMP NOT NULL,
		updated_at TIMESTAMP NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_console_role_bindings_principal
		ON console_role_bindings(principal_type, principal_id);
	CREATE INDEX IF NOT EXISTS idx_console_role_bindings_scope
		ON console_role_bindings(scope_type, scope_id);

	CREATE TABLE IF NOT EXISTS console_sessions (
		id TEXT PRIMARY KEY,
		user_id TEXT NOT NULL,
		auth_provider TEXT,
		external_subject TEXT,
		status TEXT NOT NULL,
		expires_at TIMESTAMP,
		revoked_at TIMESTAMP,
		metadata_json TEXT NOT NULL,
		created_at TIMESTAMP NOT NULL,
		updated_at TIMESTAMP NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_console_sessions_user
		ON console_sessions(user_id);
	CREATE INDEX IF NOT EXISTS idx_console_sessions_status
		ON console_sessions(status);
	`
}

func revisionSchema() string {
	return `
	CREATE TABLE IF NOT EXISTS console_config_revisions (
		id TEXT PRIMARY KEY,
		parent_revision_id TEXT,
		status TEXT NOT NULL,
		source TEXT,
		summary TEXT,
		document_json TEXT NOT NULL,
		runtime_config_yaml TEXT,
		created_by TEXT,
		activated_at TIMESTAMP,
		metadata_json TEXT NOT NULL,
		created_at TIMESTAMP NOT NULL,
		updated_at TIMESTAMP NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_console_config_revisions_status
		ON console_config_revisions(status);
	CREATE INDEX IF NOT EXISTS idx_console_config_revisions_created_at
		ON console_config_revisions(created_at DESC);
	`
}

func deploymentSchema() string {
	return `
	CREATE TABLE IF NOT EXISTS console_deploy_events (
		id TEXT PRIMARY KEY,
		revision_id TEXT NOT NULL,
		status TEXT NOT NULL,
		trigger_source TEXT,
		message TEXT,
		runtime_target TEXT,
		rollback_revision_id TEXT,
		metadata_json TEXT NOT NULL,
		started_at TIMESTAMP,
		completed_at TIMESTAMP,
		created_at TIMESTAMP NOT NULL,
		updated_at TIMESTAMP NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_console_deploy_events_revision
		ON console_deploy_events(revision_id);
	CREATE INDEX IF NOT EXISTS idx_console_deploy_events_created_at
		ON console_deploy_events(created_at DESC);
	`
}

func secretSchema() string {
	return `
	CREATE TABLE IF NOT EXISTS console_secret_refs (
		id TEXT PRIMARY KEY,
		scope_type TEXT NOT NULL,
		scope_id TEXT,
		provider TEXT,
		external_ref TEXT NOT NULL,
		version TEXT,
		redacted_label TEXT,
		last_rotated_at TIMESTAMP,
		metadata_json TEXT NOT NULL,
		created_at TIMESTAMP NOT NULL,
		updated_at TIMESTAMP NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_console_secret_refs_scope
		ON console_secret_refs(scope_type, scope_id);
	`
}

func auditSchema() string {
	return `
	CREATE TABLE IF NOT EXISTS console_audit_events (
		id TEXT PRIMARY KEY,
		actor_type TEXT NOT NULL,
		actor_id TEXT,
		action TEXT NOT NULL,
		target_type TEXT,
		target_id TEXT,
		outcome TEXT NOT NULL,
		message TEXT,
		metadata_json TEXT NOT NULL,
		occurred_at TIMESTAMP NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_console_audit_events_actor
		ON console_audit_events(actor_id);
	CREATE INDEX IF NOT EXISTS idx_console_audit_events_target
		ON console_audit_events(target_type, target_id);
	CREATE INDEX IF NOT EXISTS idx_console_audit_events_occurred_at
		ON console_audit_events(occurred_at DESC);
	`
}
