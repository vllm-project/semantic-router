// Package workflowstore provides server-owned durable state for dashboard workflows
// (ML pipeline jobs, OpenClaw collaboration entities). Live SSE/WebSocket client maps
// remain in handlers; this store holds reconstructable job and entity records.
package workflowstore

import (
	"database/sql"
	"errors"
	"fmt"
	"log"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

// Options configures opening the workflow database.
type Options struct {
	// LegacyOpenClawDir, if set, is scanned once for containers.json / teams.json /
	// rooms.json / room-messages/*.json and imported when the OpenClaw tables are empty.
	LegacyOpenClawDir string
}

// Store is a SQLite-backed workflow control-plane store.
type Store struct {
	db             *sql.DB
	filesystemPath string
	mu             sync.Mutex
}

// Open opens or creates the workflow SQLite database at dbPath.
func Open(dbPath string, opts Options) (*Store, error) {
	dsn, err := workflowSQLiteDSN(dbPath)
	if err != nil {
		return nil, err
	}
	filesystemPath, err := prepareWorkflowDatabaseStorage(dbPath)
	if err != nil {
		return nil, err
	}

	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("workflowstore: open: %w", err)
	}
	if err := db.Ping(); err != nil {
		return nil, errors.Join(
			fmt.Errorf("workflowstore: ping: %w", err),
			closeWorkflowDatabase(db, filesystemPath),
		)
	}

	s := &Store{db: db, filesystemPath: filesystemPath}
	if err := s.initSchema(); err != nil {
		return nil, errors.Join(err, s.Close())
	}

	if opts.LegacyOpenClawDir != "" {
		if err := s.maybeImportLegacyOpenClaw(opts.LegacyOpenClawDir); err != nil {
			log.Printf("workflowstore: legacy OpenClaw import: %v", err)
		}
	}
	if err := secureExistingWorkflowDatabaseFiles(filesystemPath); err != nil {
		return nil, errors.Join(err, s.Close())
	}

	if filesystemPath == "" {
		log.Printf("Workflow in-memory database initialized")
	} else {
		log.Printf("Workflow database initialized at: %s", filesystemPath)
	}
	return s, nil
}

func (s *Store) initSchema() error {
	schema := `
CREATE TABLE IF NOT EXISTS ml_pipeline_jobs (
	id TEXT PRIMARY KEY,
	job_type TEXT NOT NULL,
	status TEXT NOT NULL,
	created_at TEXT NOT NULL,
	completed_at TEXT,
	error TEXT,
	output_files_json TEXT,
	progress INTEGER NOT NULL DEFAULT 0,
	current_step TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_ml_jobs_status ON ml_pipeline_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ml_jobs_created ON ml_pipeline_jobs(created_at DESC);

CREATE TABLE IF NOT EXISTS ml_pipeline_progress_events (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	job_id TEXT NOT NULL,
	step TEXT NOT NULL DEFAULT '',
	percent INTEGER NOT NULL DEFAULT 0,
	message TEXT NOT NULL DEFAULT '',
	recorded_at TEXT NOT NULL,
	FOREIGN KEY (job_id) REFERENCES ml_pipeline_jobs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_ml_prog_job ON ml_pipeline_progress_events(job_id, id);

CREATE TABLE IF NOT EXISTS openclaw_container (
	name TEXT PRIMARY KEY,
	json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS openclaw_team (
	id TEXT PRIMARY KEY,
	json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS openclaw_room (
	id TEXT PRIMARY KEY,
	json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS openclaw_room_message (
	seq INTEGER PRIMARY KEY AUTOINCREMENT,
	room_id TEXT NOT NULL,
	message_id TEXT NOT NULL,
	json TEXT NOT NULL,
	UNIQUE(room_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_oc_msg_room ON openclaw_room_message(room_id, seq);

CREATE TABLE IF NOT EXISTS mcp_server (
	id TEXT PRIMARY KEY,
	json TEXT NOT NULL
);
`
	_, err := s.db.Exec(schema)
	return err
}

// Close releases the database handle.
func (s *Store) Close() error {
	if s == nil {
		return nil
	}
	return closeWorkflowDatabase(s.db, s.filesystemPath)
}

// DB exposes the raw connection for health checks (row queries only).
func (s *Store) DB() *sql.DB {
	return s.db
}
