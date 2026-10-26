// Package workflowstore provides server-owned durable state for dashboard workflows
// (ML pipeline jobs, MCP servers). Live SSE/WebSocket client maps remain in
// handlers; this store holds reconstructable job and entity records.
package workflowstore

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	_ "github.com/mattn/go-sqlite3"
)

// Store is a SQLite-backed workflow control-plane store.
type Store struct {
	db *sql.DB
	mu sync.Mutex
}

// Open opens or creates the workflow SQLite database at dbPath.
func Open(dbPath string) (*Store, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("workflowstore: create dir: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=1&_txlock=immediate")
	if err != nil {
		return nil, fmt.Errorf("workflowstore: open: %w", err)
	}
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("workflowstore: ping: %w", err)
	}

	s := &Store{db: db}
	if err := s.initSchema(); err != nil {
		_ = db.Close()
		return nil, err
	}

	log.Printf("Workflow database initialized at: %s", dbPath)
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

CREATE TABLE IF NOT EXISTS mcp_server (
	id TEXT PRIMARY KEY,
	json TEXT NOT NULL
);
`
	if _, err := s.db.Exec(schema); err != nil {
		return err
	}
	return s.initTrainingSchema()
}

// Close releases the database handle.
func (s *Store) Close() error {
	return s.db.Close()
}
