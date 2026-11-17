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
