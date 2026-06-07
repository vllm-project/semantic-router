// Package configprojection persists derived read models for deployed router config.
package configprojection

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// Store is a SQLite-backed deployment projection store.
type Store struct {
	db *sql.DB
	mu sync.Mutex
}

// Open opens or creates the config projection SQLite database at dbPath.
func Open(dbPath string) (*Store, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("configprojection: create dir: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=1")
	if err != nil {
		return nil, fmt.Errorf("configprojection: open: %w", err)
	}
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("configprojection: ping: %w", err)
	}

	s := &Store{db: db}
	if err := s.initSchema(); err != nil {
		_ = db.Close()
		return nil, err
	}

	log.Printf("Config projection database initialized at: %s", dbPath)
	return s, nil
}

func (s *Store) initSchema() error {
	schema := `
CREATE TABLE IF NOT EXISTS config_deployments (
	version TEXT PRIMARY KEY,
	source TEXT NOT NULL,
	created_at TEXT NOT NULL,
	dsl_snapshot TEXT NOT NULL DEFAULT '',
	yaml_hash TEXT NOT NULL,
	validation_json TEXT NOT NULL,
	models_json TEXT NOT NULL,
	signals_json TEXT NOT NULL,
	decisions_json TEXT NOT NULL,
	plugins_json TEXT NOT NULL,
	projections_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_config_deployments_created ON config_deployments(created_at DESC);

CREATE TABLE IF NOT EXISTS config_projection_active (
	id INTEGER PRIMARY KEY CHECK (id = 1),
	active_version TEXT NOT NULL DEFAULT '',
	status TEXT NOT NULL DEFAULT 'failed',
	last_error TEXT NOT NULL DEFAULT '',
	updated_at TEXT NOT NULL
);
INSERT OR IGNORE INTO config_projection_active (id, active_version, status, last_error, updated_at)
VALUES (1, '', 'failed', 'projection not initialized', datetime('now'));
`
	_, err := s.db.Exec(schema)
	return err
}

// Close closes the underlying database handle.
func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

// RefreshFromCanonical rebuilds and persists the active deployment projection.
func (s *Store) RefreshFromCanonical(input RefreshInput) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	snapshot, err := BuildSnapshot(input)
	if err != nil {
		if markErr := s.markStaleLocked(err); markErr != nil {
			return fmt.Errorf("configprojection: build failed: %w (mark stale: %w)", err, markErr)
		}
		return err
	}

	validationJSON, err := json.Marshal(snapshot.Validation)
	if err != nil {
		return fmt.Errorf("configprojection: marshal validation: %w", err)
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("configprojection: begin tx: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	if _, err := tx.Exec(`
INSERT INTO config_deployments (
	version, source, created_at, dsl_snapshot, yaml_hash, validation_json,
	models_json, signals_json, decisions_json, plugins_json, projections_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(version) DO UPDATE SET
	source = excluded.source,
	created_at = excluded.created_at,
	dsl_snapshot = excluded.dsl_snapshot,
	yaml_hash = excluded.yaml_hash,
	validation_json = excluded.validation_json,
	models_json = excluded.models_json,
	signals_json = excluded.signals_json,
	decisions_json = excluded.decisions_json,
	plugins_json = excluded.plugins_json,
	projections_json = excluded.projections_json
`,
		snapshot.Version,
		snapshot.Source,
		snapshot.CreatedAt.Format(time.RFC3339Nano),
		snapshot.DSLSnapshot,
		snapshot.YAMLHash,
		string(validationJSON),
		string(snapshot.Models),
		string(snapshot.Signals),
		string(snapshot.Decisions),
		string(snapshot.Plugins),
		string(snapshot.Projections),
	); err != nil {
		return fmt.Errorf("configprojection: upsert deployment: %w", err)
	}

	now := time.Now().UTC().Format(time.RFC3339Nano)
	if _, err := tx.Exec(`
INSERT INTO config_projection_active (id, active_version, status, last_error, updated_at)
VALUES (1, ?, ?, '', ?)
ON CONFLICT(id) DO UPDATE SET
	active_version = excluded.active_version,
	status = excluded.status,
	last_error = '',
	updated_at = excluded.updated_at
`, snapshot.Version, StatusOK, now); err != nil {
		return fmt.Errorf("configprojection: update active row: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("configprojection: commit: %w", err)
	}

	return nil
}

// MarkStale records projection drift without mutating canonical config files.
func (s *Store) MarkStale(err error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.markStaleLocked(err)
}

func (s *Store) markStaleLocked(err error) error {
	lastError := ""
	if err != nil {
		lastError = err.Error()
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	_, execErr := s.db.Exec(`
UPDATE config_projection_active
SET status = ?, last_error = ?, updated_at = ?
WHERE id = 1
`, StatusStale, lastError, now)
	if execErr != nil {
		return fmt.Errorf("configprojection: mark stale: %w", execErr)
	}
	return nil
}

// ListDeployments returns deployment summaries ordered by created_at desc.
func (s *Store) ListDeployments() ([]DeploymentSummary, error) {
	rows, err := s.db.Query(`
SELECT version, source, created_at, yaml_hash
FROM config_deployments
ORDER BY created_at DESC
`)
	if err != nil {
		return nil, fmt.Errorf("configprojection: list deployments: %w", err)
	}
	defer rows.Close()

	summaries := []DeploymentSummary{}
	for rows.Next() {
		var (
			version   string
			source    string
			createdAt string
			yamlHash  string
		)
		if scanErr := rows.Scan(&version, &source, &createdAt, &yamlHash); scanErr != nil {
			return nil, fmt.Errorf("configprojection: scan deployment: %w", scanErr)
		}
		parsedAt, parseErr := time.Parse(time.RFC3339Nano, createdAt)
		if parseErr != nil {
			parsedAt, parseErr = time.Parse(time.RFC3339, createdAt)
			if parseErr != nil {
				return nil, fmt.Errorf("configprojection: parse created_at: %w", parseErr)
			}
		}
		summaries = append(summaries, DeploymentSummary{
			Version:   version,
			Source:    source,
			CreatedAt: parsedAt,
			YAMLHash:  yamlHash,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("configprojection: iterate deployments: %w", err)
	}
	return summaries, nil
}

// GetDeployment returns one deployment projection by version.
func (s *Store) GetDeployment(version string) (*DeploymentRecord, error) {
	version = strings.TrimSpace(version)
	if version == "" {
		return nil, fmt.Errorf("configprojection: version is required")
	}

	row := s.db.QueryRow(`
SELECT version, source, created_at, dsl_snapshot, yaml_hash, validation_json,
	models_json, signals_json, decisions_json, plugins_json, projections_json
FROM config_deployments
WHERE version = ?
`, version)

	record, err := scanDeploymentRow(row, version)
	if err != nil {
		return nil, err
	}
	return record, nil
}

// GetActiveProjection returns active drift metadata and the active deployment record.
func (s *Store) GetActiveProjection() (*ActiveProjectionResponse, error) {
	status, err := s.getActiveStatus()
	if err != nil {
		return nil, err
	}

	resp := &ActiveProjectionResponse{ActiveProjectionStatus: *status}
	if strings.TrimSpace(status.ActiveVersion) == "" {
		return resp, nil
	}

	deployment, err := s.GetDeployment(status.ActiveVersion)
	if err != nil {
		return resp, nil
	}
	resp.Deployment = deployment
	return resp, nil
}

func (s *Store) getActiveStatus() (*ActiveProjectionStatus, error) {
	row := s.db.QueryRow(`
SELECT active_version, status, last_error, updated_at
FROM config_projection_active
WHERE id = 1
`)

	var (
		activeVersion string
		status        string
		lastError     string
		updatedAt     string
	)
	if err := row.Scan(&activeVersion, &status, &lastError, &updatedAt); err != nil {
		if err == sql.ErrNoRows {
			return &ActiveProjectionStatus{Status: StatusFailed}, nil
		}
		return nil, fmt.Errorf("configprojection: read active status: %w", err)
	}

	parsedAt, err := time.Parse(time.RFC3339Nano, updatedAt)
	if err != nil {
		parsedAt, err = time.Parse(time.RFC3339, updatedAt)
		if err != nil {
			return nil, fmt.Errorf("configprojection: parse updated_at: %w", err)
		}
	}

	return &ActiveProjectionStatus{
		ActiveVersion: activeVersion,
		Status:        status,
		LastError:     lastError,
		UpdatedAt:     parsedAt,
	}, nil
}

func scanDeploymentRow(row *sql.Row, version string) (*DeploymentRecord, error) {
	var (
		record         DeploymentRecord
		createdAt      string
		validationRaw  string
		modelsRaw      string
		signalsRaw     string
		decisionsRaw   string
		pluginsRaw     string
		projectionsRaw string
	)
	if err := row.Scan(
		&record.Version,
		&record.Source,
		&createdAt,
		&record.DSLSnapshot,
		&record.YAMLHash,
		&validationRaw,
		&modelsRaw,
		&signalsRaw,
		&decisionsRaw,
		&pluginsRaw,
		&projectionsRaw,
	); err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("configprojection: deployment %q not found", version)
		}
		return nil, fmt.Errorf("configprojection: scan deployment: %w", err)
	}

	parsedAt, err := time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		parsedAt, err = time.Parse(time.RFC3339, createdAt)
		if err != nil {
			return nil, fmt.Errorf("configprojection: parse created_at: %w", err)
		}
	}
	record.CreatedAt = parsedAt

	if err := json.Unmarshal([]byte(validationRaw), &record.Validation); err != nil {
		return nil, fmt.Errorf("configprojection: decode validation: %w", err)
	}
	record.Models = json.RawMessage(modelsRaw)
	record.Signals = json.RawMessage(signalsRaw)
	record.Decisions = json.RawMessage(decisionsRaw)
	record.Plugins = json.RawMessage(pluginsRaw)
	record.Projections = json.RawMessage(projectionsRaw)
	return &record, nil
}
