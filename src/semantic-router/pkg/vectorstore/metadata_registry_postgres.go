/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	_ "github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
)

const (
	defaultStoreTableName = "vector_store_registry"
	defaultFileTableName  = "file_registry"
)

// PostgresMetadataRegistry persists VectorStore and FileRecord metadata
// in PostgreSQL tables so the router can recover its inventory on restart.
type PostgresMetadataRegistry struct {
	db             *sql.DB
	storeTableName string
	fileTableName  string
}

// NewPostgresMetadataRegistry opens a Postgres connection, creates the
// required tables if they don't exist, and returns a ready-to-use registry.
func NewPostgresMetadataRegistry(cfg *postgres.Config) (*PostgresMetadataRegistry, error) {
	rc, err := postgres.NewRuntimeConfig(cfg, defaultStoreTableName)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	db, err := postgres.OpenDB(ctx, rc)
	if err != nil {
		return nil, err
	}

	reg := &PostgresMetadataRegistry{
		db:             db,
		storeTableName: rc.TableName,
		fileTableName:  defaultFileTableName,
	}

	if err := reg.createTables(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to create metadata tables: %w", err)
	}
	return reg, nil
}

func (r *PostgresMetadataRegistry) createTables(ctx context.Context) error {
	storeQ := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		id              VARCHAR(255) PRIMARY KEY,
		name            VARCHAR(255) DEFAULT '',
		created_at      BIGINT       NOT NULL,
		status          VARCHAR(50)  DEFAULT 'active',
		backend_type    VARCHAR(50)  DEFAULT '',
		fc_in_progress  INTEGER      DEFAULT 0,
		fc_completed    INTEGER      DEFAULT 0,
		fc_failed       INTEGER      DEFAULT 0,
		fc_total        INTEGER      DEFAULT 0,
		expires_anchor  VARCHAR(50)  DEFAULT '',
		expires_days    INTEGER      DEFAULT 0,
		metadata        JSONB
	)`, r.storeTableName)

	fileQ := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		id         VARCHAR(255) PRIMARY KEY,
		bytes      BIGINT       DEFAULT 0,
		created_at BIGINT       NOT NULL,
		filename   VARCHAR(512) DEFAULT '',
		purpose    VARCHAR(50)  DEFAULT '',
		status     VARCHAR(50)  DEFAULT ''
	)`, r.fileTableName)

	if _, err := r.db.ExecContext(ctx, storeQ); err != nil {
		return fmt.Errorf("create %s: %w", r.storeTableName, err)
	}
	if _, err := r.db.ExecContext(ctx, fileQ); err != nil {
		return fmt.Errorf("create %s: %w", r.fileTableName, err)
	}
	return nil
}

// --- VectorStore CRUD ---

func (r *PostgresMetadataRegistry) SaveStore(ctx context.Context, vs *VectorStore) error {
	metaJSON, err := json.Marshal(vs.Metadata)
	if err != nil {
		return fmt.Errorf("marshal metadata: %w", err)
	}

	expiresAnchor, expiresDays := flattenExpiration(vs.ExpiresAfter)

	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`INSERT INTO %s
		(id, name, created_at, status, backend_type,
		 fc_in_progress, fc_completed, fc_failed, fc_total,
		 expires_anchor, expires_days, metadata)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
		ON CONFLICT (id) DO UPDATE SET
		 name=EXCLUDED.name, status=EXCLUDED.status,
		 fc_in_progress=EXCLUDED.fc_in_progress, fc_completed=EXCLUDED.fc_completed,
		 fc_failed=EXCLUDED.fc_failed, fc_total=EXCLUDED.fc_total,
		 expires_anchor=EXCLUDED.expires_anchor, expires_days=EXCLUDED.expires_days,
		 metadata=EXCLUDED.metadata
	`, r.storeTableName)

	_, err = r.db.ExecContext(ctx, q,
		vs.ID, vs.Name, vs.CreatedAt, vs.Status, vs.BackendType,
		vs.FileCounts.InProgress, vs.FileCounts.Completed,
		vs.FileCounts.Failed, vs.FileCounts.Total,
		expiresAnchor, expiresDays, metaJSON,
	)
	return err
}

func (r *PostgresMetadataRegistry) GetStore(ctx context.Context, id string) (*VectorStore, error) {
	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`SELECT id, name, created_at, status, backend_type,
		fc_in_progress, fc_completed, fc_failed, fc_total,
		expires_anchor, expires_days, metadata
		FROM %s WHERE id = $1`, r.storeTableName)

	row := r.db.QueryRowContext(ctx, q, id)
	vs, err := scanVectorStoreRow(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("vector store not found: %s", id)
		}
		return nil, fmt.Errorf("get store %s: %w", id, err)
	}
	return vs, nil
}

func (r *PostgresMetadataRegistry) ListStores(ctx context.Context) ([]*VectorStore, error) {
	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`SELECT id, name, created_at, status, backend_type,
		fc_in_progress, fc_completed, fc_failed, fc_total,
		expires_anchor, expires_days, metadata
		FROM %s ORDER BY created_at DESC`, r.storeTableName)

	rows, err := r.db.QueryContext(ctx, q)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var stores []*VectorStore
	for rows.Next() {
		vs, err := scanVectorStoreRows(rows)
		if err != nil {
			return nil, err
		}
		stores = append(stores, vs)
	}
	return stores, rows.Err()
}

func (r *PostgresMetadataRegistry) DeleteStore(ctx context.Context, id string) error {
	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`DELETE FROM %s WHERE id = $1`, r.storeTableName)
	_, err := r.db.ExecContext(ctx, q, id)
	return err
}

// --- FileRecord CRUD ---

func (r *PostgresMetadataRegistry) SaveFile(ctx context.Context, fr *FileRecord) error {
	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`INSERT INTO %s
		(id, bytes, created_at, filename, purpose, status)
		VALUES ($1,$2,$3,$4,$5,$6)
		ON CONFLICT (id) DO UPDATE SET
		 bytes=EXCLUDED.bytes, filename=EXCLUDED.filename,
		 purpose=EXCLUDED.purpose, status=EXCLUDED.status
	`, r.fileTableName)

	_, err := r.db.ExecContext(ctx, q,
		fr.ID, fr.Bytes, fr.CreatedAt, fr.Filename, fr.Purpose, fr.Status,
	)
	return err
}

func (r *PostgresMetadataRegistry) GetFile(ctx context.Context, id string) (*FileRecord, error) {
	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`SELECT id, bytes, created_at, filename, purpose, status
		FROM %s WHERE id = $1`, r.fileTableName)

	var fr FileRecord
	err := r.db.QueryRowContext(ctx, q, id).Scan(
		&fr.ID, &fr.Bytes, &fr.CreatedAt, &fr.Filename, &fr.Purpose, &fr.Status,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("file not found: %s", id)
		}
		return nil, fmt.Errorf("get file %s: %w", id, err)
	}
	fr.Object = "file"
	return &fr, nil
}

func (r *PostgresMetadataRegistry) ListFiles(ctx context.Context) ([]*FileRecord, error) {
	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`SELECT id, bytes, created_at, filename, purpose, status
		FROM %s ORDER BY created_at DESC`, r.fileTableName)

	rows, err := r.db.QueryContext(ctx, q)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	var files []*FileRecord
	for rows.Next() {
		var fr FileRecord
		if err := rows.Scan(&fr.ID, &fr.Bytes, &fr.CreatedAt, &fr.Filename, &fr.Purpose, &fr.Status); err != nil {
			return nil, err
		}
		fr.Object = "file"
		files = append(files, &fr)
	}
	return files, rows.Err()
}

func (r *PostgresMetadataRegistry) DeleteFile(ctx context.Context, id string) error {
	//nolint:gosec // table name validated at construction
	q := fmt.Sprintf(`DELETE FROM %s WHERE id = $1`, r.fileTableName)
	_, err := r.db.ExecContext(ctx, q, id)
	return err
}

// Close shuts down the underlying database connection.
func (r *PostgresMetadataRegistry) Close() error {
	return r.db.Close()
}

// --- helpers ---

type rowScanner interface {
	Scan(dest ...interface{}) error
}

func scanVectorStoreFromRow(s rowScanner) (*VectorStore, error) {
	var (
		vs            VectorStore
		expiresAnchor string
		expiresDays   int
		metaJSON      []byte
	)
	err := s.Scan(
		&vs.ID, &vs.Name, &vs.CreatedAt, &vs.Status, &vs.BackendType,
		&vs.FileCounts.InProgress, &vs.FileCounts.Completed,
		&vs.FileCounts.Failed, &vs.FileCounts.Total,
		&expiresAnchor, &expiresDays, &metaJSON,
	)
	if err != nil {
		return nil, err
	}
	vs.Object = "vector_store"
	vs.ExpiresAfter = unflattenExpiration(expiresAnchor, expiresDays)
	if len(metaJSON) > 0 && string(metaJSON) != "null" {
		if err := json.Unmarshal(metaJSON, &vs.Metadata); err != nil {
			return nil, fmt.Errorf("unmarshal metadata for %q: %w", vs.ID, err)
		}
	}
	return &vs, nil
}

func scanVectorStoreRow(row *sql.Row) (*VectorStore, error)    { return scanVectorStoreFromRow(row) }
func scanVectorStoreRows(rows *sql.Rows) (*VectorStore, error) { return scanVectorStoreFromRow(rows) }

func flattenExpiration(ep *ExpirationPolicy) (string, int) {
	if ep == nil {
		return "", 0
	}
	return ep.Anchor, ep.Days
}

func unflattenExpiration(anchor string, days int) *ExpirationPolicy {
	if anchor == "" && days == 0 {
		return nil
	}
	return &ExpirationPolicy{Anchor: anchor, Days: days}
}
