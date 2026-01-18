package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "github.com/lib/pq"
)

const (
	DefaultPostgresTableName       = "router_replay_records"
	DefaultPostgresMaxOpenConns    = 25
	DefaultPostgresMaxIdleConns    = 5
	DefaultPostgresConnMaxLifetime = 300 // 5 minutes
)

// PostgresStore implements Storage using PostgreSQL as the backend.
type PostgresStore struct {
	db          *sql.DB
	tableName   string
	ttl         time.Duration
	asyncWrites bool
	asyncChan   chan asyncOp
	done        chan struct{}
}

// NewPostgresStore creates a new PostgreSQL storage backend.
func NewPostgresStore(cfg *PostgresConfig, ttlSeconds int, asyncWrites bool) (*PostgresStore, error) {
	if cfg == nil {
		return nil, fmt.Errorf("postgres config is required")
	}

	if cfg.Host == "" {
		cfg.Host = "localhost"
	}
	if cfg.Port == 0 {
		cfg.Port = 5432
	}
	if cfg.Database == "" {
		return nil, fmt.Errorf("postgres database name is required")
	}
	if cfg.User == "" {
		return nil, fmt.Errorf("postgres user is required")
	}

	sslMode := cfg.SSLMode
	if sslMode == "" {
		sslMode = "disable"
	}

	tableName := cfg.TableName
	if tableName == "" {
		tableName = DefaultPostgresTableName
	}

	connStr := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.Host, cfg.Port, cfg.User, cfg.Password, cfg.Database, sslMode,
	)

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to open postgres connection: %w", err)
	}

	// Set connection pool settings
	maxOpenConns := cfg.MaxOpenConns
	if maxOpenConns <= 0 {
		maxOpenConns = DefaultPostgresMaxOpenConns
	}
	db.SetMaxOpenConns(maxOpenConns)

	maxIdleConns := cfg.MaxIdleConns
	if maxIdleConns <= 0 {
		maxIdleConns = DefaultPostgresMaxIdleConns
	}
	db.SetMaxIdleConns(maxIdleConns)

	connMaxLifetime := cfg.ConnMaxLifetime
	if connMaxLifetime <= 0 {
		connMaxLifetime = DefaultPostgresConnMaxLifetime
	}
	db.SetConnMaxLifetime(time.Duration(connMaxLifetime) * time.Second)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping postgres: %w", err)
	}

	store := &PostgresStore{
		db:          db,
		tableName:   tableName,
		ttl:         time.Duration(ttlSeconds) * time.Second,
		asyncWrites: asyncWrites,
		done:        make(chan struct{}),
	}

	// Create table if not exists
	if err := store.createTable(ctx); err != nil {
		return nil, fmt.Errorf("failed to create table: %w", err)
	}

	if asyncWrites {
		store.asyncChan = make(chan asyncOp, 100)
		go store.asyncWriter()
	}

	return store, nil
}

// createTable creates the records table if it doesn't exist.
func (p *PostgresStore) createTable(ctx context.Context) error {
	query := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id VARCHAR(255) PRIMARY KEY,
			timestamp TIMESTAMP NOT NULL,
			request_id VARCHAR(255),
			decision VARCHAR(255),
			category VARCHAR(255),
			original_model VARCHAR(255),
			selected_model VARCHAR(255),
			reasoning_mode VARCHAR(255),
			signals JSONB,
			request_body TEXT,
			response_body TEXT,
			response_status INTEGER,
			from_cache BOOLEAN DEFAULT FALSE,
			streaming BOOLEAN DEFAULT FALSE,
			request_body_truncated BOOLEAN DEFAULT FALSE,
			response_body_truncated BOOLEAN DEFAULT FALSE,
			created_at TIMESTAMP DEFAULT NOW()
		);
		CREATE INDEX IF NOT EXISTS idx_%s_timestamp ON %s (timestamp DESC);
		CREATE INDEX IF NOT EXISTS idx_%s_created_at ON %s (created_at);
	`, p.tableName, p.tableName, p.tableName, p.tableName, p.tableName)

	_, err := p.db.ExecContext(ctx, query)
	return err
}

// asyncWriter processes async write operations.
func (p *PostgresStore) asyncWriter() {
	for {
		select {
		case op := <-p.asyncChan:
			err := op.fn()
			if op.err != nil {
				op.err <- err
			}
		case <-p.done:
			return
		}
	}
}

// Add inserts a new record into PostgreSQL.
func (p *PostgresStore) Add(ctx context.Context, record Record) (string, error) {
	if record.ID == "" {
		id, err := generateID()
		if err != nil {
			return "", err
		}
		record.ID = id
	}

	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now().UTC()
	}

	signalsJSON, err := json.Marshal(record.Signals)
	if err != nil {
		return "", fmt.Errorf("failed to marshal signals: %w", err)
	}

	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		INSERT INTO %s (
			id, timestamp, request_id, decision, category,
			original_model, selected_model, reasoning_mode,
			signals, request_body, response_body, response_status,
			from_cache, streaming, request_body_truncated, response_body_truncated
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
	`, p.tableName)

	fn := func() error {
		_, err := p.db.ExecContext(ctx, query,
			record.ID, record.Timestamp, record.RequestID, record.Decision, record.Category,
			record.OriginalModel, record.SelectedModel, record.ReasoningMode,
			signalsJSON, record.RequestBody, record.ResponseBody, record.ResponseStatus,
			record.FromCache, record.Streaming, record.RequestBodyTruncated, record.ResponseBodyTruncated,
		)
		return err
	}

	if p.asyncWrites {
		errChan := make(chan error, 1)
		p.asyncChan <- asyncOp{fn: fn, err: errChan}
		return record.ID, nil
	}

	if err := fn(); err != nil {
		return "", fmt.Errorf("failed to insert record: %w", err)
	}

	// Clean up old records if TTL is set
	if p.ttl > 0 {
		go func() {
			_ = p.cleanupOldRecords(context.Background())
		}()
	}

	return record.ID, nil
}

// Get retrieves a record by ID from PostgreSQL.
func (p *PostgresStore) Get(ctx context.Context, id string) (Record, bool, error) {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		SELECT id, timestamp, request_id, decision, category,
		       original_model, selected_model, reasoning_mode,
		       signals, request_body, response_body, response_status,
		       from_cache, streaming, request_body_truncated, response_body_truncated
		FROM %s WHERE id = $1
	`, p.tableName)

	var record Record
	var signalsJSON []byte

	err := p.db.QueryRowContext(ctx, query, id).Scan(
		&record.ID, &record.Timestamp, &record.RequestID, &record.Decision, &record.Category,
		&record.OriginalModel, &record.SelectedModel, &record.ReasoningMode,
		&signalsJSON, &record.RequestBody, &record.ResponseBody, &record.ResponseStatus,
		&record.FromCache, &record.Streaming, &record.RequestBodyTruncated, &record.ResponseBodyTruncated,
	)

	if err == sql.ErrNoRows {
		return Record{}, false, nil
	}
	if err != nil {
		return Record{}, false, fmt.Errorf("failed to query record: %w", err)
	}

	if err := json.Unmarshal(signalsJSON, &record.Signals); err != nil {
		return Record{}, false, fmt.Errorf("failed to unmarshal signals: %w", err)
	}

	return record, true, nil
}

// List returns all records ordered by timestamp descending.
func (p *PostgresStore) List(ctx context.Context) ([]Record, error) {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		SELECT id, timestamp, request_id, decision, category,
		       original_model, selected_model, reasoning_mode,
		       signals, request_body, response_body, response_status,
		       from_cache, streaming, request_body_truncated, response_body_truncated
		FROM %s
		ORDER BY timestamp DESC
		LIMIT 10000
	`, p.tableName)

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query records: %w", err)
	}
	defer rows.Close()

	records := []Record{}
	for rows.Next() {
		var record Record
		var signalsJSON []byte

		err := rows.Scan(
			&record.ID, &record.Timestamp, &record.RequestID, &record.Decision, &record.Category,
			&record.OriginalModel, &record.SelectedModel, &record.ReasoningMode,
			&signalsJSON, &record.RequestBody, &record.ResponseBody, &record.ResponseStatus,
			&record.FromCache, &record.Streaming, &record.RequestBodyTruncated, &record.ResponseBodyTruncated,
		)
		if err != nil {
			continue // Skip malformed records
		}

		if err := json.Unmarshal(signalsJSON, &record.Signals); err != nil {
			continue
		}

		records = append(records, record)
	}

	return records, nil
}

// UpdateStatus updates the response status and flags for a record.
func (p *PostgresStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET response_status = CASE WHEN $2 != 0 THEN $2 ELSE response_status END,
		    from_cache = from_cache OR $3,
		    streaming = streaming OR $4
		WHERE id = $1
	`, p.tableName)

	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, status, fromCache, streaming)
		if err != nil {
			return fmt.Errorf("failed to update status: %w", err)
		}

		rows, err := result.RowsAffected()
		if err != nil {
			return err
		}
		if rows == 0 {
			return fmt.Errorf("record with ID %s not found", id)
		}

		return nil
	}

	if p.asyncWrites {
		p.asyncChan <- asyncOp{fn: fn}
		return nil
	}

	return fn()
}

// AttachRequest updates the request body for a record.
func (p *PostgresStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET request_body = $2,
		    request_body_truncated = request_body_truncated OR $3
		WHERE id = $1
	`, p.tableName)

	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, body, truncated)
		if err != nil {
			return fmt.Errorf("failed to update request: %w", err)
		}

		rows, err := result.RowsAffected()
		if err != nil {
			return err
		}
		if rows == 0 {
			return fmt.Errorf("record with ID %s not found", id)
		}

		return nil
	}

	if p.asyncWrites {
		p.asyncChan <- asyncOp{fn: fn}
		return nil
	}

	return fn()
}

// AttachResponse updates the response body for a record.
func (p *PostgresStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET response_body = $2,
		    response_body_truncated = response_body_truncated OR $3
		WHERE id = $1
	`, p.tableName)

	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, body, truncated)
		if err != nil {
			return fmt.Errorf("failed to update response: %w", err)
		}

		rows, err := result.RowsAffected()
		if err != nil {
			return err
		}
		if rows == 0 {
			return fmt.Errorf("record with ID %s not found", id)
		}

		return nil
	}

	if p.asyncWrites {
		p.asyncChan <- asyncOp{fn: fn}
		return nil
	}

	return fn()
}

// cleanupOldRecords removes records older than the TTL.
func (p *PostgresStore) cleanupOldRecords(ctx context.Context) error {
	if p.ttl == 0 {
		return nil
	}

	//nolint:gosec // tableName is validated during store creation, ttl is duration
	query := fmt.Sprintf(`
		DELETE FROM %s
		WHERE created_at < NOW() - INTERVAL '%d seconds'
	`, p.tableName, int(p.ttl.Seconds()))

	_, err := p.db.ExecContext(ctx, query)
	return err
}

// Close closes the PostgreSQL connection and stops async writer.
func (p *PostgresStore) Close() error {
	if p.asyncWrites {
		close(p.done)
	}
	return p.db.Close()
}
