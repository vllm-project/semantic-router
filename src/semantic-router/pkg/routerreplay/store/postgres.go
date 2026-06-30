package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

const DefaultPostgresTableName = "router_replay_records"

// postgresInsertQueryTemplate is the parameterized INSERT statement used by
// PostgresStore.Add. The single %s slot receives the validated table name.
//
// The column list, the $N placeholders, and the slice returned by
// postgresInsertRecord.args() must stay in sync — a mismatch silently breaks
// every replay write (postgres returns "more target columns than expressions"
// or vice versa, which is then swallowed by the Recorder). The invariant is
// guarded by TestPostgresInsertQueryColumnArgsAlignment.
const postgresInsertQueryTemplate = `
		INSERT INTO %s (
			id, timestamp, request_id, decision, decision_tier, decision_priority, category,
			original_model, selected_model, reasoning_mode,
			signals, projections, projection_scores, signal_confidences, signal_values, tool_trace, projection_trace, session_policy, route_diagnostics, learning, outcomes,
			request_body, response_body, response_status,
			from_cache, streaming, request_body_truncated, response_body_truncated,
			guardrails_enabled, jailbreak_enabled, pii_enabled,
			prompt, prompt_truncated, tool_definitions, tool_definitions_truncated,
			rag_enabled, rag_backend, rag_context_length, rag_similarity_score,
			hallucination_enabled, hallucination_detected, hallucination_confidence, hallucination_spans,
			prompt_tokens, cached_prompt_tokens, completion_tokens, total_tokens,
			actual_cost, baseline_cost, cost_savings, currency, baseline_model,
			session_id, turn_index, previous_response_id, conversation_id,
			cache_similarity, context_token_count
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58)
	`

const postgresCreateTableQueryTemplate = `
		CREATE TABLE IF NOT EXISTS {{table}} (
			id VARCHAR(255) PRIMARY KEY,
			timestamp TIMESTAMP NOT NULL,
			request_id VARCHAR(255),
			decision VARCHAR(255),
			decision_tier INTEGER DEFAULT 0,
			decision_priority INTEGER DEFAULT 0,
			category VARCHAR(255),
			original_model VARCHAR(255),
			selected_model VARCHAR(255),
			reasoning_mode VARCHAR(255),
			signals JSONB,
			projections JSONB,
			projection_scores JSONB,
			signal_confidences JSONB,
			signal_values JSONB,
			tool_trace JSONB,
			projection_trace JSONB,
			session_policy JSONB,
			route_diagnostics JSONB,
			learning JSONB,
			outcomes JSONB,
			request_body TEXT,
			response_body TEXT,
			response_status INTEGER,
			from_cache BOOLEAN DEFAULT FALSE,
			streaming BOOLEAN DEFAULT FALSE,
			request_body_truncated BOOLEAN DEFAULT FALSE,
			response_body_truncated BOOLEAN DEFAULT FALSE,
			guardrails_enabled BOOLEAN DEFAULT FALSE,
			jailbreak_enabled BOOLEAN DEFAULT FALSE,
			pii_enabled BOOLEAN DEFAULT FALSE,
			prompt TEXT,
			prompt_truncated BOOLEAN DEFAULT FALSE,
			tool_definitions TEXT,
			tool_definitions_truncated BOOLEAN DEFAULT FALSE,
			rag_enabled BOOLEAN DEFAULT FALSE,
			rag_backend VARCHAR(255),
			rag_context_length INTEGER DEFAULT 0,
			rag_similarity_score REAL DEFAULT 0,
			hallucination_enabled BOOLEAN DEFAULT FALSE,
			hallucination_detected BOOLEAN DEFAULT FALSE,
			hallucination_confidence REAL DEFAULT 0,
			hallucination_spans JSONB,
			prompt_tokens INTEGER,
			cached_prompt_tokens INTEGER,
			completion_tokens INTEGER,
			total_tokens INTEGER,
			actual_cost DOUBLE PRECISION,
			baseline_cost DOUBLE PRECISION,
			cost_savings DOUBLE PRECISION,
			currency VARCHAR(32),
			baseline_model VARCHAR(255),
			created_at TIMESTAMP DEFAULT NOW()
		);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS decision_tier INTEGER DEFAULT 0;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS decision_priority INTEGER DEFAULT 0;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS projections JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS projection_scores JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS signal_confidences JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS signal_values JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS tool_trace JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS projection_trace JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS session_policy JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS route_diagnostics JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS learning JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS outcomes JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS prompt TEXT;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS prompt_truncated BOOLEAN DEFAULT FALSE;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS tool_definitions TEXT;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS tool_definitions_truncated BOOLEAN DEFAULT FALSE;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS cached_prompt_tokens INTEGER;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS completion_tokens INTEGER;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS total_tokens INTEGER;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS actual_cost DOUBLE PRECISION;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS baseline_cost DOUBLE PRECISION;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS cost_savings DOUBLE PRECISION;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS currency VARCHAR(32);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS baseline_model VARCHAR(255);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS session_id VARCHAR(255);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS turn_index INTEGER;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS previous_response_id VARCHAR(255);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(255);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS cache_similarity REAL DEFAULT 0;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS context_token_count INTEGER DEFAULT 0;
		CREATE INDEX IF NOT EXISTS idx_{{table}}_timestamp ON {{table}} (timestamp DESC);
		CREATE INDEX IF NOT EXISTS idx_{{table}}_created_at ON {{table}} (created_at);
		CREATE INDEX IF NOT EXISTS idx_{{table}}_request_id ON {{table}} (request_id);
		CREATE INDEX IF NOT EXISTS idx_{{table}}_decision_timestamp ON {{table}} (decision, timestamp DESC);
		CREATE INDEX IF NOT EXISTS idx_{{table}}_selected_model_timestamp ON {{table}} (selected_model, timestamp DESC);
		CREATE INDEX IF NOT EXISTS idx_{{table}}_session_timestamp ON {{table}} (session_id, timestamp DESC);
	`

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
	runtimeCfg, err := newPostgresRuntimeConfig(cfg)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	db, err := openConfiguredPostgresDB(ctx, runtimeCfg)
	if err != nil {
		return nil, err
	}

	store := newPostgresStoreWithDB(db, runtimeCfg.TableName, ttlSeconds, asyncWrites)
	if err := store.createTable(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("failed to create table: %w", err)
	}
	startPostgresAsyncWriter(store)
	return store, nil
}

func newPostgresStoreWithDB(db *sql.DB, tableName string, ttlSeconds int, asyncWrites bool) *PostgresStore {
	return &PostgresStore{
		db:          db,
		tableName:   tableName,
		ttl:         time.Duration(ttlSeconds) * time.Second,
		asyncWrites: asyncWrites,
		done:        make(chan struct{}),
	}
}

func startPostgresAsyncWriter(store *PostgresStore) {
	if !store.asyncWrites {
		return
	}
	store.asyncChan = make(chan asyncOp, 100)
	go store.asyncWriter()
}

// createTable creates the records table if it doesn't exist.
func (p *PostgresStore) createTable(ctx context.Context) error {
	query := postgresCreateTableQuery(p.tableName)
	_, err := p.db.ExecContext(ctx, query)
	return err
}

func postgresCreateTableQuery(tableName string) string {
	return strings.ReplaceAll(postgresCreateTableQueryTemplate, "{{table}}", tableName)
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
	insertRecord, err := newPostgresInsertRecord(record)
	if err != nil {
		return "", err
	}

	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(postgresInsertQueryTemplate, p.tableName)

	fn := func() error {
		_, err := p.db.ExecContext(ctx, query, insertRecord.args()...)
		return err
	}

	if p.asyncWrites {
		errChan := make(chan error, 1)
		p.asyncChan <- asyncOp{fn: fn, err: errChan}
		return insertRecord.record.ID, nil
	}

	if err := fn(); err != nil {
		return "", fmt.Errorf("failed to insert record: %w", err)
	}

	p.schedulePostgresCleanup()
	return insertRecord.record.ID, nil
}

// Get retrieves a record by ID from PostgreSQL.
func (p *PostgresStore) Get(ctx context.Context, id string) (Record, bool, error) {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		SELECT %s
		FROM %s WHERE id = $1
	`, postgresRecordSelectColumns, p.tableName)

	record, err := scanPostgresRecord(p.db.QueryRowContext(ctx, query, id))
	if errors.Is(err, sql.ErrNoRows) {
		return Record{}, false, nil
	}
	if err != nil {
		return Record{}, false, fmt.Errorf("failed to query record: %w", err)
	}
	return record, true, nil
}

// List returns all records ordered by timestamp descending.
func (p *PostgresStore) List(ctx context.Context) ([]Record, error) {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		SELECT %s
		FROM %s
		ORDER BY timestamp DESC
		LIMIT 10000
	`, postgresRecordSelectColumns, p.tableName)

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query records: %w", err)
	}
	return scanPostgresRecordList(rows)
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

// AppendOutcome links post-route feedback to a replay record.
func (p *PostgresStore) AppendOutcome(ctx context.Context, id string, outcome Outcome) error {
	outcomeJSON, err := json.Marshal([]Outcome{cloneOutcome(outcome)})
	if err != nil {
		return fmt.Errorf("failed to marshal outcome: %w", err)
	}

	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET outcomes = COALESCE(outcomes, '[]'::jsonb) || $2::jsonb
		WHERE id = $1
	`, p.tableName)

	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, outcomeJSON)
		if err != nil {
			return fmt.Errorf("failed to append outcome: %w", err)
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

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (p *PostgresStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string) error {
	spansJSON, err := json.Marshal(spans)
	if err != nil {
		return fmt.Errorf("failed to marshal hallucination spans: %w", err)
	}

	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET hallucination_detected = $2,
		    hallucination_confidence = $3,
		    hallucination_spans = $4
		WHERE id = $1
	`, p.tableName)

	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, detected, confidence, spansJSON)
		if err != nil {
			return fmt.Errorf("failed to update hallucination status: %w", err)
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

// UpdateUsageCost updates token usage and pricing-derived cost fields for a record.
func (p *PostgresStore) UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error {
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET prompt_tokens = $2,
		    cached_prompt_tokens = $3,
		    completion_tokens = $4,
		    total_tokens = $5,
		    actual_cost = $6,
		    baseline_cost = $7,
		    cost_savings = $8,
		    currency = $9,
		    baseline_model = $10
		WHERE id = $1
	`, p.tableName)

	fn := func() error {
		result, err := p.db.ExecContext(
			ctx,
			query,
			id,
			nullableIntArg(usage.PromptTokens),
			nullableIntArg(usage.CachedPromptTokens),
			nullableIntArg(usage.CompletionTokens),
			nullableIntArg(usage.TotalTokens),
			nullableFloat64Arg(usage.ActualCost),
			nullableFloat64Arg(usage.BaselineCost),
			nullableFloat64Arg(usage.CostSavings),
			nullableStringArg(usage.Currency),
			nullableStringArg(usage.BaselineModel),
		)
		if err != nil {
			return fmt.Errorf("failed to update usage cost: %w", err)
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

// UpdateToolTrace updates tool-calling trace details for a record.
func (p *PostgresStore) UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error {
	traceJSON, err := marshalReplayOptionalJSON(&trace)
	if err != nil {
		return fmt.Errorf("failed to marshal tool trace: %w", err)
	}

	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET tool_trace = $2
		WHERE id = $1
	`, p.tableName)

	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, traceJSON)
		if err != nil {
			return fmt.Errorf("failed to update tool trace: %w", err)
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

func (p *PostgresStore) schedulePostgresCleanup() {
	if p.ttl == 0 {
		return
	}
	go func() {
		_ = p.cleanupOldRecords(context.Background())
	}()
}
