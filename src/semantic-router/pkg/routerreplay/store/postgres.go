package store

import (
	"context"
	"database/sql"
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
			lifecycle_state, ended_at, duration_ms, terminal_reason,
			from_cache, streaming, request_body_truncated, response_body_truncated,
			guardrails_enabled, jailbreak_enabled, pii_enabled,
			prompt, prompt_truncated, tool_definitions, tool_definitions_truncated,
			rag_enabled, rag_backend, rag_context_length, rag_similarity_score,
			hallucination_enabled, hallucination_detected, hallucination_confidence, hallucination_spans,
			prompt_tokens, cached_prompt_tokens, cache_write_tokens, completion_tokens, total_tokens,
			actual_cost, baseline_cost, cost_savings, currency, baseline_model,
			session_id, turn_index, previous_response_id, conversation_id,
			cache_similarity, context_token_count, hallucination_span_details, recipe, safety_evidence, routing_metadata
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60, $61, $62, $63, $64, $65, $66, $67)
	`

// Updating in place keeps columns outside the record, such as created_at.
var postgresUpdateRecordQueryTemplate = func() string {
	head, values, _ := strings.Cut(postgresInsertQueryTemplate, ") VALUES (")
	_, columns, _ := strings.Cut(head, "INSERT INTO %s (")
	values, _, _ = strings.Cut(values, ")")
	return "UPDATE %s SET (" + columns + ") = ROW(" + values + ") WHERE id = $1"
}()

const postgresCreateTableQueryTemplate = `
		CREATE TABLE IF NOT EXISTS {{table}} (
			id VARCHAR(255) PRIMARY KEY,
			timestamp TIMESTAMP NOT NULL,
			request_id VARCHAR(255),
			recipe TEXT,
 safety_evidence JSONB,
			routing_metadata JSONB,
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
			lifecycle_state VARCHAR(32) NOT NULL DEFAULT 'unknown',
			ended_at TIMESTAMP,
			duration_ms BIGINT DEFAULT 0,
			terminal_reason TEXT,
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
			cache_write_tokens INTEGER,
			completion_tokens INTEGER,
			total_tokens INTEGER,
			actual_cost DOUBLE PRECISION,
			baseline_cost DOUBLE PRECISION,
			cost_savings DOUBLE PRECISION,
			currency VARCHAR(32),
			baseline_model VARCHAR(255),
			created_at TIMESTAMP DEFAULT NOW()
		);
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS recipe TEXT;
 ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS safety_evidence JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS routing_metadata JSONB;
		ALTER TABLE {{table}} ALTER COLUMN recipe TYPE TEXT;
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
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS lifecycle_state VARCHAR(32) NOT NULL DEFAULT 'unknown';
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS ended_at TIMESTAMP;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS duration_ms BIGINT DEFAULT 0;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS terminal_reason TEXT;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS prompt TEXT;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS prompt_truncated BOOLEAN DEFAULT FALSE;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS tool_definitions TEXT;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS tool_definitions_truncated BOOLEAN DEFAULT FALSE;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS cached_prompt_tokens INTEGER;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS cache_write_tokens INTEGER;
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
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS hallucination_span_details JSONB;
		ALTER TABLE {{table}} ADD COLUMN IF NOT EXISTS recipe VARCHAR(255);
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
	writerDone  chan struct{}
	lifecycle   *storageLifecycle
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
		lifecycle:   newStorageLifecycle(),
	}
}

func startPostgresAsyncWriter(store *PostgresStore) {
	if !store.asyncWrites {
		return
	}
	store.asyncChan = make(chan asyncOp, 100)
	store.writerDone = make(chan struct{})
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
	defer close(p.writerDone)
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
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return "", err
	}
	defer release()
	if _, err = encodeRecord(&record, MaxRecordBytes); err != nil {
		return "", err
	}
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
		if err := runAsyncOp(ctx, p.asyncChan, fn); err != nil {
			return "", fmt.Errorf("failed to insert record: %w", err)
		}
		p.schedulePostgresCleanup()
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
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return Record{}, false, err
	}
	defer release()
	return p.getRecord(ctx, id)
}

func (p *PostgresStore) getRecord(ctx context.Context, id string) (Record, bool, error) {
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
	if isExpired(record, p.ttl) {
		return Record{}, false, nil
	}
	return record, true, nil
}

// List returns all records ordered by timestamp descending.
func (p *PostgresStore) List(ctx context.Context) ([]Record, error) {
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return nil, err
	}
	defer release()
	return p.listRecords(ctx)
}

func (p *PostgresStore) listRecords(ctx context.Context) ([]Record, error) {
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
	return p.updateExistingRecord(ctx, id, func(rec *Record) bool {
		if status != 0 {
			rec.ResponseStatus = status
		}
		rec.FromCache = rec.FromCache || fromCache
		rec.Streaming = rec.Streaming || streaming
		return true
	})
}

func (p *PostgresStore) UpdateLifecycle(
	ctx context.Context,
	id string,
	state string,
	endedAt time.Time,
	durationMS int64,
	reason string,
) error {
	// Compare-and-set: a missing or already finished record is left alone.
	_, err := p.updateRecord(ctx, id, func(rec *Record) bool {
		if rec.LifecycleState != LifecycleInProgress {
			return false
		}
		rec.LifecycleState = state
		rec.EndedAt = cloneTimePtr(&endedAt)
		rec.DurationMS = durationMS
		rec.TerminalReason = reason
		return true
	})
	return err
}

// AttachRequest updates the request body for a record.
func (p *PostgresStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	return p.updateExistingRecord(ctx, id, func(rec *Record) bool {
		rec.RequestBody = body
		rec.RequestBodyTruncated = rec.RequestBodyTruncated || truncated
		return true
	})
}

// AttachResponse updates the response body for a record.
func (p *PostgresStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	return p.updateExistingRecord(ctx, id, func(rec *Record) bool {
		rec.ResponseBody = body
		rec.ResponseBodyTruncated = rec.ResponseBodyTruncated || truncated
		return true
	})
}

// AppendOutcome links post-route feedback to a record.
func (p *PostgresStore) AppendOutcome(ctx context.Context, id string, outcome Outcome) error {
	return p.updateExistingRecord(ctx, id, func(rec *Record) bool {
		rec.Outcomes = append(rec.Outcomes, cloneOutcome(outcome))
		return true
	})
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (p *PostgresStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan, score ...HallucinationScore) error {
	return p.updateExistingRecord(ctx, id, func(rec *Record) bool {
		rec.HallucinationDetected = detected
		rec.HallucinationConfidence = confidence
		applyHallucinationScore(rec, score)
		rec.HallucinationSpans = cloneStringSlice(spans)
		rec.HallucinationSpanDetails = cloneHallucinationSpanDetails(spanDetails)
		return true
	})
}

// UpdateUsageCost updates token usage and pricing-derived cost fields for a record.
func (p *PostgresStore) UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error {
	return p.updateExistingRecord(ctx, id, func(rec *Record) bool {
		rec.PromptTokens = cloneIntPtr(usage.PromptTokens)
		rec.CachedPromptTokens = cloneIntPtr(usage.CachedPromptTokens)
		rec.CacheWriteTokens = cloneIntPtr(usage.CacheWriteTokens)
		rec.CompletionTokens = cloneIntPtr(usage.CompletionTokens)
		rec.TotalTokens = cloneIntPtr(usage.TotalTokens)
		rec.ActualCost = cloneFloat64Ptr(usage.ActualCost)
		rec.BaselineCost = cloneFloat64Ptr(usage.BaselineCost)
		rec.CostSavings = cloneFloat64Ptr(usage.CostSavings)
		rec.Currency = cloneStringPtr(usage.Currency)
		rec.BaselineModel = cloneStringPtr(usage.BaselineModel)
		return true
	})
}

// UpdateToolTrace updates tool-calling trace details for a record.
func (p *PostgresStore) UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error {
	return p.updateExistingRecord(ctx, id, func(rec *Record) bool {
		rec.ToolTrace = cloneToolTrace(&trace)
		return true
	})
}

func (p *PostgresStore) updateExistingRecord(ctx context.Context, id string, mutate func(*Record) bool) error {
	found, err := p.updateRecord(ctx, id, mutate)
	if err == nil && !found {
		err = fmt.Errorf("record with ID %s not found", id)
	}
	return err
}

// updateRecord rewrites the record in one transaction so every update is
// size-checked. It reports whether the record exists; mutate returns false to
// skip the write.
func (p *PostgresStore) updateRecord(ctx context.Context, id string, mutate func(*Record) bool) (bool, error) {
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return false, err
	}
	defer release()

	found := false
	fn := func() error {
		tx, err := p.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin replay update: %w", err)
		}
		defer func() { _ = tx.Rollback() }()

		//nolint:gosec // tableName is validated during store creation
		selectQuery := fmt.Sprintf(`SELECT %s FROM %s WHERE id = $1 FOR UPDATE`, postgresRecordSelectColumns, p.tableName)
		record, err := scanPostgresRecord(tx.QueryRowContext(ctx, selectQuery, id))
		if errors.Is(err, sql.ErrNoRows) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("failed to query record: %w", err)
		}
		found = true
		if !mutate(&record) {
			return nil
		}
		if _, err := encodeRecord(&record, MaxRecordBytes); err != nil {
			return err
		}
		row, err := newPostgresInsertRecord(record)
		if err != nil {
			return err
		}
		//nolint:gosec // tableName is validated during store creation
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(postgresUpdateRecordQueryTemplate, p.tableName), row.args()...); err != nil {
			return fmt.Errorf("failed to update record: %w", err)
		}
		return tx.Commit()
	}

	if p.asyncWrites {
		err = p.runAsyncAndWait(ctx, fn)
	} else {
		err = fn()
	}
	return found, err
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
	return p.lifecycle.close(func() error {
		if p.asyncWrites {
			return closeAsyncWriter(p.asyncChan, p.done, p.writerDone, p.db.Close)
		}
		return p.db.Close()
	})
}

func (p *PostgresStore) runAsyncAndWait(ctx context.Context, fn func() error) error {
	return runAsyncOp(ctx, p.asyncChan, fn)
}

func (p *PostgresStore) schedulePostgresCleanup() {
	if p.ttl == 0 {
		return
	}
	go func() {
		release, err := p.lifecycle.beginMutation()
		if err != nil {
			return
		}
		defer release()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = p.cleanupOldRecords(ctx)
	}()
}
