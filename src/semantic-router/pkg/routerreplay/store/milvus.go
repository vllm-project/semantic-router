package store

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	DefaultMilvusCollection       = "router_replay_records"
	DefaultMilvusConsistencyLevel = "Session"
	DefaultMilvusShardNum         = 2

	milvusScanBatch = 100
	// Default Milvus limit on offset+limit for one query.
	milvusWindowRows = 16384
	// max_length of the data column.
	milvusMaxRecordBytes = 65535
)

// MilvusStore implements Storage using Milvus as the backend.
// Records are stored as entities in a Milvus collection.
type MilvusStore struct {
	client           client.Client
	collectionName   string
	consistencyLevel entity.ConsistencyLevel
	wrote            atomic.Bool
	ttl              time.Duration
	sweep            retentionSweep
	asyncWrites      bool
	asyncChan        chan asyncOp
	done             chan struct{}
	writerDone       chan struct{}
	updateMu         sync.Mutex
	lifecycle        *storageLifecycle
}

// readLevel floors Session at Bounded until this process has written, since the
// SDK degrades Session to Eventually while no session timestamp exists.
func (m *MilvusStore) readLevel() entity.ConsistencyLevel {
	if m.consistencyLevel == entity.ClSession && !m.wrote.Load() {
		return entity.ClBounded
	}
	return m.consistencyLevel
}

// NewMilvusStore creates a new Milvus storage backend.
func NewMilvusStore(cfg *MilvusConfig, ttlSeconds int, asyncWrites bool) (*MilvusStore, error) {
	if cfg == nil {
		return nil, fmt.Errorf("milvus config is required")
	}

	if cfg.Address == "" {
		cfg.Address = "localhost:19530"
	}

	collectionName := cfg.CollectionName
	if collectionName == "" {
		collectionName = DefaultMilvusCollection
	}
	consistencyLevel, validConsistencyLevel := milvusConsistencyLevel(cfg.ConsistencyLevel)
	if !validConsistencyLevel {
		logging.Warnf("unknown router replay Milvus consistency level %q; using %q", cfg.ConsistencyLevel, DefaultMilvusConsistencyLevel)
	}

	milvusClient, err := milvuslifecycle.Connect(context.Background(), client.Config{
		Address:  cfg.Address,
		Username: cfg.Username,
		Password: cfg.Password,
	}, 10*time.Second)
	if err != nil {
		return nil, err
	}

	store := &MilvusStore{
		client:           milvusClient,
		collectionName:   collectionName,
		consistencyLevel: consistencyLevel,
		ttl:              time.Duration(ttlSeconds) * time.Second,
		asyncWrites:      asyncWrites,
		done:             make(chan struct{}),
		lifecycle:        newStorageLifecycle(),
	}

	if err := milvuslifecycle.Retry(
		context.Background(),
		3,
		2*time.Second,
		"ensure router replay Milvus collection",
		func(ctx context.Context) error {
			collCtx, collCancel := context.WithTimeout(ctx, 20*time.Second)
			defer collCancel()
			return store.createCollection(collCtx, cfg)
		},
	); err != nil {
		return nil, err
	}

	if asyncWrites {
		store.asyncChan = make(chan asyncOp, 100)
		store.writerDone = make(chan struct{})
		go store.asyncWriter()
	}

	return store, nil
}

// createCollection creates the Milvus collection if it doesn't exist.
//
//nolint:gocognit
func (m *MilvusStore) createCollection(ctx context.Context, cfg *MilvusConfig) error {
	return milvuslifecycle.EnsureCollectionLoadedWithHooksRetry(
		ctx,
		m.client,
		m.collectionName,
		func(innerCtx context.Context) error {
			// Create schema
			schema := &entity.Schema{
				CollectionName: m.collectionName,
				Description:    "Router replay records",
				Fields: []*entity.Field{
					{
						Name:       "id",
						DataType:   entity.FieldTypeVarChar,
						PrimaryKey: true,
						AutoID:     false,
						TypeParams: map[string]string{
							"max_length": "255",
						},
					},
					{
						Name:     "timestamp",
						DataType: entity.FieldTypeInt64,
					},
					{
						Name:     "data",
						DataType: entity.FieldTypeVarChar,
						TypeParams: map[string]string{
							"max_length": "65535",
						},
					},
					{
						Name:     "vector",
						DataType: entity.FieldTypeFloatVector,
						TypeParams: map[string]string{
							"dim": "2",
						},
					},
				},
			}

			shardNum := cfg.ShardNum
			if shardNum <= 0 {
				shardNum = DefaultMilvusShardNum
			}
			if shardNum > 2147483647 {
				shardNum = DefaultMilvusShardNum
			}

			// Create collection
			//nolint:gosec // shardNum is validated to be within int32 range
			if err := m.client.CreateCollection(innerCtx, schema, int32(shardNum), client.WithConsistencyLevel(m.consistencyLevel)); err != nil {
				return err
			}

			// Create vector index (required by Milvus even for dummy vectors)
			vectorIdx, err := entity.NewIndexAUTOINDEX(entity.L2)
			if err != nil {
				return fmt.Errorf("failed to build vector index: %w", err)
			}

			if err := m.client.CreateIndex(innerCtx, m.collectionName, "vector", vectorIdx, false); err != nil {
				return fmt.Errorf("failed to create vector index: %w", err)
			}

			// Create index on timestamp for efficient querying
			timeIdx := entity.NewGenericIndex(
				"timestamp_idx",
				entity.Sorted,
				map[string]string{},
			)

			if err := m.client.CreateIndex(innerCtx, m.collectionName, "timestamp", timeIdx, false); err != nil {
				return fmt.Errorf("failed to create index: %w", err)
			}

			return nil
		},
		func(innerCtx context.Context) error {
			// Ensure required vector index exists (Milvus needs one before insert)
			idxes, err := m.client.DescribeIndex(innerCtx, m.collectionName, "vector")
			if err != nil {
				return fmt.Errorf("failed to describe vector index: %w", err)
			}

			if len(idxes) == 0 {
				vectorIdx, err := entity.NewIndexAUTOINDEX(entity.L2)
				if err != nil {
					return fmt.Errorf("failed to build vector index: %w", err)
				}

				if err := m.client.CreateIndex(innerCtx, m.collectionName, "vector", vectorIdx, false); err != nil {
					return fmt.Errorf("failed to create vector index: %w", err)
				}
			}
			return nil
		},
		milvuslifecycle.CollectionRetryOptions{},
	)
}

// asyncWriter processes async write operations.
func (m *MilvusStore) asyncWriter() {
	defer close(m.writerDone)
	for {
		select {
		case op := <-m.asyncChan:
			err := op.fn()
			if op.err != nil {
				op.err <- err
			}
		case <-m.done:
			return
		}
	}
}

// insertRecord writes one marshalled record as a Milvus entity.
func (m *MilvusStore) insertRecord(ctx context.Context, record Record, data []byte) error {
	idColumn := entity.NewColumnVarChar("id", []string{record.ID})
	timestampColumn := entity.NewColumnInt64("timestamp", []int64{record.Timestamp.Unix()})
	dataColumn := entity.NewColumnVarChar("data", []string{string(data)})
	vectorColumn := entity.NewColumnFloatVector("vector", 2, [][]float32{{0.0, 0.0}})

	if _, err := m.client.Insert(ctx, m.collectionName, "", idColumn, timestampColumn, dataColumn, vectorColumn); err != nil {
		return err
	}
	// Record that this client has written, so Session reads no longer
	// degrade to Eventually for want of a session timestamp.
	m.wrote.Store(true)
	return nil
}

// Add inserts a new record into Milvus.
func (m *MilvusStore) Add(ctx context.Context, record Record) (string, error) {
	release, err := m.lifecycle.beginMutation()
	if err != nil {
		return "", err
	}
	defer release()
	if record.LifecycleState == "" {
		record.LifecycleState = LifecycleInProgress
	}
	if record.ID == "" {
		generatedID, generateErr := generateID()
		if generateErr != nil {
			return "", generateErr
		}
		record.ID = generatedID
	}

	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now().UTC()
	}

	data, err := encodeRecord(&record, milvusMaxRecordBytes)
	if err != nil {
		return "", err
	}

	fn := func() error {
		if err := m.insertRecord(ctx, record, data); err != nil {
			return err
		}
		if m.ttl > 0 && m.sweep.due() {
			m.deleteExpired()
		}
		return nil
	}

	if m.asyncWrites {
		if err := runAsyncOp(ctx, m.asyncChan, func() error {
			err := fn()
			if err == nil {
				// Flush to ensure data is persisted
				err = m.client.Flush(ctx, m.collectionName, false)
			}
			return err
		}); err != nil {
			return "", fmt.Errorf("failed to insert record: %w", err)
		}
		return record.ID, nil
	}

	if err := fn(); err != nil {
		return "", fmt.Errorf("failed to insert record: %w", err)
	}

	// Flush to ensure data is persisted
	if err := m.client.Flush(ctx, m.collectionName, false); err != nil {
		return "", fmt.Errorf("failed to flush: %w", err)
	}

	return record.ID, nil
}

// Get retrieves a record by ID from Milvus.
func (m *MilvusStore) Get(ctx context.Context, id string) (Record, bool, error) {
	release, err := m.lifecycle.beginMutation()
	if err != nil {
		return Record{}, false, err
	}
	defer release()
	record, found, err := m.getRecord(ctx, id)
	if !found || isExpired(record, m.ttl) {
		return Record{}, false, err
	}
	return record, true, nil
}

func (m *MilvusStore) getRecord(ctx context.Context, id string) (Record, bool, error) {
	if m.asyncWrites {
		if err := runAsyncOp(ctx, m.asyncChan, func() error { return nil }); err != nil {
			return Record{}, false, err
		}
	}

	// Query by ID
	expr := fmt.Sprintf("id == '%s'", id)
	result, err := m.client.Query(ctx, m.collectionName, nil, expr, []string{"id", "timestamp", "data"}, client.WithSearchQueryConsistencyLevel(m.readLevel()))
	if err != nil {
		return Record{}, false, fmt.Errorf("failed to query record: %w", err)
	}

	if result.Len() == 0 {
		return Record{}, false, nil
	}

	// Get data field
	dataColumn := result.GetColumn("data")
	dataVarChar, ok := dataColumn.(*entity.ColumnVarChar)
	if !ok {
		return Record{}, false, fmt.Errorf("unexpected column type for data")
	}

	if len(dataVarChar.Data()) == 0 {
		return Record{}, false, nil
	}

	// Unmarshal record
	var record Record
	if err := json.Unmarshal([]byte(dataVarChar.Data()[0]), &record); err != nil {
		return Record{}, false, fmt.Errorf("failed to unmarshal record: %w", err)
	}

	return record, true, nil
}

// List returns all records ordered by timestamp descending.
func (m *MilvusStore) List(ctx context.Context) ([]Record, error) {
	release, err := m.lifecycle.beginMutation()
	if err != nil {
		return nil, err
	}
	defer release()
	return m.listRecords(ctx)
}

func (m *MilvusStore) listRecords(ctx context.Context) ([]Record, error) {
	if err := m.waitForWrites(ctx); err != nil {
		return nil, err
	}
	return listAll(m.scan(ctx))
}

func (m *MilvusStore) QueryPage(ctx context.Context, filters QueryFilters, limit, offset int, details bool) (RecordPage, error) {
	release, err := m.lifecycle.beginMutation()
	if err != nil {
		return RecordPage{}, err
	}
	defer release()
	if err := m.waitForWrites(ctx); err != nil {
		return RecordPage{}, err
	}
	count := func() (int, error) {
		return m.count(ctx, m.liveExpr(""))
	}
	return collectPage(filters, limit, offset, count, m.scan(ctx))
}

func (m *MilvusStore) ScanMetadata(ctx context.Context, visit func(Record) error) error {
	release, err := m.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	if err := m.waitForWrites(ctx); err != nil {
		return err
	}
	return scanAll(m.scan(ctx), visit)
}

func (m *MilvusStore) waitForWrites(ctx context.Context) error {
	if !m.asyncWrites {
		return nil
	}
	return runAsyncOp(ctx, m.asyncChan, func() error { return nil })
}

type milvusRow struct {
	id        string
	timestamp int64
}

// Milvus queries cannot sort, so scan walks timestamp ranges newest first,
// reading IDs before bodies.
func (m *MilvusStore) scan(ctx context.Context) recordScan {
	return func(from int, visit func(Record) bool) error {
		floor := int64(math.MinInt64)
		if m.ttl > 0 {
			floor = retentionCutoff(m.ttl).Unix()
		}
		// Leave room for writers whose clocks run a little ahead.
		top := time.Now().Unix() + 60
		hi, width := int64(math.MaxInt64), int64(60)
		for {
			lo := max(min(hi, top)-width, floor)
			rows, err := m.queryRows(ctx, lo, hi)
			if err != nil {
				return err
			}
			if len(rows) >= milvusWindowRows {
				if min(hi, top)-lo <= 1 {
					return fmt.Errorf("more than %d replay records share one second", milvusWindowRows)
				}
				width = (min(hi, top) - lo) / 2
				continue
			}
			slices.SortFunc(rows, func(a, b milvusRow) int {
				if a.timestamp != b.timestamp {
					return cmp.Compare(b.timestamp, a.timestamp)
				}
				return strings.Compare(b.id, a.id)
			})
			if from >= len(rows) {
				from -= len(rows)
			} else {
				more, visitErr := m.visitIDs(ctx, rows[from:], visit)
				if visitErr != nil || !more {
					return visitErr
				}
				from = 0
			}
			if lo <= floor {
				return nil
			}
			older, err := m.count(ctx, m.liveExpr(fmt.Sprintf("timestamp < %d", lo)))
			if err != nil || older == 0 {
				return err
			}
			hi = lo
			if len(rows) < milvusWindowRows/4 && width < 1<<40 {
				width *= 2
			}
		}
	}
}

func (m *MilvusStore) queryRows(ctx context.Context, lo, hi int64) ([]milvusRow, error) {
	expr := fmt.Sprintf("timestamp >= %d", lo)
	if hi != math.MaxInt64 {
		expr += fmt.Sprintf(" && timestamp < %d", hi)
	}
	result, err := m.client.Query(ctx, m.collectionName, nil, expr, []string{"id", "timestamp"},
		client.WithLimit(milvusWindowRows), client.WithSearchQueryConsistencyLevel(m.readLevel()))
	if err != nil {
		return nil, fmt.Errorf("failed to query records: %w", err)
	}
	ids, okIDs := result.GetColumn("id").(*entity.ColumnVarChar)
	timestamps, okTimestamps := result.GetColumn("timestamp").(*entity.ColumnInt64)
	if result.Len() == 0 {
		return nil, nil
	}
	if !okIDs || !okTimestamps {
		return nil, fmt.Errorf("unexpected column types for id and timestamp")
	}
	rows := make([]milvusRow, len(ids.Data()))
	for i, id := range ids.Data() {
		rows[i] = milvusRow{id: id, timestamp: timestamps.Data()[i]}
	}
	return rows, nil
}

func (m *MilvusStore) visitIDs(ctx context.Context, rows []milvusRow, visit func(Record) bool) (bool, error) {
	for start := 0; start < len(rows); start += milvusScanBatch {
		batch := rows[start:min(start+milvusScanBatch, len(rows))]
		quoted := make([]string, len(batch))
		for i, row := range batch {
			quoted[i] = fmt.Sprintf("'%s'", row.id)
		}
		expr := fmt.Sprintf("id in [%s]", strings.Join(quoted, ","))
		result, err := m.client.Query(ctx, m.collectionName, nil, expr, []string{"id", "data"},
			client.WithSearchQueryConsistencyLevel(m.readLevel()))
		if err != nil {
			return false, fmt.Errorf("failed to query records: %w", err)
		}
		byID := make(map[string]Record, result.Len())
		if data, ok := result.GetColumn("data").(*entity.ColumnVarChar); ok {
			for _, raw := range data.Data() {
				var record Record
				if json.Unmarshal([]byte(raw), &record) == nil {
					byID[record.ID] = record
				}
			}
		}
		for _, row := range batch {
			if record, ok := byID[row.id]; ok && !visit(record) {
				return false, nil
			}
		}
	}
	return true, nil
}

func (m *MilvusStore) count(ctx context.Context, expr string) (int, error) {
	result, err := m.client.Query(ctx, m.collectionName, nil, expr, []string{"count(*)"},
		client.WithSearchQueryConsistencyLevel(m.readLevel()))
	if err != nil {
		return 0, fmt.Errorf("failed to count records: %w", err)
	}
	column := result.GetColumn("count(*)")
	if column == nil {
		return 0, nil
	}
	counts, ok := column.(*entity.ColumnInt64)
	if !ok || len(counts.Data()) == 0 {
		return 0, fmt.Errorf("unexpected count result")
	}
	return int(counts.Data()[0]), nil
}

func (m *MilvusStore) liveExpr(expr string) string {
	if m.ttl <= 0 {
		return expr
	}
	live := fmt.Sprintf("timestamp >= %d", retentionCutoff(m.ttl).Unix())
	if expr == "" {
		return live
	}
	return expr + " && " + live
}

func (m *MilvusStore) deleteExpired() {
	go func() {
		release, err := m.lifecycle.beginMutation()
		if err != nil {
			return
		}
		defer release()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		expr := fmt.Sprintf("timestamp < %d", retentionCutoff(m.ttl).Unix())
		if err := m.client.Delete(ctx, m.collectionName, "", expr); err != nil {
			logging.Warnf("router replay: failed to delete expired Milvus records: %v", err)
		}
	}()
}

// UpdateStatus updates the response status and flags for a record.
func (m *MilvusStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		if status != 0 {
			record.ResponseStatus = status
		}
		record.FromCache = record.FromCache || fromCache
		record.Streaming = record.Streaming || streaming
	})
}

func (m *MilvusStore) UpdateLifecycle(
	ctx context.Context,
	id string,
	state string,
	endedAt time.Time,
	durationMS int64,
	reason string,
) error {
	return m.updateRecordIf(ctx, id, func(record *Record) bool {
		if record.LifecycleState != "" && record.LifecycleState != LifecycleInProgress {
			return false
		}
		record.LifecycleState = state
		record.EndedAt = cloneTimePtr(&endedAt)
		record.DurationMS = durationMS
		record.TerminalReason = reason
		return true
	})
}

// AttachRequest updates the request body for a record.
func (m *MilvusStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		record.RequestBody = body
		record.RequestBodyTruncated = record.RequestBodyTruncated || truncated
	})
}

// AttachResponse updates the response body for a record.
func (m *MilvusStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		record.ResponseBody = body
		record.ResponseBodyTruncated = record.ResponseBodyTruncated || truncated
	})
}

// AppendOutcome links post-route feedback to a record.
func (m *MilvusStore) AppendOutcome(ctx context.Context, id string, outcome Outcome) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		record.Outcomes = append(record.Outcomes, cloneOutcome(outcome))
	})
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (m *MilvusStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan, score ...HallucinationScore) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		record.HallucinationDetected = detected
		record.HallucinationConfidence = confidence
		applyHallucinationScore(record, score)
		record.HallucinationSpans = cloneStringSlice(spans)
		record.HallucinationSpanDetails = cloneHallucinationSpanDetails(spanDetails)
	})
}

// UpdateUsageCost updates token usage and pricing-derived cost fields for a record.
func (m *MilvusStore) UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		record.PromptTokens = cloneIntPtr(usage.PromptTokens)
		record.CachedPromptTokens = cloneIntPtr(usage.CachedPromptTokens)
		record.CacheWriteTokens = cloneIntPtr(usage.CacheWriteTokens)
		record.CompletionTokens = cloneIntPtr(usage.CompletionTokens)
		record.TotalTokens = cloneIntPtr(usage.TotalTokens)
		record.ActualCost = cloneFloat64Ptr(usage.ActualCost)
		record.BaselineCost = cloneFloat64Ptr(usage.BaselineCost)
		record.CostSavings = cloneFloat64Ptr(usage.CostSavings)
		record.Currency = cloneStringPtr(usage.Currency)
		record.BaselineModel = cloneStringPtr(usage.BaselineModel)
	})
}

// UpdateToolTrace updates tool-calling trace details for a record.
func (m *MilvusStore) UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		record.ToolTrace = cloneToolTrace(&trace)
	})
}

func (m *MilvusStore) updateRecord(ctx context.Context, id string, mutate func(*Record)) error {
	return m.updateRecordIf(ctx, id, func(record *Record) bool {
		mutate(record)
		return true
	})
}

func (m *MilvusStore) updateRecordIf(ctx context.Context, id string, mutate func(*Record) bool) error {
	release, err := m.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	m.updateMu.Lock()
	defer m.updateMu.Unlock()
	record, found, err := m.getRecord(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}
	if !mutate(&record) {
		return nil
	}
	// Milvus stores each replay as a whole JSON document. Keep every
	// read-modify-write in one critical section so concurrent enrichments do
	// not overwrite usage, trace, response, or lifecycle fields.
	return m.replaceRecord(ctx, record)
}

func (m *MilvusStore) replaceRecord(ctx context.Context, record Record) error {
	data, err := encodeRecord(&record, milvusMaxRecordBytes)
	if err != nil {
		return err
	}
	idColumn := entity.NewColumnVarChar("id", []string{record.ID})
	timestampColumn := entity.NewColumnInt64("timestamp", []int64{record.Timestamp.Unix()})
	dataColumn := entity.NewColumnVarChar("data", []string{string(data)})
	vectorColumn := entity.NewColumnFloatVector("vector", 2, [][]float32{{0.0, 0.0}})
	if _, err := m.client.Upsert(ctx, m.collectionName, "", idColumn, timestampColumn, dataColumn, vectorColumn); err != nil {
		return fmt.Errorf("failed to upsert updated record: %w", err)
	}
	m.wrote.Store(true)
	return m.client.Flush(ctx, m.collectionName, false)
}

// Close closes the Milvus client and stops async writer.
func (m *MilvusStore) Close() error {
	return m.lifecycle.close(func() error {
		if m.asyncWrites {
			return closeAsyncWriter(m.asyncChan, m.done, m.writerDone, m.client.Close)
		}
		return m.client.Close()
	})
}
