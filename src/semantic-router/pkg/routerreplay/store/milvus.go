package store

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
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
)

// MilvusStore implements Storage using Milvus as the backend.
// Records are stored as entities in a Milvus collection.
type MilvusStore struct {
	client           client.Client
	collectionName   string
	consistencyLevel entity.ConsistencyLevel
	wrote            atomic.Bool
	ttl              time.Duration
	asyncWrites      bool
	asyncChan        chan asyncOp
	done             chan struct{}
	writerDone       chan struct{}
	updateMu         sync.Mutex
	lifecycle        *storageLifecycle
}

// resolveMilvusConsistencyLevel maps the configured consistency level name
// to the SDK constant via the shared parser, falling back to
// DefaultMilvusConsistencyLevel ("Session") when the value is empty or
// unrecognized.
func resolveMilvusConsistencyLevel(name string) entity.ConsistencyLevel {
	if level, ok := milvuslifecycle.ParseConsistencyLevel(name); ok {
		return level
	}
	if strings.TrimSpace(name) != "" {
		logging.Warnf("MilvusStore: unrecognized consistency_level %q (valid: %s); using default %q", name, milvuslifecycle.ConsistencyLevelNames, DefaultMilvusConsistencyLevel)
	}
	return entity.ClSession
}

// readLevel returns the consistency level for plain reads (Get, List). The
// SDK degrades Session to Eventually until this client's first write on the
// collection (no session timestamp exists yet), which would be weaker than
// the Bounded these reads effectively ran at before the level was
// configurable — so Session is floored at Bounded until this process has
// written.
func (m *MilvusStore) readLevel() entity.ConsistencyLevel {
	if m.consistencyLevel == entity.ClSession && !m.wrote.Load() {
		return entity.ClBounded
	}
	return m.consistencyLevel
}

// updateReadLevel returns the consistency level for the read feeding an
// update's read-modify-write (updateRecordIf). updateMu serializes those
// updates in-process, but the read must also see this process's own prior
// upserts or the whole-document replace silently resurrects stale fields —
// so it never runs weaker than Session, even when the configured level is
// Bounded or Eventually; an explicit Strong is kept. Before this process's
// first write Session carries no session timestamp (the SDK would degrade it
// to Eventually), so Bounded applies instead.
func (m *MilvusStore) updateReadLevel() entity.ConsistencyLevel {
	if m.consistencyLevel == entity.ClStrong {
		return entity.ClStrong
	}
	if m.wrote.Load() {
		return entity.ClSession
	}
	return entity.ClBounded
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
		consistencyLevel: resolveMilvusConsistencyLevel(cfg.ConsistencyLevel),
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

// replayCollectionSchema returns the schema of the router replay collection.
func replayCollectionSchema(collectionName string) *entity.Schema {
	return &entity.Schema{
		CollectionName: collectionName,
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
			schema := replayCollectionSchema(m.collectionName)

			shardNum := cfg.ShardNum
			if shardNum <= 0 {
				shardNum = DefaultMilvusShardNum
			}
			if shardNum > 2147483647 {
				shardNum = DefaultMilvusShardNum
			}

			// Create collection. The consistency level is stamped only when
			// this call creates the collection; a pre-existing collection
			// keeps its original level, and the per-query options carry the
			// configured level for this store's reads instead.
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

	// Marshal record to JSON
	data, err := json.Marshal(record)
	if err != nil {
		return "", fmt.Errorf("failed to marshal record: %w", err)
	}

	fn := func() error {
		// Create columns
		idColumn := entity.NewColumnVarChar("id", []string{record.ID})
		timestampColumn := entity.NewColumnInt64("timestamp", []int64{record.Timestamp.Unix()})
		dataColumn := entity.NewColumnVarChar("data", []string{string(data)})
		vectorColumn := entity.NewColumnFloatVector("vector", 2, [][]float32{{0.0, 0.0}})

		// Insert
		_, err := m.client.Insert(ctx, m.collectionName, "", idColumn, timestampColumn, dataColumn, vectorColumn)
		if err != nil {
			return err
		}
		m.wrote.Store(true)
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
	return m.getRecord(ctx, id, m.readLevel())
}

func (m *MilvusStore) getRecord(ctx context.Context, id string, level entity.ConsistencyLevel) (Record, bool, error) {
	if m.asyncWrites {
		if err := runAsyncOp(ctx, m.asyncChan, func() error { return nil }); err != nil {
			return Record{}, false, err
		}
	}

	// Query by ID
	expr := fmt.Sprintf("id == '%s'", id)
	result, err := m.client.Query(
		ctx,
		m.collectionName,
		nil,
		expr,
		[]string{"id", "timestamp", "data"},
		client.WithSearchQueryConsistencyLevel(level),
	)
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
	// If async writes enabled, wait for all pending writes to complete
	if m.asyncWrites {
		if err := runAsyncOp(ctx, m.asyncChan, func() error { return nil }); err != nil {
			return nil, err
		}
	}

	// Query all records with a high limit (empty expression requires limit in Milvus)
	result, err := m.client.Query(
		ctx,
		m.collectionName,
		nil,
		"",
		[]string{"id", "timestamp", "data"},
		client.WithLimit(10000),
		client.WithSearchQueryConsistencyLevel(m.readLevel()),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to query records: %w", err)
	}

	if result.Len() == 0 {
		return []Record{}, nil
	}

	// Get data column
	dataColumn := result.GetColumn("data")
	dataVarChar, ok := dataColumn.(*entity.ColumnVarChar)
	if !ok {
		return nil, fmt.Errorf("unexpected column type for data")
	}

	// Parse all records
	records := make([]Record, 0, len(dataVarChar.Data()))
	for _, data := range dataVarChar.Data() {
		var record Record
		if err := json.Unmarshal([]byte(data), &record); err != nil {
			continue // Skip malformed records
		}
		records = append(records, record)
	}

	sort.Slice(records, func(i, j int) bool {
		return records[i].Timestamp.After(records[j].Timestamp)
	})

	return records, nil
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
func (m *MilvusStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan) error {
	return m.updateRecord(ctx, id, func(record *Record) {
		record.HallucinationDetected = detected
		record.HallucinationConfidence = confidence
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
	record, found, err := m.getRecord(ctx, id, m.updateReadLevel())
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
	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
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
