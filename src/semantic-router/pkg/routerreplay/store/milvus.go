package store

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	DefaultMilvusCollection       = "router_replay_records"
	DefaultMilvusConsistencyLevel = "Session"
	DefaultMilvusShardNum         = 2
)

// MilvusStore implements Storage using Milvus as the backend.
// Records are stored as entities in a Milvus collection.
type MilvusStore struct {
	client         client.Client
	collectionName string
	ttl            time.Duration
	asyncWrites    bool
	asyncChan      chan asyncOp
	done           chan struct{}
	wg             sync.WaitGroup
	pendingWrites  map[string]struct{}
	mu             sync.RWMutex
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

	// Create Milvus client
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	milvusClient, err := client.NewClient(ctx, client.Config{
		Address:  cfg.Address,
		Username: cfg.Username,
		Password: cfg.Password,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create milvus client: %w", err)
	}

	store := &MilvusStore{
		client:         milvusClient,
		collectionName: collectionName,
		ttl:            time.Duration(ttlSeconds) * time.Second,
		asyncWrites:    asyncWrites,
		done:           make(chan struct{}),
		pendingWrites:  make(map[string]struct{}),
	}

	// Create collection if not exists with retry logic
	var collErr error
	for i := 0; i < 3; i++ {
		collCtx, collCancel := context.WithTimeout(context.Background(), 20*time.Second)
		collErr = store.createCollection(collCtx, cfg)
		collCancel()

		if collErr == nil {
			break
		}

		// Wait before retry
		if i < 2 {
			time.Sleep(time.Duration(i+1) * 2 * time.Second)
		}
	}

	if collErr != nil {
		return nil, fmt.Errorf("failed to create collection after retries: %w", collErr)
	}

	if asyncWrites {
		store.asyncChan = make(chan asyncOp, 100)
		go store.asyncWriter()
	}

	return store, nil
}

// createCollection creates the Milvus collection if it doesn't exist.
func (m *MilvusStore) createCollection(ctx context.Context, cfg *MilvusConfig) error {
	// Check if collection exists
	has, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if has {
		// Ensure required vector index exists (Milvus needs one before insert)
		idxes, idxErr := m.client.DescribeIndex(ctx, m.collectionName, "vector")
		if idxErr != nil {
			return fmt.Errorf("failed to describe vector index: %w", idxErr)
		}

		if len(idxes) == 0 {
			vectorIdx, idxErr := entity.NewIndexAUTOINDEX(entity.L2)
			if idxErr != nil {
				return fmt.Errorf("failed to build vector index: %w", idxErr)
			}

			if idxErr := m.client.CreateIndex(ctx, m.collectionName, "vector", vectorIdx, false); idxErr != nil {
				return fmt.Errorf("failed to create vector index: %w", idxErr)
			}
		}

		// Load collection into memory
		return m.client.LoadCollection(ctx, m.collectionName, false)
	}

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
	err = m.client.CreateCollection(ctx, schema, int32(shardNum))
	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	// Create vector index (required by Milvus even for dummy vectors)
	vectorIdx, err := entity.NewIndexAUTOINDEX(entity.L2)
	if err != nil {
		return fmt.Errorf("failed to build vector index: %w", err)
	}

	err = m.client.CreateIndex(ctx, m.collectionName, "vector", vectorIdx, false)
	if err != nil {
		return fmt.Errorf("failed to create vector index: %w", err)
	}

	// Create index on timestamp for efficient querying
	timeIdx := entity.NewGenericIndex(
		"timestamp_idx",
		entity.Sorted,
		map[string]string{},
	)

	err = m.client.CreateIndex(ctx, m.collectionName, "timestamp", timeIdx, false)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	// Load collection
	return m.client.LoadCollection(ctx, m.collectionName, false)
}

// asyncWriter processes async write operations.
func (m *MilvusStore) asyncWriter() {
	for {
		select {
		case op := <-m.asyncChan:
			err := op.fn()
			if op.err != nil {
				op.err <- err
			}
			m.wg.Done()
		case <-m.done:
			return
		}
	}
}

// Add inserts a new record into Milvus.
func (m *MilvusStore) Add(ctx context.Context, record Record) (string, error) {
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
		return err
	}

	if m.asyncWrites {
		m.mu.Lock()
		m.pendingWrites[record.ID] = struct{}{}
		m.mu.Unlock()
		m.wg.Add(1)
		errChan := make(chan error, 1)
		m.asyncChan <- asyncOp{fn: func() error {
			err := fn()
			m.mu.Lock()
			delete(m.pendingWrites, record.ID)
			m.mu.Unlock()
			if err == nil {
				// Flush to ensure data is persisted
				_ = m.client.Flush(ctx, m.collectionName, false)
			}
			return err
		}, err: errChan}
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
	// If async writes enabled and record is pending, wait for it
	if m.asyncWrites {
		m.mu.RLock()
		_, pending := m.pendingWrites[id]
		m.mu.RUnlock()

		if pending {
			// Wait for pending writes to complete
			m.wg.Wait()
		}
	}

	// Query by ID
	expr := fmt.Sprintf("id == '%s'", id)
	result, err := m.client.Query(ctx, m.collectionName, nil, expr, []string{"id", "timestamp", "data"})
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
	// If async writes enabled, wait for all pending writes to complete
	if m.asyncWrites {
		m.wg.Wait()
	}

	// Query all records with a high limit (empty expression requires limit in Milvus)
	result, err := m.client.Query(
		ctx,
		m.collectionName,
		nil,
		"",
		[]string{"id", "timestamp", "data"},
		client.WithLimit(10000),
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

	// Sort by timestamp descending
	for i := 0; i < len(records)-1; i++ {
		for j := i + 1; j < len(records); j++ {
			if records[i].Timestamp.Before(records[j].Timestamp) {
				records[i], records[j] = records[j], records[i]
			}
		}
	}

	return records, nil
}

// UpdateStatus updates the response status and flags for a record.
func (m *MilvusStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	// Get existing record
	record, found, err := m.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	// Update fields
	if status != 0 {
		record.ResponseStatus = status
	}
	record.FromCache = record.FromCache || fromCache
	record.Streaming = record.Streaming || streaming

	// Delete old record and insert updated one
	return m.upsertRecord(ctx, record)
}

// AttachRequest updates the request body for a record.
func (m *MilvusStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	record, found, err := m.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	record.RequestBody = body
	record.RequestBodyTruncated = record.RequestBodyTruncated || truncated

	return m.upsertRecord(ctx, record)
}

// AttachResponse updates the response body for a record.
func (m *MilvusStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	record, found, err := m.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	record.ResponseBody = body
	record.ResponseBodyTruncated = record.ResponseBodyTruncated || truncated

	return m.upsertRecord(ctx, record)
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (m *MilvusStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string) error {
	record, found, err := m.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	record.HallucinationDetected = detected
	record.HallucinationConfidence = confidence
	record.HallucinationSpans = spans

	return m.upsertRecord(ctx, record)
}

// upsertRecord updates a record by deleting and reinserting.
func (m *MilvusStore) upsertRecord(ctx context.Context, record Record) error {
	// Marshal record
	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}

	fn := func() error {
		// Delete existing record
		expr := fmt.Sprintf("id == '%s'", record.ID)
		if err := m.client.Delete(ctx, m.collectionName, "", expr); err != nil {
			return fmt.Errorf("failed to delete old record: %w", err)
		}

		// Insert updated record
		idColumn := entity.NewColumnVarChar("id", []string{record.ID})
		timestampColumn := entity.NewColumnInt64("timestamp", []int64{record.Timestamp.Unix()})
		dataColumn := entity.NewColumnVarChar("data", []string{string(data)})
		vectorColumn := entity.NewColumnFloatVector("vector", 2, [][]float32{{0.0, 0.0}})

		_, err := m.client.Insert(ctx, m.collectionName, "", idColumn, timestampColumn, dataColumn, vectorColumn)
		if err != nil {
			return fmt.Errorf("failed to insert updated record: %w", err)
		}

		return m.client.Flush(ctx, m.collectionName, false)
	}

	if m.asyncWrites {
		m.wg.Add(1)
		m.asyncChan <- asyncOp{fn: func() error {
			err := fn()
			if err == nil {
				// Flush to ensure data is persisted
				_ = m.client.Flush(ctx, m.collectionName, false)
			}
			return err
		}}
		return nil
	}

	return fn()
}

// Close closes the Milvus client and stops async writer.
func (m *MilvusStore) Close() error {
	if m.asyncWrites {
		// Wait for all pending writes to complete
		m.wg.Wait()
		close(m.done)
	}
	return m.client.Close()
}
