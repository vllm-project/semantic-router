package store

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
)

const (
	DefaultQdrantCollection = "router_replay_records"
	defaultQdrantPort       = 6334
)

type QdrantStore struct {
	client         *qdrant.Client
	collectionName string
	ttl            time.Duration
	asyncWrites    bool
	asyncChan      chan asyncOp
	done           chan struct{}
	writerDone     chan struct{}
	updateMu       sync.Mutex
	lifecycle      *storageLifecycle
}

func NewQdrantStore(cfg *QdrantConfig, ttlSeconds int, asyncWrites bool) (*QdrantStore, error) {
	if cfg == nil {
		return nil, fmt.Errorf("qdrant config is required")
	}
	if cfg.Host == "" {
		return nil, fmt.Errorf("qdrant host is required")
	}

	port := cfg.Port
	if port == 0 {
		port = defaultQdrantPort
	}

	collectionName := cfg.CollectionName
	if collectionName == "" {
		collectionName = DefaultQdrantCollection
	}

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:                   cfg.Host,
		Port:                   port,
		APIKey:                 cfg.APIKey,
		UseTLS:                 cfg.UseTLS,
		SkipCompatibilityCheck: true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create qdrant client: %w", err)
	}

	s := &QdrantStore{
		client:         client,
		collectionName: collectionName,
		ttl:            time.Duration(ttlSeconds) * time.Second,
		asyncWrites:    asyncWrites,
		done:           make(chan struct{}),
		lifecycle:      newStorageLifecycle(),
	}

	if err := s.ensureCollection(context.Background()); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("failed to ensure qdrant collection: %w", err)
	}

	if asyncWrites {
		s.asyncChan = make(chan asyncOp, 100)
		s.writerDone = make(chan struct{})
		go s.asyncWriter()
	}

	return s, nil
}

// Qdrant only allows UUIDs and +ve integers.
// Ref: https://qdrant.tech/documentation/concepts/points/#point-ids
// So we create a deterministic UUID based on the original ID.
func arbitraryIDToUUID(id string) *qdrant.PointId {
	// If already a valid UUID, use it directly
	if _, err := uuid.Parse(id); err == nil {
		return qdrant.NewIDUUID(id)
	}

	// Otherwise create a deterministic UUID based on the ID
	deterministicUUID := uuid.NewSHA1(uuid.NameSpaceURL, []byte(id))
	return qdrant.NewIDUUID(deterministicUUID.String())
}

func (q *QdrantStore) ensureCollection(ctx context.Context) error {
	exists, err := q.client.CollectionExists(ctx, q.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}
	if exists {
		return nil
	}
	return q.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: q.collectionName,
		VectorsConfig:  qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{}),
	})
}

func (q *QdrantStore) asyncWriter() {
	defer close(q.writerDone)
	for {
		select {
		case op := <-q.asyncChan:
			err := op.fn()
			if op.err != nil {
				op.err <- err
			}
		case <-q.done:
			return
		}
	}
}

func (q *QdrantStore) Add(ctx context.Context, record Record) (string, error) {
	release, err := q.lifecycle.beginMutation()
	if err != nil {
		return "", err
	}
	defer release()
	if record.LifecycleState == "" {
		record.LifecycleState = LifecycleInProgress
	}
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

	fn := func() error { return q.upsert(ctx, record) }

	if q.asyncWrites {
		if err := runAsyncOp(ctx, q.asyncChan, fn); err != nil {
			return "", fmt.Errorf("failed to add record: %w", err)
		}
		return record.ID, nil
	}

	if err := fn(); err != nil {
		return "", fmt.Errorf("failed to add record: %w", err)
	}
	return record.ID, nil
}

func (q *QdrantStore) Get(ctx context.Context, id string) (Record, bool, error) {
	release, err := q.lifecycle.beginMutation()
	if err != nil {
		return Record{}, false, err
	}
	defer release()
	return q.getRecord(ctx, id)
}

func (q *QdrantStore) getRecord(ctx context.Context, id string) (Record, bool, error) {
	if q.asyncWrites {
		if err := runAsyncOp(ctx, q.asyncChan, func() error { return nil }); err != nil {
			return Record{}, false, err
		}
	}

	pts, err := q.client.Get(ctx, &qdrant.GetPoints{
		CollectionName: q.collectionName,
		Ids:            []*qdrant.PointId{arbitraryIDToUUID(id)},
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return Record{}, false, fmt.Errorf("failed to get record %q: %w", id, err)
	}
	if len(pts) == 0 {
		return Record{}, false, nil
	}
	return q.decodePoint(pts[0])
}

func (q *QdrantStore) List(ctx context.Context) ([]Record, error) {
	release, err := q.lifecycle.beginMutation()
	if err != nil {
		return nil, err
	}
	defer release()
	return q.listRecords(ctx)
}

func (q *QdrantStore) listRecords(ctx context.Context) ([]Record, error) {
	if q.asyncWrites {
		if err := runAsyncOp(ctx, q.asyncChan, func() error { return nil }); err != nil {
			return nil, err
		}
	}

	var records []Record
	var offset *qdrant.PointId

	for {
		result, nextOffset, err := q.client.ScrollAndOffset(ctx, &qdrant.ScrollPoints{
			CollectionName: q.collectionName,
			Limit:          qdrant.PtrOf(uint32(100)),
			WithPayload:    qdrant.NewWithPayload(true),
			Offset:         offset,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to scroll records: %w", err)
		}
		for _, pt := range result {
			r, ok, err := q.decodePoint(pt)
			if err != nil || !ok {
				continue
			}
			records = append(records, r)
		}
		if nextOffset == nil {
			break
		}
		offset = nextOffset
	}

	sort.Slice(records, func(i, j int) bool {
		return records[i].Timestamp.After(records[j].Timestamp)
	})
	return records, nil
}

func (q *QdrantStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	return q.updateRecord(ctx, id, func(r *Record) {
		if status != 0 {
			r.ResponseStatus = status
		}
		r.FromCache = r.FromCache || fromCache
		r.Streaming = r.Streaming || streaming
	})
}

func (q *QdrantStore) UpdateLifecycle(
	ctx context.Context,
	id string,
	state string,
	endedAt time.Time,
	durationMS int64,
	reason string,
) error {
	return q.updateRecordIf(ctx, id, func(record *Record) bool {
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

func (q *QdrantStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	return q.updateRecord(ctx, id, func(r *Record) {
		r.RequestBody = body
		r.RequestBodyTruncated = r.RequestBodyTruncated || truncated
	})
}

func (q *QdrantStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	return q.updateRecord(ctx, id, func(r *Record) {
		r.ResponseBody = body
		r.ResponseBodyTruncated = r.ResponseBodyTruncated || truncated
	})
}

func (q *QdrantStore) AppendOutcome(ctx context.Context, id string, outcome Outcome) error {
	return q.updateRecord(ctx, id, func(r *Record) {
		r.Outcomes = append(r.Outcomes, cloneOutcome(outcome))
	})
}

func (q *QdrantStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan) error {
	return q.updateRecord(ctx, id, func(r *Record) {
		r.HallucinationDetected = detected
		r.HallucinationConfidence = confidence
		r.HallucinationSpans = spans
		r.HallucinationSpanDetails = spanDetails
	})
}

func (q *QdrantStore) UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error {
	return q.updateRecord(ctx, id, func(r *Record) {
		r.PromptTokens = cloneIntPtr(usage.PromptTokens)
		r.CachedPromptTokens = cloneIntPtr(usage.CachedPromptTokens)
		r.CacheWriteTokens = cloneIntPtr(usage.CacheWriteTokens)
		r.CompletionTokens = cloneIntPtr(usage.CompletionTokens)
		r.TotalTokens = cloneIntPtr(usage.TotalTokens)
		r.ActualCost = cloneFloat64Ptr(usage.ActualCost)
		r.BaselineCost = cloneFloat64Ptr(usage.BaselineCost)
		r.CostSavings = cloneFloat64Ptr(usage.CostSavings)
		r.Currency = cloneStringPtr(usage.Currency)
		r.BaselineModel = cloneStringPtr(usage.BaselineModel)
	})
}

func (q *QdrantStore) UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error {
	return q.updateRecord(ctx, id, func(r *Record) {
		r.ToolTrace = cloneToolTrace(&trace)
	})
}

func (q *QdrantStore) Close() error {
	return q.lifecycle.close(func() error {
		if q.asyncWrites {
			return closeAsyncWriter(q.asyncChan, q.done, q.writerDone, q.client.Close)
		}
		return q.client.Close()
	})
}

func (q *QdrantStore) updateRecord(ctx context.Context, id string, mutate func(*Record)) error {
	return q.updateRecordIf(ctx, id, func(record *Record) bool {
		mutate(record)
		return true
	})
}

func (q *QdrantStore) updateRecordIf(ctx context.Context, id string, mutate func(*Record) bool) error {
	release, err := q.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	q.updateMu.Lock()
	defer q.updateMu.Unlock()
	record, found, err := q.getRecord(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}
	if !mutate(&record) {
		return nil
	}
	// Read-modify-write updates are synchronous even when bulk writes are
	// configured async. Qdrant stores a whole JSON record, so queueing stale
	// snapshots can otherwise drop usage, trace, or terminal fields.
	return q.upsert(ctx, record)
}

func (q *QdrantStore) upsert(ctx context.Context, record Record) error {
	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}
	wait := true
	_, err = q.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: q.collectionName,
		Wait:           &wait,
		Points: []*qdrant.PointStruct{
			{
				Id:      arbitraryIDToUUID(record.ID),
				Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{}),
				Payload: map[string]*qdrant.Value{
					"id":        qdrant.NewValueString(record.ID),
					"timestamp": qdrant.NewValueInt(record.Timestamp.Unix()),
					"data":      qdrant.NewValueString(string(data)),
				},
			},
		},
	})
	return err
}

func (q *QdrantStore) decodePoint(pt *qdrant.RetrievedPoint) (Record, bool, error) {
	dataVal, ok := pt.Payload["data"]
	if !ok {
		return Record{}, false, nil
	}
	var record Record
	if err := json.Unmarshal([]byte(dataVal.GetStringValue()), &record); err != nil {
		return Record{}, false, fmt.Errorf("failed to unmarshal record: %w", err)
	}
	return record, true, nil
}
