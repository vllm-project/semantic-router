package store

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	DefaultRedisKeyPrefix = "router_replay:"
	DefaultRedisPoolSize  = 10
)

// RedisStore implements Storage using Redis as the backend.
// Records are stored as JSON with optional TTL expiration.
type RedisStore struct {
	client      *redis.Client
	keyPrefix   string
	ttl         time.Duration
	asyncWrites bool
	asyncChan   chan asyncOp
	done        chan struct{}
	writerDone  chan struct{}
	updateMu    sync.Mutex
	lifecycle   *storageLifecycle
}

// NewRedisStore creates a new Redis storage backend.
func NewRedisStore(cfg *RedisConfig, ttlSeconds int, asyncWrites bool) (*RedisStore, error) {
	if cfg == nil {
		return nil, fmt.Errorf("redis config is required")
	}

	if cfg.Address == "" {
		cfg.Address = "localhost:6379"
	}

	keyPrefix := cfg.KeyPrefix
	if keyPrefix == "" {
		keyPrefix = DefaultRedisKeyPrefix
	}

	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = DefaultRedisPoolSize
	}

	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}

	opts := &redis.Options{
		Addr:       cfg.Address,
		DB:         cfg.DB,
		Password:   cfg.Password,
		PoolSize:   poolSize,
		MaxRetries: maxRetries,
	}

	if cfg.UseTLS {
		opts.TLSConfig = &tls.Config{
			InsecureSkipVerify: cfg.TLSSkipVerify,
		}
	}

	client := redis.NewClient(opts)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to redis: %w", err)
	}

	store := &RedisStore{
		client:      client,
		keyPrefix:   keyPrefix,
		ttl:         time.Duration(ttlSeconds) * time.Second,
		asyncWrites: asyncWrites,
		done:        make(chan struct{}),
		lifecycle:   newStorageLifecycle(),
	}

	if asyncWrites {
		store.asyncChan = make(chan asyncOp, 100)
		store.writerDone = make(chan struct{})
		go store.asyncWriter()
	}

	return store, nil
}

// asyncWriter processes async write operations in the background.
func (r *RedisStore) asyncWriter() {
	defer close(r.writerDone)
	for {
		select {
		case op := <-r.asyncChan:
			err := op.fn()
			if op.err != nil {
				op.err <- err
			}
		case <-r.done:
			return
		}
	}
}

// Add inserts a new record into Redis.
func (r *RedisStore) Add(ctx context.Context, record Record) (string, error) {
	release, err := r.lifecycle.beginMutation()
	if err != nil {
		return "", err
	}
	defer release()
	if record.LifecycleState == "" {
		record.LifecycleState = LifecycleInProgress
	}
	if record.ID == "" {
		generatedID, generateErr := generateRedisID()
		if generateErr != nil {
			return "", generateErr
		}
		record.ID = generatedID
	}

	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now().UTC()
	}

	data, err := json.Marshal(record)
	if err != nil {
		return "", fmt.Errorf("failed to marshal record: %w", err)
	}

	key := r.keyPrefix + record.ID

	fn := func() error {
		if r.ttl > 0 {
			return r.client.Set(ctx, key, data, r.ttl).Err()
		}
		return r.client.Set(ctx, key, data, 0).Err()
	}

	if r.asyncWrites {
		if err := runAsyncOp(ctx, r.asyncChan, fn); err != nil {
			return "", fmt.Errorf("failed to store record: %w", err)
		}
		return record.ID, nil
	}

	if err := fn(); err != nil {
		return "", fmt.Errorf("failed to store record: %w", err)
	}

	return record.ID, nil
}

// Get retrieves a record by ID from Redis.
func (r *RedisStore) Get(ctx context.Context, id string) (Record, bool, error) {
	release, err := r.lifecycle.beginMutation()
	if err != nil {
		return Record{}, false, err
	}
	defer release()
	return r.getRecord(ctx, id)
}

func (r *RedisStore) getRecord(ctx context.Context, id string) (Record, bool, error) {
	key := r.keyPrefix + id
	data, err := r.client.Get(ctx, key).Bytes()
	if errors.Is(err, redis.Nil) {
		return Record{}, false, nil
	}
	if err != nil {
		return Record{}, false, fmt.Errorf("failed to get record: %w", err)
	}

	var record Record
	if err := json.Unmarshal(data, &record); err != nil {
		return Record{}, false, fmt.Errorf("failed to unmarshal record: %w", err)
	}

	return record, true, nil
}

// List returns all records ordered by timestamp descending.
func (r *RedisStore) List(ctx context.Context) ([]Record, error) {
	release, err := r.lifecycle.beginMutation()
	if err != nil {
		return nil, err
	}
	defer release()
	return r.listRecords(ctx)
}

func (r *RedisStore) listRecords(ctx context.Context) ([]Record, error) {
	// Scan for all keys with our prefix
	var cursor uint64
	var keys []string

	for {
		var batch []string
		var err error
		batch, cursor, err = r.client.Scan(ctx, cursor, r.keyPrefix+"*", 100).Result()
		if err != nil {
			return nil, fmt.Errorf("failed to scan keys: %w", err)
		}
		keys = append(keys, batch...)
		if cursor == 0 {
			break
		}
	}

	if len(keys) == 0 {
		return []Record{}, nil
	}

	// Get all records
	pipe := r.client.Pipeline()
	cmds := make([]*redis.StringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.Get(ctx, key)
	}

	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		return nil, fmt.Errorf("failed to get records: %w", err)
	}

	records := make([]Record, 0, len(keys))
	for _, cmd := range cmds {
		data, err := cmd.Bytes()
		if errors.Is(err, redis.Nil) {
			continue
		}
		if err != nil {
			continue // Skip failed records
		}

		var record Record
		if err := json.Unmarshal(data, &record); err != nil {
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
func (r *RedisStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		if status != 0 {
			record.ResponseStatus = status
		}
		record.FromCache = record.FromCache || fromCache
		record.Streaming = record.Streaming || streaming
		return true
	})
}

func (r *RedisStore) UpdateLifecycle(
	ctx context.Context,
	id string,
	state string,
	endedAt time.Time,
	durationMS int64,
	reason string,
) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
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
func (r *RedisStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		record.RequestBody = body
		record.RequestBodyTruncated = record.RequestBodyTruncated || truncated
		return true
	})
}

// AttachResponse updates the response body for a record.
func (r *RedisStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		record.ResponseBody = body
		record.ResponseBodyTruncated = record.ResponseBodyTruncated || truncated
		return true
	})
}

// AppendOutcome links post-route feedback to a record.
func (r *RedisStore) AppendOutcome(ctx context.Context, id string, outcome Outcome) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		record.Outcomes = append(record.Outcomes, cloneOutcome(outcome))
		return true
	})
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (r *RedisStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		record.HallucinationDetected = detected
		record.HallucinationConfidence = confidence
		record.HallucinationSpans = cloneStringSlice(spans)
		record.HallucinationSpanDetails = cloneHallucinationSpanDetails(spanDetails)
		return true
	})
}

// UpdateUsageCost updates token usage and pricing-derived cost fields for a record.
func (r *RedisStore) UpdateUsageCost(ctx context.Context, id string, usage UsageCost) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
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
		return true
	})
}

// UpdateToolTrace updates tool-calling trace details for a record.
func (r *RedisStore) UpdateToolTrace(ctx context.Context, id string, trace ToolTrace) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		record.ToolTrace = cloneToolTrace(&trace)
		return true
	})
}

// updateRecord serializes Redis read-modify-write operations through the
// async writer. Reading inside the queued operation prevents a later update
// from overwriting fields written by an earlier queued operation with a stale
// snapshot. Every update waits for persistence so a completed lifecycle never
// claims that response, tool, or usage metadata was stored when it was not.
func (r *RedisStore) updateRecord(
	ctx context.Context,
	id string,
	mutate func(*Record) bool,
) error {
	release, err := r.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	fn := func() error {
		r.updateMu.Lock()
		defer r.updateMu.Unlock()
		record, found, err := r.getRecord(ctx, id)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("record with ID %s not found", id)
		}
		if !mutate(&record) {
			return nil
		}
		data, err := json.Marshal(record)
		if err != nil {
			return fmt.Errorf("failed to marshal record: %w", err)
		}
		key := r.keyPrefix + id
		if r.ttl > 0 {
			return r.client.Set(ctx, key, data, r.ttl).Err()
		}
		return r.client.Set(ctx, key, data, 0).Err()
	}
	if !r.asyncWrites {
		return fn()
	}
	return runAsyncOp(ctx, r.asyncChan, fn)
}

// Close closes the Redis client and stops async writer.
func (r *RedisStore) Close() error {
	return r.lifecycle.close(func() error {
		if r.asyncWrites {
			return closeAsyncWriter(r.asyncChan, r.done, r.writerDone, r.client.Close)
		}
		return r.client.Close()
	})
}

func generateRedisID() (string, error) {
	return generateID()
}
