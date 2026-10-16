package store

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	DefaultRedisKeyPrefix = "router_replay:"
	DefaultRedisPoolSize  = 10

	redisScanBatch = 100
)

// RedisStore implements Storage using Redis as the backend.
// Records are stored as JSON with optional TTL expiration.
type RedisStore struct {
	client      *redis.Client
	keyPrefix   string
	indexKey    string
	readyKey    string
	ttl         time.Duration
	sweep       retentionSweep
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
		indexKey:    redisIndexKey(keyPrefix),
		readyKey:    redisIndexKey(keyPrefix) + "_ready",
		ttl:         time.Duration(ttlSeconds) * time.Second,
		asyncWrites: asyncWrites,
		done:        make(chan struct{}),
		lifecycle:   newStorageLifecycle(),
	}

	indexCtx, indexCancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer indexCancel()
	if err := store.indexExisting(indexCtx); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("failed to index existing replay records: %w", err)
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

	data, err := encodeRecord(&record, MaxRecordBytes)
	if err != nil {
		return "", err
	}

	key := r.keyPrefix + record.ID

	fn := func() error {
		pipe := r.client.TxPipeline()
		pipe.Set(ctx, key, data, r.ttl)
		pipe.ZAdd(ctx, r.indexKey, redis.Z{Score: redisIndexScore(record.Timestamp), Member: record.ID})
		if _, err := pipe.Exec(ctx); err != nil {
			return err
		}
		if r.ttl > 0 && r.sweep.due() {
			return r.trimIndex(ctx)
		}
		return nil
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
	record, found, err := r.getRecord(ctx, id)
	if !found || isExpired(record, r.ttl) {
		return Record{}, false, err
	}
	return record, true, nil
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
	if err := r.trimIndex(ctx); err != nil {
		return nil, err
	}
	return listAll(r.scan(ctx))
}

func (r *RedisStore) QueryPage(ctx context.Context, filters QueryFilters, limit, offset int, details bool) (RecordPage, error) {
	release, err := r.lifecycle.beginMutation()
	if err != nil {
		return RecordPage{}, err
	}
	defer release()
	if err := r.trimIndex(ctx); err != nil {
		return RecordPage{}, err
	}
	count := func() (int, error) {
		n, err := r.client.ZCard(ctx, r.indexKey).Result()
		return int(n), err
	}
	return collectPage(filters, limit, offset, count, r.scan(ctx))
}

func (r *RedisStore) ScanMetadata(ctx context.Context, visit func(Record) error) error {
	release, err := r.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	if err := r.trimIndex(ctx); err != nil {
		return err
	}
	return scanAll(r.scan(ctx), visit)
}

// Equal scores come back in reverse ID order, same as PostgreSQL.
func (r *RedisStore) scan(ctx context.Context) recordScan {
	return func(from int, visit func(Record) bool) error {
		var missing []interface{}
		defer func() {
			if len(missing) > 0 {
				_ = r.client.ZRem(ctx, r.indexKey, missing...).Err()
			}
		}()
		for start := int64(from); ; start += redisScanBatch {
			ids, err := r.client.ZRevRange(ctx, r.indexKey, start, start+redisScanBatch-1).Result()
			if err != nil {
				return fmt.Errorf("failed to read replay index: %w", err)
			}
			if len(ids) == 0 {
				return nil
			}
			keys := make([]string, len(ids))
			for i, id := range ids {
				keys[i] = r.keyPrefix + id
			}
			values, err := r.client.MGet(ctx, keys...).Result()
			if err != nil {
				return fmt.Errorf("failed to get records: %w", err)
			}
			for i, value := range values {
				data, ok := value.(string)
				if !ok {
					missing = append(missing, ids[i])
					continue
				}
				var record Record
				if err := json.Unmarshal([]byte(data), &record); err != nil {
					continue
				}
				if !visit(record) {
					return nil
				}
			}
			if len(ids) < redisScanBatch {
				return nil
			}
		}
	}
}

func (r *RedisStore) trimIndex(ctx context.Context) error {
	if r.ttl <= 0 {
		return nil
	}
	cutoff := "(" + strconv.FormatFloat(redisIndexScore(retentionCutoff(r.ttl)), 'f', -1, 64)
	return r.client.ZRemRangeByScore(ctx, r.indexKey, "-inf", cutoff).Err()
}

// indexExisting indexes records written before the index existed. Writes also
// create the index, so the ready key marks a finished pass.
func (r *RedisStore) indexExisting(ctx context.Context) error {
	ready, err := r.client.Exists(ctx, r.readyKey).Result()
	if err != nil || ready > 0 {
		return err
	}
	var cursor uint64
	for {
		keys, next, err := r.client.Scan(ctx, cursor, r.keyPrefix+"*", redisScanBatch).Result()
		if err != nil {
			return err
		}
		if err := r.indexKeys(ctx, keys); err != nil {
			return err
		}
		if next == 0 {
			return r.client.Set(ctx, r.readyKey, "1", 0).Err()
		}
		cursor = next
	}
}

func (r *RedisStore) indexKeys(ctx context.Context, keys []string) error {
	keys = slices.DeleteFunc(keys, func(key string) bool { return key == r.indexKey })
	if len(keys) == 0 {
		return nil
	}
	values, err := r.client.MGet(ctx, keys...).Result()
	if err != nil {
		return err
	}
	var members []redis.Z
	for i, value := range values {
		data, ok := value.(string)
		if !ok {
			continue
		}
		var record struct {
			Timestamp time.Time `json:"timestamp"`
		}
		if json.Unmarshal([]byte(data), &record) != nil {
			continue
		}
		members = append(members, redis.Z{Score: redisIndexScore(record.Timestamp), Member: strings.TrimPrefix(keys[i], r.keyPrefix)})
	}
	if len(members) == 0 {
		return nil
	}
	return r.client.ZAdd(ctx, r.indexKey, members...).Err()
}

// redisIndexKey stays outside the default record key pattern.
func redisIndexKey(prefix string) string {
	return strings.TrimSuffix(prefix, ":") + "_index"
}

// Microseconds match PostgreSQL and stay exact in a float64 score.
func redisIndexScore(t time.Time) float64 {
	return float64(t.UnixMicro())
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
func (r *RedisStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string, spanDetails []HallucinationSpan, score ...HallucinationScore) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		record.HallucinationDetected = detected
		record.HallucinationConfidence = confidence
		applyHallucinationScore(record, score)
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
		data, err := encodeRecord(&record, MaxRecordBytes)
		if err != nil {
			return err
		}
		return r.client.Set(ctx, r.keyPrefix+id, data, r.ttl).Err()
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
