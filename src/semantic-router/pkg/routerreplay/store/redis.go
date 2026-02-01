package store

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
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
}

type asyncOp struct {
	fn  func() error
	err chan error
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
	}

	if asyncWrites {
		store.asyncChan = make(chan asyncOp, 100)
		go store.asyncWriter()
	}

	return store, nil
}

// asyncWriter processes async write operations in the background.
func (r *RedisStore) asyncWriter() {
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
	if record.ID == "" {
		id, err := generateRedisID()
		if err != nil {
			return "", err
		}
		record.ID = id
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
		errChan := make(chan error, 1)
		r.asyncChan <- asyncOp{fn: fn, err: errChan}
		// Non-blocking for async writes
		return record.ID, nil
	}

	if err := fn(); err != nil {
		return "", fmt.Errorf("failed to store record: %w", err)
	}

	return record.ID, nil
}

// Get retrieves a record by ID from Redis.
func (r *RedisStore) Get(ctx context.Context, id string) (Record, bool, error) {
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
func (r *RedisStore) UpdateStatus(ctx context.Context, id string, status int, fromCache bool, streaming bool) error {
	record, found, err := r.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	if status != 0 {
		record.ResponseStatus = status
	}
	record.FromCache = record.FromCache || fromCache
	record.Streaming = record.Streaming || streaming

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}

	key := r.keyPrefix + id
	fn := func() error {
		if r.ttl > 0 {
			return r.client.Set(ctx, key, data, r.ttl).Err()
		}
		return r.client.Set(ctx, key, data, 0).Err()
	}

	if r.asyncWrites {
		r.asyncChan <- asyncOp{fn: fn}
		return nil
	}

	return fn()
}

// AttachRequest updates the request body for a record.
func (r *RedisStore) AttachRequest(ctx context.Context, id string, body string, truncated bool) error {
	record, found, err := r.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	record.RequestBody = body
	record.RequestBodyTruncated = record.RequestBodyTruncated || truncated

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}

	key := r.keyPrefix + id
	fn := func() error {
		if r.ttl > 0 {
			return r.client.Set(ctx, key, data, r.ttl).Err()
		}
		return r.client.Set(ctx, key, data, 0).Err()
	}

	if r.asyncWrites {
		r.asyncChan <- asyncOp{fn: fn}
		return nil
	}

	return fn()
}

// AttachResponse updates the response body for a record.
func (r *RedisStore) AttachResponse(ctx context.Context, id string, body string, truncated bool) error {
	record, found, err := r.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	record.ResponseBody = body
	record.ResponseBodyTruncated = record.ResponseBodyTruncated || truncated

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}

	key := r.keyPrefix + id
	fn := func() error {
		if r.ttl > 0 {
			return r.client.Set(ctx, key, data, r.ttl).Err()
		}
		return r.client.Set(ctx, key, data, 0).Err()
	}

	if r.asyncWrites {
		r.asyncChan <- asyncOp{fn: fn}
		return nil
	}

	return fn()
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (r *RedisStore) UpdateHallucinationStatus(ctx context.Context, id string, detected bool, confidence float32, spans []string) error {
	record, found, err := r.Get(ctx, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("record with ID %s not found", id)
	}

	record.HallucinationDetected = detected
	record.HallucinationConfidence = confidence
	record.HallucinationSpans = spans

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}

	key := r.keyPrefix + id
	fn := func() error {
		if r.ttl > 0 {
			return r.client.Set(ctx, key, data, r.ttl).Err()
		}
		return r.client.Set(ctx, key, data, 0).Err()
	}

	if r.asyncWrites {
		r.asyncChan <- asyncOp{fn: fn}
		return nil
	}

	return fn()
}

// Close closes the Redis client and stops async writer.
func (r *RedisStore) Close() error {
	if r.asyncWrites {
		close(r.done)
	}
	return r.client.Close()
}

func generateRedisID() (string, error) {
	return generateID()
}
