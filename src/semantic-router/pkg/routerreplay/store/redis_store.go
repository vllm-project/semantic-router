package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// RedisStore implements ReplayStore using Redis.
// It stores records as JSON and uses sorted sets for time-based indexing.
//
// Key patterns:
//   - {prefix}{id}                    -> JSON record data
//   - {prefix}index:time              -> Sorted set by timestamp (score = unix nano)
//   - {prefix}index:decision:{name}   -> Set of record IDs for a decision
//   - {prefix}index:category:{name}   -> Set of record IDs for a category
//   - {prefix}index:model:{name}      -> Set of record IDs for a model
type RedisStore struct {
	client    *redis.Client
	keyPrefix string
	ttl       time.Duration
	enabled   bool
}

// NewRedisStore creates a new Redis-backed replay store.
func NewRedisStore(config RedisStoreConfig) (*RedisStore, error) {
	if config.Address == "" {
		return nil, fmt.Errorf("redis address is required")
	}

	keyPrefix := config.KeyPrefix
	if keyPrefix == "" {
		keyPrefix = "replay:"
	}

	ttl := time.Duration(config.TTLSeconds) * time.Second
	if config.TTLSeconds <= 0 {
		ttl = DefaultTTL
	}

	client := redis.NewClient(&redis.Options{
		Addr:     config.Address,
		Password: config.Password,
		DB:       config.Database,
	})

	store := &RedisStore{
		client:    client,
		keyPrefix: keyPrefix,
		ttl:       ttl,
		enabled:   true,
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := store.CheckConnection(ctx); err != nil {
		client.Close()
		return nil, fmt.Errorf("redis connection failed: %w", err)
	}

	logging.Infof("RedisStore connected to %s with prefix %s, TTL %v",
		config.Address, keyPrefix, ttl)

	return store, nil
}

// IsEnabled returns whether the store is enabled.
func (r *RedisStore) IsEnabled() bool {
	return r.enabled
}

// CheckConnection verifies the store connection is healthy.
func (r *RedisStore) CheckConnection(ctx context.Context) error {
	if !r.enabled {
		return nil
	}
	return r.client.Ping(ctx).Err()
}

// Close releases resources held by the store.
func (r *RedisStore) Close() error {
	if r.client != nil {
		return r.client.Close()
	}
	return nil
}

// Key generation helpers

func (r *RedisStore) recordKey(id string) string {
	return r.keyPrefix + id
}

func (r *RedisStore) timeIndexKey() string {
	return r.keyPrefix + "index:time"
}

func (r *RedisStore) decisionIndexKey(decision string) string {
	return r.keyPrefix + "index:decision:" + decision
}

func (r *RedisStore) categoryIndexKey(category string) string {
	return r.keyPrefix + "index:category:" + category
}

func (r *RedisStore) modelIndexKey(model string) string {
	return r.keyPrefix + "index:model:" + model
}

// StoreRecord stores a new routing record.
func (r *RedisStore) StoreRecord(ctx context.Context, record *RoutingRecord) error {
	if !r.enabled {
		return ErrStoreDisabled
	}
	if record == nil || record.ID == "" {
		return ErrInvalidInput
	}

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}

	key := r.recordKey(record.ID)

	// Use pipeline for atomic operations
	pipe := r.client.Pipeline()

	// Store the record as JSON with TTL
	pipe.Set(ctx, key, data, r.ttl)

	// Add to time index (sorted by timestamp)
	pipe.ZAdd(ctx, r.timeIndexKey(), redis.Z{
		Score:  float64(record.Timestamp.UnixNano()),
		Member: record.ID,
	})

	// Add to secondary indexes for filtering
	if record.Decision != "" {
		pipe.SAdd(ctx, r.decisionIndexKey(record.Decision), record.ID)
		pipe.Expire(ctx, r.decisionIndexKey(record.Decision), r.ttl)
	}
	if record.Category != "" {
		pipe.SAdd(ctx, r.categoryIndexKey(record.Category), record.ID)
		pipe.Expire(ctx, r.categoryIndexKey(record.Category), r.ttl)
	}
	if record.SelectedModel != "" {
		pipe.SAdd(ctx, r.modelIndexKey(record.SelectedModel), record.ID)
		pipe.Expire(ctx, r.modelIndexKey(record.SelectedModel), r.ttl)
	}

	_, err = pipe.Exec(ctx)
	if err != nil {
		return fmt.Errorf("failed to store record: %w", err)
	}

	return nil
}

// GetRecord retrieves a routing record by ID.
func (r *RedisStore) GetRecord(ctx context.Context, recordID string) (*RoutingRecord, error) {
	if !r.enabled {
		return nil, ErrStoreDisabled
	}
	if recordID == "" {
		return nil, ErrInvalidID
	}

	data, err := r.client.Get(ctx, r.recordKey(recordID)).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get record: %w", err)
	}

	var record RoutingRecord
	if err := json.Unmarshal(data, &record); err != nil {
		return nil, fmt.Errorf("failed to unmarshal record: %w", err)
	}

	return &record, nil
}

// UpdateRecord updates an existing routing record.
func (r *RedisStore) UpdateRecord(ctx context.Context, record *RoutingRecord) error {
	if !r.enabled {
		return ErrStoreDisabled
	}
	if record == nil || record.ID == "" {
		return ErrInvalidInput
	}

	key := r.recordKey(record.ID)

	// Check if record exists
	exists, err := r.client.Exists(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to check record existence: %w", err)
	}
	if exists == 0 {
		return ErrNotFound
	}

	data, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("failed to marshal record: %w", err)
	}

	// Update preserving TTL
	err = r.client.Set(ctx, key, data, redis.KeepTTL).Err()
	if err != nil {
		return fmt.Errorf("failed to update record: %w", err)
	}

	return nil
}

// DeleteRecord deletes a routing record by ID.
func (r *RedisStore) DeleteRecord(ctx context.Context, recordID string) error {
	if !r.enabled {
		return ErrStoreDisabled
	}
	if recordID == "" {
		return ErrInvalidID
	}

	// Get record first to clean up indexes
	record, err := r.GetRecord(ctx, recordID)
	if err != nil {
		return err
	}

	pipe := r.client.Pipeline()

	// Delete record
	pipe.Del(ctx, r.recordKey(recordID))

	// Clean up indexes
	pipe.ZRem(ctx, r.timeIndexKey(), recordID)
	if record.Decision != "" {
		pipe.SRem(ctx, r.decisionIndexKey(record.Decision), recordID)
	}
	if record.Category != "" {
		pipe.SRem(ctx, r.categoryIndexKey(record.Category), recordID)
	}
	if record.SelectedModel != "" {
		pipe.SRem(ctx, r.modelIndexKey(record.SelectedModel), recordID)
	}

	_, err = pipe.Exec(ctx)
	if err != nil {
		return fmt.Errorf("failed to delete record: %w", err)
	}

	return nil
}

// ListRecords lists routing records with pagination and filtering.
func (r *RedisStore) ListRecords(ctx context.Context, opts ListOptions) (*ListResult, error) {
	if !r.enabled {
		return nil, ErrStoreDisabled
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	if limit > MaxListLimit {
		limit = MaxListLimit
	}

	// Get record IDs from time index
	var ids []string
	var err error

	// Fetch more than needed to allow for filtering
	fetchCount := int64(limit * 3)

	if opts.Order == "asc" {
		ids, err = r.client.ZRange(ctx, r.timeIndexKey(), 0, fetchCount-1).Result()
	} else {
		// Default: descending (newest first)
		ids, err = r.client.ZRevRange(ctx, r.timeIndexKey(), 0, fetchCount-1).Result()
	}
	if err != nil {
		return nil, fmt.Errorf("failed to list record IDs: %w", err)
	}

	// Apply cursor pagination
	startIdx := 0
	if opts.After != "" {
		for i, id := range ids {
			if id == opts.After {
				startIdx = i + 1
				break
			}
		}
	}

	if startIdx >= len(ids) {
		return &ListResult{Records: nil, HasMore: false}, nil
	}
	ids = ids[startIdx:]

	// Fetch and filter records
	var results []*RoutingRecord
	for _, id := range ids {
		if len(results) >= limit+1 { // +1 to check hasMore
			break
		}

		record, err := r.GetRecord(ctx, id)
		if err != nil {
			// Skip missing records (may have expired)
			continue
		}

		if r.matchesFilters(record, opts) {
			results = append(results, record)
		}
	}

	// Check if there are more results
	hasMore := len(results) > limit
	if hasMore {
		results = results[:limit]
	}

	result := &ListResult{
		Records: results,
		HasMore: hasMore,
	}

	if len(results) > 0 {
		result.FirstID = results[0].ID
		result.LastID = results[len(results)-1].ID
	}

	return result, nil
}

// matchesFilters checks if a record matches the filter criteria.
func (r *RedisStore) matchesFilters(rec *RoutingRecord, opts ListOptions) bool {
	if opts.DecisionName != "" && rec.Decision != opts.DecisionName {
		return false
	}
	if opts.Category != "" && rec.Category != opts.Category {
		return false
	}
	if opts.Model != "" && rec.SelectedModel != opts.Model {
		return false
	}
	if opts.StartTime != nil && rec.Timestamp.Before(*opts.StartTime) {
		return false
	}
	if opts.EndTime != nil && rec.Timestamp.After(*opts.EndTime) {
		return false
	}
	if opts.FromCache != nil && rec.FromCache != *opts.FromCache {
		return false
	}
	if opts.RequestID != "" && rec.RequestID != opts.RequestID {
		return false
	}
	return true
}

// RecordCount returns the total number of records in the store (for testing).
func (r *RedisStore) RecordCount(ctx context.Context) (int64, error) {
	if !r.enabled {
		return 0, ErrStoreDisabled
	}
	return r.client.ZCard(ctx, r.timeIndexKey()).Result()
}

// CleanupExpired removes expired records from indexes.
// This is called periodically to clean up stale index entries
// for records that have been removed by Redis TTL.
func (r *RedisStore) CleanupExpired(ctx context.Context) error {
	if !r.enabled {
		return ErrStoreDisabled
	}

	// Get all IDs from time index
	ids, err := r.client.ZRange(ctx, r.timeIndexKey(), 0, -1).Result()
	if err != nil {
		return fmt.Errorf("failed to get IDs for cleanup: %w", err)
	}

	// Check each ID and remove from index if record no longer exists
	pipe := r.client.Pipeline()
	for _, id := range ids {
		exists, existsErr := r.client.Exists(ctx, r.recordKey(id)).Result()
		if existsErr != nil {
			continue
		}
		if exists == 0 {
			pipe.ZRem(ctx, r.timeIndexKey(), id)
		}
	}

	_, err = pipe.Exec(ctx)
	return err
}
