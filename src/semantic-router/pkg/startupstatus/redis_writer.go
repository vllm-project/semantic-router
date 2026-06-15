package startupstatus

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

const defaultStatusKey = "vllm-sr:startup-status"

var _ StatusWriter = (*RedisWriter)(nil)

// RedisWriter persists startup state to a Redis key so the dashboard
// and other consumers can read it without sharing a filesystem.
type RedisWriter struct {
	client *redis.Client
	key    string
	ttl    time.Duration
	mu     sync.Mutex
}

// RedisWriterConfig holds the parameters for creating a RedisWriter.
type RedisWriterConfig struct {
	Address  string
	Password string
	DB       int
	TTL      time.Duration
}

// NewRedisWriter creates a Redis-backed StatusWriter. It pings the server
// on creation to verify connectivity.
func NewRedisWriter(cfg RedisWriterConfig) (*RedisWriter, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     cfg.Address,
		Password: cfg.Password,
		DB:       cfg.DB,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("startup status redis ping failed: %w", err)
	}

	ttl := cfg.TTL
	if ttl == 0 {
		ttl = 5 * time.Minute
	}

	return &RedisWriter{
		client: client,
		key:    defaultStatusKey,
		ttl:    ttl,
	}, nil
}

// Write serializes the state and stores it in Redis with a TTL.
func (w *RedisWriter) Write(state State) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	state.UpdatedAt = time.Now().UTC().Format(time.RFC3339)

	payload, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal startup status for redis: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	return w.client.Set(ctx, w.key, payload, w.ttl).Err()
}

// Close releases the Redis connection.
func (w *RedisWriter) Close() error {
	return w.client.Close()
}

// LoadFromRedis reads the startup status from Redis using the standard key.
func LoadFromRedis(client *redis.Client) (*State, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	data, err := client.Get(ctx, defaultStatusKey).Bytes()
	if err != nil {
		return nil, err
	}

	var state State
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("decode startup status from redis: %w", err)
	}

	return &state, nil
}
