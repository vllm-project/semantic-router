package startupstatus

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	// DefaultRedisTTL bounds how long a stopped process remains observable.
	DefaultRedisTTL       = 30 * time.Second
	redisStatusPrefix     = "vllm-sr:startup-status:"
	redisOperationTimeout = 3 * time.Second
)

var _ StatusWriter = (*RedisWriter)(nil)

// RedisWriter publishes one process incarnation. Heartbeats retain its latest
// observation; they do not imply that dependencies or the Router are ready.
type RedisWriter struct {
	client     *redis.Client
	key        string
	instanceID string
	ttl        time.Duration
	ctx        context.Context
	cancel     context.CancelFunc
	done       chan struct{}
	mu         sync.Mutex
	payload    []byte
	closeOnce  sync.Once
	closeErr   error
}

type RedisWriterConfig struct {
	Address  string
	Password string
	DB       int
	// InstanceID must identify one process incarnation, not a reusable pod name.
	InstanceID string
	// TTL defaults to DefaultRedisTTL. The writer refreshes it every TTL/3.
	TTL time.Duration
}

func redisStatusKey(instanceID string) (string, error) {
	if instanceID == "" || len(instanceID) > 128 || strings.IndexFunc(instanceID, func(r rune) bool {
		return !strings.ContainsRune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_", r)
	}) >= 0 {
		return "", fmt.Errorf("startup status requires a bounded alphanumeric instance ID (hyphen and underscore allowed)")
	}
	return redisStatusPrefix + instanceID, nil
}

// NewRedisWriter starts a publisher without requiring Redis to be reachable.
// Failed writes retain the latest observation for the next heartbeat retry.
// Call Close on every return path to stop publishing and release the client.
func NewRedisWriter(cfg RedisWriterConfig) (*RedisWriter, error) {
	key, err := redisStatusKey(cfg.InstanceID)
	if err != nil {
		return nil, err
	}
	if cfg.TTL == 0 {
		cfg.TTL = DefaultRedisTTL
	}
	if cfg.TTL < 3*time.Millisecond {
		return nil, fmt.Errorf("startup status TTL must be at least 3ms")
	}
	ctx, cancel := context.WithCancel(context.Background())
	w := &RedisWriter{
		client: redis.NewClient(&redis.Options{
			Addr: cfg.Address, Password: cfg.Password, DB: cfg.DB,
			ContextTimeoutEnabled: true,
		}),
		key: key, instanceID: cfg.InstanceID, ttl: cfg.TTL,
		ctx: ctx, cancel: cancel, done: make(chan struct{}),
	}
	go w.heartbeat()
	return w, nil
}

func (w *RedisWriter) Write(state State) error {
	state.InstanceID = w.instanceID
	state.UpdatedAt = time.Now().UTC().Format(time.RFC3339Nano)
	payload, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal startup status for redis: %w", err)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.ctx.Err(); err != nil {
		return err
	}
	// Save serialized bytes so callers cannot mutate a future heartbeat.
	w.payload = payload
	return w.publishLocked()
}

func (w *RedisWriter) publishLocked() error {
	if len(w.payload) == 0 {
		return nil
	}
	ctx, cancel := context.WithTimeout(w.ctx, redisOperationTimeout)
	defer cancel()
	return w.client.Set(ctx, w.key, w.payload, w.ttl).Err()
}

func (w *RedisWriter) heartbeat() {
	defer close(w.done)
	ticker := time.NewTicker(w.ttl / 3)
	defer ticker.Stop()
	for {
		select {
		case <-w.ctx.Done():
			return
		case <-ticker.C:
			w.mu.Lock()
			_ = w.publishLocked() // A failed refresh expires externally; the next tick retries.
			w.mu.Unlock()
		}
	}
}

// Close stops heartbeats. The record expires naturally within its remaining TTL.
func (w *RedisWriter) Close() error {
	w.closeOnce.Do(func() {
		w.cancel()
		<-w.done
		w.mu.Lock()
		defer w.mu.Unlock()
		w.closeErr = w.client.Close()
	})
	return w.closeErr
}

// LoadFromRedis reads only the requested live process incarnation. It never
// falls back to the old shared key or accepts a persistent, non-expiring record.
func LoadFromRedis(client *redis.Client, instanceID string) (*State, error) {
	key, err := redisStatusKey(instanceID)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), redisOperationTimeout)
	defer cancel()
	var data *redis.StringCmd
	var ttl *redis.DurationCmd
	_, err = client.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
		data = pipe.Get(ctx, key)
		ttl = pipe.PTTL(ctx, key)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if ttl.Val() <= 0 {
		return nil, redis.Nil
	}
	var state State
	if err := json.Unmarshal([]byte(data.Val()), &state); err != nil {
		return nil, fmt.Errorf("decode startup status from redis: %w", err)
	}
	if state.InstanceID != instanceID {
		return nil, fmt.Errorf("startup status instance identity mismatch")
	}
	return &state, nil
}
