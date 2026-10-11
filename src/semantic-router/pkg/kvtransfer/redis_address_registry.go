package kvtransfer

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RedisAddressRegistryOptions configures a Redis/Valkey-backed address registry.
type RedisAddressRegistryOptions struct {
	Address  string
	Password string
	Database int
	Timeout  time.Duration
	TTL      time.Duration
}

type redisAddressRegistry struct {
	client  *redis.Client
	timeout time.Duration
	ttl     time.Duration
}

// NewRedisAddressRegistry stores kv_addr entries in Redis or Valkey.
func NewRedisAddressRegistry(options RedisAddressRegistryOptions) (AddressRegistry, error) {
	if strings.TrimSpace(options.Address) == "" {
		return nil, fmt.Errorf("redis address registry address is required")
	}
	timeout := options.Timeout
	if timeout <= 0 {
		timeout = 50 * time.Millisecond
	}
	ttl := options.TTL
	if ttl <= 0 {
		ttl = defaultAddressRegistryTTL
	}
	client := redis.NewClient(&redis.Options{
		Addr:         options.Address,
		Password:     options.Password,
		DB:           options.Database,
		DialTimeout:  timeout,
		ReadTimeout:  timeout,
		WriteTimeout: timeout,
	})
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("redis address registry ping failed: %w", err)
	}
	return &redisAddressRegistry{
		client:  client,
		timeout: timeout,
		ttl:     ttl,
	}, nil
}

// NewAddressRegistryFromResponseCacheStore reuses the semantic response-cache
// Redis/Valkey connection settings when available.
func NewAddressRegistryFromResponseCacheStore(
	store config.ResponseCacheStoreConfig,
) (AddressRegistry, error) {
	switch strings.ToLower(strings.TrimSpace(store.BackendType)) {
	case "", "redis":
		if store.Redis == nil {
			return NoopAddressRegistry{}, nil
		}
		return NewRedisAddressRegistry(redisOptionsFromConfig(store.Redis))
	case "valkey":
		if store.Valkey == nil {
			return NoopAddressRegistry{}, nil
		}
		return NewRedisAddressRegistry(valkeyOptionsFromConfig(store.Valkey))
	default:
		return NoopAddressRegistry{}, nil
	}
}

func redisOptionsFromConfig(cfg *config.RedisConfig) RedisAddressRegistryOptions {
	if cfg == nil {
		return RedisAddressRegistryOptions{}
	}
	timeout := time.Duration(cfg.Connection.Timeout) * time.Millisecond
	if timeout <= 0 {
		timeout = 50 * time.Millisecond
	}
	return RedisAddressRegistryOptions{
		Address:  fmt.Sprintf("%s:%d", cfg.Connection.Host, cfg.Connection.Port),
		Password: cfg.Connection.Password,
		Database: cfg.Connection.Database,
		Timeout:  timeout,
	}
}

func valkeyOptionsFromConfig(cfg *config.ValkeyConfig) RedisAddressRegistryOptions {
	if cfg == nil {
		return RedisAddressRegistryOptions{}
	}
	timeout := time.Duration(cfg.Connection.Timeout) * time.Millisecond
	if timeout <= 0 {
		timeout = 50 * time.Millisecond
	}
	return RedisAddressRegistryOptions{
		Address:  fmt.Sprintf("%s:%d", cfg.Connection.Host, cfg.Connection.Port),
		Password: cfg.Connection.Password,
		Database: cfg.Connection.Database,
		Timeout:  timeout,
	}
}

func (r *redisAddressRegistry) Write(ctx context.Context, record AddressRecord) error {
	if r == nil || r.client == nil {
		return nil
	}
	if err := validateAddressRecord(record); err != nil {
		return err
	}
	payload, err := encodeAddressRecord(record)
	if err != nil {
		return err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	writeCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	key := AddressKey(record.Namespace, record.SessionID)
	if err := r.client.Set(writeCtx, key, payload, r.ttl).Err(); err != nil {
		return fmt.Errorf("redis kv address write failed: %w", err)
	}
	return nil
}

func (r *redisAddressRegistry) Lookup(ctx context.Context, namespace, sessionID string) (*AddressRecord, error) {
	if r == nil || r.client == nil {
		return nil, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	readCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	payload, err := r.client.Get(readCtx, AddressKey(namespace, sessionID)).Bytes()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("redis kv address lookup failed: %w", err)
	}
	record, err := decodeAddressRecord(payload)
	if err != nil {
		return nil, err
	}
	if record.Namespace != namespace {
		return nil, nil
	}
	copy := record
	return &copy, nil
}

func (r *redisAddressRegistry) Close() error {
	if r == nil || r.client == nil {
		return nil
	}
	return r.client.Close()
}
