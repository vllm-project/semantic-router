package contextcompression

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

// StorageIntegration: redis
func TestRedisRecoveryStoreRoundTripAndScopeIsolation(t *testing.T) {
	storagetest.Require(t, "redis")
	store, ctx := integrationRecoveryStore(t)
	defer store.Close()
	entry := RecoveryEntry{
		Key:       "key",
		Scope:     "scope-a",
		Content:   "original context",
		CreatedAt: time.Now().UTC(),
		ExpiresAt: time.Now().UTC().Add(time.Minute),
	}
	if err := store.Put(ctx, entry, time.Minute); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	got, err := store.Get(ctx, entry.Scope, entry.Key)
	if err != nil || got.Content != entry.Content {
		t.Fatalf("Get() = %#v, %v", got, err)
	}
	if _, scopeErr := store.Get(ctx, "scope-b", entry.Key); scopeErr == nil {
		t.Fatal("Get() crossed recovery scope")
	}
	deleted, err := store.InvalidateScope(ctx, entry.Scope)
	if err != nil || deleted != 1 {
		t.Fatalf("InvalidateScope() = %d, %v", deleted, err)
	}
	if _, postInvalidateErr := store.Get(
		ctx,
		entry.Scope,
		entry.Key,
	); postInvalidateErr == nil {
		t.Fatal("InvalidateScope() retained recovery content")
	}
}

func integrationRecoveryStore(
	t *testing.T,
) (*RedisRecoveryStore, context.Context) {
	t.Helper()
	host := os.Getenv("REDIS_HOST")
	if host == "" {
		host = "127.0.0.1"
	}
	port := 6379
	if raw := os.Getenv("REDIS_PORT"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil {
			port = parsed
		}
	}
	store, storeErr := NewRedisRecoveryStore(RedisRecoveryStoreOptions{
		Address:       fmt.Sprintf("%s:%d", host, port),
		KeyPrefix:     fmt.Sprintf("vsr:test:context-recovery:%d:", time.Now().UnixNano()),
		MaxTotalBytes: 4096,
	})
	if storeErr != nil {
		t.Fatalf("NewRedisRecoveryStore() error = %v", storeErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)
	if err := store.Health(ctx); err != nil {
		_ = store.Close()
		storagetest.Unavailable(t, "redis", fmt.Sprintf("Redis unavailable: %v", err))
	}
	return store, ctx
}
