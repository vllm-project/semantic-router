package cache

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// fakeRedisCacheClient embeds the interface so only the commands exercised by
// addEntry need real behavior; everything else panics if reached.
type fakeRedisCacheClient struct {
	redisCacheClient
	expireErr error
}

func (f *fakeRedisCacheClient) HSet(ctx context.Context, key string, values ...interface{}) *redis.IntCmd {
	cmd := redis.NewIntCmd(ctx)
	cmd.SetVal(1)
	return cmd
}

func (f *fakeRedisCacheClient) Expire(ctx context.Context, key string, ttl time.Duration) *redis.BoolCmd {
	cmd := redis.NewBoolCmd(ctx)
	if f.expireErr != nil {
		cmd.SetErr(f.expireErr)
	} else {
		cmd.SetVal(true)
	}
	return cmd
}

type fakeExpireEmbeddingProvider struct {
	embedding.Provider
}

func (f fakeExpireEmbeddingProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}

// captureLogging swaps the global zap logger for one writing to a buffer and
// restores it via t.Cleanup.
func captureLogging(t *testing.T, run func()) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	core := zapcore.NewCore(
		zapcore.NewConsoleEncoder(zap.NewDevelopmentEncoderConfig()),
		zapcore.AddSync(&buf),
		zap.DebugLevel,
	)
	restore := zap.ReplaceGlobals(zap.New(core))
	t.Cleanup(restore)
	run()
	return &buf
}

func newExpireTestCache(client redisCacheClient) *RedisCache {
	cfg := &config.RedisConfig{}
	cfg.Index.Prefix = "test-cache:"
	cfg.Index.VectorField.Name = "embedding"
	return &RedisCache{
		embeddingProvider: fakeExpireEmbeddingProvider{},
		client:            client,
		config:            cfg,
		indexName:         "test-index",
		ttlSeconds:        60,
		enabled:           true,
	}
}

func TestAddEntryLogsExpireFailure(t *testing.T) {
	cache := newExpireTestCache(&fakeRedisCacheClient{expireErr: errors.New("boom")})
	var addErr error
	buf := captureLogging(t, func() {
		addErr = cache.addEntry(
			context.Background(), "", "req-1", "model", "query",
			[]byte("request"), []byte("response"), 30,
		)
	})
	if addErr != nil {
		t.Fatalf("addEntry should succeed even when Expire fails: %v", addErr)
	}
	if !strings.Contains(buf.String(), "failed to set TTL") {
		t.Fatalf("expected a warn log for the failed Expire, got:\n%s", buf.String())
	}
}

func TestAddEntryQuietWhenExpireSucceeds(t *testing.T) {
	cache := newExpireTestCache(&fakeRedisCacheClient{})
	var addErr error
	buf := captureLogging(t, func() {
		addErr = cache.addEntry(
			context.Background(), "", "req-1", "model", "query",
			[]byte("request"), []byte("response"), 30,
		)
	})
	if addErr != nil {
		t.Fatalf("addEntry failed: %v", addErr)
	}
	if strings.Contains(buf.String(), "failed to set TTL") {
		t.Fatalf("unexpected TTL failure log:\n%s", buf.String())
	}
}
