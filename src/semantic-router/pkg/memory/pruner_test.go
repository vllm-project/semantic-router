//go:build !windows && cgo

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func setupTestStoreWithConfig(cfg config.MemoryConfig) (*MilvusStore, *MockMilvusClient) {
	mockClient := &MockMilvusClient{}
	testEmbeddingConfig := EmbeddingConfig{
		Model: EmbeddingModelBERT,
	}
	// Ensure EmbeddingModel is set so NewMilvusStore doesn't replace cfg with defaults
	if cfg.EmbeddingModel == "" {
		cfg.EmbeddingModel = "bert"
	}
	options := MilvusStoreOptions{
		Client:          mockClient,
		CollectionName:  "test_memories",
		Config:          cfg,
		Enabled:         true,
		EmbeddingConfig: &testEmbeddingConfig,
	}
	store, _ := NewMilvusStore(options)
	return store, mockClient
}

func TestPruneIfOverCap_TriggersWhenOverCap(t *testing.T) {
	cfg := DefaultMemoryConfig()
	cfg.QualityScoring.MaxMemoriesPerUser = 3
	cfg.QualityScoring.PruneThreshold = 0.1
	cfg.QualityScoring.InitialStrengthDays = 30

	store, mockClient := setupTestStoreWithConfig(cfg)

	queryCall := 0
	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		queryCall++
		if queryCall == 1 {
			// countUserMemories call: return 5 IDs (over cap of 3)
			return []entity.Column{
				entity.NewColumnVarChar("id", []string{"m1", "m2", "m3", "m4", "m5"}),
			}, nil
		}
		// ListForPrune call: return entries with metadata
		now := time.Now()
		old := now.Add(-90 * 24 * time.Hour)
		meta1, _ := json.Marshal(map[string]interface{}{"last_accessed": old.Unix(), "access_count": 0})
		meta2, _ := json.Marshal(map[string]interface{}{"last_accessed": old.Unix(), "access_count": 0})
		meta3, _ := json.Marshal(map[string]interface{}{"last_accessed": now.Unix(), "access_count": 5})
		meta4, _ := json.Marshal(map[string]interface{}{"last_accessed": now.Unix(), "access_count": 10})
		meta5, _ := json.Marshal(map[string]interface{}{"last_accessed": now.Unix(), "access_count": 3})

		return []entity.Column{
			entity.NewColumnVarChar("id", []string{"m1", "m2", "m3", "m4", "m5"}),
			entity.NewColumnVarChar("metadata", []string{string(meta1), string(meta2), string(meta3), string(meta4), string(meta5)}),
			entity.NewColumnInt64("created_at", []int64{old.Unix(), old.Unix(), now.Unix(), now.Unix(), now.Unix()}),
		}, nil
	}

	store.pruneIfOverCap(context.Background(), "user1")

	// PruneUser should have been called (via ListForPrune query + delete calls)
	assert.GreaterOrEqual(t, mockClient.QueryCallCount, 2, "should have queried for count and then for prune entries")
	assert.Positive(t, mockClient.DeleteCallCount, "should have deleted some memories")
}

func TestPruneIfOverCap_SkipsWhenUnderCap(t *testing.T) {
	cfg := DefaultMemoryConfig()
	cfg.QualityScoring.MaxMemoriesPerUser = 10

	store, mockClient := setupTestStoreWithConfig(cfg)

	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		return []entity.Column{
			entity.NewColumnVarChar("id", []string{"m1", "m2", "m3"}),
		}, nil
	}

	store.pruneIfOverCap(context.Background(), "user1")

	assert.Equal(t, 1, mockClient.QueryCallCount, "should only query for count, not prune")
	assert.Equal(t, 0, mockClient.DeleteCallCount, "should not delete anything")
}

func TestPruneIfOverCap_SkipsWhenCapIsZero(t *testing.T) {
	cfg := DefaultMemoryConfig()
	cfg.QualityScoring.MaxMemoriesPerUser = 0

	store, mockClient := setupTestStoreWithConfig(cfg)

	store.pruneIfOverCap(context.Background(), "user1")

	assert.Equal(t, 0, mockClient.QueryCallCount, "should not query at all when cap is 0")
}

func TestListStaleUserIDs_ReturnsUniqueUsers(t *testing.T) {
	store, mockClient := setupTestStore()

	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		assert.Contains(t, expr, "created_at <", "should filter by created_at")
		assert.Equal(t, []string{"user_id"}, out, "should request user_id field")

		return []entity.Column{
			entity.NewColumnVarChar("user_id", []string{"alice", "bob", "alice", "charlie", "bob", "alice"}),
		}, nil
	}

	cutoff := time.Now().Add(-30 * 24 * time.Hour).Unix()
	userIDs, err := store.ListStaleUserIDs(context.Background(), cutoff)
	require.NoError(t, err)

	assert.Len(t, userIDs, 3, "should deduplicate to 3 unique users")
	seen := make(map[string]bool)
	for _, uid := range userIDs {
		seen[uid] = true
	}
	assert.True(t, seen["alice"])
	assert.True(t, seen["bob"])
	assert.True(t, seen["charlie"])
}

func TestListStaleUserIDs_ReturnsEmptyWhenNoResults(t *testing.T) {
	store, mockClient := setupTestStore()

	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		return []entity.Column{}, nil
	}

	userIDs, err := store.ListStaleUserIDs(context.Background(), time.Now().Unix())
	require.NoError(t, err)
	assert.Empty(t, userIDs)
}

func TestListStaleUserIDs_ReturnsErrorOnQueryFailure(t *testing.T) {
	store, mockClient := setupTestStore()

	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		return nil, fmt.Errorf("connection refused")
	}

	userIDs, err := store.ListStaleUserIDs(context.Background(), time.Now().Unix())
	assert.Error(t, err)
	assert.Nil(t, userIDs)
}

func TestCountUserMemories(t *testing.T) {
	store, mockClient := setupTestStore()

	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		assert.Contains(t, expr, `user_id == "user42"`)
		assert.Equal(t, []string{"id"}, out)
		return []entity.Column{
			entity.NewColumnVarChar("id", []string{"m1", "m2", "m3", "m4", "m5"}),
		}, nil
	}

	count, err := store.countUserMemories(context.Background(), "user42")
	require.NoError(t, err)
	assert.Equal(t, 5, count)
}

func TestStartPruneSweep_StartsAndStops(t *testing.T) {
	cfg := config.MemoryQualityScoringConfig{
		PruneInterval:       "100ms",
		PruneBatchSize:      10,
		InitialStrengthDays: 30,
		PruneThreshold:      0.1,
	}

	store, mockClient := setupTestStore()

	var sweepCount atomic.Int32
	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		sweepCount.Add(1)
		return []entity.Column{}, nil
	}

	stop := StartPruneSweep(cfg, store)
	require.NotNil(t, stop)

	// Wait for at least one tick
	time.Sleep(350 * time.Millisecond)

	stop()

	count := sweepCount.Load()
	assert.GreaterOrEqual(t, count, int32(1), "sweep should have run at least once")
}

func TestStartPruneSweep_DisabledWithEmptyInterval(t *testing.T) {
	cfg := config.MemoryQualityScoringConfig{
		PruneInterval: "",
	}

	store, _ := setupTestStore()

	stop := StartPruneSweep(cfg, store)
	require.NotNil(t, stop, "should return a no-op stop function")

	// Should not panic
	stop()
}

func TestStartPruneSweep_DisabledWithInvalidInterval(t *testing.T) {
	cfg := config.MemoryQualityScoringConfig{
		PruneInterval: "invalid",
	}

	store, _ := setupTestStore()

	stop := StartPruneSweep(cfg, store)
	require.NotNil(t, stop, "should return a no-op stop function")

	stop()
}
