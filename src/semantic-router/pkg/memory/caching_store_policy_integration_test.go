package memory

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

type retrievalPolicyStoreDelegate interface{ Store }
type retrievalPolicyStore struct {
	retrievalPolicyStoreDelegate
	calls int
}

func (s *retrievalPolicyStore) Retrieve(_ context.Context, o RetrieveOptions) ([]*RetrieveResult, error) {
	s.calls++
	return []*RetrieveResult{{Memory: &Memory{ID: fmt.Sprintf("hybrid=%t,mode=%s,adaptive=%t", o.HybridSearch, o.HybridMode, o.AdaptiveThreshold), UserID: o.UserID}, Score: 1}}, nil
}

// StorageIntegration: redis
func TestCachingStoreRetrievalPolicyIsolation(t *testing.T) {
	storagetest.Require(t, "redis")
	ctx := context.Background()
	base := RetrieveOptions{Query: "coffee", UserID: "release-test-user", Limit: 5, Threshold: 0.5}
	for _, tc := range []struct {
		name string
		opts RetrieveOptions
	}{
		{"hybrid", func() RetrieveOptions { o := base; o.HybridSearch = true; return o }()},
		{"hybrid_mode", func() RetrieveOptions { o := base; o.HybridSearch = true; o.HybridMode = "rrf"; return o }()},
		{"adaptive", func() RetrieveOptions { o := base; o.AdaptiveThreshold = true; return o }()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			baseline := base
			if tc.name == "hybrid_mode" {
				baseline.HybridSearch = true
				baseline.HybridMode = "weighted"
			}
			cache, err := NewRedisCache(ctx, &RedisCacheConfig{Address: storageRedisAddress(), KeyPrefix: "retrieval-policy:" + t.Name(), TTLSeconds: 60})
			if err != nil {
				t.Fatalf("required isolated Redis: %v", err)
			}
			defer cache.Close()
			cache.InvalidateByUser(ctx, base.UserID)
			store := &retrievalPolicyStore{}
			wrapped := NewCachingStore(store, cache, "valkey")
			first, err := wrapped.Retrieve(ctx, baseline)
			if err != nil {
				t.Fatal(err)
			}
			control, err := wrapped.Retrieve(ctx, baseline)
			if err != nil {
				t.Fatal(err)
			}
			if store.calls != 1 || control[0].Memory.ID != first[0].Memory.ID {
				t.Fatal("same-policy control failed")
			}
			second, err := wrapped.Retrieve(ctx, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("first=%s second=%s backend_calls=%d first_key=%s second_key=%s", first[0].Memory.ID, second[0].Memory.ID, store.calls, cacheKey(cache.prefix, baseline.UserID, baseline), cacheKey(cache.prefix, base.UserID, tc.opts))
			if store.calls != 2 || second[0].Memory.ID == first[0].Memory.ID {
				t.Errorf("different policy reused cached result; calls=%d", store.calls)
			}
		})
	}
}
