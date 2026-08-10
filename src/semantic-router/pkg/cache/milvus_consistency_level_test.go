package cache

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMilvusConsistencyLevelMapping(t *testing.T) {
	cases := []struct {
		name  string
		input string
		level entity.ConsistencyLevel
		ok    bool
	}{
		{"strong", "Strong", entity.ClStrong, true},
		{"session lowercase", "session", entity.ClSession, true},
		{"bounded uppercase", "BOUNDED", entity.ClBounded, true},
		{"eventually padded", " Eventually ", entity.ClEventually, true},
		{"empty keeps default", "", entity.DefaultConsistencyLevel, false},
		{"unknown keeps default", "customized", entity.DefaultConsistencyLevel, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			level, ok := milvusConsistencyLevel(tc.input)
			if level != tc.level || ok != tc.ok {
				t.Errorf("milvusConsistencyLevel(%q) = (%v, %v), want (%v, %v)",
					tc.input, level, ok, tc.level, tc.ok)
			}
		})
	}
}

// appliedConsistencyLevel replays the options against a sentinel value that
// differs from every mapped level, so "option did nothing" (ClStrong is the
// zero value) cannot masquerade as an explicit Strong.
func appliedConsistencyLevel(t *testing.T, opts []client.SearchQueryOptionFunc) entity.ConsistencyLevel {
	t.Helper()
	applied := &client.SearchQueryOption{ConsistencyLevel: entity.ClCustomized}
	for _, opt := range opts {
		opt(applied)
	}
	return applied.ConsistencyLevel
}

func TestMilvusSearchQueryOptions(t *testing.T) {
	if opts := (&MilvusCache{}).searchQueryOptions(); opts != nil {
		t.Errorf("expected no options without config, got %d", len(opts))
	}

	cfg := &config.MilvusConfig{}
	cache := &MilvusCache{config: cfg}
	if opts := cache.searchQueryOptions(); opts != nil {
		t.Errorf("expected no options for unset consistency level, got %d", len(opts))
	}

	cfg.Search.ConsistencyLevel = "Strong"
	if got := appliedConsistencyLevel(t, cache.searchQueryOptions()); got != entity.ClStrong {
		t.Errorf("searchQueryOptions applied level = %v, want ClStrong", got)
	}

	cfg.Search.ConsistencyLevel = "Eventually"
	if got := appliedConsistencyLevel(t, cache.searchQueryOptions()); got != entity.ClEventually {
		t.Errorf("searchQueryOptions applied level = %v, want ClEventually", got)
	}
}

func TestMilvusInternalReadOptions(t *testing.T) {
	cfg := &config.MilvusConfig{}
	cache := &MilvusCache{config: cfg}

	// The pending lookup reads this process's own write: floored at Session
	// even when the configured level is weaker or unset.
	for _, lvl := range []string{"", "Bounded", "Eventually", "Session"} {
		cfg.Search.ConsistencyLevel = lvl
		if got := appliedConsistencyLevel(t, cache.pendingLookupOptions()); got != entity.ClSession {
			t.Errorf("pendingLookupOptions(%q) applied level = %v, want ClSession", lvl, got)
		}
	}
	cfg.Search.ConsistencyLevel = "Strong"
	if got := appliedConsistencyLevel(t, cache.pendingLookupOptions()); got != entity.ClStrong {
		t.Errorf("pendingLookupOptions(Strong) applied level = %v, want ClStrong", got)
	}

	// The startup rebuild runs before this process's first write, where
	// Session degrades to Eventually: floored at Bounded, other levels pass
	// through unchanged.
	cfg.Search.ConsistencyLevel = "Session"
	if got := appliedConsistencyLevel(t, cache.rebuildQueryOptions()); got != entity.ClBounded {
		t.Errorf("rebuildQueryOptions(Session) applied level = %v, want ClBounded", got)
	}
	cfg.Search.ConsistencyLevel = "Eventually"
	if got := appliedConsistencyLevel(t, cache.rebuildQueryOptions()); got != entity.ClEventually {
		t.Errorf("rebuildQueryOptions(Eventually) applied level = %v, want ClEventually", got)
	}
	cfg.Search.ConsistencyLevel = "Strong"
	if got := appliedConsistencyLevel(t, cache.rebuildQueryOptions()); got != entity.ClStrong {
		t.Errorf("rebuildQueryOptions(Strong) applied level = %v, want ClStrong", got)
	}
}
