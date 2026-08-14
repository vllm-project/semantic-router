package store

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestResolveMilvusConsistencyLevel(t *testing.T) {
	cases := []struct {
		name  string
		input string
		level entity.ConsistencyLevel
		ok    bool
	}{
		{"unset resolves to store default", "", entity.ClSession, true},
		{"strong", "Strong", entity.ClStrong, true},
		{"session lowercase", "session", entity.ClSession, true},
		{"bounded uppercase", "BOUNDED", entity.ClBounded, true},
		{"eventually padded", " Eventually ", entity.ClEventually, true},
		{"unknown falls back to default", "customized", entity.ClSession, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			level, ok := resolveMilvusConsistencyLevel(tc.input)
			if level != tc.level || ok != tc.ok {
				t.Errorf("resolveMilvusConsistencyLevel(%q) = (%v, %v), want (%v, %v)",
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

func TestMilvusStoreQueryOptions(t *testing.T) {
	cases := []struct {
		name  string
		level entity.ConsistencyLevel
	}{
		{"store default", entity.ClSession},
		{"strong", entity.ClStrong},
		{"bounded", entity.ClBounded},
		{"eventually", entity.ClEventually},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := &MilvusStore{consistencyLevel: tc.level}
			if got := appliedConsistencyLevel(t, store.queryOptions()); got != tc.level {
				t.Errorf("queryOptions() applied level = %v, want %v", got, tc.level)
			}
		})
	}
}
