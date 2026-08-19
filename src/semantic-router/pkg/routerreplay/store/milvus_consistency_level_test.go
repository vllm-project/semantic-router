package store

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func TestResolveMilvusConsistencyLevel(t *testing.T) {
	cases := []struct {
		name string
		want entity.ConsistencyLevel
	}{
		{"Strong", entity.ClStrong},
		{"Session", entity.ClSession},
		{"Bounded", entity.ClBounded},
		{"Eventually", entity.ClEventually},
		{"strong", entity.ClStrong},
		{"  session  ", entity.ClSession},
		{"EVENTUALLY", entity.ClEventually},
		{"", entity.ClSession},
		{"   ", entity.ClSession},
		{"banana", entity.ClSession},
	}

	for _, tc := range cases {
		if got := resolveMilvusConsistencyLevel(tc.name); got != tc.want {
			t.Fatalf("resolveMilvusConsistencyLevel(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}

	if got := resolveMilvusConsistencyLevel(DefaultMilvusConsistencyLevel); got != entity.ClSession {
		t.Fatalf("DefaultMilvusConsistencyLevel resolves to %v, want %v", got, entity.ClSession)
	}
}

func TestMilvusStoreReadLevels(t *testing.T) {
	cases := []struct {
		configured entity.ConsistencyLevel
		wrote      bool
		wantRead   entity.ConsistencyLevel
		wantUpdate entity.ConsistencyLevel
	}{
		// Session floors to Bounded until the first write: the SDK degrades
		// Session to Eventually when no session timestamp exists.
		{entity.ClSession, false, entity.ClBounded, entity.ClBounded},
		{entity.ClSession, true, entity.ClSession, entity.ClSession},
		// Strong applies everywhere unchanged.
		{entity.ClStrong, false, entity.ClStrong, entity.ClStrong},
		{entity.ClStrong, true, entity.ClStrong, entity.ClStrong},
		// Weaker configured levels apply to plain reads, but update-chain
		// reads are floored at Session once this process has written.
		{entity.ClBounded, false, entity.ClBounded, entity.ClBounded},
		{entity.ClBounded, true, entity.ClBounded, entity.ClSession},
		{entity.ClEventually, false, entity.ClEventually, entity.ClBounded},
		{entity.ClEventually, true, entity.ClEventually, entity.ClSession},
	}

	for _, tc := range cases {
		m := &MilvusStore{consistencyLevel: tc.configured}
		m.wrote.Store(tc.wrote)
		if got := m.readLevel(); got != tc.wantRead {
			t.Fatalf("readLevel(configured=%v, wrote=%v) = %v, want %v", tc.configured, tc.wrote, got, tc.wantRead)
		}
		if got := m.updateReadLevel(); got != tc.wantUpdate {
			t.Fatalf("updateReadLevel(configured=%v, wrote=%v) = %v, want %v", tc.configured, tc.wrote, got, tc.wantUpdate)
		}
	}
}
