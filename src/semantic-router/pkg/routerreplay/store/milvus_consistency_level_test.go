package store

import (
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// The name→level mapping itself is covered in pkg/milvus; these tests pin the
// store's own policies layered on top of it.

func TestResolveMilvusConsistencyLevel(t *testing.T) {
	cases := []struct {
		name string
		want entity.ConsistencyLevel
	}{
		{"Strong", entity.ClStrong},                       // valid names pass through the shared parser
		{DefaultMilvusConsistencyLevel, entity.ClSession}, // the documented default maps to Session
		{"", entity.ClSession},                            // unset -> default
		{"banana", entity.ClSession},                      // unrecognized -> warn + default
	}

	for _, tc := range cases {
		if got := resolveMilvusConsistencyLevel(tc.name); got != tc.want {
			t.Errorf("resolveMilvusConsistencyLevel(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestMilvusStoreReadLevels(t *testing.T) {
	cases := []struct {
		configured entity.ConsistencyLevel
		wrote      bool
		wantRead   entity.ConsistencyLevel
		wantUpdate entity.ConsistencyLevel
	}{
		// Session floors to Bounded until the first write (the SDK would
		// degrade it to Eventually); update-chain reads floor at Session once
		// this process has written, even under weaker configured levels; an
		// explicit Strong applies everywhere unchanged.
		{entity.ClSession, false, entity.ClBounded, entity.ClBounded},
		{entity.ClSession, true, entity.ClSession, entity.ClSession},
		{entity.ClStrong, false, entity.ClStrong, entity.ClStrong},
		{entity.ClStrong, true, entity.ClStrong, entity.ClStrong},
		{entity.ClBounded, false, entity.ClBounded, entity.ClBounded},
		{entity.ClBounded, true, entity.ClBounded, entity.ClSession},
		{entity.ClEventually, false, entity.ClEventually, entity.ClBounded},
		{entity.ClEventually, true, entity.ClEventually, entity.ClSession},
	}

	for _, tc := range cases {
		m := &MilvusStore{consistencyLevel: tc.configured}
		m.wrote.Store(tc.wrote)
		if got := m.readLevel(); got != tc.wantRead {
			t.Errorf("readLevel(configured=%v, wrote=%v) = %v, want %v", tc.configured, tc.wrote, got, tc.wantRead)
		}
		if got := m.updateReadLevel(); got != tc.wantUpdate {
			t.Errorf("updateReadLevel(configured=%v, wrote=%v) = %v, want %v", tc.configured, tc.wrote, got, tc.wantUpdate)
		}
	}
}
