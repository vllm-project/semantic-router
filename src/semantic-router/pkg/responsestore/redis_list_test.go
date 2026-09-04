package responsestore

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// TestRedisConversationIndexListing covers the index-backed read path: what
// it returns, in what order, and how it converges when index and payloads
// disagree. Subtests that care about relative order pass Order: "asc"
// explicitly — the default order is "desc" (newest first, blueprint §3.6,
// matching the ListOptions.Order contract in interface.go), which
// TestRedisListDefaultOrderIsDescending covers directly.
func TestRedisConversationIndexListing(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	now := time.Now().Unix()
	storeResponse := func(t *testing.T, id, convID string, createdAt int64) {
		t.Helper()
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID:             id,
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      createdAt,
			Output:         []responseapi.OutputItem{{ID: "item_" + id}},
		}))
	}

	t.Run("returns only the requested conversation, oldest first", func(t *testing.T) {
		storeResponse(t, "resp_idx_a1", "conv_idx_a", now)
		storeResponse(t, "resp_idx_a2", "conv_idx_a", now+1)
		storeResponse(t, "resp_idx_b1", "conv_idx_b", now)
		// A response with no conversation must not land in any index.
		storeResponse(t, "resp_idx_orphan", "", now)

		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_a", ListOptions{Order: "asc"})
		require.NoError(t, err)
		require.Len(t, responses, 2)
		// The index is scored by created_at, so asc reads back chronologically.
		assert.Equal(t, "resp_idx_a1", responses[0].ID)
		assert.Equal(t, "resp_idx_a2", responses[1].ID)

		assert.Equal(t, []string{"resp_idx_a1", "resp_idx_a2"}, conversationIndexMembers(t, store, "conv_idx_a"))
		assert.Equal(t, []string{"resp_idx_b1"}, conversationIndexMembers(t, store, "conv_idx_b"))
	})

	t.Run("unknown conversation returns nothing", func(t *testing.T) {
		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_missing", ListOptions{})
		assert.NoError(t, err)
		assert.Empty(t, responses)
	})

	t.Run("empty conversation ID is rejected", func(t *testing.T) {
		_, err := store.ListResponsesByConversation(ctx, "", ListOptions{})
		assert.ErrorIs(t, err, ErrInvalidInput)
	})

	t.Run("deleting a response drops its index entry", func(t *testing.T) {
		require.NoError(t, store.DeleteResponse(ctx, "resp_idx_a1"))

		assert.Equal(t, []string{"resp_idx_a2"}, conversationIndexMembers(t, store, "conv_idx_a"))

		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_a", ListOptions{Order: "asc"})
		require.NoError(t, err)
		require.Len(t, responses, 1)
		assert.Equal(t, "resp_idx_a2", responses[0].ID)
	})

	t.Run("moving a response between conversations reindexes it", func(t *testing.T) {
		moved, err := store.GetResponse(ctx, "resp_idx_a2")
		require.NoError(t, err)

		moved.ConversationID = "conv_idx_b"
		require.NoError(t, store.UpdateResponse(ctx, moved))

		fromOld, err := store.ListResponsesByConversation(ctx, "conv_idx_a", ListOptions{Order: "asc"})
		require.NoError(t, err)
		assert.Empty(t, fromOld)
		assert.Empty(t, conversationIndexMembers(t, store, "conv_idx_a"))

		toNew, err := store.ListResponsesByConversation(ctx, "conv_idx_b", ListOptions{Order: "asc"})
		require.NoError(t, err)
		require.Len(t, toNew, 2)
		assert.Equal(t, []string{"resp_idx_b1", "resp_idx_a2"}, []string{toNew[0].ID, toNew[1].ID})
	})

	t.Run("listing prunes entries whose payload is gone", func(t *testing.T) {
		// Drop the payload behind the store's back, the way a TTL expiry would.
		require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+"resp_idx_b1")).Err())

		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_b", ListOptions{Order: "asc"})
		require.NoError(t, err)
		require.Len(t, responses, 1)
		assert.Equal(t, "resp_idx_a2", responses[0].ID)

		assert.Equal(t, []string{"resp_idx_a2"}, conversationIndexMembers(t, store, "conv_idx_b"))
	})
}

// TestRedisListResponsesByConversationLazyBackfill covers the upgrade path
// (blueprint §6.2): responses written before the index existed — simulated
// with directSetResponsePayload, which bypasses StoreResponse's indexing —
// must still be discoverable on first read, get backfilled into the index,
// and must not trigger a second O(N) scan on the next read of the same
// conversation.
func TestRedisListResponsesByConversationLazyBackfill(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	now := time.Now().Unix()
	legacy := []*responseapi.StoredResponse{
		{ID: "resp_legacy_1", ConversationID: "conv_legacy", Status: "completed", CreatedAt: now},
		{ID: "resp_legacy_2", ConversationID: "conv_legacy", Status: "completed", CreatedAt: now + 1},
		{ID: "resp_legacy_3", ConversationID: "conv_legacy", Status: "completed", CreatedAt: now + 2},
	}
	for _, response := range legacy {
		directSetResponsePayload(t, store, response)
	}
	require.Empty(t, conversationIndexMembers(t, store, "conv_legacy"),
		"precondition: no index should exist yet for legacy data")

	responses, err := store.ListResponsesByConversation(ctx, "conv_legacy", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Equal(t, []string{"resp_legacy_1", "resp_legacy_2", "resp_legacy_3"},
		[]string{responses[0].ID, responses[1].ID, responses[2].ID})
	assert.Equal(t, 1, store.scanInvocations, "first read must run exactly one legacy scan")

	assert.ElementsMatch(t, []string{"resp_legacy_1", "resp_legacy_2", "resp_legacy_3"},
		conversationIndexMembers(t, store, "conv_legacy"))
	assert.Zero(t, exists(t, store, store.emptyConversationIndexMarkerKey("conv_legacy")))

	// The index now exists, so a second read must go straight through it
	// without scanning the keyspace again.
	responses, err = store.ListResponsesByConversation(ctx, "conv_legacy", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Equal(t, 1, store.scanInvocations, "second read of an already-backfilled conversation must not scan again")
}

// TestRedisListResponsesByConversationEmptyMarker covers blueprint §2.4/§3.4:
// a legitimately empty or unknown conversation must not force a full
// keyspace scan on every read — only once, after which the empty marker
// short-circuits subsequent reads.
func TestRedisListResponsesByConversationEmptyMarker(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	// Seed unrelated data so the scan has something to walk past.
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_unrelated", ConversationID: "conv_other", Status: "completed", CreatedAt: time.Now().Unix(),
	})

	responses, err := store.ListResponsesByConversation(ctx, "conv_never_existed", ListOptions{})
	require.NoError(t, err)
	assert.Empty(t, responses)
	assert.Equal(t, 1, store.scanInvocations)

	markerKey := store.emptyConversationIndexMarkerKey("conv_never_existed")
	assert.EqualValues(t, 1, exists(t, store, markerKey))
	ttl, err := store.client.TTL(ctx, markerKey).Result()
	require.NoError(t, err)
	assert.Positive(t, ttl, "empty marker must carry a positive TTL, not be immortal")

	// Repeated reads of the same empty conversation must not scan again.
	for i := 0; i < 3; i++ {
		responses, err = store.ListResponsesByConversation(ctx, "conv_never_existed", ListOptions{})
		require.NoError(t, err)
		assert.Empty(t, responses)
	}
	assert.Equal(t, 1, store.scanInvocations, "repeated reads of an empty conversation must not force repeated scans")
}

// TestRedisLazyBackfillConcurrentWriteNotHidden covers blueprint §2.2: a
// response indexed normally by a concurrent StoreResponse call, landing
// while a lazy legacy scan for the same conversation is in flight, must
// survive — the scan's own findings only ever add to the index, and
// indexResponse always clears the empty marker, so neither ordering can make
// the concurrent write disappear behind a "confirmed empty" marker.
func TestRedisLazyBackfillConcurrentWriteNotHidden(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	legacy := &responseapi.StoredResponse{
		ID: "resp_race_legacy", ConversationID: "conv_race", Status: "completed", CreatedAt: time.Now().Unix(),
	}
	directSetResponsePayload(t, store, legacy)

	concurrent := &responseapi.StoredResponse{
		ID: "resp_race_concurrent", ConversationID: "conv_race", Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}
	store.lazyBackfillPreScanHook = func() {
		// Runs once, before the scan walks the keyspace: lands a normal
		// indexed write for the same conversation the backfill is about to
		// scan for, so the scan observes both the legacy and the
		// concurrently-indexed payload.
		require.NoError(t, store.StoreResponse(ctx, concurrent))
	}

	responses, err := store.ListResponsesByConversation(ctx, "conv_race", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 2)
	assert.Equal(t, []string{"resp_race_legacy", "resp_race_concurrent"},
		[]string{responses[0].ID, responses[1].ID})

	assert.ElementsMatch(t, []string{"resp_race_legacy", "resp_race_concurrent"},
		conversationIndexMembers(t, store, "conv_race"))
	assert.Zero(t, exists(t, store, store.emptyConversationIndexMarkerKey("conv_race")),
		"the concurrently indexed write must clear any empty marker, not be hidden behind one")
}

// TestNormalizeResponseListOptions needs no Redis: pure validation/defaulting.
func TestNormalizeResponseListOptions(t *testing.T) {
	tests := []struct {
		name      string
		opts      ListOptions
		want      normalizedListOptions
		expectErr error
	}{
		{
			name: "zero value defaults to desc and DefaultListLimit",
			opts: ListOptions{},
			want: normalizedListOptions{Limit: DefaultListLimit, Order: "desc"},
		},
		{
			name: "negative limit defaults",
			opts: ListOptions{Limit: -5},
			want: normalizedListOptions{Limit: DefaultListLimit, Order: "desc"},
		},
		{
			name: "limit above MaxListLimit is clamped",
			opts: ListOptions{Limit: MaxListLimit + 500},
			want: normalizedListOptions{Limit: MaxListLimit, Order: "desc"},
		},
		{
			name: "explicit asc is kept",
			opts: ListOptions{Order: "asc"},
			want: normalizedListOptions{Limit: DefaultListLimit, Order: "asc"},
		},
		{
			name: "explicit desc is kept",
			opts: ListOptions{Order: "desc"},
			want: normalizedListOptions{Limit: DefaultListLimit, Order: "desc"},
		},
		{
			name: "after cursor is preserved",
			opts: ListOptions{After: "resp_a"},
			want: normalizedListOptions{Limit: DefaultListLimit, Order: "desc", After: "resp_a"},
		},
		{
			name: "before cursor is preserved",
			opts: ListOptions{Before: "resp_b"},
			want: normalizedListOptions{Limit: DefaultListLimit, Order: "desc", Before: "resp_b"},
		},
		{
			name:      "unrecognized order is rejected",
			opts:      ListOptions{Order: "sideways"},
			expectErr: ErrInvalidInput,
		},
		{
			name:      "after and before together are rejected",
			opts:      ListOptions{After: "resp_a", Before: "resp_b"},
			expectErr: ErrInvalidInput,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeResponseListOptions(tt.opts)
			if tt.expectErr != nil {
				assert.ErrorIs(t, err, tt.expectErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestRedisListDefaultOrderIsDescending is the direct regression test for the
// Phase 4 default-order fix: an unspecified Order must return newest first,
// per the ListOptions.Order contract in interface.go (blueprint §3.6),
// which neither store honored before this issue.
func TestRedisListDefaultOrderIsDescending(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	seedPageResponses(t, store, "conv_default_order", 3)

	responses, err := store.ListResponsesByConversation(ctx, "conv_default_order", ListOptions{})
	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Equal(t, []string{"resp_page_2", "resp_page_1", "resp_page_0"}, responseIDsOf(responses))
}

// TestRedisListBoundedLimit covers blueprint §6.5: default and clamped
// limits, without reading the full conversation.
func TestRedisListBoundedLimit(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	total := DefaultListLimit + 5
	seedPageResponses(t, store, "conv_bounded", total)

	t.Run("default limit caps at DefaultListLimit, not the full conversation", func(t *testing.T) {
		responses, err := store.ListResponsesByConversation(ctx, "conv_bounded", ListOptions{Order: "asc"})
		require.NoError(t, err)
		assert.Len(t, responses, DefaultListLimit)
		// Bounded read: the oldest page, not an arbitrary DefaultListLimit
		// subset of a fully-read-then-truncated slice.
		assert.Equal(t, "resp_page_0", responses[0].ID)
	})

	t.Run("limit above MaxListLimit is clamped", func(t *testing.T) {
		responses, err := store.ListResponsesByConversation(ctx, "conv_bounded", ListOptions{Order: "asc", Limit: MaxListLimit + 1000})
		require.NoError(t, err)
		assert.LessOrEqual(t, len(responses), MaxListLimit)
	})
}

// TestRedisListCursors covers blueprint §3.6/§6.5's After/Before rank-window
// semantics in both directions, plus the invalid-input and unknown-cursor
// edge cases.
func TestRedisListCursors(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	seedPageResponses(t, store, "conv_cursor", 5) // resp_page_0 (oldest) .. resp_page_4 (newest)

	list := func(t *testing.T, opts ListOptions) []string {
		t.Helper()
		responses, err := store.ListResponsesByConversation(ctx, "conv_cursor", opts)
		require.NoError(t, err)
		return responseIDsOf(responses)
	}

	t.Run("ascending pages walk oldest to newest via After", func(t *testing.T) {
		assert.Equal(t, []string{"resp_page_0", "resp_page_1"}, list(t, ListOptions{Order: "asc", Limit: 2}))
		assert.Equal(t, []string{"resp_page_2", "resp_page_3"},
			list(t, ListOptions{Order: "asc", Limit: 2, After: "resp_page_1"}))
		assert.Equal(t, []string{"resp_page_4"},
			list(t, ListOptions{Order: "asc", Limit: 2, After: "resp_page_3"}))
		assert.Empty(t, list(t, ListOptions{Order: "asc", Limit: 2, After: "resp_page_4"}))
	})

	t.Run("descending pages walk newest to oldest via After", func(t *testing.T) {
		assert.Equal(t, []string{"resp_page_4", "resp_page_3"}, list(t, ListOptions{Order: "desc", Limit: 2}))
		assert.Equal(t, []string{"resp_page_2", "resp_page_1"},
			list(t, ListOptions{Order: "desc", Limit: 2, After: "resp_page_3"}))
	})

	t.Run("before returns the page preceding the cursor in the selected order", func(t *testing.T) {
		assert.Equal(t, []string{"resp_page_1", "resp_page_2"},
			list(t, ListOptions{Order: "asc", Limit: 2, Before: "resp_page_3"}))
		// Nothing precedes the very first element.
		assert.Empty(t, list(t, ListOptions{Order: "asc", Limit: 2, Before: "resp_page_0"}))
	})

	t.Run("a cursor naming a non-member returns an empty page, not an error", func(t *testing.T) {
		assert.Empty(t, list(t, ListOptions{Order: "asc", After: "resp_does_not_exist"}))
		assert.Empty(t, list(t, ListOptions{Order: "asc", Before: "resp_does_not_exist"}))
	})

	t.Run("after and before together are rejected", func(t *testing.T) {
		_, err := store.ListResponsesByConversation(ctx, "conv_cursor", ListOptions{After: "resp_page_0", Before: "resp_page_4"})
		assert.ErrorIs(t, err, ErrInvalidInput)
	})

	t.Run("unrecognized order is rejected", func(t *testing.T) {
		_, err := store.ListResponsesByConversation(ctx, "conv_cursor", ListOptions{Order: "sideways"})
		assert.ErrorIs(t, err, ErrInvalidInput)
	})
}
