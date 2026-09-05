package responsestore

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
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
	assert.Equal(t, int64(1), store.scanInvocations.Load(), "first read must run exactly one legacy scan")

	assert.ElementsMatch(t, []string{"resp_legacy_1", "resp_legacy_2", "resp_legacy_3"},
		conversationIndexMembers(t, store, "conv_legacy"))
	assert.EqualValues(t, 1, exists(t, store, store.conversationIndexMigratedKey("conv_legacy")),
		"a completed backfill that found responses must mark the conversation migrated")

	// The index now exists, so a second read must go straight through it
	// without scanning the keyspace again.
	responses, err = store.ListResponsesByConversation(ctx, "conv_legacy", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Equal(t, int64(1), store.scanInvocations.Load(), "second read of an already-backfilled conversation must not scan again")
}

// TestRedisListResponsesByConversationEmptyMarker covers blueprint §2.4/§3.4:
// a legitimately empty or unknown conversation must not force a full
// keyspace scan on every read — only once, after which the migrated marker
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
	assert.Equal(t, int64(1), store.scanInvocations.Load())

	markerKey := store.conversationIndexMigratedKey("conv_never_existed")
	assert.EqualValues(t, 1, exists(t, store, markerKey))
	ttl, err := store.client.TTL(ctx, markerKey).Result()
	require.NoError(t, err)
	assert.Positive(t, ttl, "migrated marker for a confirmed-empty conversation must carry a positive TTL, not be immortal")

	// Repeated reads of the same empty conversation must not scan again.
	for i := 0; i < 3; i++ {
		responses, err = store.ListResponsesByConversation(ctx, "conv_never_existed", ListOptions{})
		require.NoError(t, err)
		assert.Empty(t, responses)
	}
	assert.Equal(t, int64(1), store.scanInvocations.Load(), "repeated reads of an empty conversation must not force repeated scans")
}

// TestRedisLazyBackfillConcurrentWriteNotHidden covers blueprint §2.2: a
// response indexed normally by a concurrent StoreResponse call, landing
// while a lazy legacy scan for the same conversation is in flight, must
// survive — the scan walks every response payload regardless of whether it
// is already indexed, so it discovers and idempotently re-adds the
// concurrent write alongside the legacy one, and marks the conversation
// migrated once both are captured.
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
	writer := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	t.Cleanup(func() { _ = writer.Close() })
	store.client.AddHook(&beforeCommandHook{name: "scan", once: true, before: func() {
		// Runs once, before the scan walks the keyspace: lands a normal
		// indexed write for the same conversation the backfill is about to
		// scan for, so the scan observes both the legacy and the
		// concurrently-indexed payload.
		payload := mustMarshalResponse(t, concurrent)
		require.NoError(t, writer.Set(ctx, store.buildKey(ResponseKeyPrefix+concurrent.ID), payload, store.ttl).Err())
		require.NoError(t, writer.ZAdd(ctx, store.conversationIndexKey(concurrent.ConversationID),
			redis.Z{Score: float64(concurrent.CreatedAt), Member: concurrent.ID}).Err())
	}})

	responses, err := store.ListResponsesByConversation(ctx, "conv_race", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 2)
	assert.Equal(t, []string{"resp_race_legacy", "resp_race_concurrent"},
		[]string{responses[0].ID, responses[1].ID})

	assert.ElementsMatch(t, []string{"resp_race_legacy", "resp_race_concurrent"},
		conversationIndexMembers(t, store, "conv_race"))
	assert.EqualValues(t, 1, exists(t, store, store.conversationIndexMigratedKey("conv_race")),
		"the completed backfill must mark the conversation migrated once both responses are captured")
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

// TestRedisListResponsesByConversationConcurrentMissingScansOnce is the
// concurrency counterpart to TestRedisListResponsesByConversationEmptyMarker:
// scanInvocations must be an atomic counter, not a plain int, because
// scanResponsePayloads is reachable from concurrent requests missing the
// same conversation's index in real production traffic, not just from
// sequential test calls. Run with `go test -race` to catch a regression
// back to a plain int directly; the count assertion below additionally
// proves the migration lock (Phase 3) meaningfully reduces duplicate scans
// under real concurrency, not just in the serialized tests elsewhere.
func TestRedisListResponsesByConversationConcurrentMissingScansOnce(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const goroutines = 20

	var wg sync.WaitGroup
	errs := make([]error, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := store.ListResponsesByConversation(ctx, "conv_concurrent_missing", ListOptions{})
			errs[i] = err
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		assert.NoErrorf(t, err, "goroutine %d", i)
	}

	// The lock is an optimization, not a guarantee (blueprint §4.5): a
	// generous upper bound, not exactly 1, keeps this robust against
	// scheduling variance while still proving most of the herd converged on
	// the lock holder's result instead of each scanning independently.
	assert.Lessf(t, store.scanInvocations.Load(), int64(goroutines),
		"the migration lock should stop most concurrent readers from scanning independently")
}

// TestRedisConversationMigratedMarkerTTLBoundedWhenEmpty covers a
// rolling-upgrade blind spot: a migrated marker set after a scan finds
// nothing must not inherit the store's full data-retention TTL. A marker
// sharing a long data TTL (a day, 30 days) could hide a response written by
// an older, pre-index pod during a rolling deployment for that entire
// window — capping the marker independently bounds the blind spot to
// emptyConversationIndexMarkerMaxTTL regardless of how long data itself
// lives.
func TestRedisConversationMigratedMarkerTTLBoundedWhenEmpty(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 24*60*60) // 24h data TTL
	ctx := context.Background()

	_, err := store.ListResponsesByConversation(ctx, "conv_ttl_bound", ListOptions{})
	require.NoError(t, err)

	markerKey := store.conversationIndexMigratedKey("conv_ttl_bound")
	ttl, err := store.client.TTL(ctx, markerKey).Result()
	require.NoError(t, err)
	assert.Positive(t, ttl)
	assert.LessOrEqualf(t, ttl, emptyConversationIndexMarkerMaxTTL,
		"migrated marker TTL for a confirmed-empty conversation must be capped, not inherit the store's %s data TTL", store.ttl)
}

// TestRedisConversationMigratedMarkerExpiryRevealsLegacyWrite simulates the
// empty-case marker's TTL cap actually elapsing (by deleting it directly
// rather than waiting out the real duration in a unit test): a response
// written by an indexing-unaware writer while the marker was still valid
// must be discovered on the next read once the marker is gone, rather than
// stay permanently hidden behind it.
func TestRedisConversationMigratedMarkerExpiryRevealsLegacyWrite(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	responses, err := store.ListResponsesByConversation(ctx, "conv_marker_expiry", ListOptions{})
	require.NoError(t, err)
	assert.Empty(t, responses)

	markerKey := store.conversationIndexMigratedKey("conv_marker_expiry")
	require.EqualValues(t, 1, exists(t, store, markerKey))

	// A write from an older, indexing-unaware pod: lands a payload with no
	// index entry and no marker awareness, exactly as pre-#2814 code would.
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_after_marker", ConversationID: "conv_marker_expiry", Status: "completed", CreatedAt: time.Now().Unix(),
	})

	// Simulate the marker's TTL cap elapsing, rather than waiting it out.
	require.NoError(t, store.client.Del(ctx, markerKey).Err())

	responses, err = store.ListResponsesByConversation(ctx, "conv_marker_expiry", ListOptions{})
	require.NoError(t, err)
	require.Len(t, responses, 1)
	assert.Equal(t, "resp_after_marker", responses[0].ID)
}

// TestRedisListResponsesByConversationPartiallyMigrated is the direct
// regression test for the "index exists" != "migration complete" bug: a
// conversation with unindexed legacy responses that then receives an
// ordinary post-upgrade write has an index — created by that one write's
// indexResponse call — containing only the new response. Before the
// migrated marker decoupled these two facts, every subsequent read trusted
// that index as complete and never discovered resp_old, silently and
// permanently.
func TestRedisListResponsesByConversationPartiallyMigrated(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	// A legacy response, bypassing StoreResponse's indexing entirely.
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_old", ConversationID: "conv_1", Status: "completed", CreatedAt: time.Now().Unix(),
	})

	// An ordinary post-upgrade write to the *same* conversation, before any
	// read has ever triggered a backfill. This creates the index directly,
	// containing only resp_new.
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_new", ConversationID: "conv_1", Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}))
	require.Equal(t, []string{"resp_new"}, conversationIndexMembers(t, store, "conv_1"),
		"precondition: the index exists but is not yet exhaustive")
	require.Zero(t, exists(t, store, store.conversationIndexMigratedKey("conv_1")),
		"precondition: no backfill has run yet, despite the index already existing")

	responses, err := store.ListResponsesByConversation(ctx, "conv_1", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 2, "both the legacy and the post-upgrade response must be returned")
	assert.Equal(t, []string{"resp_old", "resp_new"}, responseIDsOf(responses))

	assert.ElementsMatch(t, []string{"resp_old", "resp_new"}, conversationIndexMembers(t, store, "conv_1"))
	assert.EqualValues(t, 1, exists(t, store, store.conversationIndexMigratedKey("conv_1")))

	// A second read must not re-scan: migrated is now true.
	_, err = store.ListResponsesByConversation(ctx, "conv_1", ListOptions{})
	require.NoError(t, err)
	assert.Equal(t, int64(1), store.scanInvocations.Load(), "once migrated, a second read must not scan again")
}
