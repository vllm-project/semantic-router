package responsestore

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// TestRedisFinalizeConversationIndex covers the cluster-wide sweep
// itself: legacy responses scattered across several conversations, none of
// them indexed or even touched by a read yet, must all be discoverable in
// their respective conversation indexes once the sweep completes, and the
// global status key must record that completion.
func TestRedisFinalizeConversationIndex(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	now := time.Now().Unix()
	legacy := []*responseapi.StoredResponse{
		{ID: "resp_fin_a1", ConversationID: "conv_fin_a", Status: "completed", CreatedAt: now},
		{ID: "resp_fin_a2", ConversationID: "conv_fin_a", Status: "completed", CreatedAt: now + 1},
		{ID: "resp_fin_b1", ConversationID: "conv_fin_b", Status: "completed", CreatedAt: now},
		// No ConversationID: must not end up indexed anywhere.
		{ID: "resp_fin_orphan", Status: "completed", CreatedAt: now},
	}
	for _, response := range legacy {
		directSetResponsePayload(t, store, response)
	}

	statusKey := store.conversationIndexCompletionKey()
	require.Zero(t, exists(t, store, statusKey), "precondition: migration not yet finalized")

	stats, err := store.FinalizeConversationIndex(ctx)
	require.NoError(t, err)
	assert.EqualValues(t, len(legacy), stats.ResponsesScanned)
	assert.EqualValues(t, 3, stats.ResponsesIndexed)

	value, err := store.client.Get(ctx, statusKey).Result()
	require.NoError(t, err)
	assert.Equal(t, conversationIndexCompletionValue, value)

	assert.Equal(t, []string{"resp_fin_a1", "resp_fin_a2"}, conversationIndexMembers(t, store, "conv_fin_a"))
	assert.Equal(t, []string{"resp_fin_b1"}, conversationIndexMembers(t, store, "conv_fin_b"))

	// Discoverable through the ordinary read path too, not just directly
	// against the index.
	responses, err := store.ListResponsesByConversation(ctx, "conv_fin_a", ListOptions{Order: "asc"})
	require.NoError(t, err)
	assert.Equal(t, []string{"resp_fin_a1", "resp_fin_a2"}, responseIDsOf(responses))

	// Idempotent: calling it again once complete is a no-op, not an error.
	stats, err = store.FinalizeConversationIndex(ctx)
	require.NoError(t, err)
	assert.Zero(t, stats)
}

func TestRedisFinalizeConversationIndexAbortsOnMalformedPayload(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	key := store.buildKey(ResponseKeyPrefix + "resp_finalize_malformed")
	require.NoError(t, store.client.Set(ctx, key, "not-json", store.ttl).Err())

	stats, err := store.FinalizeConversationIndex(ctx)
	require.Error(t, err)
	assert.Zero(t, stats)
	assert.Zero(t, exists(t, store, store.conversationIndexCompletionKey()))
}

func TestRedisFinalizeConversationIndexAbortsOnIndexFailure(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const conversationID = "conv_finalize_wrongtype"
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_finalize_wrongtype", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix(),
	})
	require.NoError(t, store.client.Set(ctx, store.conversationIndexKey(conversationID), "wrong-type", 0).Err())

	stats, err := store.FinalizeConversationIndex(ctx)
	require.Error(t, err)
	assert.Zero(t, stats)
	assert.Zero(t, exists(t, store, store.conversationIndexCompletionKey()))
}

func TestRedisFinalizeConversationIndexRefreshesIndexTTL(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_finalize_ttl", ConversationID: "conv_finalize_ttl",
		Status: "completed", CreatedAt: time.Now().Unix(),
	})

	_, err := store.FinalizeConversationIndex(ctx)
	require.NoError(t, err)
	ttl, err := store.client.TTL(ctx, store.conversationIndexKey("conv_finalize_ttl")).Result()
	require.NoError(t, err)
	assert.Greater(t, ttl, time.Duration(0))
	assert.LessOrEqual(t, ttl, store.ttl)
}

func TestRedisFinalizeConversationIndexRejectsUnknownCompletionValue(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(), "future:v2", 0).Err())
	complete, err := store.conversationIndexFinalized(ctx)
	require.NoError(t, err)
	assert.False(t, complete)
}

func TestRedisFinalizeConversationIndexMissingCompletionIsNotError(t *testing.T) {
	store := newConversationIndexStore(t)
	complete, err := store.conversationIndexFinalized(context.Background())
	require.NoError(t, err)
	assert.False(t, complete)
}

// TestRedisSteadyStateNeverScansOnceMigrationComplete is the direct
// regression test for the steady-state performance goal this closure
// exists for: once ConversationIndexCompletionKeySuffix is set, reading an
// unknown or empty conversation — the one case that could still force a
// per-conversation legacy scan under the Phase 8 design alone — must
// consult the index only, never scanResponsePayloads.
func TestRedisSteadyStateNeverScansOnceMigrationComplete(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	// Seed unrelated data so a scan, if one incorrectly ran, would have
	// something to walk past.
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_steady_unrelated", ConversationID: "conv_steady_other", Status: "completed", CreatedAt: time.Now().Unix(),
	})

	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(), conversationIndexCompletionValue, 0).Err())

	responses, err := store.ListResponsesByConversation(ctx, "conv_steady_never_existed", ListOptions{})
	require.NoError(t, err)
	assert.Empty(t, responses)
	assert.Equal(t, int64(0), store.scanInvocations.Load(),
		"a finalized store must never scan, even for a conversation ID nothing has ever indexed")

	// Repeated reads: still no scan, and no per-conversation marker is
	// needed or written — the store-wide flag alone is authoritative.
	for i := 0; i < 3; i++ {
		responses, err = store.ListResponsesByConversation(ctx, "conv_steady_never_existed", ListOptions{})
		require.NoError(t, err)
		assert.Empty(t, responses)
	}
	assert.Equal(t, int64(0), store.scanInvocations.Load())
	assert.Zero(t, exists(t, store, store.conversationIndexMigratedKey("conv_steady_never_existed")),
		"the per-conversation marker is never consulted or written once the store is finalized")

	// Cascade delete of an unknown conversation must not scan either.
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: "conv_steady_empty_delete", CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.DeleteConversation(ctx, "conv_steady_empty_delete", true))
	assert.Equal(t, int64(0), store.scanInvocations.Load())
}

// TestRedisSteadyStateReadsRealDataOnceMigrationComplete confirms the
// finalized fast path still returns real indexed data, not just correctly
// short-circuiting to empty.
func TestRedisSteadyStateReadsRealDataOnceMigrationComplete(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_steady_real", ConversationID: "conv_steady_real", Status: "completed", CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(), conversationIndexCompletionValue, 0).Err())

	responses, err := store.ListResponsesByConversation(ctx, "conv_steady_real", ListOptions{})
	require.NoError(t, err)
	require.Len(t, responses, 1)
	assert.Equal(t, "resp_steady_real", responses[0].ID)
	assert.Equal(t, int64(0), store.scanInvocations.Load())
}

func TestRedisFirstReadAfterFinalizedWriteNeverScans(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(), conversationIndexCompletionValue, 0).Err())
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_finalized_first_read", ConversationID: "conv_finalized_first_read",
		Status: "completed", CreatedAt: time.Now().Unix(),
	}))

	responses, err := store.ListResponsesByConversation(ctx, "conv_finalized_first_read", ListOptions{})
	require.NoError(t, err)
	require.Len(t, responses, 1)
	assert.Equal(t, "resp_finalized_first_read", responses[0].ID)
	assert.Zero(t, store.scanInvocations.Load())
}
