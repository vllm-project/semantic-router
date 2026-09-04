package responsestore

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// TestRedisStoreResponseRejectsDuplicate covers the SET NX path: it enforces the
// no-duplicate contract and orders the payload write ahead of the index write.
func TestRedisStoreResponseRejectsDuplicate(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	response := &responseapi.StoredResponse{
		ID:             "resp_duplicate",
		ConversationID: "conv_duplicate",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, response))

	duplicate := *response
	duplicate.Status = "overwritten"
	assert.ErrorIs(t, store.StoreResponse(ctx, &duplicate), ErrAlreadyExists)

	// The original payload must be untouched, and indexed exactly once.
	stored, err := store.GetResponse(ctx, response.ID)
	require.NoError(t, err)
	assert.Equal(t, "completed", stored.Status)
	assert.Equal(t, []string{"resp_duplicate"}, conversationIndexMembers(t, store, "conv_duplicate"))
}

// TestRedisStoreResponseRollsBackIndexFailure covers blueprint §2.3/§3.5: if
// the payload write succeeds but indexing fails, StoreResponse must roll the
// payload back via compare-delete rather than leave an orphan that a retry
// can never repair (SETNX would just see it exists and hit ErrAlreadyExists
// with no index to fix).
func TestRedisStoreResponseRollsBackIndexFailure(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	injectedErr := errors.New("injected index failure")
	store.indexResponseOverride = func(context.Context, string, string, int64) error {
		return injectedErr
	}

	response := &responseapi.StoredResponse{
		ID:             "resp_rollback",
		ConversationID: "conv_rollback",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}

	err := store.StoreResponse(ctx, response)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	// The rolled-back payload must be gone, not orphaned: GetResponse sees a
	// clean slate, and a retry does not hit ErrAlreadyExists.
	_, err = store.GetResponse(ctx, response.ID)
	assert.ErrorIs(t, err, ErrNotFound)
	assert.Empty(t, conversationIndexMembers(t, store, "conv_rollback"))

	// Remove the injected failure and retry: must succeed cleanly.
	store.indexResponseOverride = nil
	require.NoError(t, store.StoreResponse(ctx, response))

	stored, err := store.GetResponse(ctx, response.ID)
	require.NoError(t, err)
	assert.Equal(t, "completed", stored.Status)
	assert.Equal(t, []string{"resp_rollback"}, conversationIndexMembers(t, store, "conv_rollback"))
}

// TestRedisCompareDeleteResponsePayload covers the ABA race the blueprint
// calls out (§2.3): rollback must never delete a payload that changed after
// the failed write, e.g. because a concurrent writer replaced it once the
// original value's TTL expired.
func TestRedisCompareDeleteResponsePayload(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	key := store.buildKey(ResponseKeyPrefix + "resp_cas")

	t.Run("deletes when value matches", func(t *testing.T) {
		require.NoError(t, store.client.Set(ctx, key, []byte("payload-a"), store.ttl).Err())

		deleted, err := store.compareDeleteResponsePayload(ctx, key, []byte("payload-a"))
		require.NoError(t, err)
		assert.True(t, deleted)

		exists, err := store.client.Exists(ctx, key).Result()
		require.NoError(t, err)
		assert.Zero(t, exists)
	})

	t.Run("leaves a changed value untouched", func(t *testing.T) {
		require.NoError(t, store.client.Set(ctx, key, []byte("payload-a"), store.ttl).Err())

		deleted, err := store.compareDeleteResponsePayload(ctx, key, []byte("payload-b"))
		require.NoError(t, err)
		assert.False(t, deleted)

		value, err := store.client.Get(ctx, key).Result()
		require.NoError(t, err)
		assert.Equal(t, "payload-a", value)
	})

	t.Run("no-op against a missing key", func(t *testing.T) {
		require.NoError(t, store.client.Del(ctx, key).Err())

		deleted, err := store.compareDeleteResponsePayload(ctx, key, []byte("anything"))
		require.NoError(t, err)
		assert.False(t, deleted)
	})
}

// TestRedisStoreResponseDuplicateRepairsOrphanedIndex covers the repair half
// of blueprint §3.5: a retry that lands on an existing payload whose index
// entry is missing (e.g. left behind by a prior partial failure, or never
// indexed pre-upgrade) must repair the index from the stored payload before
// returning the duplicate error.
func TestRedisStoreResponseDuplicateRepairsOrphanedIndex(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	response := &responseapi.StoredResponse{
		ID:             "resp_orphan",
		ConversationID: "conv_orphan",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}
	directSetResponsePayload(t, store, response)
	require.Empty(t, conversationIndexMembers(t, store, "conv_orphan"))

	retry := *response
	err := store.StoreResponse(ctx, &retry)
	assert.ErrorIs(t, err, ErrAlreadyExists)

	assert.Equal(t, []string{"resp_orphan"}, conversationIndexMembers(t, store, "conv_orphan"))
}

// TestRedisStoreResponseDuplicateDifferentConversationNotRepaired covers the
// poisoning guard from blueprint §2.3/§3.5: a duplicate ID whose stored
// payload belongs to a different conversation than the retry attempted must
// not be indexed under the attempted conversation.
func TestRedisStoreResponseDuplicateDifferentConversationNotRepaired(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	stored := &responseapi.StoredResponse{
		ID:             "resp_mismatch",
		ConversationID: "conv_real",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}
	directSetResponsePayload(t, store, stored)

	attempted := *stored
	attempted.ConversationID = "conv_wrong"
	err := store.StoreResponse(ctx, &attempted)
	assert.ErrorIs(t, err, ErrAlreadyExists)

	assert.Empty(t, conversationIndexMembers(t, store, "conv_wrong"))
	// The real conversation was never indexed either: repair only runs
	// against the attempted conversation, and stored != attempted here.
	assert.Empty(t, conversationIndexMembers(t, store, "conv_real"))
}

// TestRedisStoreResponseDuplicateNoConversationNotRepaired covers the "no
// index expected" branch of repairExistingResponseIndex: a stored response
// with no ConversationID has nothing to repair.
func TestRedisStoreResponseDuplicateNoConversationNotRepaired(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	stored := &responseapi.StoredResponse{
		ID:        "resp_no_conv",
		Status:    "completed",
		CreatedAt: time.Now().Unix(),
	}
	directSetResponsePayload(t, store, stored)

	attempted := *stored
	attempted.ConversationID = "conv_should_not_exist"
	err := store.StoreResponse(ctx, &attempted)
	assert.ErrorIs(t, err, ErrAlreadyExists)

	assert.Empty(t, conversationIndexMembers(t, store, "conv_should_not_exist"))
}

// TestRedisUpdateResponseRestoresOnIndexFailure covers the UpdateResponse
// repairability the blueprint asks for on top of StoreResponse's (§5 Phase
// 5): if the new conversation's index write fails, the previous payload
// bytes must be restored and reindexed under the previous conversation,
// rather than left pointing at a conversation whose index was never
// actually written.
func TestRedisUpdateResponseRestoresOnIndexFailure(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	original := &responseapi.StoredResponse{
		ID:             "resp_update_rollback",
		ConversationID: "conv_update_from",
		Status:         "original",
		CreatedAt:      time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, original))
	require.Equal(t, []string{"resp_update_rollback"}, conversationIndexMembers(t, store, "conv_update_from"))

	injectedErr := errors.New("injected update index failure")
	store.indexResponseOverride = func(context.Context, string, string, int64) error {
		return injectedErr
	}

	updated := *original
	updated.ConversationID = "conv_update_to"
	updated.Status = "updated"
	err := store.UpdateResponse(ctx, &updated)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	// The payload restore itself does not go through indexResponseOverride
	// (it's a plain SET), so it must have succeeded even with the override
	// still failing indexResponse.
	store.indexResponseOverride = nil

	restored, err := store.GetResponse(ctx, original.ID)
	require.NoError(t, err)
	assert.Equal(t, "original", restored.Status, "payload must be restored to its pre-update value")
	assert.Equal(t, "conv_update_from", restored.ConversationID)

	assert.Empty(t, conversationIndexMembers(t, store, "conv_update_to"),
		"the failed new conversation must never gain an index entry")
	assert.Equal(t, []string{"resp_update_rollback"}, conversationIndexMembers(t, store, "conv_update_from"),
		"the previous conversation must be reindexed after rollback")
}
