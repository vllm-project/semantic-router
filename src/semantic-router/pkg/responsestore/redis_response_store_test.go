package responsestore

import (
	"context"
	"encoding/json"
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
	store.client.AddHook(&pipelineFailureHook{
		name: "zadd", key: store.conversationIndexKey("conv_rollback"), err: injectedErr,
	})

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
	store.client.AddHook(&pipelineFailureHook{
		name: "zadd", key: store.conversationIndexKey("conv_update_to"), err: injectedErr,
	})

	updated := *original
	updated.ConversationID = "conv_update_to"
	updated.Status = "updated"
	err := store.UpdateResponse(ctx, &updated)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	restored, err := store.GetResponse(ctx, original.ID)
	require.NoError(t, err)
	assert.Equal(t, "original", restored.Status, "payload must be restored to its pre-update value")
	assert.Equal(t, "conv_update_from", restored.ConversationID)

	assert.Empty(t, conversationIndexMembers(t, store, "conv_update_to"),
		"the failed new conversation must never gain an index entry")
	assert.Equal(t, []string{"resp_update_rollback"}, conversationIndexMembers(t, store, "conv_update_from"),
		"the previous conversation must be reindexed after rollback")
}

// TestRedisUpdateResponseRollbackConflictPreservesNewerWrite is the direct
// regression test for the CAS-safe rollback (Phase 1): if a newer write
// lands on top of a failed update's own payload before that update's
// rollback runs, the rollback's compare-and-swap must detect the mismatch,
// leave the newer payload untouched, and must not reindex the stale
// snapshot's previous conversation.
func TestRedisUpdateResponseRollbackConflictPreservesNewerWrite(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const responseID = "resp_update_conflict"
	key := store.buildKey(ResponseKeyPrefix + responseID)

	// Written unindexed (not via StoreResponse), so "conv_update_conflict_from"
	// starts with an empty index — making "rollback must not reindex it"
	// an observable assertion below, rather than one an unrelated prior
	// index write would satisfy by coincidence.
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_conflict_from", Status: "original", CreatedAt: time.Now().Unix(),
	}
	directSetResponsePayload(t, store, original)

	snapshot, err := store.readPreviousResponseForUpdate(ctx, key, responseID)
	require.NoError(t, err)
	require.Equal(t, "conv_update_conflict_from", snapshot.conversationID)
	require.Empty(t, conversationIndexMembers(t, store, "conv_update_conflict_from"))

	// The failed update's own write (what its rollback will try to
	// compare-and-swap against), then a newer, fully successful concurrent
	// update landing on top of it before the first update's rollback runs.
	failedUpdateData, err := json.Marshal(&responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_conflict_failed", Status: "failed-update", CreatedAt: time.Now().Unix(),
	})
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, key, failedUpdateData, store.ttl).Err())

	newer := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_conflict_newer", Status: "newer-update", CreatedAt: time.Now().Unix(),
	}
	newerData, err := json.Marshal(newer)
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, key, newerData, store.ttl).Err())
	require.NoError(t, store.indexResponse(ctx, newer.ConversationID, responseID, newer.CreatedAt))

	injectedErr := errors.New("injected index failure")
	err = store.rollbackUpdatePayload(ctx, key, responseID, failedUpdateData, snapshot, injectedErr)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	current, err := store.GetResponse(ctx, responseID)
	require.NoError(t, err)
	assert.Equal(t, "newer-update", current.Status, "a newer concurrent write must survive an older update's failed rollback")
	assert.Equal(t, "conv_update_conflict_newer", current.ConversationID)

	assert.Empty(t, conversationIndexMembers(t, store, "conv_update_conflict_from"),
		"a CAS conflict must not reindex the stale snapshot's previous conversation")
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, "conv_update_conflict_newer"))
}

// TestRedisUpdateResponseRollbackRestoresImmediatePredecessor covers the
// opposite overlap from the conflict test: two updates begin from the same
// original payload, the first replacement succeeds, and the second
// replacement later fails its index write. The failed update must restore the
// first update, not the value that existed before both calls began.
func TestRedisUpdateResponseRollbackRestoresImmediatePredecessor(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const responseID = "resp_update_immediate_predecessor"
	key := store.buildKey(ResponseKeyPrefix + responseID)
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_original", Status: "original", CreatedAt: time.Now().Unix(),
	}
	directSetResponsePayload(t, store, original)

	first := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_first", Status: "first-success", CreatedAt: time.Now().Unix() + 1,
	}
	firstData, err := json.Marshal(first)
	require.NoError(t, err)
	_, err = store.replaceResponseAndSnapshot(ctx, key, responseID, firstData)
	require.NoError(t, err)

	second := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_second", Status: "second-failed", CreatedAt: time.Now().Unix() + 2,
	}
	secondData, err := json.Marshal(second)
	require.NoError(t, err)
	secondSnapshot, err := store.replaceResponseAndSnapshot(ctx, key, responseID, secondData)
	require.NoError(t, err)
	assert.JSONEq(t, string(firstData), string(secondSnapshot.data),
		"the atomic replacement must snapshot its immediate predecessor")

	// The first update can finish its index write while the second update is
	// still in flight. The second update then fails and rolls itself back.
	require.NoError(t, store.indexResponse(ctx, first.ConversationID, responseID, first.CreatedAt))
	injectedErr := errors.New("injected second update index failure")
	err = store.rollbackUpdatePayload(ctx, key, responseID, secondData, secondSnapshot, injectedErr)
	require.ErrorIs(t, err, injectedErr)

	current, err := store.GetResponse(ctx, responseID)
	require.NoError(t, err)
	assert.Equal(t, "first-success", current.Status)
	assert.Equal(t, "conv_first", current.ConversationID)
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, "conv_first"))
}

// TestRedisUpdateResponseRollbackPreservesRemainingTTL covers that a
// rollback restores the snapshot's approximate remaining lifetime, not a
// freshly reset full retention TTL.
func TestRedisUpdateResponseRollbackPreservesRemainingTTL(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const responseID = "resp_update_ttl"
	key := store.buildKey(ResponseKeyPrefix + responseID)

	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_ttl", Status: "original", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, original))
	require.NoError(t, store.client.Expire(ctx, key, 100*time.Second).Err())

	snapshot, err := store.readPreviousResponseForUpdate(ctx, key, responseID)
	require.NoError(t, err)
	require.InDelta(t, (100 * time.Second).Milliseconds(), snapshot.pttlMillis, 2000)

	failedData, err := json.Marshal(&responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_ttl_failed", Status: "failed", CreatedAt: time.Now().Unix(),
	})
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, key, failedData, store.ttl).Err())

	injectedErr := errors.New("injected index failure")
	err = store.rollbackUpdatePayload(ctx, key, responseID, failedData, snapshot, injectedErr)
	require.ErrorIs(t, err, injectedErr)

	ttl, err := store.client.TTL(ctx, key).Result()
	require.NoError(t, err)
	assert.InDelta(t, 100.0, ttl.Seconds(), 5,
		"restored TTL must approximate the snapshot's remaining lifetime (100s), not the store's full 300s retention TTL")
}

// TestRedisUpdateResponseRollbackPreservesPersistentTTL covers the
// persistent (no-expiry) case: a snapshot with pttlMillis == -1 must be
// restored persistent, not given the store's default TTL.
func TestRedisUpdateResponseRollbackPreservesPersistentTTL(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const responseID = "resp_update_persistent"
	key := store.buildKey(ResponseKeyPrefix + responseID)

	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_persistent", Status: "original", CreatedAt: time.Now().Unix(),
	}
	data, err := json.Marshal(original)
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, key, data, 0).Err())

	snapshot, err := store.readPreviousResponseForUpdate(ctx, key, responseID)
	require.NoError(t, err)
	require.EqualValues(t, -1, snapshot.pttlMillis)

	failedData, err := json.Marshal(&responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_persistent_failed", Status: "failed", CreatedAt: time.Now().Unix(),
	})
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, key, failedData, store.ttl).Err())

	injectedErr := errors.New("injected index failure")
	err = store.rollbackUpdatePayload(ctx, key, responseID, failedData, snapshot, injectedErr)
	require.ErrorIs(t, err, injectedErr)

	ttl, err := store.client.TTL(ctx, key).Result()
	require.NoError(t, err)
	assert.Equal(t, time.Duration(-1), ttl, "a persistent payload restored by rollback must remain persistent")
}

// TestRedisUpdateResponseRollbackDeletesWhenSnapshotTTLElapsed covers
// compareRestoreExpired: if the snapshot's own remaining TTL has run out by
// the time rollback runs, the failed update's payload is deleted rather
// than resurrected with a value whose intended lifetime already ended, and
// nothing is reindexed.
func TestRedisUpdateResponseRollbackDeletesWhenSnapshotTTLElapsed(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const responseID = "resp_update_expired"
	key := store.buildKey(ResponseKeyPrefix + responseID)

	// Written unindexed, so "conv_update_expired" starts with an empty
	// index — making "rollback must not reindex it" observable below.
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_expired", Status: "original", CreatedAt: time.Now().Unix(),
	}
	directSetResponsePayload(t, store, original)

	snapshot, err := store.readPreviousResponseForUpdate(ctx, key, responseID)
	require.NoError(t, err)
	require.Empty(t, conversationIndexMembers(t, store, "conv_update_expired"))

	// Force the snapshot to look like its captured TTL has (almost)
	// elapsed, then let a moment actually pass, rather than waiting out a
	// real TTL in a unit test.
	snapshot.pttlMillis = 1
	time.Sleep(5 * time.Millisecond)

	failedData, err := json.Marshal(&responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_update_expired_failed", Status: "failed", CreatedAt: time.Now().Unix(),
	})
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, key, failedData, store.ttl).Err())

	injectedErr := errors.New("injected index failure")
	err = store.rollbackUpdatePayload(ctx, key, responseID, failedData, snapshot, injectedErr)
	require.ErrorIs(t, err, injectedErr)

	_, getErr := store.GetResponse(ctx, responseID)
	assert.ErrorIs(t, getErr, ErrNotFound,
		"a rollback whose snapshot TTL already elapsed must delete rather than resurrect the payload")
	assert.Empty(t, conversationIndexMembers(t, store, "conv_update_expired"),
		"an expired rollback must not reindex the deleted payload")
}
