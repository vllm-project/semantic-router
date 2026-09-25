package responsestore

import (
	"context"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// TestStoreResponseRollsBackAfterContextCancellation covers the first half of
// the detached-rollback rule. A cancellation arriving after the payload is
// durably stored but before it is indexed fails the index write, and a
// rollback sharing that context fails for the very same reason — leaving a
// live payload no index entry names. Nothing lists it, and once
// FinalizeConversationIndex has sealed the store nothing rescans to find it.
func TestStoreResponseRollsBackAfterContextCancellation(t *testing.T) {
	store := newConversationIndexStore(t)
	background := context.Background()

	const (
		conversationID = "conv_store_cancel"
		responseID     = "resp_store_cancel"
	)

	ctx, cancel := context.WithCancel(background)
	defer cancel()
	// Cancel the instant the payload has committed, before it can be indexed:
	// exactly what an HTTP client disconnecting mid-request does to the
	// request context.
	store.client.AddHook(&afterCommandHook{name: "set", after: cancel})

	err := store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: responseID, ConversationID: conversationID, Status: "completed", CreatedAt: time.Now().Unix(),
	})
	require.Error(t, err)

	_, getErr := store.GetResponse(background, responseID)
	assert.ErrorIs(t, getErr, ErrNotFound,
		"the payload must be rolled back, not stranded by the same cancellation that failed its index write")
	assert.Empty(t, conversationIndexMembers(t, store, conversationID))
}

// TestUpdateResponseRollsBackAfterContextCancellation is the same rule for
// updates, and the state the reviewer reproduced: the payload replacement
// commits, the cancellation lands, the new conversation's index write fails,
// and a rollback on the caller's context fails too — stranding the response
// under its new conversation while both conversations' listings come back
// empty.
func TestUpdateResponseRollsBackAfterContextCancellation(t *testing.T) {
	store := newConversationIndexStore(t)
	background := context.Background()

	const (
		fromConversation = "conv_update_cancel_from"
		toConversation   = "conv_update_cancel_to"
		responseID       = "resp_update_cancel"
	)
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: fromConversation, Status: "original", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(background, original))
	// Warm the update path's script, so the hook below cannot fire on a
	// NOSCRIPT probe that replaced nothing.
	require.NoError(t, store.UpdateResponse(background, original))
	require.Equal(t, []string{responseID}, conversationIndexMembers(t, store, fromConversation))

	ctx, cancel := context.WithCancel(background)
	defer cancel()
	store.client.AddHook(&afterCommandHook{name: "evalsha", after: cancel})

	moved := *original
	moved.ConversationID = toConversation
	moved.Status = "moved"
	require.Error(t, store.UpdateResponse(ctx, &moved))

	restored, err := store.GetResponse(background, responseID)
	require.NoError(t, err, "the response must not vanish from the store")
	assert.Equal(t, fromConversation, restored.ConversationID,
		"the payload must be rolled back to the conversation whose index still names it")
	assert.Equal(t, "original", restored.Status)

	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, fromConversation))
	assert.Empty(t, conversationIndexMembers(t, store, toConversation),
		"the conversation whose index write failed must not keep an entry")

	listed, err := store.ListResponsesByConversation(background, fromConversation, ListOptions{})
	require.NoError(t, err)
	require.Len(t, listed, 1,
		"a rolled-back update must leave the response discoverable, not stranded between two conversations")
}

// TestUpdateResponseKeepsMembershipRestoredByConcurrentUpdate is the ABA
// regression the reviewer described. An A->B update pauses before cleaning up
// A; a B->A update completes and legitimately re-adds A's entry; the first
// update resumes and, with an unconditional ZREM, deletes the membership the
// second update just created. Both calls return success, GetResponse still
// answers with the response in A, and listing A returns nothing — a state no
// prune can repair, since pruning only ever removes entries, and no scan
// repairs once the store is finalized.
func TestUpdateResponseKeepsMembershipRestoredByConcurrentUpdate(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	background := context.Background()

	const (
		fromConversation = "conv_aba_from"
		toConversation   = "conv_aba_to"
		responseID       = "resp_aba"
	)
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: fromConversation, Status: "original", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(background, original))
	require.NoError(t, store.ensureConversationIndexResolved(background, fromConversation))

	movedBack := *original
	movedBack.ConversationID = fromConversation
	movedBack.Status = "moved-back"
	movedBack.CreatedAt++
	var injectedErr error
	hook := &commandInterleavingHook{
		match: func(cmd redis.Cmder) bool {
			return commandRunsScriptOnKey(cmd, store.conversationIndexKey(toConversation))
		},
		inject: func() { injectedErr = writer.UpdateResponse(context.Background(), &movedBack) },
	}
	store.client.AddHook(hook)

	moved := *original
	moved.ConversationID = toConversation
	require.NoError(t, store.UpdateResponse(background, &moved))
	require.True(t, hook.fired.Load(), "the competing update must actually have been injected for this to mean anything")
	require.NoError(t, injectedErr)

	assertResponseListedIn(t, store, fromConversation, responseID)
}

// assertResponseListedIn checks the three facts that have to agree once a
// racing update settles: the payload says the response is in this
// conversation, the index names it, and a listing returns it. The bug these
// tests guard against is precisely the three disagreeing.
func assertResponseListedIn(t *testing.T, store *RedisStore, conversationID, responseID string) {
	t.Helper()
	background := context.Background()

	stored, err := store.GetResponse(background, responseID)
	require.NoError(t, err)
	require.Equal(t, conversationID, stored.ConversationID,
		"precondition: the competing update owns the payload")

	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, conversationID),
		"the membership the competing update created must survive the other update's cleanup")

	listed, err := store.ListResponsesByConversation(background, conversationID, ListOptions{})
	require.NoError(t, err)
	require.Len(t, listed, 1, "a response GetResponse reports in this conversation must also be listed by it")
	assert.Equal(t, responseID, listed[0].ID)
}

// TestUpdateResponseClusterCrossSlotSafe pins the constraint that shapes the
// membership cleanup above, against a real (single-node) Redis Cluster.
//
// A response payload key and a conversation index key share no hash tag, so
// they hash to different slots — sr:response:* and sr:conversation-index:*
// land in unrelated slots by construction. That is why the cleanup verifies
// the payload with its own read instead of a compare-and-remove Lua script
// spanning both keys: Redis rejects such a script outright with CROSSSLOT.
//
// Moving a response between conversations therefore has to stay a sequence of
// single-key commands, and this proves it still is. Redis enforces slot
// locality per command regardless of node count, so one node in cluster mode
// is a faithful testbed.
func TestUpdateResponseClusterCrossSlotSafe(t *testing.T) {
	store := newConversationIndexClusterStore(t)
	ctx := context.Background()

	const (
		fromConversation = "conv_cluster_from"
		toConversation   = "conv_cluster_to"
		responseID       = "resp_cluster_move"
	)

	// The three keys this exercises hash to different slots, which is the
	// whole point: assert that rather than assume it.
	slots := map[string]int64{}
	for _, key := range []string{
		store.buildKey(ResponseKeyPrefix + responseID),
		store.conversationIndexKey(fromConversation),
		store.conversationIndexKey(toConversation),
	} {
		slot, err := store.client.ClusterKeySlot(ctx, key).Result()
		require.NoError(t, err)
		slots[key] = slot
	}
	require.Len(t, slots, 3)
	distinct := map[int64]struct{}{}
	for _, slot := range slots {
		distinct[slot] = struct{}{}
	}
	require.Greaterf(t, len(distinct), 1,
		"the payload and index keys must genuinely span slots for this test to prove anything: %v", slots)

	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: fromConversation, Status: "original", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, original))

	moved := *original
	moved.ConversationID = toConversation
	moved.Status = "moved"
	require.NoError(t, store.UpdateResponse(ctx, &moved),
		"moving a response between conversations must issue only single-key commands")

	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, toConversation))
	assert.Empty(t, conversationIndexMembers(t, store, fromConversation))
}
