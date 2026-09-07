package responsestore

import (
	"context"
	"encoding/json"
	"sync/atomic"
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

// updateCleanupRaceInjectionPoint says where a competing update is made to
// land relative to the cleanup it races.
type updateCleanupRaceInjectionPoint int

const (
	// beforeCleanup lands it after the new conversation's index write and so
	// before the cleanup starts — the reviewer's scenario, where the
	// competing update had already run to completion. Anchored on a command
	// every implementation issues, so this reproduces the bug against an
	// unconditional removal as well as demonstrating the fix.
	beforeCleanup updateCleanupRaceInjectionPoint = iota
	// insideCleanup lands it between the cleanup's read of the payload and
	// its removal — the window that stays open because Redis Cluster will not
	// let the two be one atomic command.
	insideCleanup
)

// updateCleanupRaceHook lands a complete, committed competing update — one
// that moved the response back into the conversation an in-flight update is
// about to prune it from — at one of those two points.
//
// The write goes through a genuinely separate client, as a real concurrent
// writer's would: reusing the client the hook is installed on makes the
// injected commands reentrant into that same hook chain, which go-redis does
// not handle as a simple nested call.
type updateCleanupRaceHook struct {
	at            updateCleanupRaceInjectionPoint
	payloadKey    string
	previousIndex string
	newIndex      string
	response      *responseapi.StoredResponse
	client        *redis.Client
	fired         atomic.Bool
}

func (h *updateCleanupRaceHook) DialHook(next redis.DialHook) redis.DialHook { return next }

// ProcessHook carries the beforeCleanup anchor: the index write that moves the
// response into its new conversation, which is the last thing an update does
// before cleaning up the old one, whatever that cleanup looks like.
func (h *updateCleanupRaceHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		err := next(ctx, cmd)
		if h.at != beforeCleanup || err != nil || !isIndexWriteFor(cmd, h.newIndex) || !h.fired.CompareAndSwap(false, true) {
			return err
		}
		h.inject()
		return err
	}
}

// ProcessPipelineHook carries the insideCleanup anchor: the cleanup's own read
// of the payload, the only pipelined GET of that key an update issues.
func (h *updateCleanupRaceHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		err := next(ctx, cmds)
		if h.at != insideCleanup || !readsKey(cmds, h.payloadKey) || !h.fired.CompareAndSwap(false, true) {
			return err
		}
		h.inject()
		return err
	}
}

// isIndexWriteFor reports whether cmd is conversationIndexAddScript aimed at
// indexKey. EVAL/EVALSHA lay the key out after the verb, the script (or its
// SHA), and the key count.
func isIndexWriteFor(cmd redis.Cmder, indexKey string) bool {
	if cmd.Name() != "eval" && cmd.Name() != "evalsha" {
		return false
	}
	args := cmd.Args()
	if len(args) < 4 {
		return false
	}
	key, ok := args[3].(string)
	return ok && key == indexKey
}

func readsKey(cmds []redis.Cmder, key string) bool {
	for _, cmd := range cmds {
		args := cmd.Args()
		if cmd.Name() != "get" || len(args) < 2 {
			continue
		}
		if got, ok := args[1].(string); ok && got == key {
			return true
		}
	}
	return false
}

// inject writes the payload first and the membership second, exactly as
// UpdateResponse orders them, so what lands is a competing update that has
// genuinely committed rather than a half-applied one.
func (h *updateCleanupRaceHook) inject() {
	ctx := context.Background()

	payload, err := json.Marshal(h.response)
	if err != nil {
		panic(err)
	}
	if setErr := h.client.Set(ctx, h.payloadKey, payload, time.Minute).Err(); setErr != nil {
		panic(setErr)
	}
	if zaddErr := h.client.ZAdd(ctx, h.previousIndex,
		redis.Z{Score: float64(h.response.CreatedAt), Member: h.response.ID}).Err(); zaddErr != nil {
		panic(zaddErr)
	}
}

func newUpdateCleanupRaceHook(t *testing.T, store *RedisStore, previousConversationID, newConversationID, responseID string, at updateCleanupRaceInjectionPoint) *updateCleanupRaceHook {
	t.Helper()

	writer := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	t.Cleanup(func() { _ = writer.Close() })

	return &updateCleanupRaceHook{
		at:            at,
		payloadKey:    store.buildKey(ResponseKeyPrefix + responseID),
		previousIndex: store.conversationIndexKey(previousConversationID),
		newIndex:      store.conversationIndexKey(newConversationID),
		response: &responseapi.StoredResponse{
			ID: responseID, ConversationID: previousConversationID,
			Status: "moved-back", CreatedAt: time.Now().Unix() + 1,
		},
		client: writer,
	}
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

	hook := newUpdateCleanupRaceHook(t, store, fromConversation, toConversation, responseID, beforeCleanup)
	store.client.AddHook(hook)

	moved := *original
	moved.ConversationID = toConversation
	require.NoError(t, store.UpdateResponse(background, &moved))
	require.True(t, hook.fired.Load(), "the competing update must actually have been injected for this to mean anything")

	assertResponseListedIn(t, store, fromConversation, responseID)
}

// TestUpdateResponseRestoresMembershipLostToCleanupRace covers the window the
// check above cannot close on its own. Redis Cluster forbids a
// compare-and-remove spanning the payload and index keys — they hash to
// different slots, so such a script is rejected with CROSSSLOT — which leaves
// the read and the removal as two round trips. A competing update landing
// between them is removed anyway, and only the re-read afterwards puts it
// back.
func TestUpdateResponseRestoresMembershipLostToCleanupRace(t *testing.T) {
	store := newConversationIndexStore(t)
	background := context.Background()

	const (
		fromConversation = "conv_aba_window_from"
		toConversation   = "conv_aba_window_to"
		responseID       = "resp_aba_window"
	)
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: fromConversation, Status: "original", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(background, original))
	require.NoError(t, store.ensureConversationIndexResolved(background, fromConversation))

	hook := newUpdateCleanupRaceHook(t, store, fromConversation, toConversation, responseID, insideCleanup)
	store.client.AddHook(hook)

	moved := *original
	moved.ConversationID = toConversation
	require.NoError(t, store.UpdateResponse(background, &moved))
	require.True(t, hook.fired.Load(), "the competing update must actually have been injected for this to mean anything")

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
