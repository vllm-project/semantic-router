package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// commandFailureHook injects err, at most once, in place of one specific
// single-key command (matched by name and its first key argument) — a
// one-shot transient failure, not a permanent block, since a hook installed
// for the rest of a test's life would also intercept that test's own later
// verification reads/writes against the same key. Covers both the
// non-pipelined path (ProcessHook: the real command is never even sent,
// since injecting a synthetic failure alongside a real write actually
// happening would make assertions about the resulting state meaningless)
// and the pipelined path (ProcessPipelineHook: the pipeline still runs for
// real — go-redis has no per-command skip within one Exec — and only the
// matched command's own result is overwritten afterward).
type commandFailureHook struct {
	name string
	key  string
	err  error
	used bool
}

func (h *commandFailureHook) matches(cmd redis.Cmder) bool {
	if h.used || cmd.Name() != h.name {
		return false
	}
	args := cmd.Args()
	if len(args) < 2 {
		return false
	}
	key, ok := args[1].(string)
	return ok && key == h.key
}

func (h *commandFailureHook) DialHook(next redis.DialHook) redis.DialHook { return next }

func (h *commandFailureHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if h.matches(cmd) {
			h.used = true
			cmd.SetErr(h.err)
			return h.err
		}
		return next(ctx, cmd)
	}
}

func (h *commandFailureHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		err := next(ctx, cmds)
		for _, cmd := range cmds {
			if h.matches(cmd) {
				h.used = true
				cmd.SetErr(h.err)
			}
		}
		return err
	}
}

// raceInjectingHook lets the real GET for targetKey execute and return its
// genuine value, then immediately overwrites the key with newValue — a
// real, wire-level simulation of "a concurrent write landed right after
// this read," rather than a mocked-out CAS conflict. Fires at most once.
type raceInjectingHook struct {
	targetKey string
	client    redis.UniversalClient
	newValue  []byte
	fired     bool
}

type recreateAfterDeleteHook struct {
	targetKey string
	indexKey  string
	response  *responseapi.StoredResponse
	client    *redis.Client
	fired     atomic.Bool
}

func (h *recreateAfterDeleteHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *recreateAfterDeleteHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}
func (h *recreateAfterDeleteHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		err := next(ctx, cmd)
		if err != nil || (cmd.Name() != "eval" && cmd.Name() != "evalsha") || !commandContainsArg(cmd, h.targetKey) || !h.fired.CompareAndSwap(false, true) {
			return err
		}
		payload, marshalErr := json.Marshal(h.response)
		if marshalErr != nil {
			panic(marshalErr)
		}
		if setErr := h.client.Set(context.Background(), h.targetKey, payload, time.Minute).Err(); setErr != nil {
			panic(setErr)
		}
		if zaddErr := h.client.ZAdd(context.Background(), h.indexKey,
			redis.Z{Score: float64(h.response.CreatedAt), Member: h.response.ID}).Err(); zaddErr != nil {
			panic(zaddErr)
		}
		return nil
	}
}

func commandContainsArg(cmd redis.Cmder, target string) bool {
	for _, arg := range cmd.Args() {
		if value, ok := arg.(string); ok && value == target {
			return true
		}
	}
	return false
}

func (h *raceInjectingHook) DialHook(next redis.DialHook) redis.DialHook { return next }

// matchesTarget reports whether cmd is the not-yet-fired GET this hook is
// waiting for.
func (h *raceInjectingHook) matchesTarget(cmd redis.Cmder) bool {
	if h.fired || cmd.Name() != "get" {
		return false
	}
	args := cmd.Args()
	if len(args) < 2 {
		return false
	}
	key, ok := args[1].(string)
	return ok && key == h.targetKey
}

// inject marks the race fired and overwrites targetKey with newValue.
func (h *raceInjectingHook) inject() {
	h.fired = true
	if setErr := h.client.Set(context.Background(), h.targetKey, h.newValue, 0).Err(); setErr != nil {
		panic(fmt.Sprintf("raceInjectingHook: failed to inject concurrent write: %v", setErr))
	}
}

func (h *raceInjectingHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		err := next(ctx, cmd)
		if h.matchesTarget(cmd) {
			h.inject()
		}
		return err
	}
}

func (h *raceInjectingHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		err := next(ctx, cmds)
		for _, cmd := range cmds {
			if h.matchesTarget(cmd) {
				h.inject()
				break
			}
		}
		return err
	}
}

// TestCascadeDeleteMissingPayloadPrunesIndexMember covers "missing payload
// -> stale index member pruned, not an error": a response whose payload
// expired or was otherwise deleted out from under its index entry must not
// block the rest of the cascade, and the conversation must still delete
// cleanly.
func TestCascadeDeleteMissingPayloadPrunesIndexMember(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_missing"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: convID, CreatedAt: time.Now().Unix()}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_missing", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))

	// Simulate the payload having already expired/been deleted while the
	// index entry survives (TTLs on the two keys are independent).
	require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+"resp_missing")).Err())
	require.Equal(t, []string{"resp_missing"}, conversationIndexMembers(t, store, convID),
		"precondition: the stale index member is still there")

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	_, err := store.GetConversation(ctx, convID)
	assert.ErrorIs(t, err, ErrNotFound)
}

// TestCascadeDeleteGetFailurePreservesAndReports covers "GET failed for a
// reason other than missing -> preserve payload+member, record error": a
// transient Redis-level GET failure (not redis.Nil) must be reported and
// leave both the payload and the index member untouched for a retry.
func TestCascadeDeleteGetFailurePreservesAndReports(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_get_failure"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: convID, CreatedAt: time.Now().Unix()}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_get_failure", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))
	// Resolve the conversation's migration proof before installing the
	// fault: StoreResponse indexes but never itself certifies exhaustiveness
	// (see indexResponse's doc comment), so the very first read/delete would
	// otherwise trigger its own legacy scan first — which also GETs every
	// response key, including this one, and would consume this one-shot
	// fault injection before deleteConversationResponseBatch's own fetch.
	require.NoError(t, store.ensureConversationIndexResolved(ctx, convID))

	responseKey := store.buildKey(ResponseKeyPrefix + "resp_get_failure")
	injectedErr := errors.New("injected transient GET failure")
	store.client.AddHook(&commandFailureHook{name: "get", key: responseKey, err: injectedErr})

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err)

	_, getErr := store.GetConversation(ctx, convID)
	assert.NoError(t, getErr, "the conversation record must survive a reported cascade failure")
	assert.Equal(t, []string{"resp_get_failure"}, conversationIndexMembers(t, store, convID))
	_, getErr = store.GetResponse(ctx, "resp_get_failure")
	assert.NoError(t, getErr, "the payload must be untouched by a failed GET")
}

// TestCascadeDeleteZremFailureReported covers "ZREM failed -> reported":
// candidate membership is removed before touching payloads, so a failure
// must preserve the payload and surface for retry.
func TestCascadeDeleteZremFailureReported(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_zrem_failure"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: convID, CreatedAt: time.Now().Unix()}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_zrem_failure", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))

	indexKey := store.conversationIndexKey(convID)
	injectedErr := errors.New("injected ZREM failure")
	store.client.AddHook(&commandFailureHook{name: "zrem", key: indexKey, err: injectedErr})

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	// Pre-removal failed, so payload and conversation both survive intact.
	_, getErr := store.GetResponse(ctx, "resp_zrem_failure")
	assert.NoError(t, getErr)
	_, getErr = store.GetConversation(ctx, convID)
	assert.NoError(t, getErr, "the conversation record must survive a reported ZREM failure")
}

// TestCascadeDeleteStaleMovedMemberPreservesNewOwnerPayload covers "stored
// ConversationID differs from conversationID -> preserve payload, prune
// only the stale target-index member": a response indexed under an old
// conversation, whose payload has since moved to a different conversation
// (e.g. because the old index's best-effort cleanup in UpdateResponse
// hadn't run, or failed), must never be deleted by a cascade of the old
// conversation — only the stale membership is pruned.
func TestCascadeDeleteStaleMovedMemberPreservesNewOwnerPayload(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const (
		oldConvID = "conv_cascade_old_owner"
		newConvID = "conv_cascade_new_owner"
		respID    = "resp_moved_owner"
	)
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: oldConvID, CreatedAt: time.Now().Unix()}))
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: newConvID, CreatedAt: time.Now().Unix()}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: respID, ConversationID: oldConvID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))

	// The response's payload moves to newConvID, but the old index's member
	// is left stale on purpose — simulating UpdateResponse's best-effort
	// unindex of the previous conversation having failed or not yet run.
	require.NoError(t, store.client.Set(ctx, store.buildKey(ResponseKeyPrefix+respID),
		mustMarshalResponse(t, &responseapi.StoredResponse{
			ID: respID, ConversationID: newConvID, Status: "completed", CreatedAt: time.Now().Unix(),
		}), store.ttl).Err())
	require.NoError(t, store.indexResponse(ctx, newConvID, respID, time.Now().Unix()))
	require.Equal(t, []string{respID}, conversationIndexMembers(t, store, oldConvID),
		"precondition: the stale member is still in the old conversation's index")

	require.NoError(t, store.DeleteConversation(ctx, oldConvID, true))

	// The old conversation is gone, and its stale index member with it, but
	// the response itself — now legitimately owned by newConvID — survives.
	_, err := store.GetConversation(ctx, oldConvID)
	assert.ErrorIs(t, err, ErrNotFound)
	stored, err := store.GetResponse(ctx, respID)
	require.NoError(t, err, "a response that moved to a different conversation must survive the old conversation's cascade")
	assert.Equal(t, newConvID, stored.ConversationID)
	assert.Equal(t, []string{respID}, conversationIndexMembers(t, store, newConvID),
		"the new conversation's own index entry must be untouched")
}

// TestCascadeDeleteConcurrentUpdatePreservesNewerPayload covers "update
// between ownership GET and CAS delete preserves updated payload" and
// "retry after partial failure completes safely": a concurrent write
// landing in the narrow window between deleteConversationResponseBatch's
// ownership-verifying GET and its compare-delete must survive that first
// cascade attempt (CAS conflict, reported, retryable) — and a subsequent
// retry, with no further interference, must complete cleanly.
func TestCascadeDeleteConcurrentUpdatePreservesNewerPayload(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_concurrent_update"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: convID, CreatedAt: time.Now().Unix()}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_concurrent_update", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))
	// Resolve the conversation's migration proof before injecting the race
	// (see the identical comment in TestCascadeDeleteGetFailurePreservesAndReports):
	// otherwise DeleteConversation's own first-touch legacy scan would GET
	// this same key before deleteConversationResponseBatch's fetch does,
	// firing the one-shot race injection too early.
	require.NoError(t, store.ensureConversationIndexResolved(ctx, convID))

	responseKey := store.buildKey(ResponseKeyPrefix + "resp_concurrent_update")
	updatedPayload := mustMarshalResponse(t, &responseapi.StoredResponse{
		ID: "resp_concurrent_update", ConversationID: convID, Status: "completed",
		CreatedAt: time.Now().Unix(), Model: "concurrently-updated",
	})
	// A genuinely separate client/connection injects the concurrent write —
	// a real concurrent writer would never share the connection whose
	// in-flight command this race targets, and reusing store.client here
	// (the very client the hook is installed on) makes the injected Set
	// reentrant into that same client's hook chain, which go-redis does not
	// handle as a simple nested call.
	concurrentWriter := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	t.Cleanup(func() { _ = concurrentWriter.Close() })
	hook := &raceInjectingHook{targetKey: responseKey, client: concurrentWriter, newValue: updatedPayload}
	store.client.AddHook(hook)

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err, "the first attempt must see a CAS conflict against the concurrently-updated payload")
	assert.True(t, hook.fired, "the race must actually have been injected for this assertion to be meaningful")

	_, getErr := store.GetConversation(ctx, convID)
	assert.NoError(t, getErr, "the conversation record must survive a reported CAS conflict")
	stored, getErr := store.GetResponse(ctx, "resp_concurrent_update")
	require.NoError(t, getErr, "the concurrently-updated payload must survive the failed cascade attempt")
	assert.Equal(t, "concurrently-updated", stored.Model)
	assert.Equal(t, []string{"resp_concurrent_update"}, conversationIndexMembers(t, store, convID))

	// Retry, with no further interference: the now-stable payload (still
	// owned by convID) deletes cleanly.
	require.NoError(t, store.DeleteConversation(ctx, convID, true))
	_, getErr = store.GetConversation(ctx, convID)
	assert.ErrorIs(t, getErr, ErrNotFound)
	_, getErr = store.GetResponse(ctx, "resp_concurrent_update")
	assert.ErrorIs(t, getErr, ErrNotFound)
}

// TestCascadeDeleteDoesNotEraseRecreatedMembership covers the ordering race
// between payload deletion and index cleanup. A writer that recreates and
// indexes the response immediately after CAS deletion must not have that new
// membership removed by a later ZREM from the old cascade attempt.
func TestCascadeDeleteDoesNotEraseRecreatedMembership(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const (
		conversationID = "conv_cascade_recreated"
		responseID     = "resp_cascade_recreated"
	)
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: conversationID, CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: responseID, ConversationID: conversationID, Status: "original", CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.ensureConversationIndexResolved(ctx, conversationID))

	writer := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	t.Cleanup(func() { _ = writer.Close() })
	hook := &recreateAfterDeleteHook{
		targetKey: store.buildKey(ResponseKeyPrefix + responseID),
		indexKey:  store.conversationIndexKey(conversationID),
		response: &responseapi.StoredResponse{
			ID: responseID, ConversationID: conversationID, Status: "recreated", CreatedAt: time.Now().Unix() + 1,
		},
		client: writer,
	}
	store.client.AddHook(hook)

	require.NoError(t, store.DeleteConversation(ctx, conversationID, true))
	assert.True(t, hook.fired.Load(), "the response must have been recreated in the target race window")
	_, err := store.GetResponse(ctx, responseID)
	assert.ErrorIs(t, err, ErrNotFound, "the recreated indexed response must be observed by the next cascade batch")
}

// TestCascadeDeleteClusterCrossSlotSafe covers the Cluster-safety
// requirement directly, against a real (single-node) Redis Cluster: a
// conversation whose response IDs are spread across many different hash
// slots must still cascade-delete cleanly, since every command
// deleteConversationResponseBatch issues is single-key
// (fetchResponsePayloadsPipelined's independent pipelined GETs,
// compareDeleteResponsePayload's single-key Lua script, and ZREM against
// one index key) — never a cross-slot command that a real cluster would
// reject with CROSSSLOT.
func TestCascadeDeleteClusterCrossSlotSafe(t *testing.T) {
	store := newConversationIndexClusterStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_cluster_crossslot"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: convID, CreatedAt: time.Now().Unix()}))

	const responseCount = 40
	ids := make([]string, responseCount)
	for i := 0; i < responseCount; i++ {
		id := fmt.Sprintf("resp_crossslot_%d_%d", i, time.Now().UnixNano())
		ids[i] = id
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID: id, ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix() + int64(i),
		}))
	}

	require.NoError(t, store.DeleteConversation(ctx, convID, true), "cascade delete must not fail with CROSSSLOT")

	for _, id := range ids {
		_, err := store.GetResponse(ctx, id)
		assert.ErrorIsf(t, err, ErrNotFound, "response %s should have been deleted", id)
	}
	_, err := store.GetConversation(ctx, convID)
	assert.ErrorIs(t, err, ErrNotFound)
}
