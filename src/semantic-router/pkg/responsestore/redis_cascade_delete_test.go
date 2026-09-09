package responsestore

import (
	"context"
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

// scriptFailureHook fails one warmed EVALSHA by script identity. Warming the
// script first avoids mistaking a harmless NOSCRIPT probe for the operation
// whose failure the test intends to inject.
type scriptFailureHook struct {
	hash string
	err  error
	used atomic.Bool
}

func (h *scriptFailureHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *scriptFailureHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *scriptFailureHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		args := cmd.Args()
		if cmd.Name() == "evalsha" && len(args) > 1 && args[1] == h.hash && h.used.CompareAndSwap(false, true) {
			return h.err
		}
		return next(ctx, cmd)
	}
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

func commandContainsArg(cmd redis.Cmder, target string) bool {
	for _, arg := range cmd.Args() {
		if value, ok := arg.(string); ok && value == target {
			return true
		}
	}
	return false
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

// TestCascadeDeleteUnindexFailureReported covers a failure of the atomic
// conditional ZREM+HDEL after payload deletion. The stale witness remains a
// retry anchor, so the failure is safe and recoverable.
func TestCascadeDeleteUnindexFailureReported(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_zrem_failure"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{ID: convID, CreatedAt: time.Now().Unix()}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_zrem_failure", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))

	require.NoError(t, store.ensureConversationIndexResolved(ctx, convID))
	// Warm the conditional script so the injected EVALSHA is the actual
	// unindex attempt, not a NOSCRIPT probe.
	require.NoError(t, store.unindexResponseGenerations(ctx, convID,
		responseGenerationWitness{responseID: "not-present", generation: newResponseGeneration()}))
	injectedErr := errors.New("injected conditional unindex failure")
	store.client.AddHook(&scriptFailureHook{hash: conditionalUnindexScript.Hash(), err: injectedErr})

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	// The payload CAS committed before unindex failed. The membership remains
	// deliberately, allowing the next cascade attempt to prune it as missing.
	_, getErr := store.GetResponse(ctx, "resp_zrem_failure")
	assert.ErrorIs(t, getErr, ErrNotFound)
	assert.Equal(t, []string{"resp_zrem_failure"}, conversationIndexMembers(t, store, convID))
	_, getErr = store.GetConversation(ctx, convID)
	assert.NoError(t, getErr, "the conversation record must survive a reported unindex failure")
	require.NoError(t, store.DeleteConversation(ctx, convID, true))
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
	movedPayload, movedGeneration := mustMarshalGeneratedResponse(t, &responseapi.StoredResponse{
		ID: respID, ConversationID: newConvID, Status: "completed", CreatedAt: time.Now().Unix(),
	})
	require.NoError(t, store.client.Set(ctx, store.buildKey(ResponseKeyPrefix+respID), movedPayload, store.ttl).Err())
	require.NoError(t, store.indexResponse(ctx, newConvID, respID, movedGeneration, time.Now().Unix(), store.ttlMillis()))
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
	updated := &responseapi.StoredResponse{
		ID: "resp_concurrent_update", ConversationID: convID, Status: "completed",
		CreatedAt: time.Now().Unix(), Model: "concurrently-updated",
	}
	writer := newConcurrentRedisStore(t, store)
	var injectedErr error
	hook := &commandInterleavingHook{
		pipeline: true,
		match:    func(cmd redis.Cmder) bool { return commandReadsKey(cmd, responseKey) },
		inject:   func() { injectedErr = writer.UpdateResponse(context.Background(), updated) },
	}
	store.client.AddHook(hook)

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err, "the first attempt must see a CAS conflict against the concurrently-updated payload")
	assert.True(t, hook.fired.Load(), "the race must actually have been injected for this assertion to be meaningful")
	require.NoError(t, injectedErr)

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

	writer := newConcurrentRedisStore(t, store)
	recreated := &responseapi.StoredResponse{
		ID: responseID, ConversationID: conversationID, Status: "recreated", CreatedAt: time.Now().Unix() + 1,
	}
	var injectedErr error
	hook := &commandInterleavingHook{
		match: func(cmd redis.Cmder) bool {
			return commandRunsScriptOnKey(cmd, store.buildKey(ResponseKeyPrefix+responseID))
		},
		inject: func() { injectedErr = writer.StoreResponse(context.Background(), recreated) },
	}
	store.client.AddHook(hook)

	require.NoError(t, store.DeleteConversation(ctx, conversationID, true))
	assert.True(t, hook.fired.Load(), "the response must have been recreated in the target race window")
	require.NoError(t, injectedErr)
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

// TestCascadeDeleteCancellationKeepsRetryWitness cancels after payload CAS but
// before conditional unindex. The stale witness must remain so a retry can
// observe the missing payload and finish cleanup.
func TestCascadeDeleteCancellationKeepsRetryWitness(t *testing.T) {
	store := newConversationIndexStore(t)
	baseCtx := context.Background()

	const convID = "conv_cascade_cancelled"
	const respID = "resp_cascade_cancelled"
	require.NoError(t, store.CreateConversation(baseCtx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.StoreResponse(baseCtx, &responseapi.StoredResponse{
		ID: respID, ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))
	// Resolve the proof first so the hook targets the cascade payload CAS.
	require.NoError(t, store.ensureConversationIndexResolved(baseCtx, convID))

	ctx, cancel := context.WithCancel(baseCtx)
	defer cancel()

	hook := &commandInterleavingHook{
		match: func(cmd redis.Cmder) bool {
			return commandRunsScriptOnKey(cmd, store.buildKey(ResponseKeyPrefix+respID))
		},
		inject: cancel,
	}
	store.client.AddHook(hook)

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err, "a cancelled cascade must report failure, not silently succeed")

	assert.True(t, hook.fired.Load())
	assert.Equal(t, []string{respID}, conversationIndexMembers(t, store, convID),
		"a cancelled conditional unindex must retain the stale retry witness")
	assert.Zero(t, exists(t, store, store.buildKey(ResponseKeyPrefix+respID)),
		"the payload generation was safely deleted before cancellation")

	// The conversation record survived as the retry anchor, and a retry on a
	// live context completes the cascade for real rather than reporting
	// success over an emptied index.
	_, getErr := store.GetConversation(baseCtx, convID)
	require.NoError(t, getErr)

	require.NoError(t, store.DeleteConversation(baseCtx, convID, true))
	_, getErr = store.GetResponse(baseCtx, respID)
	assert.ErrorIs(t, getErr, ErrNotFound, "the retry must actually delete the payload, not orphan it")
}

// cascadeRaceHook lands a complete, indexed response for conversationID the
// moment the cascade has read its index empty — the window between that read
// and whatever the cascade does about it.
//
// Anchoring on the empty read rather than on the delete that follows is
// deliberate: it is the window the race actually lives in, and it exists in
// any implementation, so this same hook reproduces the orphan against an
// unconditional delete and shows it fixed against a guarded one.
//
// The write goes through a genuinely separate client, as a real concurrent
// writer's would: reusing the client the hook is installed on makes the
// injected commands reentrant into that same hook chain, which go-redis does
// not handle as a simple nested call.
type cascadeRaceHook struct {
	indexKey string
	response *responseapi.StoredResponse
	writer   *RedisStore
	always   bool
	fired    atomic.Int64
}

func (h *cascadeRaceHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *cascadeRaceHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *cascadeRaceHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		err := next(ctx, cmd)
		if err != nil || !h.emptiedIndexRead(cmd) {
			return err
		}
		if !h.always && h.fired.Load() > 0 {
			return err
		}
		h.fired.Add(1)

		if storeErr := h.writer.StoreResponse(context.Background(), h.response); storeErr != nil {
			panic(storeErr)
		}
		return err
	}
}

// emptiedIndexRead reports whether cmd is the cascade's generation-snapshotted
// candidate script, come back empty.
func (h *cascadeRaceHook) emptiedIndexRead(cmd redis.Cmder) bool {
	if cmd.Name() != "evalsha" {
		return false
	}
	args := cmd.Args()
	if len(args) < 4 || args[1] != readCascadeCandidatesScript.Hash() {
		return false
	}
	key, keyOK := args[3].(string)
	result, resultOK := cmd.(*redis.Cmd)
	if !keyOK || key != h.indexKey || !resultOK {
		return false
	}
	items, ok := result.Val().([]interface{})
	return ok && len(items) == 0
}

func newCascadeRaceHook(t *testing.T, store *RedisStore, conversationID, responseID string, always bool) *cascadeRaceHook {
	t.Helper()

	// Populate Redis's script cache before installing the hook, so the hook
	// observes the successful EVALSHA result rather than a NOSCRIPT probe.
	_, err := store.readCascadeCandidates(context.Background(), conversationID)
	require.NoError(t, err)
	writer := newConcurrentRedisStore(t, store)

	return &cascadeRaceHook{
		indexKey: store.conversationIndexKey(conversationID),
		response: &responseapi.StoredResponse{
			ID: responseID, ConversationID: conversationID,
			Status: "completed", CreatedAt: time.Now().Unix() + 1,
		},
		writer: writer,
		always: always,
		fired:  atomic.Int64{},
	}
}

// TestCascadeDeleteDrainsWriteCommittedAfterFinalEmptyRead is the regression
// for the cascade's last step. Reading the index empty and then deleting it
// are two separate operations, and a StoreResponse that commits in between has
// written a live payload and indexed it. Deleting the index key
// unconditionally erases that membership, and since DeleteConversation goes on
// to remove the conversation record too, nothing is left pointing at the
// payload — permanently, once the store is finalized and no read ever rescans.
//
// Deterministic by construction: the racing write is injected in the instant
// before the delete command reaches Redis, which is precisely the window that
// cannot be hit reliably by timing.
func TestCascadeDeleteDrainsWriteCommittedAfterFinalEmptyRead(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const (
		conversationID = "conv_cascade_final_race"
		originalID     = "resp_cascade_original"
		racingID       = "resp_cascade_racer"
	)
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: conversationID, CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: originalID, ConversationID: conversationID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.ensureConversationIndexResolved(ctx, conversationID))

	hook := newCascadeRaceHook(t, store, conversationID, racingID, false)
	store.client.AddHook(hook)

	require.NoError(t, store.DeleteConversation(ctx, conversationID, true))
	require.EqualValues(t, 1, hook.fired.Load(), "the race must actually have been injected for this assertion to mean anything")

	_, err := store.GetResponse(ctx, racingID)
	assert.ErrorIs(t, err, ErrNotFound,
		"a response committed after the cascade's final empty read must be deleted with the rest, never left behind with its index entry erased")
	assert.Empty(t, conversationIndexMembers(t, store, conversationID))
	assert.Zero(t, exists(t, store, store.conversationIndexKey(conversationID)))
}

// TestCascadeDeleteReportsUnendingConcurrentWrites covers the other end of the
// same loop: a conversation being written to faster than it can be drained
// must be reported rather than either spun on forever or quietly reported as a
// successful delete. What it must never do is leave a payload no index names —
// the retry has to have something to find.
func TestCascadeDeleteReportsUnendingConcurrentWrites(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const (
		conversationID = "conv_cascade_unending"
		racingID       = "resp_cascade_unending"
	)
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: conversationID, CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.ensureConversationIndexResolved(ctx, conversationID))

	hook := newCascadeRaceHook(t, store, conversationID, racingID, true)
	store.client.AddHook(hook)

	err := store.DeleteConversation(ctx, conversationID, true)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "kept receiving responses during cascade delete")
	assert.EqualValues(t, conversationIndexCascadeMaxRaceRounds+1, hook.fired.Load(),
		"the cascade must give up after a bounded number of racing writes, not spin on them")

	_, getErr := store.GetConversation(ctx, conversationID)
	assert.NoError(t, getErr, "the conversation record must survive a reported cascade failure, as the anchor for the retry")
	assert.Equal(t, []string{racingID}, conversationIndexMembers(t, store, conversationID),
		"the response that outran the cascade must still be indexed, so a retry can find and delete it")
}
