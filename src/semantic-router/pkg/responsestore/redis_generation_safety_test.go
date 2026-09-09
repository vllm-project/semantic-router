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

func TestResponseGenerationUsesFlatBackwardCompatibleJSON(t *testing.T) {
	response := &responseapi.StoredResponse{
		ID: "resp_flat_generation", ConversationID: "conv_flat_generation", Status: "completed", CreatedAt: 123,
	}
	payload, generation := mustMarshalGeneratedResponse(t, response)

	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(payload, &fields))
	assert.Contains(t, fields, "id")
	assert.Contains(t, fields, responseGenerationField)
	assert.NotContains(t, fields, "response", "the persisted format must not introduce a wrapper")

	var oldReader responseapi.StoredResponse
	require.NoError(t, json.Unmarshal(payload, &oldReader))
	assert.Equal(t, *response, oldReader, "older readers must ignore the additive generation field")

	legacy, err := decodeResponseRecord(mustMarshalResponse(t, response))
	require.NoError(t, err)
	assert.Empty(t, legacy.generation)
	assert.Equal(t, *response, *legacy.response)
	assert.NotEmpty(t, generation)
}

// commandInterleavingHook runs inject at a precise Redis command boundary.
// A separate client performs the injected write, matching a real concurrent
// caller without recursively entering the hooked client's command chain.
type commandInterleavingHook struct {
	before   bool
	pipeline bool
	match    func(redis.Cmder) bool
	inject   func()
	fired    atomic.Bool
}

func (h *commandInterleavingHook) DialHook(next redis.DialHook) redis.DialHook { return next }

func (h *commandInterleavingHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		candidate := !h.pipeline && h.match(cmd)
		if candidate && h.before && h.fired.CompareAndSwap(false, true) {
			h.inject()
		}
		err := next(ctx, cmd)
		if candidate && !h.before && err == nil && h.fired.CompareAndSwap(false, true) {
			h.inject()
		}
		return err
	}
}

func (h *commandInterleavingHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		matched := false
		if h.pipeline && !h.fired.Load() {
			for _, cmd := range cmds {
				if h.match(cmd) {
					matched = h.fired.CompareAndSwap(false, true)
					break
				}
			}
		}
		if matched && h.before {
			h.inject()
		}
		err := next(ctx, cmds)
		if matched && !h.before {
			h.inject()
		}
		return err
	}
}

func newConcurrentRedisStore(t *testing.T, store *RedisStore) *RedisStore {
	t.Helper()
	client := redis.NewClient(&redis.Options{
		Addr:     store.config.Address,
		Password: store.config.Password,
		DB:       store.config.DB,
	})
	t.Cleanup(func() { _ = client.Close() })
	return &RedisStore{
		client:    client,
		config:    store.config,
		keyPrefix: store.keyPrefix,
		ttl:       store.ttl,
		enabled:   true,
	}
}

func commandReadsKey(cmd redis.Cmder, key string) bool {
	args := cmd.Args()
	if cmd.Name() != "get" || len(args) < 2 {
		return false
	}
	got, ok := args[1].(string)
	return ok && got == key
}

func commandRunsScriptOnKey(cmd redis.Cmder, key string) bool {
	return (cmd.Name() == "eval" || cmd.Name() == "evalsha") && commandContainsArg(cmd, key)
}

func indexedGeneration(t *testing.T, store *RedisStore, conversationID, responseID string) string {
	t.Helper()
	generation, err := store.client.HGet(
		context.Background(), store.conversationIndexGenerationKey(conversationID), responseID,
	).Result()
	require.NoError(t, err)
	return generation
}

// TestListPruneDoesNotRemoveRecreatedGeneration deterministically lands a
// complete StoreResponse after list's payload GET observed the old generation
// missing but before its conditional ZREM+HDEL. The stale prune must not erase
// the recreated generation's membership.
func TestListPruneDoesNotRemoveRecreatedGeneration(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	ctx := context.Background()

	const conversationID = "conv_list_prune_recreate"
	const responseID = "resp_list_prune_recreate"
	old := &responseapi.StoredResponse{
		ID: responseID, ConversationID: conversationID, Status: "old", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, old))
	require.NoError(t, store.ensureConversationIndexResolved(ctx, conversationID))
	require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+responseID)).Err())

	recreated := *old
	recreated.Status = "recreated"
	recreated.CreatedAt++
	var injectedErr error
	hook := &commandInterleavingHook{
		pipeline: true,
		match: func(cmd redis.Cmder) bool {
			return commandReadsKey(cmd, store.buildKey(ResponseKeyPrefix+responseID))
		},
		inject: func() { injectedErr = writer.StoreResponse(context.Background(), &recreated) },
	}
	store.client.AddHook(hook)

	first, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{})
	require.NoError(t, err)
	require.NoError(t, injectedErr)
	assert.True(t, hook.fired.Load(), "the recreation must land in the GET-to-prune window")
	assert.Empty(t, first, "the in-flight page is allowed to reflect its older payload snapshot")
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, conversationID))

	raw, err := store.client.Get(ctx, store.buildKey(ResponseKeyPrefix+responseID)).Bytes()
	require.NoError(t, err)
	record, err := decodeResponseRecord(raw)
	require.NoError(t, err)
	assert.Equal(t, "recreated", record.response.Status)
	assert.Equal(t, record.generation, indexedGeneration(t, store, conversationID, responseID))

	second, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{})
	require.NoError(t, err)
	require.Len(t, second, 1)
	assert.Equal(t, "recreated", second[0].Status)
}

// TestListPruneDoesNotRemoveMovedBackGeneration covers the other list-prune
// predicate. The window observes a stale A/G1 witness and a payload in B/G2;
// before prune, a complete B->A update installs G3. Conditional cleanup of G1
// must not erase G3 after this move-away-and-back ABA.
func TestListPruneDoesNotRemoveMovedBackGeneration(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	ctx := context.Background()

	const conversationA = "conv_list_move_a"
	const conversationB = "conv_list_move_b"
	const responseID = "resp_list_move_back"
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: conversationA, Status: "in-a", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, original))
	require.NoError(t, store.ensureConversationIndexResolved(ctx, conversationA))
	oldGeneration := indexedGeneration(t, store, conversationA, responseID)

	moved := *original
	moved.ConversationID = conversationB
	moved.Status = "in-b"
	require.NoError(t, writer.UpdateResponse(ctx, &moved))
	// Recreate the stale A/G1 membership a delayed pre-sidecar cleanup would
	// have left. The current payload is B/G2.
	require.NoError(t, store.indexResponse(
		ctx, conversationA, responseID, oldGeneration, original.CreatedAt, store.ttlMillis(),
	))

	movedBack := moved
	movedBack.ConversationID = conversationA
	movedBack.Status = "back-in-a"
	movedBack.CreatedAt++
	var injectedErr error
	hook := &commandInterleavingHook{
		pipeline: true,
		match: func(cmd redis.Cmder) bool {
			return commandReadsKey(cmd, store.buildKey(ResponseKeyPrefix+responseID))
		},
		inject: func() { injectedErr = writer.UpdateResponse(context.Background(), &movedBack) },
	}
	store.client.AddHook(hook)

	first, err := store.ListResponsesByConversation(ctx, conversationA, ListOptions{})
	require.NoError(t, err)
	require.NoError(t, injectedErr)
	assert.True(t, hook.fired.Load(), "the move-back must land in the GET-to-prune window")
	assert.Empty(t, first, "the in-flight page may reflect the older B/G2 payload snapshot")
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, conversationA))

	raw, err := store.client.Get(ctx, store.buildKey(ResponseKeyPrefix+responseID)).Bytes()
	require.NoError(t, err)
	record, err := decodeResponseRecord(raw)
	require.NoError(t, err)
	assert.Equal(t, "back-in-a", record.response.Status)
	assert.NotEqual(t, oldGeneration, record.generation)
	assert.Equal(t, record.generation, indexedGeneration(t, store, conversationA, responseID))

	second, err := store.ListResponsesByConversation(ctx, conversationA, ListOptions{})
	require.NoError(t, err)
	require.Len(t, second, 1)
	assert.Equal(t, "back-in-a", second[0].Status)
}

// TestDeleteResponseDoesNotUnindexConcurrentStore deterministically recreates
// the same response ID after atomic take deleted G1 but before DeleteResponse
// conditionally unindexes G1. G2 must remain both present and indexed.
func TestDeleteResponseDoesNotUnindexConcurrentStore(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	ctx := context.Background()

	const conversationID = "conv_delete_recreate"
	const responseID = "resp_delete_recreate"
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: conversationID, Status: "original", CreatedAt: time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, original))

	recreated := *original
	recreated.Status = "recreated"
	recreated.CreatedAt++
	var injectedErr error
	hook := &commandInterleavingHook{
		match: func(cmd redis.Cmder) bool {
			return commandRunsScriptOnKey(cmd, store.buildKey(ResponseKeyPrefix+responseID))
		},
		inject: func() { injectedErr = writer.StoreResponse(context.Background(), &recreated) },
	}
	store.client.AddHook(hook)

	require.NoError(t, store.DeleteResponse(ctx, responseID))
	require.NoError(t, injectedErr)
	assert.True(t, hook.fired.Load(), "the StoreResponse must land between atomic take and unindex")

	stored, err := store.GetResponse(ctx, responseID)
	require.NoError(t, err)
	assert.Equal(t, "recreated", stored.Status)
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, conversationID))

	raw, err := store.client.Get(ctx, store.buildKey(ResponseKeyPrefix+responseID)).Bytes()
	require.NoError(t, err)
	record, err := decodeResponseRecord(raw)
	require.NoError(t, err)
	assert.Equal(t, record.generation, indexedGeneration(t, store, conversationID, responseID))
}

// TestRollbackGenerationCASRejectsByteIdenticalConcurrentWrite proves a
// successful concurrent update with the same public JSON fields receives a
// distinct generation and cannot be overwritten by a stale rollback.
func TestRollbackGenerationCASRejectsByteIdenticalConcurrentWrite(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	ctx := context.Background()

	const responseID = "resp_rollback_identical"
	key := store.buildKey(ResponseKeyPrefix + responseID)
	original := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_rollback_original", Status: "original", CreatedAt: time.Now().Unix(),
	}
	originalData, _ := mustMarshalGeneratedResponse(t, original)
	require.NoError(t, store.client.Set(ctx, key, originalData, store.ttl).Err())

	failed := &responseapi.StoredResponse{
		ID: responseID, ConversationID: "conv_rollback_identical", Status: "same-bytes", CreatedAt: time.Now().Unix() + 1,
	}
	failedData, failedGeneration := mustMarshalGeneratedResponse(t, failed)
	snapshot, err := store.replaceResponseAndSnapshot(ctx, key, responseID, failedData)
	require.NoError(t, err)

	var injectedErr error
	hook := &commandInterleavingHook{
		before: true,
		match: func(cmd redis.Cmder) bool {
			return commandRunsScriptOnKey(cmd, key)
		},
		inject: func() { injectedErr = writer.UpdateResponse(context.Background(), failed) },
	}
	store.client.AddHook(hook)

	injectedIndexErr := assert.AnError
	err = store.rollbackUpdatePayload(ctx, key, responseID, failedGeneration, snapshot, injectedIndexErr)
	require.ErrorIs(t, err, injectedIndexErr)
	require.NoError(t, injectedErr)
	assert.True(t, hook.fired.Load(), "the byte-identical update must land before rollback CAS")

	raw, err := store.client.Get(ctx, key).Bytes()
	require.NoError(t, err)
	record, err := decodeResponseRecord(raw)
	require.NoError(t, err)
	assert.Equal(t, *failed, *record.response)
	assert.NotEqual(t, failedGeneration, record.generation)
	assert.Empty(t, conversationIndexMembers(t, store, original.ConversationID),
		"a rejected stale rollback must not reindex its predecessor")
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, failed.ConversationID))
	assert.Equal(t, record.generation, indexedGeneration(t, store, failed.ConversationID, responseID))
}
