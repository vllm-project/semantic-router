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
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// afterCommandHook runs after() once, immediately after the named command
// has actually committed at Redis — the seam for asserting on what happens
// in the window *between* two commands a single operation issues (e.g. a
// context cancelled after a cascade batch's ZREM has already landed).
type afterCommandHook struct {
	name  string
	after func()
	used  atomic.Bool
}

func (h *afterCommandHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *afterCommandHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *afterCommandHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		err := next(ctx, cmd)
		if cmd.Name() == h.name && h.used.CompareAndSwap(false, true) {
			h.after()
		}
		return err
	}
}

type beforeCommandHook struct {
	name   string
	once   bool
	before func()
	used   atomic.Bool
}

func (h *beforeCommandHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *beforeCommandHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if cmd.Name() == h.name && (!h.once || h.used.CompareAndSwap(false, true)) {
			h.before()
		}
		return next(ctx, cmd)
	}
}

func (h *beforeCommandHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

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
	used atomic.Bool
}

func (h *commandFailureHook) matches(cmd redis.Cmder) bool {
	if cmd.Name() != h.name {
		return false
	}
	args := cmd.Args()
	if len(args) < 2 {
		return false
	}
	key, ok := args[1].(string)
	return ok && key == h.key && h.used.CompareAndSwap(false, true)
}

func (h *commandFailureHook) DialHook(next redis.DialHook) redis.DialHook { return next }

func (h *commandFailureHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if h.matches(cmd) {
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
				cmd.SetErr(h.err)
			}
		}
		return err
	}
}

// pttlValueHook rewrites one key's pipelined PTTL reply to a fixed duration,
// at most once. Redis will not produce a key's final millisecond on demand, so
// the boundary where a live payload answers PTTL 0 has to be injected rather
// than waited for. Only the reply is rewritten; the payload itself keeps the
// long retention the test gave it.
type pttlValueHook struct {
	key   string
	value time.Duration
	used  atomic.Bool
}

func (h *pttlValueHook) DialHook(next redis.DialHook) redis.DialHook          { return next }
func (h *pttlValueHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook { return next }

func (h *pttlValueHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return func(ctx context.Context, cmds []redis.Cmder) error {
		err := next(ctx, cmds)
		for _, cmd := range cmds {
			ttlCmd, isDuration := cmd.(*redis.DurationCmd)
			if !isDuration || cmd.Name() != "pttl" {
				continue
			}
			args := cmd.Args()
			if len(args) < 2 {
				continue
			}
			if key, ok := args[1].(string); ok && key == h.key && h.used.CompareAndSwap(false, true) {
				ttlCmd.SetVal(h.value)
			}
		}
		return err
	}
}

// conversationIndexMembers reads the index directly, so tests can assert on it
// and not only on what a listing happens to return.
func conversationIndexMembers(t *testing.T, store *RedisStore, conversationID string) []string {
	t.Helper()

	members, err := store.client.ZRange(context.Background(), store.conversationIndexKey(conversationID), 0, -1).Result()
	require.NoError(t, err)
	return members
}

// newConversationIndexStore scopes the store to a key prefix unique to this run,
// so it cannot collide with the other suites sharing DB 0. Skips without Redis.
//
// Takes testing.TB rather than *testing.T so benchmarks can use it too — see
// BenchmarkListResponsesByConversation, which needs a scoped, self-cleaning
// store just as much as any test does.
func newConversationIndexStore(tb testing.TB) *RedisStore {
	tb.Helper()
	return newConversationIndexStoreWithTTLSeconds(tb, 300)
}

// newConversationIndexStoreWithTTLSeconds is newConversationIndexStore with
// an explicit data-retention TTL, for tests asserting behavior that depends
// on how the configured TTL compares to a fixed internal bound (e.g. the
// empty-marker TTL cap).
func newConversationIndexStoreWithTTLSeconds(tb testing.TB, ttlSeconds int) *RedisStore {
	tb.Helper()

	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  ttlSeconds,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address:   "localhost:6379",
			DB:        0,
			KeyPrefix: fmt.Sprintf("srtest:%d:", time.Now().UnixNano()),
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		tb.Skipf("Redis not available: %v", err)
	}

	tb.Cleanup(func() {
		deleteScopedKeys(store)
		_ = store.Close()
	})

	return store
}

// deleteScopedKeys removes everything under a store's own key prefix, in
// pipelined batches rather than one round trip per key: a benchmark seeded
// with a hundred thousand responses would otherwise spend far longer being
// cleaned up than measured.
func deleteScopedKeys(store *RedisStore) {
	ctx := context.Background()
	pipe := store.client.Pipeline()
	queued := 0

	iter := store.client.Scan(ctx, 0, store.buildKey("*"), 0).Iterator()
	for iter.Next(ctx) {
		pipe.Del(ctx, iter.Val())
		queued++
		if queued >= redisCleanupBatchSize {
			_, _ = pipe.Exec(ctx)
			queued = 0
		}
	}
	if queued > 0 {
		_, _ = pipe.Exec(ctx)
	}
}

// redisCleanupBatchSize bounds how many DELs one cleanup pipeline carries.
const redisCleanupBatchSize = 500

// directSetResponsePayload writes a response payload straight to Redis,
// bypassing StoreResponse and its indexing entirely — the same shape as data
// written before this index existed, or as an in-flight writer that landed
// its payload but has not indexed it yet.
func directSetResponsePayload(t *testing.T, store *RedisStore, response *responseapi.StoredResponse) {
	t.Helper()

	data, err := json.Marshal(response)
	require.NoError(t, err)

	key := store.buildKey(ResponseKeyPrefix + response.ID)
	require.NoError(t, store.client.Set(context.Background(), key, data, store.ttl).Err())
}

// directSetResponsePayloadWithTTL is directSetResponsePayload with an
// explicit payload lifetime, for tests that need the payload to outlive the
// store's own (deliberately tiny) data TTL — e.g. proving what happens once
// a proof capped by that TTL expires, while the responses it described are
// still there to be rediscovered.
func directSetResponsePayloadWithTTL(t *testing.T, store *RedisStore, response *responseapi.StoredResponse, ttl time.Duration) {
	t.Helper()

	data, err := json.Marshal(response)
	require.NoError(t, err)

	key := store.buildKey(ResponseKeyPrefix + response.ID)
	require.NoError(t, store.client.Set(context.Background(), key, data, ttl).Err())
}

// directSetGeneratedResponsePayloadWithTTL writes the current generated
// payload format straight to Redis while deliberately bypassing its
// conversation index. Finalization tests use this when legacy promotion must
// not provide a second, atomic lifetime observation that masks a failed
// pipelined PTTL.
func directSetGeneratedResponsePayloadWithTTL(t *testing.T, store *RedisStore, response *responseapi.StoredResponse, ttl time.Duration) string {
	t.Helper()

	data, generation := mustMarshalGeneratedResponse(t, response)
	key := store.buildKey(ResponseKeyPrefix + response.ID)
	require.NoError(t, store.client.Set(context.Background(), key, data, ttl).Err())
	return generation
}

// exists reports whether a raw Redis key exists, for asserting on marker/
// index presence directly rather than only through store methods.
func exists(t *testing.T, store *RedisStore, key string) int64 {
	t.Helper()
	n, err := store.client.Exists(context.Background(), key).Result()
	require.NoError(t, err)
	return n
}

// seedPageResponses stores count responses in convID with strictly
// increasing CreatedAt (now+i) and IDs "resp_page_<i>", so ascending index
// order is resp_page_0..resp_page_(count-1) and descending is the reverse.
func seedPageResponses(t *testing.T, store *RedisStore, convID string, count int) []string {
	t.Helper()
	ctx := context.Background()
	now := time.Now().Unix()

	ids := make([]string, count)
	for i := 0; i < count; i++ {
		id := fmt.Sprintf("resp_page_%d", i)
		ids[i] = id
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID:             id,
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      now + int64(i),
		}))
	}
	return ids
}

// clusterTestAddr is the seed address of a single-node Redis Cluster
// started alongside this suite's standalone sr-test-redis instance,
// specifically for tests that must prove no command they issue is ever
// cross-slot: Redis enforces per-command slot-locality independent of node
// count, so even one node in cluster mode genuinely rejects a multi-key
// command (e.g. MGET) spanning different slots with CROSSSLOT, making it a
// faithful (if minimal) Cluster-safety testbed.
const clusterTestAddr = "127.0.0.1:7000"

// newConversationIndexClusterStore builds a RedisStore backed by a real
// Redis Cluster client (see clusterTestAddr). Skips (not fails) if that
// cluster isn't reachable, since most environments running this package's
// tests only provide the standalone sr-test-redis container.
func newConversationIndexClusterStore(t *testing.T) *RedisStore {
	t.Helper()

	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  300,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			ClusterMode:      true,
			ClusterAddresses: []string{clusterTestAddr},
			KeyPrefix:        fmt.Sprintf("srtestcluster:%d:", time.Now().UnixNano()),
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis Cluster not available at %s: %v", clusterTestAddr, err)
	}

	t.Cleanup(func() {
		ctx := context.Background()
		clusterClient, ok := store.client.(interface {
			ForEachMaster(ctx context.Context, fn func(context.Context, *redis.Client) error) error
		})
		if ok {
			_ = clusterClient.ForEachMaster(ctx, func(ctx context.Context, master *redis.Client) error {
				iter := master.Scan(ctx, 0, store.buildKey("*"), 0).Iterator()
				for iter.Next(ctx) {
					master.Del(ctx, iter.Val())
				}
				return nil
			})
		}
		_ = store.Close()
	})

	return store
}

// mustMarshalResponse JSON-marshals a StoredResponse for tests writing a
// payload's raw bytes directly (e.g. repairing a deliberately corrupted
// payload, or constructing a raw value to compare-delete/compare-restore
// against), failing the test immediately on a marshal error rather than
// letting a malformed fixture masquerade as a real one.
func mustMarshalResponse(t *testing.T, response *responseapi.StoredResponse) []byte {
	t.Helper()
	data, err := json.Marshal(response)
	require.NoError(t, err)
	return data
}

// mustMarshalGeneratedResponse writes the production flat JSON shape and
// returns both its bytes and UUID-v4 generation for generation-CAS tests.
func mustMarshalGeneratedResponse(t *testing.T, response *responseapi.StoredResponse) ([]byte, string) {
	t.Helper()
	generation := newResponseGeneration()
	data, err := marshalResponseRecord(response, generation)
	require.NoError(t, err)
	return data, generation
}

func responseIDsOf(responses []*responseapi.StoredResponse) []string {
	ids := make([]string, len(responses))
	for i, r := range responses {
		ids[i] = r.ID
	}
	return ids
}

// seedLegacyIndexMember installs a membership with no generation witness —
// exactly what request-path backfill deliberately leaves behind while legacy
// writers may still exist, and what finalization leaves if a best-effort
// promotion fails. Goes through the production add script rather than a raw
// ZADD, so the fixture cannot drift from what the store really writes.
func seedLegacyIndexMember(t *testing.T, store *RedisStore, conversationID, responseID string, createdAt int64) {
	t.Helper()

	_, err := store.addConversationIndexMembers(
		context.Background(), conversationID, witnessRepair, store.ttlMillis(),
		[]conversationIndexMember{{responseID: responseID, generation: "", score: float64(createdAt)}},
	)
	require.NoError(t, err)
	require.Empty(t, optionalIndexedGeneration(t, store, conversationID, responseID),
		"precondition: the seeded member must carry no witness")
}

// optionalIndexedGeneration reads a member's sidecar witness, reporting an
// absent field as the empty string rather than failing — the distinction
// blank-witness cleanup turns on.
func optionalIndexedGeneration(t *testing.T, store *RedisStore, conversationID, responseID string) string {
	t.Helper()

	value, err := store.client.HGet(
		context.Background(), store.conversationIndexGenerationKey(conversationID), responseID,
	).Result()
	if errors.Is(err, redis.Nil) {
		return ""
	}
	require.NoError(t, err)
	return value
}
