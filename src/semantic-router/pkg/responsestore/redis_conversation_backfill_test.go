package responsestore

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// indexWriteObserverHook is a go-redis Hook that records the member count of
// every conversation index write it sees and, if callCount reaches failAt,
// fails that call (short-circuiting before the real command runs) instead of
// letting it through — a real fault-injection mechanism operating at the
// actual wire command boundary, not a package-private override.
//
// The write it watches for is conversationIndexAddScript, which carries the
// ZADD: adding members and covering the index's lifetime have to happen in one
// atomic step, since the script decides the lifetime from the key's expiry as
// it stood *before* the members landed.
type indexWriteObserverHook struct {
	mu           sync.Mutex
	memberCounts []int
	callCount    int
	failAt       int // 0 disables fault injection
}

func (h *indexWriteObserverHook) DialHook(next redis.DialHook) redis.DialHook { return next }

func (h *indexWriteObserverHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		count, ok := indexWriteMemberCount(cmd)
		if !ok {
			return next(ctx, cmd)
		}

		h.mu.Lock()
		h.callCount++
		call := h.callCount
		h.memberCounts = append(h.memberCounts, count)
		h.mu.Unlock()

		if h.failAt != 0 && call == h.failAt {
			return errors.New("injected index write failure")
		}
		return next(ctx, cmd)
	}
}

func (h *indexWriteObserverHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

// indexWriteMemberCount reports how many members one conversationIndexAddScript
// call is adding. EVAL/EVALSHA lay out as: verb, script (or SHA), numkeys, the
// two keys this script takes, the requested lifetime, then
// score/member/generation triples.
func indexWriteMemberCount(cmd redis.Cmder) (int, bool) {
	if cmd.Name() != "eval" && cmd.Name() != "evalsha" {
		return 0, false
	}
	args := cmd.Args()
	if len(args) < 6 {
		return 0, false
	}
	key, ok := args[3].(string)
	if !ok || !strings.Contains(key, ConversationIndexKeyPrefix) {
		return 0, false
	}
	return (len(args) - 6) / 3, true
}

func (h *indexWriteObserverHook) totalMembers() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	total := 0
	for _, c := range h.memberCounts {
		total += c
	}
	return total
}

func (h *indexWriteObserverHook) maxBatch() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	max := 0
	for _, c := range h.memberCounts {
		if c > max {
			max = c
		}
	}
	return max
}

// TestLazyBackfillBoundedZaddBatches covers the streaming backfill's
// batching directly, at the wire command level: more than
// redisBackfillBatchSize matching legacy responses must produce multiple
// ZADD calls, none exceeding that cap, rather than one oversized command.
func TestLazyBackfillBoundedZaddBatches(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	hook := &indexWriteObserverHook{}
	store.client.AddHook(hook)

	const total = redisBackfillBatchSize + 50
	for i := 0; i < total; i++ {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: fmt.Sprintf("resp_bounded_%d", i), ConversationID: "conv_bounded_backfill",
			Status: "completed", CreatedAt: time.Now().Unix(),
		})
	}

	found, err := store.lazyBackfillConversationIndex(ctx, "conv_bounded_backfill")
	require.NoError(t, err)
	assert.EqualValues(t, total, found)

	assert.GreaterOrEqualf(t, hook.callCount, 2, "more than %d responses must span at least 2 ZADD calls", redisBackfillBatchSize)
	assert.LessOrEqual(t, hook.maxBatch(), redisBackfillBatchSize, "no single ZADD call may exceed the batch cap")
	assert.Equal(t, total, hook.totalMembers())
	assert.Len(t, conversationIndexMembers(t, store, "conv_bounded_backfill"), total)
}

// TestLazyBackfillEqualTimestampsUseRedisMemberOrdering covers the
// sort.Slice removal: with sort.Slice gone, ordering for equal scores is
// left entirely to Redis's own ZSET tie-break (lexicographic by member),
// not a client-side pre-sort — and ZADD's own argument order does not
// affect that, since a ZSET's ordering is a property of the stored
// (score, member) pairs, never of the order ZADD received them in.
func TestLazyBackfillEqualTimestampsUseRedisMemberOrdering(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	sameCreatedAt := time.Now().Unix()
	// Seeded deliberately out of lexicographic order, so a passing test
	// proves Redis's tie-break is doing the sorting, not incidental scan
	// discovery order.
	for _, id := range []string{"resp_equal_c", "resp_equal_a", "resp_equal_b"} {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: id, ConversationID: "conv_equal_ts", Status: "completed", CreatedAt: sameCreatedAt,
		})
	}

	responses, err := store.ListResponsesByConversation(ctx, "conv_equal_ts", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Equal(t, []string{"resp_equal_a", "resp_equal_b", "resp_equal_c"}, responseIDsOf(responses))
}

// TestLazyBackfillPartialZaddFailureLeavesNoProof covers "partial indexing
// without a proof is acceptable and retryable": a ZADD failure partway
// through a multi-batch backfill must leave whatever batches already
// succeeded in the index, but must not mark the conversation migrated —
// the next call is a safe, idempotent retry rather than one that trusts an
// exhaustive-looking index that is actually incomplete.
func TestLazyBackfillPartialZaddFailureLeavesNoProof(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	hook := &indexWriteObserverHook{failAt: 2} // let the first batch through, fail the second
	store.client.AddHook(hook)

	const total = redisBackfillBatchSize + 50
	for i := 0; i < total; i++ {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: fmt.Sprintf("resp_partial_%d", i), ConversationID: "conv_partial_backfill",
			Status: "completed", CreatedAt: time.Now().Unix(),
		})
	}

	_, err := store.lazyBackfillConversationIndex(ctx, "conv_partial_backfill")
	require.Error(t, err)

	_, resolved, err := store.conversationIndexProof(ctx, "conv_partial_backfill")
	require.NoError(t, err)
	assert.False(t, resolved, "a partially failed backfill must not publish a migration proof")

	// The first, successful batch's members are still indexed: partial
	// progress is retained, not rolled back.
	assert.Len(t, conversationIndexMembers(t, store, "conv_partial_backfill"), redisBackfillBatchSize)
}

// TestIndexBackfillBatchConcurrentCallersRaceFree simulates what
// ForEachMaster's concurrent per-master callbacks each independently do:
// call indexBackfillBatch — the actual primitive lazyBackfillConversationIndex
// calls once per flushed batch per callback invocation — from several
// goroutines at once, targeting the same conversation. Run under
// `go test -race`, this is the direct regression test for the removed
// shared, unsynchronized accumulation slice: indexBackfillBatch has no
// shared mutable state to race on, only independent, idempotent ZADD calls.
func TestIndexBackfillBatchConcurrentCallersRaceFree(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const goroutines = 8
	const perGoroutine = 50
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			members := make([]conversationIndexMember, perGoroutine)
			now := time.Now().Unix()
			for i := 0; i < perGoroutine; i++ {
				members[i] = conversationIndexMember{
					responseID: fmt.Sprintf("resp_concurrent_batch_%d_%d", g, i),
					score:      float64(now),
				}
			}
			assert.NoError(t, store.indexBackfillBatch(ctx, "conv_concurrent_backfill", members, store.ttlMillis()))
		}(g)
	}
	wg.Wait()

	assert.Len(t, conversationIndexMembers(t, store, "conv_concurrent_backfill"), goroutines*perGoroutine)
}

func TestLazyBackfillDoesNotCertifyIncompleteScan(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	require.NoError(t, store.client.Set(ctx,
		store.buildKey(ResponseKeyPrefix+"resp_corrupt_unrelated"), "not-json", store.ttl).Err())

	_, err := store.ListResponsesByConversation(ctx, "conv_scan_must_be_complete", ListOptions{})
	require.Error(t, err)
	_, resolved, proofErr := store.conversationIndexProof(ctx, "conv_scan_must_be_complete")
	require.NoError(t, proofErr)
	assert.False(t, resolved, "a scan that skipped an unreadable response cannot prove any conversation exhaustive")
}
