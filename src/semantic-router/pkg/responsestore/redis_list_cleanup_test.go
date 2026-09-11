package responsestore

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// listPruneContentionHook changes one member's witness immediately before
// every conditional-unindex script. The page therefore keeps observing a
// stale candidate but can never claim the generation it just read, modeling
// a continuously refreshed membership without recursively entering the
// hooked client.
type listPruneContentionHook struct {
	client        redis.UniversalClient
	generationKey string
	responseID    string
	fired         atomic.Int64
	err           error
}

func (h *listPruneContentionHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *listPruneContentionHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *listPruneContentionHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		args := cmd.Args()
		if cmd.Name() == "evalsha" && len(args) > 1 && args[1] == conditionalUnindexScript.Hash() {
			h.fired.Add(1)
			if err := h.client.HSet(context.Background(), h.generationKey, h.responseID, newResponseGeneration()).Err(); err != nil && h.err == nil {
				h.err = err
			}
		}
		return next(ctx, cmd)
	}
}

// listWindowObserverHook records the requested width of each warmed
// readIndexWindowScript call without counting Script.Run's possible NOSCRIPT
// probe. Tests use it to pin that dead-only progress grows beyond the public
// Limit while every individual Redis window stays capped.
//
// The width is the script's fourth argument: evalsha, sha, key count, the two
// keys, direction, cursor mode, cursor, width.
type listWindowObserverHook struct {
	mu     sync.Mutex
	widths []int64
}

func (h *listWindowObserverHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *listWindowObserverHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *listWindowObserverHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		args := cmd.Args()
		if cmd.Name() == "evalsha" && len(args) > 8 && args[1] == readIndexWindowScript.Hash() {
			if width, ok := args[8].(int); ok {
				h.mu.Lock()
				h.widths = append(h.widths, int64(width))
				h.mu.Unlock()
			}
		}
		return next(ctx, cmd)
	}
}

func (h *listWindowObserverHook) snapshot() []int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]int64(nil), h.widths...)
}

// TestListWalksPastMultipleGeneratedTombstoneWindows complements the
// finalized legacy regression with ordinary generated witnesses and pins all
// supported cursor directions. Each case has at least two Limit-sized stale
// windows before the live result, so a fixed one-refill implementation would
// return an empty terminal page.
func TestListWalksPastMultipleGeneratedTombstoneWindows(t *testing.T) {
	const limit = 2

	t.Run("ascending", func(t *testing.T) {
		store := newConversationIndexStore(t)
		ctx := context.Background()
		const conversationID = "conv_generated_tombstones_asc"

		ids := seedPageResponses(t, store, conversationID, 2*limit+1)
		for _, responseID := range ids[:2*limit] {
			require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+responseID)).Err())
		}
		require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
			conversationIndexCompletionValue, 0).Err())
		require.NoError(t, readIndexWindowScript.Load(ctx, store.client).Err())
		observer := &listWindowObserverHook{}
		store.client.AddHook(observer)

		responses, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{Order: "asc", Limit: limit})
		require.NoError(t, err)
		assert.Equal(t, []string{ids[len(ids)-1]}, responseIDsOf(responses))
		assert.Equal(t, []string{ids[len(ids)-1]}, conversationIndexMembers(t, store, conversationID))

		widths := observer.snapshot()
		require.NotEmpty(t, widths)
		assert.EqualValues(t, limit, widths[0])
		assert.Contains(t, widths, int64(2*limit), "a removed dead-only window should grow the next bounded read")
		for _, width := range widths {
			assert.LessOrEqual(t, width, int64(listIndexScanMaxStride))
		}
	})

	t.Run("descending", func(t *testing.T) {
		store := newConversationIndexStore(t)
		ctx := context.Background()
		const conversationID = "conv_generated_tombstones_desc"

		ids := seedPageResponses(t, store, conversationID, 2*limit+1)
		for _, responseID := range ids[1:] {
			require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+responseID)).Err())
		}
		require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
			conversationIndexCompletionValue, 0).Err())

		responses, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{Order: "desc", Limit: limit})
		require.NoError(t, err)
		assert.Equal(t, []string{ids[0]}, responseIDsOf(responses))
		assert.Equal(t, []string{ids[0]}, conversationIndexMembers(t, store, conversationID))
	})

	t.Run("before cursor", func(t *testing.T) {
		store := newConversationIndexStore(t)
		ctx := context.Background()
		const conversationID = "conv_generated_tombstones_before"

		// Keep several live responses before an odd-sized dead run. Once
		// cleanup widens the Before window, more than Limit live responses
		// fit in it; the result must retain the ones nearest the cursor.
		const livePrefix = 5
		const tombstones = 2*limit + 1
		ids := seedPageResponses(t, store, conversationID, livePrefix+tombstones+1)
		for _, responseID := range ids[livePrefix : livePrefix+tombstones] {
			require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+responseID)).Err())
		}
		require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
			conversationIndexCompletionValue, 0).Err())

		responses, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{
			Order: "asc", Limit: limit, Before: ids[len(ids)-1],
		})
		require.NoError(t, err)
		assert.Equal(t, ids[livePrefix-limit:livePrefix], responseIDsOf(responses),
			"Before must retain the live responses nearest its cursor after widening")
		wantMembers := append(append([]string{}, ids[:livePrefix]...), ids[len(ids)-1])
		assert.Equal(t, wantMembers, conversationIndexMembers(t, store, conversationID))
	})
}

// TestListReturnsContentionErrorInsteadOfAmbiguousPartialPage proves that a
// short non-empty page is not returned as apparent exhaustion while a stale
// candidate repeatedly changes generation. The full page can be retried once
// the writer quiesces; returning the one visible member with nil error would
// let an ordinary paginator hide everything behind the contended slot.
func TestListReturnsContentionErrorInsteadOfAmbiguousPartialPage(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	ctx := context.Background()

	const conversationID = "conv_list_contention"
	const liveID = "resp_list_contention_live"
	const staleID = "resp_list_contention_stale"
	now := time.Now().Unix()
	for _, response := range []*responseapi.StoredResponse{
		{ID: liveID, ConversationID: conversationID, Status: "completed", CreatedAt: now},
		{ID: staleID, ConversationID: conversationID, Status: "completed", CreatedAt: now + 1},
	} {
		require.NoError(t, store.StoreResponse(ctx, response))
	}
	require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+staleID)).Err())
	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
		conversationIndexCompletionValue, 0).Err())
	require.NoError(t, conditionalUnindexScript.Load(ctx, store.client).Err())

	hook := &listPruneContentionHook{
		client:        writer.client,
		generationKey: store.conversationIndexGenerationKey(conversationID),
		responseID:    staleID,
	}
	store.client.AddHook(hook)

	responses, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{Order: "asc", Limit: 2})
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrIndexContended)
	assert.Empty(t, responses, "an underfilled page must not be returned with a contention error")
	assert.NoError(t, hook.err)
	assert.EqualValues(t, listIndexMaxContentionRounds, hook.fired.Load())
	assert.Equal(t, []string{liveID, staleID}, conversationIndexMembers(t, store, conversationID),
		"the contended membership remains a retry anchor")
}

// TestListWalksPastBlockedLegacyTombstonesBeforeFinalization is the
// pre-finalization regression. A blank-witness member whose payload is gone is
// deliberately non-prunable until an operator finalizes, and proof expiry never
// clears it: a rescan adds live members but removes nothing. A full window of
// them therefore yielded no responses and no prune candidates, which the loop
// reported as an empty successful page — terminal to any ordinary paginator,
// with every retained response sitting behind it.
//
// The traversal must step over them without removing them, so both halves are
// asserted: the live response comes back, and every tombstone is still a member.
func TestListWalksPastBlockedLegacyTombstonesBeforeFinalization(t *testing.T) {
	const limit = 3

	for _, tombstones := range []int{limit, 2 * limit} {
		t.Run(fmt.Sprintf("%d blocked tombstones", tombstones), func(t *testing.T) {
			store := newConversationIndexStore(t)
			ctx := context.Background()
			conversationID := fmt.Sprintf("conv_blocked_legacy_%d", tombstones)
			liveID := fmt.Sprintf("resp_blocked_legacy_live_%d", tombstones)

			now := time.Now().Unix()
			blocked := seedLegacyIndexMembers(t, store, conversationID, "resp_blocked_legacy", tombstones, now)
			require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
				ID: liveID, ConversationID: conversationID, Status: "completed",
				CreatedAt: now + int64(tombstones),
			}))
			require.Zero(t, exists(t, store, store.conversationIndexCompletionKey()),
				"precondition: the store must not be finalized")
			require.Equal(t, append(append([]string{}, blocked...), liveID),
				conversationIndexMembers(t, store, conversationID),
				"precondition: every tombstone sorts ahead of the live response")

			responses, err := store.ListResponsesByConversation(ctx, conversationID,
				ListOptions{Order: "asc", Limit: limit})
			require.NoError(t, err)
			assert.Equal(t, []string{liveID}, responseIDsOf(responses),
				"the page must reach the live response behind the blocked tombstones")

			assert.Equal(t, append(append([]string{}, blocked...), liveID),
				conversationIndexMembers(t, store, conversationID),
				"stepping over a blocked member must never remove it before finalization")
		})
	}
}

// TestListReportsBlockedTraversalBudgetExhausted pins the other half of the
// contract: the walk is bounded, and spending the budget reports an explicit
// retryable error rather than the empty page the traversal exists to prevent.
// At Limit 1 the widening strides sum to 255 members over the budget, so a
// longer blocked run cannot be crossed in one call.
func TestListReportsBlockedTraversalBudgetExhausted(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const conversationID = "conv_blocked_budget"
	const liveID = "resp_blocked_budget_live"

	now := time.Now().Unix()
	blocked := seedLegacyIndexMembers(t, store, conversationID, "resp_blocked_budget", 300, now)
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: liveID, ConversationID: conversationID, Status: "completed", CreatedAt: now + 1000,
	}))

	responses, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{Order: "asc", Limit: 1})
	require.Error(t, err)
	assert.ErrorIs(t, err, ErrIndexTraversalBlocked)
	assert.Empty(t, responses, "a budget-exhausted traversal must not answer with a page at all")
	assert.Len(t, conversationIndexMembers(t, store, conversationID), len(blocked)+1,
		"a blocked traversal removes nothing")
}

// TestListFillsPageAcrossMixedBlockedWindow covers a full window that
// underfills the page. With Limit 3 over [blocked, blocked, live-A] the first
// window is full, so its single response is no evidence that the conversation
// ended — live-B and live-C sit in the very next ranks. Returning that short
// page let a paginator testing len(page) < Limit stop and hide them.
//
// Only a window that comes back shorter than the ranks it covered proves
// exhaustion; a full one must widen instead.
func TestListFillsPageAcrossMixedBlockedWindow(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const conversationID = "conv_mixed_blocked_window"
	const limit = 3

	now := time.Now().Unix()
	blocked := seedLegacyIndexMembers(t, store, conversationID, "resp_mixed_blocked", 2, now)
	live := make([]string, 3)
	for i := range live {
		live[i] = fmt.Sprintf("resp_mixed_live_%d", i)
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID: live[i], ConversationID: conversationID, Status: "completed",
			CreatedAt: now + int64(len(blocked)+i),
		}))
	}
	require.Zero(t, exists(t, store, store.conversationIndexCompletionKey()),
		"precondition: the store must not be finalized, so the blank members stay blocked")
	require.Equal(t, append(append([]string{}, blocked...), live...),
		conversationIndexMembers(t, store, conversationID),
		"precondition: the first Limit-wide window must be blocked, blocked, live")

	responses, err := store.ListResponsesByConversation(ctx, conversationID,
		ListOptions{Order: "asc", Limit: limit})
	require.NoError(t, err)
	assert.Equal(t, live, responseIDsOf(responses),
		"a full window that underfills the page must widen, not return early")

	assert.Equal(t, append(append([]string{}, blocked...), live...),
		conversationIndexMembers(t, store, conversationID),
		"widening past a blocked member must never remove it before finalization")
}

// TestListWideningSurvivesBlockedMemberRecreation pins why the traversal
// widens from the caller's anchor instead of re-anchoring onto a member it
// stepped over.
//
// A response ID is not a stable index position. Here the blocked member is
// recreated by an ordinary StoreResponse while the page is being built, and
// conversationIndexAddScript's unqualified ZADD re-scores its membership past
// the live response that never moved. An anchor placed on that member would
// follow it — ZRANK returns the new incarnation's rank — and the next window
// would begin after the live response, reporting an empty page while a
// retained response sat in the index. A fixed anchor cannot be carried
// anywhere, so widening simply re-reads both.
func TestListWideningSurvivesBlockedMemberRecreation(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	ctx := context.Background()

	const conversationID = "conv_blocked_recreated"
	const blockedID = "resp_blocked_recreated"
	const liveID = "resp_blocked_recreated_live"

	now := time.Now().Unix()
	seedLegacyIndexMember(t, store, conversationID, blockedID, now)
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: liveID, ConversationID: conversationID, Status: "completed", CreatedAt: now + 1,
	}))
	require.Equal(t, []string{blockedID, liveID}, conversationIndexMembers(t, store, conversationID),
		"precondition: the blocked member sorts ahead of the retained response")

	// Recreated with a later created_at, so its membership is re-scored past
	// the live response rather than staying where the reader observed it.
	var injectedErr error
	hook := &commandInterleavingHook{
		pipeline: true,
		match: func(cmd redis.Cmder) bool {
			return commandReadsKey(cmd, store.buildKey(ResponseKeyPrefix+blockedID))
		},
		inject: func() {
			injectedErr = writer.StoreResponse(context.Background(), &responseapi.StoredResponse{
				ID: blockedID, ConversationID: conversationID, Status: "recreated", CreatedAt: now + 2,
			})
		},
	}
	store.client.AddHook(hook)

	responses, err := store.ListResponsesByConversation(ctx, conversationID,
		ListOptions{Order: "asc", Limit: 1})
	require.NoError(t, err)
	require.NoError(t, injectedErr)
	assert.True(t, hook.fired.Load(), "the recreation must land while the first window is being resolved")
	assert.Equal(t, []string{liveID}, responseIDsOf(responses),
		"a re-scored blocked member must not carry the window past a response that never moved")

	assert.Equal(t, []string{liveID, blockedID}, conversationIndexMembers(t, store, conversationID),
		"precondition check: the recreation really did move the member past the live response")
}

// cursorRankObserverHook counts standalone ZRANK/ZREVRANK commands. The
// listing path must never issue one: a cursor's rank is only meaningful
// against the membership it was measured in, so it is resolved inside the
// same script that reads the window.
type cursorRankObserverHook struct {
	fired atomic.Int64
}

func (h *cursorRankObserverHook) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *cursorRankObserverHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *cursorRankObserverHook) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if name := cmd.Name(); name == "zrank" || name == "zrevrank" {
			h.fired.Add(1)
		}
		return next(ctx, cmd)
	}
}

// TestListCursorPageIsOneSnapshot proves the cursor's rank and the page it
// positions are read from a single membership. A member inserted or removed
// before the cursor between a standalone ZRANK and the later ZRANGE shifts
// every subsequent rank: a removal made the saved window skip the cursor's
// true successor, and an insertion made it begin on the cursor itself and
// return a response the caller already held.
//
// The mutation is injected immediately before the window script runs. Against
// a two-command implementation that boundary is after the rank was already
// taken, which is exactly the race; against the atomic one the mutation
// simply precedes the snapshot and the page is correct either way.
func TestListCursorPageIsOneSnapshot(t *testing.T) {
	const limit = 2

	cases := []struct {
		name     string
		order    string
		mutate   func(ctx context.Context, writer *RedisStore, indexKey string, now int64) error
		wantPage []string
	}{
		{
			name:  "ascending after cursor, member before cursor removed",
			order: "asc",
			mutate: func(ctx context.Context, writer *RedisStore, indexKey string, _ int64) error {
				return writer.client.ZRem(ctx, indexKey, "resp_page_0").Err()
			},
			wantPage: []string{"resp_page_3", "resp_page_4"},
		},
		{
			name:  "ascending after cursor, member before cursor inserted",
			order: "asc",
			mutate: func(ctx context.Context, writer *RedisStore, indexKey string, now int64) error {
				return writer.client.ZAdd(ctx, indexKey, redis.Z{Score: float64(now - 1), Member: "resp_page_inserted"}).Err()
			},
			wantPage: []string{"resp_page_3", "resp_page_4"},
		},
		{
			name:  "descending after cursor, member before cursor removed",
			order: "desc",
			mutate: func(ctx context.Context, writer *RedisStore, indexKey string, _ int64) error {
				return writer.client.ZRem(ctx, indexKey, "resp_page_4").Err()
			},
			wantPage: []string{"resp_page_1", "resp_page_0"},
		},
		{
			name:  "descending after cursor, member before cursor inserted",
			order: "desc",
			mutate: func(ctx context.Context, writer *RedisStore, indexKey string, now int64) error {
				return writer.client.ZAdd(ctx, indexKey, redis.Z{Score: float64(now + 10), Member: "resp_page_inserted"}).Err()
			},
			wantPage: []string{"resp_page_1", "resp_page_0"},
		},
	}

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := newConversationIndexStore(t)
			writer := newConcurrentRedisStore(t, store)
			ctx := context.Background()
			conversationID := fmt.Sprintf("conv_cursor_snapshot_%d", i)

			now := time.Now().Unix()
			seedPageResponsesAt(t, store, conversationID, 5, now)
			require.NoError(t, store.ensureConversationIndexResolved(ctx, conversationID))
			require.NoError(t, readIndexWindowScript.Load(ctx, store.client).Err())

			var injectedErr error
			hook := &commandInterleavingHook{
				before: true,
				match: func(cmd redis.Cmder) bool {
					args := cmd.Args()
					return cmd.Name() == "evalsha" && len(args) > 1 && args[1] == readIndexWindowScript.Hash()
				},
				inject: func() {
					injectedErr = tc.mutate(context.Background(), writer, store.conversationIndexKey(conversationID), now)
				},
			}
			store.client.AddHook(hook)

			responses, err := store.ListResponsesByConversation(ctx, conversationID,
				ListOptions{Order: tc.order, Limit: limit, After: "resp_page_2"})
			require.NoError(t, err)
			require.NoError(t, injectedErr)
			assert.True(t, hook.fired.Load(), "the mutation must land right before the window read")
			assert.Equal(t, tc.wantPage, responseIDsOf(responses),
				"a rank shift before the cursor must neither skip nor repeat a live response")
		})
	}
}

// TestListCursorRankIsNeverAStandaloneCommand pins the property directly:
// no cursor read, in either direction or with either cursor kind, resolves the
// rank as a command of its own.
func TestListCursorRankIsNeverAStandaloneCommand(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()
	const conversationID = "conv_cursor_atomic"

	seedPageResponses(t, store, conversationID, 5)
	require.NoError(t, store.ensureConversationIndexResolved(ctx, conversationID))

	observer := &cursorRankObserverHook{}
	store.client.AddHook(observer)

	for _, opts := range []ListOptions{
		{Order: "asc", Limit: 2, After: "resp_page_1"},
		{Order: "asc", Limit: 2, Before: "resp_page_3"},
		{Order: "desc", Limit: 2, After: "resp_page_3"},
		{Order: "desc", Limit: 2, Before: "resp_page_1"},
	} {
		_, err := store.ListResponsesByConversation(ctx, conversationID, opts)
		require.NoError(t, err)
	}
	assert.Zero(t, observer.fired.Load(),
		"cursor rank must be resolved inside the window script, never as a separate command")
}
