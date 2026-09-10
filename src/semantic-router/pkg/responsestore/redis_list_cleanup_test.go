package responsestore

import (
	"context"
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

// listWindowObserverHook records the width of each warmed
// readIndexWindowScript call without counting Script.Run's possible NOSCRIPT
// probe. Tests use it to pin that dead-only progress grows beyond the public
// Limit while every individual Redis window stays capped.
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
		if cmd.Name() == "evalsha" && len(args) > 6 && args[1] == readIndexWindowScript.Hash() {
			start, startOK := args[5].(int64)
			end, endOK := args[6].(int64)
			if startOK && endOK {
				h.mu.Lock()
				h.widths = append(h.widths, end-start+1)
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
