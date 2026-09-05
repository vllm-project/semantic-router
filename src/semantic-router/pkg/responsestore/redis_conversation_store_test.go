package responsestore

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// TestRedisDeleteConversationCascade uses more responses than one page: the
// cascade reads the index directly, so it must not stop at DefaultListLimit.
func TestRedisDeleteConversationCascade(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const (
		convID      = "conv_cascade"
		responseCnt = DefaultListLimit + 5
		otherConvID = "conv_cascade_other"
		otherRespID = "resp_cascade_other"
	)

	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID:        convID,
		CreatedAt: time.Now().Unix(),
	}))

	for i := 0; i < responseCnt; i++ {
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID:             fmt.Sprintf("resp_cascade_%d", i),
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      time.Now().Unix() + int64(i),
			Output:         []responseapi.OutputItem{{ID: fmt.Sprintf("item_%d", i)}},
		}))
	}

	// A response in a different conversation must survive the cascade.
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID:             otherRespID,
		ConversationID: otherConvID,
		Status:         "kept",
		CreatedAt:      time.Now().Unix(),
	}))

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	for i := 0; i < responseCnt; i++ {
		_, err := store.GetResponse(ctx, fmt.Sprintf("resp_cascade_%d", i))
		assert.ErrorIsf(t, err, ErrNotFound, "response %d should have been deleted with its conversation", i)
	}
	assert.Empty(t, conversationIndexMembers(t, store, convID))

	survivor, err := store.GetResponse(ctx, otherRespID)
	require.NoError(t, err)
	assert.Equal(t, "kept", survivor.Status)
}

// TestRedisAddResponseToConversation covers blueprint §3.7/§5 Phase 5: the
// stored response's own ConversationID is the only source of truth for
// whether it may be indexed under conversationID.
func TestRedisAddResponseToConversation(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	t.Run("stored response matching the conversation is indexed", func(t *testing.T) {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: "resp_add_match", ConversationID: "conv_add", Status: "completed", CreatedAt: time.Now().Unix(),
		})
		require.Empty(t, conversationIndexMembers(t, store, "conv_add"))

		require.NoError(t, store.AddResponseToConversation(ctx, "conv_add", "resp_add_match"))
		assert.Equal(t, []string{"resp_add_match"}, conversationIndexMembers(t, store, "conv_add"))
	})

	t.Run("missing response is ErrNotFound", func(t *testing.T) {
		err := store.AddResponseToConversation(ctx, "conv_add", "resp_add_missing")
		assert.ErrorIs(t, err, ErrNotFound)
	})

	t.Run("response belonging to a different conversation is rejected", func(t *testing.T) {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: "resp_add_other", ConversationID: "conv_add_real", Status: "completed", CreatedAt: time.Now().Unix(),
		})

		err := store.AddResponseToConversation(ctx, "conv_add_wrong", "resp_add_other")
		assert.ErrorIs(t, err, ErrInvalidInput)
		assert.Empty(t, conversationIndexMembers(t, store, "conv_add_wrong"))
	})

	t.Run("response with no stored conversation is rejected", func(t *testing.T) {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: "resp_add_no_conv", Status: "completed", CreatedAt: time.Now().Unix(),
		})

		err := store.AddResponseToConversation(ctx, "conv_add_none", "resp_add_no_conv")
		assert.ErrorIs(t, err, ErrInvalidInput)
		assert.Empty(t, conversationIndexMembers(t, store, "conv_add_none"))
	})
}

// TestRedisDeleteConversationCascadeBatched exceeds redisDeleteBatchSize so
// the cascade must loop more than once (blueprint §5 Phase 5 / §6.6).
func TestRedisDeleteConversationCascadeBatched(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const (
		convID      = "conv_cascade_batched"
		responseCnt = redisDeleteBatchSize + 50
	)

	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))
	for i := 0; i < responseCnt; i++ {
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID:             fmt.Sprintf("resp_cascade_batched_%d", i),
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      time.Now().Unix() + int64(i),
		}))
	}

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	for i := 0; i < responseCnt; i++ {
		_, err := store.GetResponse(ctx, fmt.Sprintf("resp_cascade_batched_%d", i))
		assert.ErrorIsf(t, err, ErrNotFound, "response %d should have been deleted across batches", i)
	}
	assert.Empty(t, conversationIndexMembers(t, store, convID))
	assert.Zero(t, exists(t, store, store.conversationIndexMigratedKey(convID)))
	_, err := store.GetConversation(ctx, convID)
	assert.ErrorIs(t, err, ErrNotFound)
}

// TestRedisDeleteConversationCascadeFailureLeavesConversation covers
// blueprint §5 Phase 5, and Phase 5's ownership-verified rewrite's "malformed
// payload blocks completion, leaves conversation record" requirement: a
// cascade failure must be reported to the caller, and the conversation
// record must survive as the retry anchor rather than be deleted ahead of
// (and regardless of) the cascade's outcome. The failure here is a genuinely
// malformed payload (not decodable as a StoredResponse) sitting behind an
// otherwise-valid index member — deleteConversationResponseBatch must
// preserve both the payload and the index member rather than guess at
// ownership it cannot actually verify.
func TestRedisDeleteConversationCascadeFailureLeavesConversation(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_failure"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_cascade_failure", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	}))
	// Corrupt the payload in place, after indexing, so the index member is
	// genuine but the payload it points at can no longer be parsed to
	// verify ownership.
	require.NoError(t, store.client.Set(ctx, store.buildKey(ResponseKeyPrefix+"resp_cascade_failure"),
		[]byte("not valid json"), store.ttl).Err())

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err)

	// The conversation record must still be there: the caller can retry.
	_, getErr := store.GetConversation(ctx, convID)
	assert.NoError(t, getErr)
	// The malformed payload and its index entry are untouched.
	assert.Equal(t, []string{"resp_cascade_failure"}, conversationIndexMembers(t, store, convID))
	raw, getErr := store.client.Get(ctx, store.buildKey(ResponseKeyPrefix+"resp_cascade_failure")).Bytes()
	require.NoError(t, getErr)
	assert.Equal(t, []byte("not valid json"), raw)

	// Repair the payload and retry: the cascade completes cleanly.
	require.NoError(t, store.client.Set(ctx, store.buildKey(ResponseKeyPrefix+"resp_cascade_failure"),
		mustMarshalResponse(t, &responseapi.StoredResponse{
			ID: "resp_cascade_failure", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
		}), store.ttl).Err())
	require.NoError(t, store.DeleteConversation(ctx, convID, true))
	_, getErr = store.GetConversation(ctx, convID)
	assert.ErrorIs(t, getErr, ErrNotFound)
}

// TestRedisDeleteConversationCascadeLegacyUnindexed covers a cascade delete
// against a conversation whose responses were never indexed — pre-#2814
// legacy data, or a write from an indexing-unaware pod mid rolling upgrade.
// Before deleteConversationResponses resolved the index first, this case
// regressed the pre-#2814 scan-based cascade: an absent index made the
// batch-delete loop exit immediately having deleted nothing, silently
// orphaning the legacy payload forever once the conversation record itself
// was gone.
func TestRedisDeleteConversationCascadeLegacyUnindexed(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_legacy"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))

	// Bypasses StoreResponse's indexing entirely, same shape as data written
	// before the index existed.
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_cascade_legacy", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	})
	require.Empty(t, conversationIndexMembers(t, store, convID),
		"precondition: no index should exist yet for legacy data")

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	_, err := store.GetResponse(ctx, "resp_cascade_legacy")
	assert.ErrorIs(t, err, ErrNotFound, "the legacy unindexed response must be deleted by the cascade, not orphaned")
	_, err = store.GetConversation(ctx, convID)
	assert.ErrorIs(t, err, ErrNotFound)
}

// TestRedisDeleteConversationCascadePartiallyMigrated is the cascade-delete
// counterpart to TestRedisListResponsesByConversationPartiallyMigrated: a
// conversation whose index exists (from an ordinary post-upgrade write) but
// still has an older, unindexed legacy response sitting alongside it must
// have *both* responses deleted, not just the one the index happens to
// already list.
func TestRedisDeleteConversationCascadePartiallyMigrated(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_2"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))

	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_old", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix(),
	})
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_new", ConversationID: convID, Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}))
	require.Equal(t, []string{"resp_new"}, conversationIndexMembers(t, store, convID),
		"precondition: the index exists but is not yet exhaustive")
	require.Zero(t, exists(t, store, store.conversationIndexMigratedKey(convID)),
		"precondition: no backfill has run yet, despite the index already existing")

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	_, err := store.GetResponse(ctx, "resp_old")
	assert.ErrorIs(t, err, ErrNotFound, "the legacy response must be deleted, not orphaned because the index didn't list it")
	_, err = store.GetResponse(ctx, "resp_new")
	assert.ErrorIs(t, err, ErrNotFound)
	_, err = store.GetConversation(ctx, convID)
	assert.ErrorIs(t, err, ErrNotFound)
}
