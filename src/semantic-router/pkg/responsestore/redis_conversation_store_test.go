package responsestore

import (
	"context"
	"errors"
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
	assert.Zero(t, exists(t, store, store.emptyConversationIndexMarkerKey(convID)))
	_, err := store.GetConversation(ctx, convID)
	assert.ErrorIs(t, err, ErrNotFound)
}

// TestRedisDeleteConversationCascadeFailureLeavesConversation covers
// blueprint §5 Phase 5: a cascade failure must be reported to the caller,
// and the conversation record must survive as the retry anchor rather than
// be deleted ahead of (and regardless of) the cascade's outcome.
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

	injectedErr := errors.New("injected cascade delete failure")
	store.deleteResponseBatchOverride = func(context.Context, []string) error {
		return injectedErr
	}

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	// The conversation record must still be there: the caller can retry.
	_, getErr := store.GetConversation(ctx, convID)
	assert.NoError(t, getErr)
	// The response and its index entry are untouched by the failed batch.
	_, getErr = store.GetResponse(ctx, "resp_cascade_failure")
	assert.NoError(t, getErr)
	assert.Equal(t, []string{"resp_cascade_failure"}, conversationIndexMembers(t, store, convID))

	store.deleteResponseBatchOverride = nil
	require.NoError(t, store.DeleteConversation(ctx, convID, true))
	_, getErr = store.GetConversation(ctx, convID)
	assert.ErrorIs(t, getErr, ErrNotFound)
}
