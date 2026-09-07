package responsestore

import (
	"context"
	"testing"
	"time"

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
