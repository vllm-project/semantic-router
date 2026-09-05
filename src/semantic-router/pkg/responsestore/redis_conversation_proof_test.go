package responsestore

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// TestRedisConversationIndexProofEmptyValueAndTTL covers Phase 2's typed
// proof directly: a scan that finds nothing writes exactly
// conversationIndexProofEmpty (not a generic "v1" marker), read back via GET
// (conversationIndexProof), with a TTL capped at
// emptyConversationIndexMarkerMaxTTL.
func TestRedisConversationIndexProofEmptyValueAndTTL(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 24*60*60) // 24h data TTL
	ctx := context.Background()

	_, err := store.ListResponsesByConversation(ctx, "conv_proof_empty", ListOptions{})
	require.NoError(t, err)

	proof, resolved, err := store.conversationIndexProof(ctx, "conv_proof_empty")
	require.NoError(t, err)
	require.True(t, resolved)
	assert.Equal(t, conversationIndexProofEmpty, proof)

	ttl, err := store.client.TTL(ctx, store.conversationIndexMigratedKey("conv_proof_empty")).Result()
	require.NoError(t, err)
	assert.Positive(t, ttl)
	assert.LessOrEqual(t, ttl, emptyConversationIndexMarkerMaxTTL)
}

// TestRedisConversationIndexProofPopulatedValueAndRefresh covers the
// populated side: a scan that finds responses writes
// conversationIndexProofPopulated, and a later ordinary indexed write
// refreshes that proof's TTL back up (indexResponse ->
// refreshPopulatedConversationProof).
func TestRedisConversationIndexProofPopulatedValueAndRefresh(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_proof_pop_legacy", ConversationID: "conv_proof_pop", Status: "completed", CreatedAt: time.Now().Unix(),
	})
	_, err := store.ListResponsesByConversation(ctx, "conv_proof_pop", ListOptions{})
	require.NoError(t, err)

	proof, resolved, err := store.conversationIndexProof(ctx, "conv_proof_pop")
	require.NoError(t, err)
	require.True(t, resolved)
	require.Equal(t, conversationIndexProofPopulated, proof)

	markerKey := store.conversationIndexMigratedKey("conv_proof_pop")
	require.NoError(t, store.client.PExpire(ctx, markerKey, 5*time.Second).Err())

	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_proof_pop_new", ConversationID: "conv_proof_pop", Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}))

	ttl, err := store.client.TTL(ctx, markerKey).Result()
	require.NoError(t, err)
	assert.Greaterf(t, ttl, 6*time.Second, "an ordinary indexed write must refresh an existing populated proof's TTL back toward the full store TTL")

	proof, resolved, err = store.conversationIndexProof(ctx, "conv_proof_pop")
	require.NoError(t, err)
	require.True(t, resolved)
	assert.Equal(t, conversationIndexProofPopulated, proof, "the refresh must not change the proof's value")
}

// TestRedisConversationIndexProofEmptyNeverExtendedByIndexedWrite covers the
// asymmetry Phase 2 requires: indexResponse may refresh a *populated* proof,
// but must never refresh (or otherwise touch) an *empty* one, even though an
// ordinary write to that same conversation ID now makes the "empty" value
// stale. StoreResponse, duplicate repair (via a direct-set + retry), and
// UpdateResponse are all exercised, since all three ultimately call
// indexResponse.
func TestRedisConversationIndexProofEmptyNeverExtendedByIndexedWrite(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	setup := func(t *testing.T, conversationID string) string {
		t.Helper()
		_, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{})
		require.NoError(t, err)
		proof, resolved, err := store.conversationIndexProof(ctx, conversationID)
		require.NoError(t, err)
		require.True(t, resolved)
		require.Equal(t, conversationIndexProofEmpty, proof)

		markerKey := store.conversationIndexMigratedKey(conversationID)
		require.NoError(t, store.client.PExpire(ctx, markerKey, 2*time.Second).Err())
		return markerKey
	}

	assertStillEmptyAndNotExtended := func(t *testing.T, conversationID, markerKey string) {
		t.Helper()
		ttl, err := store.client.TTL(ctx, markerKey).Result()
		require.NoError(t, err)
		assert.LessOrEqualf(t, ttl, 2*time.Second, "an indexed write must never extend an empty proof's TTL")
		proof, resolved, err := store.conversationIndexProof(ctx, conversationID)
		require.NoError(t, err)
		require.True(t, resolved)
		assert.Equal(t, conversationIndexProofEmpty, proof, "an indexed write must never change an empty proof's value")
	}

	t.Run("StoreResponse", func(t *testing.T) {
		markerKey := setup(t, "conv_proof_empty_store")
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID: "resp_proof_empty_store", ConversationID: "conv_proof_empty_store", Status: "completed", CreatedAt: time.Now().Unix(),
		}))
		assertStillEmptyAndNotExtended(t, "conv_proof_empty_store", markerKey)
	})

	t.Run("duplicate repair", func(t *testing.T) {
		markerKey := setup(t, "conv_proof_empty_repair")
		orphan := &responseapi.StoredResponse{
			ID: "resp_proof_empty_repair", ConversationID: "conv_proof_empty_repair", Status: "completed", CreatedAt: time.Now().Unix(),
		}
		directSetResponsePayload(t, store, orphan)
		retry := *orphan
		assert.ErrorIs(t, store.StoreResponse(ctx, &retry), ErrAlreadyExists)
		assertStillEmptyAndNotExtended(t, "conv_proof_empty_repair", markerKey)
	})

	t.Run("UpdateResponse", func(t *testing.T) {
		markerKey := setup(t, "conv_proof_empty_update")
		original := &responseapi.StoredResponse{
			ID: "resp_proof_empty_update", ConversationID: "conv_proof_empty_update", Status: "original", CreatedAt: time.Now().Unix(),
		}
		directSetResponsePayload(t, store, original)
		updated := *original
		updated.Status = "updated"
		require.NoError(t, store.UpdateResponse(ctx, &updated))
		assertStillEmptyAndNotExtended(t, "conv_proof_empty_update", markerKey)
	})
}

// TestRedisConversationIndexProofUnknownValueNotTrusted covers the
// fail-safe rule: an unrecognized marker value must not be treated as
// resolved, so a read or cascade delete falls back to migration rather than
// trusting a value this code doesn't understand.
func TestRedisConversationIndexProofUnknownValueNotTrusted(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	markerKey := store.conversationIndexMigratedKey("conv_proof_unknown")
	require.NoError(t, store.client.Set(ctx, markerKey, "v2:something-future", store.ttl).Err())

	proof, resolved, err := store.conversationIndexProof(ctx, "conv_proof_unknown")
	require.NoError(t, err)
	assert.False(t, resolved, "an unrecognized proof value must not be trusted as resolved")
	assert.Empty(t, proof)

	// Seed a legacy response so the resulting forced migration has
	// something real to discover, proving the fallback actually runs the
	// scan rather than silently doing nothing.
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_proof_unknown", ConversationID: "conv_proof_unknown", Status: "completed", CreatedAt: time.Now().Unix(),
	})

	responses, err := store.ListResponsesByConversation(ctx, "conv_proof_unknown", ListOptions{})
	require.NoError(t, err)
	require.Len(t, responses, 1)
	assert.Equal(t, "resp_proof_unknown", responses[0].ID)

	proof, resolved, err = store.conversationIndexProof(ctx, "conv_proof_unknown")
	require.NoError(t, err)
	require.True(t, resolved)
	assert.Equal(t, conversationIndexProofPopulated, proof, "the forced re-migration must overwrite the unknown value with a real proof")
}
