package responsestore

import (
	"context"
	"fmt"
	"sync"
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
// conversationIndexProofMaxTTL.
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
	assert.LessOrEqual(t, ttl, conversationIndexProofMaxTTL)
}

// TestRedisConversationIndexProofPopulatedIsBoundedAndNeverExtended covers
// the populated side of the rolling-upgrade bound: a scan that finds
// responses writes conversationIndexProofPopulated with a TTL capped at
// conversationIndexProofMaxTTL — not the store's full data-retention TTL —
// and no ordinary indexed write may push that deadline out.
//
// The superseded behavior gave a populated proof the full store TTL and let
// indexResponse refresh it on every subsequent write, on the reasoning that
// a conversation with discovered, indexed data has no blind spot left. It
// does: an index-unaware writer mid rolling upgrade can land an unindexed
// response into an already-populated conversation, and on a conversation
// written to more often than the store TTL that refresh made the proof
// immortal, hiding the response for good.
func TestRedisConversationIndexProofPopulatedIsBoundedAndNeverExtended(t *testing.T) {
	// A 24h data TTL, so "capped at conversationIndexProofMaxTTL" is
	// distinguishable from "given the store's own TTL".
	store := newConversationIndexStoreWithTTLSeconds(t, 24*60*60)
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
	ttl, err := store.client.TTL(ctx, markerKey).Result()
	require.NoError(t, err)
	assert.Positive(t, ttl)
	assert.LessOrEqualf(t, ttl, conversationIndexProofMaxTTL,
		"a populated proof must be capped like an empty one, not given the store's 24h data TTL")

	// Shorten it, then let an ordinary indexed write land on the same
	// conversation: the write must leave the deadline exactly where it is.
	require.NoError(t, store.client.PExpire(ctx, markerKey, 5*time.Second).Err())
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_proof_pop_new", ConversationID: "conv_proof_pop", Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}))

	ttl, err = store.client.TTL(ctx, markerKey).Result()
	require.NoError(t, err)
	assert.LessOrEqualf(t, ttl, 5*time.Second,
		"an ordinary indexed write must not extend a populated proof — that is what made the blind spot permanent")

	proof, resolved, err = store.conversationIndexProof(ctx, "conv_proof_pop")
	require.NoError(t, err)
	require.True(t, resolved)
	assert.Equal(t, conversationIndexProofPopulated, proof, "an indexed write must not change the proof's value either")
}

// TestRedisPopulatedProofExpiryRevealsLateUnindexedWrite is the
// populated-scan regression Xunzhuo asked for, end to end: a response that
// an index-unaware writer lands into an *already populated* conversation
// after that conversation's backfill scan has passed is hidden only until
// the bounded proof expires, and the next read revalidates by scanning
// again rather than trusting the stale proof forever.
func TestRedisPopulatedProofExpiryRevealsLateUnindexedWrite(t *testing.T) {
	// A 2s data TTL collapses conversationIndexProofMaxTTL's cap to 2s, so
	// the bounded window is observable in a unit test rather than 5 minutes.
	// The payloads themselves are written with a long explicit TTL, so they
	// outlive the proof they are described by.
	store := newConversationIndexStoreWithTTLSeconds(t, 2)
	ctx := context.Background()

	const convID = "conv_proof_late_write"
	now := time.Now().Unix()

	directSetResponsePayloadWithTTL(t, store, &responseapi.StoredResponse{
		ID: "resp_late_early", ConversationID: convID, Status: "completed", CreatedAt: now,
	}, time.Hour)

	// First read: the legacy scan finds the existing response and publishes
	// a populated proof.
	responses, err := store.ListResponsesByConversation(ctx, convID, ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Equal(t, []string{"resp_late_early"}, responseIDsOf(responses))

	proof, resolved, err := store.conversationIndexProof(ctx, convID)
	require.NoError(t, err)
	require.True(t, resolved)
	require.Equal(t, conversationIndexProofPopulated, proof)
	scansAfterFirstRead := store.scanInvocations.Load()

	// An index-unaware writer now lands a response into that same, already
	// populated conversation — payload only, no index entry — exactly the
	// rolling-upgrade case a populated proof used to claim was impossible.
	directSetResponsePayloadWithTTL(t, store, &responseapi.StoredResponse{
		ID: "resp_late_arrival", ConversationID: convID, Status: "completed", CreatedAt: now + 1,
	}, time.Hour)

	// While the proof stands it is invisible, and no scan is repeated: the
	// blind spot is real, which is precisely why it must be time-bounded.
	responses, err = store.ListResponsesByConversation(ctx, convID, ListOptions{Order: "asc"})
	require.NoError(t, err)
	assert.Equal(t, []string{"resp_late_early"}, responseIDsOf(responses))
	assert.Equal(t, scansAfterFirstRead, store.scanInvocations.Load(),
		"a standing proof must be trusted without re-scanning")

	// Meanwhile the conversation keeps taking ordinary indexed traffic from
	// already-upgraded pods, continuously, for longer than the proof's own
	// lifetime. This is the exact shape that used to make the blind spot
	// permanent: every one of these writes refreshed the populated proof,
	// so on a busy conversation it never expired and resp_late_arrival was
	// hidden for good.
	stopTraffic := startIndexedWriteTraffic(t, store, convID, now)
	defer stopTraffic()

	require.Eventuallyf(t, func() bool {
		_, stillResolved, proofErr := store.conversationIndexProof(ctx, convID)
		return proofErr == nil && !stillResolved
	}, 6*time.Second, 50*time.Millisecond, "the populated proof must expire on schedule even under continuous indexed writes")

	stopTraffic()

	// The next read revalidates: it re-scans and discovers the late write.
	responses, err = store.ListResponsesByConversation(ctx, convID, ListOptions{Order: "desc", Limit: MaxListLimit})
	require.NoError(t, err)
	assert.Containsf(t, responseIDsOf(responses), "resp_late_arrival",
		"once the bounded proof lapses, the late unindexed write must become visible")
	assert.Contains(t, responseIDsOf(responses), "resp_late_early")
	assert.Greaterf(t, store.scanInvocations.Load(), scansAfterFirstRead,
		"the lapsed proof must have forced a revalidating re-scan")
}

// startIndexedWriteTraffic keeps an ordinary index-aware writer busy on
// conversationID until the returned stop is called, simulating an
// already-upgraded pod serving that conversation while a proof's lifetime
// elapses. Safe to call stop more than once.
func startIndexedWriteTraffic(t *testing.T, store *RedisStore, conversationID string, baseCreatedAt int64) func() {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})

	go func() {
		defer close(done)
		for i := 0; ctx.Err() == nil; i++ {
			// Best-effort: this goroutine exists to keep touching the
			// conversation, not to assert on any individual write, and a
			// write racing the test's own teardown is not a failure.
			_ = store.StoreResponse(ctx, &responseapi.StoredResponse{
				ID:             fmt.Sprintf("resp_traffic_%d", i),
				ConversationID: conversationID,
				Status:         "completed",
				CreatedAt:      baseCreatedAt + int64(i) + 2,
			})
			select {
			case <-ctx.Done():
			case <-time.After(100 * time.Millisecond):
			}
		}
	}()

	var once sync.Once
	return func() {
		once.Do(func() {
			cancel()
			<-done
		})
	}
}

// TestRedisConversationIndexProofEmptyNeverExtendedByIndexedWrite covers the
// empty side of the same rule the populated tests above cover: indexResponse
// must never refresh (or otherwise touch) a proof, even though an ordinary
// write to that conversation ID makes an "empty" value stale the moment it
// lands. StoreResponse, duplicate repair (via a direct-set + retry), and
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
