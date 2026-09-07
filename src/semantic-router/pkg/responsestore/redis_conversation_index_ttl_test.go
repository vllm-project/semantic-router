package responsestore

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// indexPTTL reads a conversation index key's remaining lifetime directly, so
// these tests assert on the key's actual expiry rather than on a listing that
// would still succeed right up until the moment it silently stops.
func indexPTTL(t *testing.T, store *RedisStore, conversationID string) time.Duration {
	t.Helper()

	pttl, err := store.client.PTTL(context.Background(), store.conversationIndexKey(conversationID)).Result()
	require.NoError(t, err)
	return pttl
}

// TestFinalizedIndexOutlivesLongerLivedPayloads covers the finalization sweep's
// half of the rule: an index must never be stamped with the store's currently
// configured TTL when the payloads it names have longer to live. Once the
// completion key is set, an index that expires first is not a degraded read —
// a missing index means "empty", nothing ever rescans, and the surviving
// payloads are hidden permanently.
func TestFinalizedIndexOutlivesLongerLivedPayloads(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 60)
	ctx := context.Background()

	const conversationID = "conv_sweep_long_payload"
	const payloadTTL = 30 * time.Minute
	directSetResponsePayloadWithTTL(t, store, &responseapi.StoredResponse{
		ID: "resp_sweep_long", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix(),
	}, payloadTTL)

	stats, err := store.FinalizeConversationIndex(ctx)
	require.NoError(t, err)
	require.EqualValues(t, 1, stats.ResponsesIndexed)

	pttl := indexPTTL(t, store, conversationID)
	assert.Greater(t, pttl, store.ttl,
		"the index must not be retired on the store TTL while a much longer-lived payload is still indexed under it")
	assert.InDelta(t, payloadTTL.Seconds(), pttl.Seconds(), 30,
		"the index must cover the payload it names")
}

// TestBackfilledIndexOutlivesLongerLivedPayloads is the same rule on the lazy
// backfill path, which discovers exactly the same kind of legacy payload one
// conversation at a time.
func TestBackfilledIndexOutlivesLongerLivedPayloads(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 60)
	ctx := context.Background()

	const conversationID = "conv_backfill_long_payload"
	const payloadTTL = 30 * time.Minute
	directSetResponsePayloadWithTTL(t, store, &responseapi.StoredResponse{
		ID: "resp_backfill_long", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix(),
	}, payloadTTL)

	responses, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{})
	require.NoError(t, err)
	require.Len(t, responses, 1)

	assert.Greater(t, indexPTTL(t, store, conversationID), store.ttl)
}

// TestIndexLifetimeNeverShortenedByOrdinaryWrite is the reason the index write
// extends rather than sets. A backfill that correctly gave the index a 30-day
// reach would otherwise be undone by the very next ordinary write, which knows
// only the store's own TTL — leaving the long-lived payloads the backfill just
// discovered facing exactly the expiry the backfill existed to prevent.
func TestIndexLifetimeNeverShortenedByOrdinaryWrite(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 60)
	ctx := context.Background()

	const conversationID = "conv_no_shortening"
	const payloadTTL = 30 * time.Minute
	directSetResponsePayloadWithTTL(t, store, &responseapi.StoredResponse{
		ID: "resp_long_lived", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix(),
	}, payloadTTL)
	require.NoError(t, store.ensureConversationIndex(ctx, conversationID))
	require.Greater(t, indexPTTL(t, store, conversationID), store.ttl, "precondition: the backfill covered the long-lived payload")

	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_fresh", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}))

	assert.Greater(t, indexPTTL(t, store, conversationID), store.ttl,
		"an ordinary write knows only the store TTL and must never pull the index's expiry back down to it")
}

// TestIndexLifetimeCoversPersistentPayload covers the other end of the range:
// a payload with no expiry at all needs an index with no expiry, since any
// finite one would eventually retire ahead of it.
func TestIndexLifetimeCoversPersistentPayload(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 60)
	ctx := context.Background()

	const conversationID = "conv_persistent_payload"
	directSetResponsePayloadWithTTL(t, store, &responseapi.StoredResponse{
		ID: "resp_persistent", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix(),
	}, 0)

	require.NoError(t, store.ensureConversationIndex(ctx, conversationID))
	assert.EqualValues(t, -1, indexPTTL(t, store, conversationID),
		"an index naming a payload that never expires must not expire either")

	// And an ordinary write must not put a TTL back on it.
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: "resp_after_persistent", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}))
	assert.EqualValues(t, -1, indexPTTL(t, store, conversationID))
}

// TestFinalizedStoreDoesNotHideSurvivingPayloads is the end-to-end version of
// the rule, and the one that shows what the bug actually costs: with the store
// finalized and the index expired, a read of a conversation whose payloads are
// still very much alive returns nothing at all, forever, because nothing
// rescans after completion.
//
// The one sleep in these tests is unavoidable — the failure is an expiry, and
// only real elapsed time produces it.
func TestFinalizedStoreDoesNotHideSurvivingPayloads(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 1)
	ctx := context.Background()

	const conversationID = "conv_survives_index_ttl"
	directSetResponsePayloadWithTTL(t, store, &responseapi.StoredResponse{
		ID: "resp_survivor", ConversationID: conversationID,
		Status: "completed", CreatedAt: time.Now().Unix(),
	}, 5*time.Minute)

	_, err := store.FinalizeConversationIndex(ctx)
	require.NoError(t, err)
	scansAfterSweep := store.scanInvocations.Load()

	// Well past the store's own TTL, which is what the index used to be
	// stamped with, and nowhere near the payload's.
	time.Sleep(1500 * time.Millisecond)

	responses, err := store.ListResponsesByConversation(ctx, conversationID, ListOptions{})
	require.NoError(t, err)
	require.Len(t, responses, 1,
		"a finalized store must not report a conversation empty while its payloads are still live: nothing will ever rescan to correct it")
	assert.Equal(t, "resp_survivor", responses[0].ID)
	assert.Equal(t, scansAfterSweep, store.scanInvocations.Load(),
		"the read path must not have rescanned; past finalization the index alone has to be right")
}

// TestLongerIndexLifetime needs no Redis: it pins the "non-positive means
// never expires, and therefore wins" convention every index-lifetime argument
// in this package shares.
func TestLongerIndexLifetime(t *testing.T) {
	assert.EqualValues(t, 500, longerIndexLifetime(100, 500))
	assert.EqualValues(t, 500, longerIndexLifetime(500, 100))
	assert.EqualValues(t, 0, longerIndexLifetime(0, 500), "an unbounded lifetime is never narrowed by a finite one")
	assert.EqualValues(t, 0, longerIndexLifetime(500, -1), "a payload that never expires makes its index unbounded")
}

// TestScannedResponseIndexLifetime pins the treatment of a payload Redis could
// not measure: it contributes nothing rather than being read as unbounded,
// which would silently make every index touched by an expiring-payload race
// immortal.
func TestScannedResponseIndexLifetime(t *testing.T) {
	const storeTTL = 60_000

	assert.EqualValues(t, storeTTL, scannedResponse{ttlMillis: unknownPayloadTTL}.indexLifetime(storeTTL))
	assert.EqualValues(t, storeTTL, scannedResponse{ttlMillis: 1_000}.indexLifetime(storeTTL))
	assert.EqualValues(t, 90_000, scannedResponse{ttlMillis: 90_000}.indexLifetime(storeTTL))
	assert.EqualValues(t, 0, scannedResponse{ttlMillis: -1}.indexLifetime(storeTTL))
}
