package responsestore

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// TestLegacyPayloadUpgradePreservesContentAndLifetime pins the promotion
// primitive every legacy path depends on: the record gains a generation and
// nothing else changes, least of all how long it has left. A legacy payload
// routinely outlives the store's current TTL — re-stamping it with s.ttl would
// silently retire data early — and the upgrade is conditional on the exact
// bytes read, so a stale caller can never overwrite a payload that moved on.
func TestLegacyPayloadUpgradePreservesContentAndLifetime(t *testing.T) {
	store := newConversationIndexStoreWithTTLSeconds(t, 30)
	ctx := context.Background()

	const responseID = "resp_legacy_upgrade"
	legacy := &responseapi.StoredResponse{
		ID:             responseID,
		ConversationID: "conv_legacy_upgrade",
		Status:         "completed",
		OutputText:     "written before generations existed",
		CreatedAt:      time.Now().Unix(),
	}
	// Deliberately far longer than the store's own TTL.
	directSetResponsePayloadWithTTL(t, store, legacy, 10*time.Minute)

	key := store.buildKey(ResponseKeyPrefix + responseID)
	raw, err := store.client.Get(ctx, key).Bytes()
	require.NoError(t, err)
	record, err := decodeResponseRecord(raw)
	require.NoError(t, err)
	require.Empty(t, record.generation, "precondition: the payload must be generation-less")

	generation, ttlMillis, upgraded, err := promoteLegacyResponsePayload(ctx, store.client, key, record)
	require.NoError(t, err)
	require.True(t, upgraded)
	require.NoError(t, validateResponseGeneration(generation))

	upgradedRaw, err := store.client.Get(ctx, key).Bytes()
	require.NoError(t, err)
	decoded, err := decodeResponseRecord(upgradedRaw)
	require.NoError(t, err)
	assert.Equal(t, generation, decoded.generation)
	assert.Equal(t, *legacy, *decoded.response, "the upgrade must change nothing but the generation")

	// The lifetime must be the one observed inside the upgrade itself, not a
	// fresh store TTL and not a value captured before the write.
	assert.Greater(t, ttlMillis, (9 * time.Minute).Milliseconds(),
		"the upgrade must report the payload's own remaining lifetime")
	assert.LessOrEqual(t, ttlMillis, (10 * time.Minute).Milliseconds())
	pttl, err := store.client.PTTL(ctx, key).Result()
	require.NoError(t, err)
	assert.InDelta(t, ttlMillis, pttl.Milliseconds(), 2000)

	// Replaying the now-stale bytes must not resurrect them over the upgrade.
	_, _, replayed, err := promoteLegacyResponsePayload(ctx, store.client, key, record)
	require.NoError(t, err)
	assert.False(t, replayed, "an upgrade that lost its race must report so, not overwrite")
	after, err := store.client.Get(ctx, key).Bytes()
	require.NoError(t, err)
	assert.Equal(t, upgradedRaw, after)
}

// TestLegacyPayloadUpgradePreservesUnknownFields is the data-integrity half of
// the same contract. Remarshaling through StoredResponse would silently drop
// any field this build does not know about — one written by a newer binary, or
// one retired from the struct but still present in stored data — which would
// make "non-destructive upgrade" false.
func TestLegacyPayloadUpgradePreservesUnknownFields(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const responseID = "resp_legacy_unknown_fields"
	key := store.buildKey(ResponseKeyPrefix + responseID)
	stored := map[string]any{
		"id":                responseID,
		"conversation_id":   "conv_legacy_unknown_fields",
		"status":            "completed",
		"created_at":        time.Now().Unix(),
		"future_scalar":     "written by a newer binary",
		"future_object":     map[string]any{"nested": []any{1, 2, 3}},
		"retired_from_type": true,
	}
	raw, err := json.Marshal(stored)
	require.NoError(t, err)
	require.NoError(t, store.client.Set(ctx, key, raw, store.ttl).Err())

	record, err := decodeResponseRecord(raw)
	require.NoError(t, err)
	_, _, upgraded, err := promoteLegacyResponsePayload(ctx, store.client, key, record)
	require.NoError(t, err)
	require.True(t, upgraded)

	upgradedRaw, err := store.client.Get(ctx, key).Bytes()
	require.NoError(t, err)
	var got map[string]any
	require.NoError(t, json.Unmarshal(upgradedRaw, &got))

	for field, want := range stored {
		assert.EqualValues(t, want, got[field], "upgrade dropped or altered %q", field)
	}
	assert.Contains(t, got, responseGenerationField)
	assert.Len(t, got, len(stored)+1, "the upgrade must add exactly one field")
}

// TestScanUpgradeDoesNotClobberNewerWitness covers the interleaving that makes
// scan writes compare-and-set rather than blind. A scan reads a payload,
// a concurrent update then commits a newer payload and installs its own
// witness, and the scan's delayed index write must not stamp the generation it
// read over the live one — after which a conditional prune would, quite
// correctly, remove a membership whose payload is alive.
func TestScanUpgradeDoesNotClobberNewerWitness(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_witness_clobber"
	const responseID = "resp_witness_clobber"
	createdAt := time.Now().Unix()

	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: responseID, ConversationID: convID, Status: "live", CreatedAt: createdAt,
	}))
	live := indexedGeneration(t, store, convID, responseID)

	// A scan that read some earlier generation now tries to index it.
	stale := newResponseGeneration()
	installed, err := store.repairResponseWitness(ctx, convID, responseID, stale, "", createdAt, store.ttlMillis())
	require.NoError(t, err)
	assert.Zero(t, installed, "a scan must not install over a witness a live writer owns")
	assert.Equal(t, live, indexedGeneration(t, store, convID, responseID))

	// A writer that owns the generation it just wrote still overwrites.
	owned := newResponseGeneration()
	require.NoError(t, store.indexResponse(ctx, convID, responseID, owned, createdAt, store.ttlMillis()))
	assert.Equal(t, owned, indexedGeneration(t, store, convID, responseID))

	// And a repair that names the witness it observed wins its compare-and-set.
	repaired := newResponseGeneration()
	installed, err = store.repairResponseWitness(ctx, convID, responseID, repaired, owned, createdAt, store.ttlMillis())
	require.NoError(t, err)
	assert.Equal(t, 1, installed)
	assert.Equal(t, repaired, indexedGeneration(t, store, convID, responseID))
}

// TestFinalizeThenCascadeDeletesLegacyResponses is the finalize-then-cascade
// regression the review asked for. Finalization used to index pre-upgrade
// payloads without a generation witness, after which cascade delete refused
// every one of them — leaving conversations that could never be deleted, with
// no rescan left to repair them. The sweep now upgrades each legacy payload as
// it indexes it, so finalization produces an index whose members are all
// deletable.
func TestFinalizeThenCascadeDeletesLegacyResponses(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_finalize_legacy"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))

	now := time.Now().Unix()
	legacyIDs := []string{"resp_finalize_legacy_1", "resp_finalize_legacy_2"}
	for i, id := range legacyIDs {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: id, ConversationID: convID, Status: "completed", CreatedAt: now + int64(i),
		})
	}
	require.Empty(t, conversationIndexMembers(t, store, convID),
		"precondition: legacy payloads are not indexed")

	stats, err := store.FinalizeConversationIndex(ctx)
	require.NoError(t, err)
	assert.EqualValues(t, len(legacyIDs), stats.ResponsesIndexed)

	assert.ElementsMatch(t, legacyIDs, conversationIndexMembers(t, store, convID))
	for _, id := range legacyIDs {
		assert.NotEmpty(t, optionalIndexedGeneration(t, store, convID, id),
			"finalization must upgrade legacy payloads, not index them witness-less")
	}

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	for _, id := range legacyIDs {
		_, err := store.GetResponse(ctx, id)
		assert.ErrorIs(t, err, ErrNotFound, "finalized legacy responses must be cascade-deletable")
	}
	assert.Empty(t, conversationIndexMembers(t, store, convID))
	assert.Zero(t, exists(t, store, store.conversationIndexGenerationKey(convID)))
}

// TestCascadeDeleteUpgradesLiveLegacyPayloadBeforeFinalization covers the
// residual case finalization's best-effort upgrade can leave behind: a member
// indexed without a witness because its upgrade failed or lost a race. A
// *live* legacy payload needs no finalization gate, because it is never
// removed on the strength of its blankness — it is upgraded, its witness is
// installed, and a later round deletes it under the ordinary generation CAS.
func TestCascadeDeleteUpgradesLiveLegacyPayloadBeforeFinalization(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_live_legacy"
	const responseID = "resp_cascade_live_legacy"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))

	createdAt := time.Now().Unix()
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: responseID, ConversationID: convID, Status: "completed", CreatedAt: createdAt,
	})
	seedLegacyIndexMember(t, store, convID, responseID, createdAt)
	// Resolve the conversation so the cascade reads the index as seeded rather
	// than running a backfill that would upgrade the payload on its own.
	require.NoError(t, store.markConversationMigrated(ctx, convID, conversationIndexProofPopulated))
	require.Zero(t, exists(t, store, store.conversationIndexCompletionKey()),
		"precondition: the store must not be finalized")

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	_, err := store.GetResponse(ctx, responseID)
	assert.ErrorIs(t, err, ErrNotFound)
	assert.Empty(t, conversationIndexMembers(t, store, convID))
}

// TestCascadeDeleteRefusesBlankTombstoneBeforeFinalization is the gate itself.
// The payload behind a blank member is already gone, so nothing can be
// upgraded and only the blankness could authorize the removal — but before
// finalization an index-unaware writer can recreate that payload without
// touching the sidecar, and cascade would then delete the conversation and its
// migrated marker on top of a live, unindexed response that no later scan
// would ever rediscover. Failing closed keeps the conversation as the retry
// anchor instead.
func TestCascadeDeleteRefusesBlankTombstoneBeforeFinalization(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_blank_gate"
	const responseID = "resp_cascade_blank_gate"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))

	// A member whose legacy payload has already expired: no payload to upgrade.
	seedLegacyIndexMember(t, store, convID, responseID, time.Now().Unix())
	require.NoError(t, store.markConversationMigrated(ctx, convID, conversationIndexProofPopulated))

	err := store.DeleteConversation(ctx, convID, true)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not finalized")

	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, convID),
		"the membership must survive as the retry anchor")
	_, err = store.GetConversation(ctx, convID)
	assert.NoError(t, err, "the conversation remains until the store is finalized")
}

// TestCascadeDeleteDropsBlankTombstoneAfterFinalization is the other side of
// the gate: once every writer is generation-aware, a blank-expected removal
// can no longer be defeated, and the tombstone must stop blocking the cascade.
func TestCascadeDeleteDropsBlankTombstoneAfterFinalization(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_blank_finalized"
	const responseID = "resp_cascade_blank_finalized"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))

	seedLegacyIndexMember(t, store, convID, responseID, time.Now().Unix())
	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
		conversationIndexCompletionValue, 0).Err())

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	assert.Empty(t, conversationIndexMembers(t, store, convID))
	_, err := store.GetConversation(ctx, convID)
	assert.ErrorIs(t, err, ErrNotFound)
}

// TestCascadeDeleteRepairsStaleWitnessInsteadOfRemoving covers the rule that a
// live payload this conversation still owns is never unindexed, whatever its
// witness says. Removing the membership instead would leave a live response
// indexed nowhere, which past finalization nothing rediscovers.
func TestCascadeDeleteRepairsStaleWitnessInsteadOfRemoving(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_cascade_stale_witness"
	const responseID = "resp_cascade_stale_witness"
	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID: convID, CreatedAt: time.Now().Unix(),
	}))

	createdAt := time.Now().Unix()
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: responseID, ConversationID: convID, Status: "completed", CreatedAt: createdAt,
	}))
	// Force the sidecar to disagree with the payload, the state a delayed scan
	// write or an in-flight index write produces.
	stale := newResponseGeneration()
	require.NoError(t, store.indexResponse(ctx, convID, responseID, stale, createdAt, store.ttlMillis()))
	require.Equal(t, stale, indexedGeneration(t, store, convID, responseID))

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	_, err := store.GetResponse(ctx, responseID)
	assert.ErrorIs(t, err, ErrNotFound, "the live payload must be deleted, not merely unindexed")
	assert.Empty(t, conversationIndexMembers(t, store, convID))
}

// TestListPrunesExpiredLegacyMember is the pagination regression the review
// asked for. A backfilled legacy member has no witness, so once its payload
// expires nothing could remove it: the tombstone sat at the head of the window
// and underfilled every page that read it, forever. Here the whole first
// window is that tombstone, so before the fix the call returned an empty page
// — which terminates a client's pagination — while a live response sat one
// rank further along.
func TestListPrunesExpiredLegacyMember(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_list_expired_legacy"
	const expiredID = "resp_list_expired_legacy"
	const liveID = "resp_list_live"

	now := time.Now().Unix()
	// A member whose legacy payload has already expired: seeded with no
	// witness and no payload behind it.
	seedLegacyIndexMember(t, store, convID, expiredID, now)
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID: liveID, ConversationID: convID, Status: "completed", CreatedAt: now + 1,
	}))
	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
		conversationIndexCompletionValue, 0).Err())
	require.Equal(t, []string{expiredID, liveID}, conversationIndexMembers(t, store, convID),
		"precondition: the tombstone sorts ahead of the live response")

	responses, err := store.ListResponsesByConversation(ctx, convID, ListOptions{Order: "asc", Limit: 1})
	require.NoError(t, err)
	require.Len(t, responses, 1, "the page must list past the expired legacy member")
	assert.Equal(t, liveID, responses[0].ID)

	assert.Equal(t, []string{liveID}, conversationIndexMembers(t, store, convID),
		"the tombstone must be gone, not merely skipped")
}

// TestListKeepsBlankTombstoneBeforeFinalization documents the residual the
// gate deliberately accepts. During a rolling upgrade an index-unaware writer
// can still put a payload back under the same ID without touching the sidecar,
// so a blank-expected prune could unindex it; a short page for the duration of
// the migration window is the accepted price of never orphaning a record.
func TestListKeepsBlankTombstoneBeforeFinalization(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_list_blank_gate"
	const expiredID = "resp_list_blank_gate"

	seedLegacyIndexMember(t, store, convID, expiredID, time.Now().Unix())
	require.NoError(t, store.markConversationMigrated(ctx, convID, conversationIndexProofPopulated))

	responses, err := store.ListResponsesByConversation(ctx, convID, ListOptions{Order: "asc"})
	require.NoError(t, err)
	assert.Empty(t, responses)
	assert.Equal(t, []string{expiredID}, conversationIndexMembers(t, store, convID),
		"before finalization the tombstone is left in place rather than risking an orphan")
}

// TestListBlankWitnessPruneKeepsRecreatedGeneration proves the blank-witness
// cleanup is still generation-safe where it is allowed to run. The page
// observes a witness-less member with no payload behind it and decides to
// prune; a complete StoreResponse then lands in the GET-to-prune window and
// installs a real witness. The stale blank-expected cleanup must find the
// mismatch and no-op, leaving the recreated response both present and indexed.
func TestListBlankWitnessPruneKeepsRecreatedGeneration(t *testing.T) {
	store := newConversationIndexStore(t)
	writer := newConcurrentRedisStore(t, store)
	ctx := context.Background()

	const convID = "conv_blank_prune_race"
	const responseID = "resp_blank_prune_race"

	now := time.Now().Unix()
	seedLegacyIndexMember(t, store, convID, responseID, now)
	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
		conversationIndexCompletionValue, 0).Err())

	recreated := &responseapi.StoredResponse{
		ID: responseID, ConversationID: convID, Status: "recreated", CreatedAt: now + 1,
	}
	var injectedErr error
	hook := &commandInterleavingHook{
		pipeline: true,
		match: func(cmd redis.Cmder) bool {
			return commandReadsKey(cmd, store.buildKey(ResponseKeyPrefix+responseID))
		},
		inject: func() { injectedErr = writer.StoreResponse(context.Background(), recreated) },
	}
	store.client.AddHook(hook)

	responses, err := store.ListResponsesByConversation(ctx, convID, ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.NoError(t, injectedErr)
	assert.True(t, hook.fired.Load(), "the recreation must land in the GET-to-prune window")

	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, convID),
		"a blank-expected prune must not erase a membership a real generation just claimed")
	assert.NotEmpty(t, optionalIndexedGeneration(t, store, convID, responseID))
	require.Len(t, responses, 1, "the refill pass must return the recreated response")
	assert.Equal(t, "recreated", responses[0].Status)
}

// TestDeleteResponseUnindexesLegacyMemberOnceFinalized covers the same
// tombstone reached through the public delete API rather than through expiry.
// The gate applies here too, and for the sharpest version of the reason:
// DeleteResponse frees the payload key, so an index-unaware SETNX can win it
// back between the take and the cleanup.
func TestDeleteResponseUnindexesLegacyMemberOnceFinalized(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const convID = "conv_delete_legacy_member"
	const responseID = "resp_delete_legacy_member"

	createdAt := time.Now().Unix()
	seed := func() {
		directSetResponsePayload(t, store, &responseapi.StoredResponse{
			ID: responseID, ConversationID: convID, Status: "completed", CreatedAt: createdAt,
		})
		seedLegacyIndexMember(t, store, convID, responseID, createdAt)
	}

	seed()
	require.NoError(t, store.DeleteResponse(ctx, responseID))
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, convID),
		"before finalization the membership is left rather than risking an orphaned recreation")

	seed()
	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
		conversationIndexCompletionValue, 0).Err())
	require.NoError(t, store.DeleteResponse(ctx, responseID))

	_, err := store.GetResponse(ctx, responseID)
	assert.ErrorIs(t, err, ErrNotFound)
	assert.Empty(t, conversationIndexMembers(t, store, convID),
		"a finalized store must not leave the deleted response's membership behind")
}

// TestUpdateResponseUnindexesLegacyPreviousConversation covers the last
// blank-witness leak: moving a legacy response between conversations. The
// displaced payload had no generation, so under the gate the old membership
// survives during the migration window and is dropped once finalized.
func TestUpdateResponseUnindexesLegacyPreviousConversation(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const fromConv = "conv_legacy_move_from"
	const toConv = "conv_legacy_move_to"
	const responseID = "resp_legacy_move"

	createdAt := time.Now().Unix()
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: responseID, ConversationID: fromConv, Status: "completed", CreatedAt: createdAt,
	})
	seedLegacyIndexMember(t, store, fromConv, responseID, createdAt)
	require.NoError(t, store.client.Set(ctx, store.conversationIndexCompletionKey(),
		conversationIndexCompletionValue, 0).Err())

	require.NoError(t, store.UpdateResponse(ctx, &responseapi.StoredResponse{
		ID: responseID, ConversationID: toConv, Status: "moved", CreatedAt: createdAt,
	}))

	assert.Empty(t, conversationIndexMembers(t, store, fromConv),
		"the legacy membership the move displaced must be dropped")
	assert.Equal(t, []string{responseID}, conversationIndexMembers(t, store, toConv))
	assert.NotEmpty(t, optionalIndexedGeneration(t, store, toConv, responseID))
}
