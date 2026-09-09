package responsestore

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// responseUpdateSnapshot captures a response payload and remaining lifetime at
// one instant. A failed update restores the logical snapshot with a fresh
// generation and its projected TTL; the failed write's generation is the CAS
// witness, never serialized byte equality.
type responseUpdateSnapshot struct {
	data           []byte
	response       *responseapi.StoredResponse
	generation     string
	conversationID string
	createdAt      int64
	pttlMillis     int64 // -1 persistent, -2/absent handled as ErrNotFound before a snapshot exists
	capturedAt     time.Time
}

// remainingTTLMillis projects the snapshot's captured PTTL forward by the
// wall-clock time elapsed since it was taken, so a rollback that runs some
// time after the snapshot restores an approximately-correct remaining
// lifetime rather than either the stale original PTTL or a full TTL reset.
// A persistent snapshot (-1) stays persistent; a TTL that has since elapsed
// clamps to 0 (compareRestoreResponsePayload treats 0 as "delete instead of
// restore" — restoring a value whose TTL already ran out would resurrect
// data that was supposed to have expired).
func (snapshot responseUpdateSnapshot) remainingTTLMillis() int64 {
	if snapshot.pttlMillis < 0 {
		return -1
	}
	remaining := snapshot.pttlMillis - time.Since(snapshot.capturedAt).Milliseconds()
	if remaining < 0 {
		return 0
	}
	return remaining
}

// compareRestoreResult reports what compareRestoreResponsePayload actually
// did, since "the CAS didn't match" and "the CAS matched but the snapshot's
// TTL had elapsed" both need different handling from the caller (see
// rollbackUpdatePayload).
type compareRestoreResult int64

const (
	// compareRestoreConflict means the key's current value no longer
	// carries the expected generation: a newer write already landed, and rollback
	// must not clobber it or reindex the snapshot it was about to restore.
	compareRestoreConflict compareRestoreResult = 0
	// compareRestoreRestored means the logical snapshot was written under a
	// fresh generation with its projected remaining TTL.
	compareRestoreRestored compareRestoreResult = 1
	// compareRestoreExpired means the CAS matched, but the snapshot's TTL
	// had already elapsed by the time of the restore, so the key was
	// deleted instead of resurrected with a stale value past its own
	// intended lifetime.
	compareRestoreExpired compareRestoreResult = 2
)

// replaceAndSnapshotResponseScript atomically captures the payload and PTTL
// that this update is replacing and installs the update's new payload. Keeping
// the read and write in one single-key script prevents two overlapping updates
// from both retaining the same stale rollback snapshot. ARGV[2] is the new
// payload TTL in milliseconds; zero means persistent. Single-key: KEYS[1]
// only, so the script is Redis Cluster safe.
var replaceAndSnapshotResponseScript = redis.NewScript(`
local previous = redis.call("GET", KEYS[1])
if not previous then
	return nil
end
local previous_ttl = redis.call("PTTL", KEYS[1])
local new_ttl = tonumber(ARGV[2])
if new_ttl > 0 then
	redis.call("SET", KEYS[1], ARGV[1], "PX", new_ttl)
else
	redis.call("SET", KEYS[1], ARGV[1])
end
return {previous, previous_ttl}
`)

func (s *RedisStore) StoreResponse(ctx context.Context, response *responseapi.StoredResponse) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if response == nil || response.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + response.ID)

	generation := newResponseGeneration()
	data, err := marshalResponseRecord(response, generation)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	// Atomic existence check, and it lands the payload before the index entry, so
	// a member always postdates its payload — what makes prune-on-missing safe.
	stored, err := s.client.SetNX(ctx, key, data, s.ttl).Result()
	if err != nil {
		return fmt.Errorf("failed to store response in Redis: %w", err)
	}
	if !stored {
		return s.handleDuplicateResponse(ctx, response)
	}

	if response.ConversationID == "" {
		return nil
	}

	if err := s.indexResponse(ctx, response.ConversationID, response.ID, generation, response.CreatedAt, s.ttlMillis()); err != nil {
		return s.rollbackStoredPayload(ctx, key, generation, err)
	}

	return nil
}

// handleDuplicateResponse runs when StoreResponse's SETNX finds the
// response ID already stored. The payload already there is the source of
// truth for whether this retry needs an index repair, not the caller's
// attempted request: verified inside repairExistingResponseIndex.
func (s *RedisStore) handleDuplicateResponse(ctx context.Context, response *responseapi.StoredResponse) error {
	if response.ConversationID != "" {
		if repairErr := s.repairExistingResponseIndex(ctx, response); repairErr != nil {
			return repairErr
		}
	}
	return ErrAlreadyExists
}

// rollbackStoredPayload runs when a freshly stored response's index write
// fails. Compare-delete rollback: never a blind DEL. Only removes the
// payload if it still carries the generation this call wrote, so a concurrent
// writer that stored a new value after this payload's TTL expired is never
// clobbered (the ABA race the blueprint calls out).
//
// Detached from the caller's context, with its own deadline. The likeliest
// reason the index write failed at all is that the caller's context was
// cancelled — an HTTP client disconnecting cancels the request context
// mid-call — and rolling back on that same context fails for exactly the same
// reason, leaving the payload durably stored with no index entry naming it.
// That response is invisible to every listing, and once
// FinalizeConversationIndex has sealed the store, to every scan that could
// have rediscovered it.
func (s *RedisStore) rollbackStoredPayload(ctx context.Context, key, generation string, indexErr error) error {
	wrapped := fmt.Errorf("failed to index response in Redis: %w", indexErr)

	rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), responseCompensationTimeout)
	defer cancel()

	deleted, rollbackErr := s.compareDeleteResponsePayload(rollbackCtx, key, generation)
	if rollbackErr != nil {
		return fmt.Errorf("%w (rollback failed: %w)", wrapped, rollbackErr)
	}
	if !deleted {
		return fmt.Errorf("%w (payload changed before rollback, left in place)", wrapped)
	}
	return wrapped
}

// compareDeleteResponsePayload deletes only the generation owned by the
// caller. The Lua script extracts _vsr_generation with cjson; legacy or
// malformed payloads fail closed and are never compared by serialized bytes.
func (s *RedisStore) compareDeleteResponsePayload(ctx context.Context, key, expectedGeneration string) (bool, error) {
	if expectedGeneration == "" {
		return false, nil
	}
	res, err := compareDeleteGenerationScript.Run(ctx, s.client, []string{key}, expectedGeneration).Result()
	if err != nil {
		return false, fmt.Errorf("failed to compare-delete response payload %s: %w", key, err)
	}

	deleted, ok := res.(int64)
	if !ok {
		return false, fmt.Errorf("unexpected compare-delete result type %T for %s", res, key)
	}

	return deleted > 0, nil
}

// repairExistingResponseIndex runs when StoreResponse's SETNX finds the
// response ID already stored. It never trusts the caller's attempted
// payload: the stored response is read back and is the only source of truth
// for whether — and under which conversation — the index should be
// repaired. A duplicate ID whose stored payload belongs to a different
// conversation than the one the caller attempted must not poison that
// conversation's index.
func (s *RedisStore) repairExistingResponseIndex(ctx context.Context, attempted *responseapi.StoredResponse) error {
	stored, lifetimeMillis, err := s.getResponseWithLifetime(ctx, attempted.ID)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			// SETNX reported existence, but the payload is gone now (raced
			// with a delete, or expired). Nothing to repair from; keep the
			// duplicate contract.
			return nil
		}
		return fmt.Errorf("failed to read stored response %s for index repair: %w", attempted.ID, err)
	}

	if stored.response.ConversationID == "" || stored.response.ConversationID != attempted.ConversationID {
		// Either no index is expected, or the stored payload proves this
		// duplicate belongs to a different conversation than attempted.
		// Repairing the attempted conversation's index here would be
		// indexing a response that conversation does not actually own.
		return nil
	}

	if err := s.indexResponse(ctx, stored.response.ConversationID, stored.response.ID, stored.generation, stored.response.CreatedAt, lifetimeMillis); err != nil {
		return fmt.Errorf("response already exists but failed to repair conversation index: %w", err)
	}

	return nil
}

func (s *RedisStore) GetResponse(ctx context.Context, responseID string) (*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if responseID == "" {
		return nil, ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + responseID)

	data, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to get response from Redis: %w", err)
	}

	record, err := decodeResponseRecord(data)
	if err != nil {
		return nil, err
	}

	return record.response, nil
}

// UpdateResponse replaces a response's payload and keeps its conversation
// index in step: indexes under the new ConversationID (if any), then
// best-effort unindexes the previous one if it changed. If the new index
// write fails, restores the previous logical payload with a fresh generation and best-effort
// reindexes the previous conversation before returning the error — an
// update must never leave a payload pointing at a conversation whose index
// was never actually written, matching the repairability blueprint §5 Phase
// 5 asks for on top of StoreResponse's existing rollback/repair (Phase 2).
//
// Deliberate ordering deviation from the blueprint's UpdateResponse
// pseudocode: this checks the new index write's outcome *before*
// unindexing the previous conversation, not after. Unindexing first and
// only then discovering the new write failed would leave a window where
// neither the old nor the new conversation's index has the response, closed
// here only by the subsequent best-effort reindex; checking first avoids
// that window ever opening.
func (s *RedisStore) UpdateResponse(ctx context.Context, response *responseapi.StoredResponse) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if response == nil || response.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + response.ID)

	generation := newResponseGeneration()
	data, err := marshalResponseRecord(response, generation)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	snapshot, err := s.replaceResponseAndSnapshot(ctx, key, response.ID, data)
	if err != nil {
		return err
	}

	if response.ConversationID != "" {
		if err := s.indexResponse(ctx, response.ConversationID, response.ID, generation, response.CreatedAt, s.ttlMillis()); err != nil {
			return s.rollbackUpdatePayload(ctx, key, response.ID, generation, snapshot, err)
		}
	}

	// The previous generation owns its sidecar witness. Removing it is one
	// conditional same-slot ZREM+HDEL, so a move back that has already
	// installed a newer generation cannot be erased by this stale cleanup.
	if snapshot.conversationID != "" && snapshot.conversationID != response.ConversationID {
		witness := responseGenerationWitness{responseID: response.ID, generation: snapshot.generation}
		if err := s.unindexResponseGenerations(ctx, snapshot.conversationID, witness); err != nil {
			logging.Warnf("RedisStore: failed to remove response %s from previous conversation %s index: %v",
				response.ID, snapshot.conversationID, err)
		}
	}

	return nil
}

// replaceResponseAndSnapshot serializes concurrent payload replacements at
// Redis and returns the exact state displaced by this update. A failed update
// can therefore restore only its immediate predecessor, never a snapshot that
// another successful overlapping update had already superseded.
func (s *RedisStore) replaceResponseAndSnapshot(ctx context.Context, key, responseID string, data []byte) (responseUpdateSnapshot, error) {
	capturedAt := time.Now()

	result, err := replaceAndSnapshotResponseScript.Run(ctx, s.client, []string{key}, data, s.ttlMillis()).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return responseUpdateSnapshot{}, ErrNotFound
		}
		return responseUpdateSnapshot{}, fmt.Errorf("failed to replace response %s: %w", responseID, err)
	}

	return decodeResponseUpdateSnapshot(result, capturedAt, responseID)
}

// decodeResponseUpdateSnapshot turns replaceAndSnapshotResponseScript's
// {value, pttl} reply into the snapshot a failed update's rollback needs,
// best-effort parsing the displaced payload. A parse failure is not fatal to
// the update, but rollback will fail closed by deleting only the failed
// update's generation instead of resurrecting undecodable bytes.
func decodeResponseUpdateSnapshot(result interface{}, capturedAt time.Time, responseID string) (responseUpdateSnapshot, error) {
	items, ok := result.([]interface{})
	if !ok || len(items) != 2 {
		return responseUpdateSnapshot{}, fmt.Errorf("unexpected snapshot result shape for response %s: %#v", responseID, result)
	}
	data, ok := items[0].(string)
	if !ok {
		return responseUpdateSnapshot{}, fmt.Errorf("unexpected snapshot payload type for response %s: %T", responseID, items[0])
	}
	pttlMillis, ok := items[1].(int64)
	if !ok {
		return responseUpdateSnapshot{}, fmt.Errorf("unexpected snapshot PTTL type for response %s: %T", responseID, items[1])
	}

	snapshot := responseUpdateSnapshot{data: []byte(data), pttlMillis: pttlMillis, capturedAt: capturedAt}

	record, err := decodeResponseRecord(snapshot.data)
	if err != nil {
		logging.Warnf("RedisStore: failed to parse previous stored response %s during update: %v", responseID, err)
		return snapshot, nil
	}
	snapshot.response = record.response
	snapshot.generation = record.generation
	snapshot.conversationID = record.response.ConversationID
	snapshot.createdAt = record.response.CreatedAt

	return snapshot, nil
}

// compareRestoreResponsePayload restores previous with the snapshot's
// projected remaining TTL, but only if key's current payload still carries
// expectedGeneration — the failed update's unique token — so a rollback
// can never clobber a newer concurrent write. remainingTTLMillis follows
// responseUpdateSnapshot.remainingTTLMillis's convention: -1 persistent, 0
// means the snapshot's own TTL has since elapsed (delete rather than
// resurrect), positive is milliseconds remaining.
//
// Single-key Lua script — touches only KEYS[1] — so it stays legal in Redis
// Cluster, matching compareDeleteResponsePayload.
func (s *RedisStore) compareRestoreResponsePayload(ctx context.Context, key, expectedGeneration string, previous []byte, remainingTTLMillis int64) (compareRestoreResult, error) {
	res, err := compareRestoreGenerationScript.Run(ctx, s.client, []string{key}, expectedGeneration, previous, remainingTTLMillis).Result()
	if err != nil {
		return compareRestoreConflict, fmt.Errorf("failed to compare-restore response payload %s: %w", key, err)
	}

	code, ok := res.(int64)
	if !ok {
		return compareRestoreConflict, fmt.Errorf("unexpected compare-restore result type %T for %s", res, key)
	}

	return compareRestoreResult(code), nil
}

// rollbackUpdatePayload runs when an update's new-conversation index write
// fails. Restores the pre-update snapshot under a newly minted generation via
// compare-and-swap against the failed write's generation — never a blind SET — so a
// newer concurrent update landing between this update's write and its
// failed rollback is never overwritten: compareRestoreConflict means
// exactly that happened, and this leaves the newer payload alone rather
// than reindexing the stale snapshot it was about to restore.
// compareRestoreExpired means the CAS matched but the snapshot's own TTL
// had elapsed by the time of the restore, so the key was deleted instead —
// also not reindexed, since there is nothing left to point an index at.
// Only compareRestoreRestored reindexes the previous conversation, using
// its own stored CreatedAt, never a fabricated timestamp.
func (s *RedisStore) rollbackUpdatePayload(ctx context.Context, key, responseID, failedGeneration string, snapshot responseUpdateSnapshot, indexErr error) error {
	wrapped := fmt.Errorf("failed to index updated response in Redis: %w", indexErr)

	// Detached, for the reason spelled out on rollbackStoredPayload: a
	// cancellation that arrives after the payload replacement commits fails
	// the index write and would fail this rollback too, stranding the response
	// under its new conversation with neither conversation's index naming it.
	// The compensating reindex below runs on the same detached context, since
	// restoring the payload without restoring its membership just moves the
	// inconsistency rather than repairing it.
	rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), responseCompensationTimeout)
	defer cancel()

	if snapshot.response == nil {
		_, deleteErr := s.compareDeleteResponsePayload(rollbackCtx, key, failedGeneration)
		if deleteErr != nil {
			return fmt.Errorf("%w (rollback failed: %w)", wrapped, deleteErr)
		}
		return fmt.Errorf("%w (previous payload was not restorable)", wrapped)
	}

	restoredGeneration := newResponseGeneration()
	restoredData, marshalErr := marshalResponseRecord(snapshot.response, restoredGeneration)
	if marshalErr != nil {
		return fmt.Errorf("%w (rollback failed: %w)", wrapped, marshalErr)
	}
	remainingTTL := snapshot.remainingTTLMillis()
	result, restoreErr := s.compareRestoreResponsePayload(rollbackCtx, key, failedGeneration, restoredData, remainingTTL)
	if restoreErr != nil {
		return fmt.Errorf("%w (rollback failed: %w)", wrapped, restoreErr)
	}

	switch result {
	case compareRestoreRestored:
		if snapshot.conversationID != "" {
			// The restored payload carries the snapshot's own remaining
			// lifetime, not a fresh store TTL, so the index is extended to
			// match what was actually put back.
			if reindexErr := s.indexResponse(rollbackCtx, snapshot.conversationID, responseID, restoredGeneration, snapshot.createdAt, remainingTTL); reindexErr != nil {
				logging.Warnf("RedisStore: failed to reindex restored response %s under previous conversation %s after update rollback: %v",
					responseID, snapshot.conversationID, reindexErr)
			}
		}
	case compareRestoreConflict:
		logging.Debugf("RedisStore: update rollback for response %s found a newer payload already in place; left it alone", responseID)
	case compareRestoreExpired:
		logging.Debugf("RedisStore: update rollback for response %s found its snapshot's TTL already elapsed; deleted rather than restored", responseID)
	}

	return wrapped
}

func (s *RedisStore) DeleteResponse(ctx context.Context, responseID string) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if responseID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + responseID)

	// The single-key script is the deletion linearization point. Its returned
	// bytes identify the exact generation whose sidecar witness this call owns;
	// a StoreResponse that recreates the ID afterwards receives a fresh witness.
	result, err := takeResponsePayloadScript.Run(ctx, s.client, []string{key}).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return ErrNotFound
		}
		return fmt.Errorf("failed to delete response from Redis: %w", err)
	}
	data, ok := result.(string)
	if !ok {
		return fmt.Errorf("unexpected response delete result type %T", result)
	}

	record, decodeErr := decodeResponseRecord([]byte(data))
	if decodeErr != nil {
		logging.Warnf("RedisStore: deleted response %s but could not decode its index ownership: %v", responseID, decodeErr)
		return nil
	}

	// Best-effort: the payload delete above is the user-visible operation, and
	// only the deleted generation may authorize removal. Legacy payloads carry
	// no witness and therefore deliberately leave their member untouched.
	witness := responseGenerationWitness{responseID: responseID, generation: record.generation}
	if err := s.unindexResponseGenerations(ctx, record.response.ConversationID, witness); err != nil {
		logging.Warnf("RedisStore: failed to remove response %s from conversation %s index: %v",
			responseID, record.response.ConversationID, err)
	}

	return nil
}

// GetConversationChain retrieves the full conversation chain for a response.
// It follows the previous_response_id links backwards to build the complete history.
func (s *RedisStore) GetConversationChain(ctx context.Context, responseID string) ([]*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if responseID == "" {
		return nil, ErrInvalidInput
	}

	// Phase 1: Collect response IDs by following the chain
	responseIDs, err := s.collectChainIDs(ctx, responseID)
	if err != nil {
		return nil, err
	}

	if len(responseIDs) == 0 {
		return []*responseapi.StoredResponse{}, nil
	}

	// Phase 2: Fetch all responses using pipelining
	chain, _, err := s.fetchResponsesPipelined(ctx, responseIDs)
	if err != nil {
		return nil, err
	}

	// Phase 3: Reverse chain to get chronological order (oldest first)
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}

	return chain, nil
}
