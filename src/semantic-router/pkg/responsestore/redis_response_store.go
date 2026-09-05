package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// responseUpdateSnapshot captures a response payload's exact bytes and
// remaining lifetime at one instant, so a failed update's rollback can
// restore both the value and its TTL — not just the value with a
// freshly-reset TTL — and so the restore can be a compare-and-swap against
// the exact bytes this snapshot saw, never a blind write.
type responseUpdateSnapshot struct {
	data           []byte
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
	// matched expectedCurrent: a newer write already landed, and rollback
	// must not clobber it or reindex the snapshot it was about to restore.
	compareRestoreConflict compareRestoreResult = 0
	// compareRestoreRestored means the snapshot's bytes were written back
	// with its (projected) remaining TTL.
	compareRestoreRestored compareRestoreResult = 1
	// compareRestoreExpired means the CAS matched, but the snapshot's TTL
	// had already elapsed by the time of the restore, so the key was
	// deleted instead of resurrected with a stale value past its own
	// intended lifetime.
	compareRestoreExpired compareRestoreResult = 2
)

// snapshotResponseScript atomically reads a key's value and remaining PTTL
// in one round trip, so the two are consistent with each other (a separate
// GET then PTTL could observe the key expiring, or a concurrent write
// changing its TTL, in between the two commands). Single-key: KEYS[1] only.
var snapshotResponseScript = redis.NewScript(`
local value = redis.call("GET", KEYS[1])
if not value then
	return nil
end
return {value, redis.call("PTTL", KEYS[1])}
`)

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

// compareRestoreScript is StoreResponse's compareDeleteScript's counterpart
// for UpdateResponse's rollback: restore ARGV[2] with TTL ARGV[3]
// (milliseconds; 0 deletes instead, negative means persistent) only if the
// key's current value still equals ARGV[1] — never a blind SET, for the
// same ABA-race reason compareDeleteScript exists. Single-key: KEYS[1] only.
var compareRestoreScript = redis.NewScript(`
local current = redis.call("GET", KEYS[1])
if current ~= ARGV[1] then
	return 0
end

local ttl = tonumber(ARGV[3])
if ttl == 0 then
	redis.call("DEL", KEYS[1])
	return 2
elseif ttl < 0 then
	redis.call("SET", KEYS[1], ARGV[2])
else
	redis.call("SET", KEYS[1], ARGV[2], "PX", ttl)
end
return 1
`)

func (s *RedisStore) StoreResponse(ctx context.Context, response *responseapi.StoredResponse) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if response == nil || response.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + response.ID)

	data, err := json.Marshal(response)
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

	if err := s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt); err != nil {
		return s.rollbackStoredPayload(ctx, key, data, err)
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
// payload if it is still exactly what this call wrote, so a concurrent
// writer that stored a new value after this payload's TTL expired is never
// clobbered (the ABA race the blueprint calls out).
func (s *RedisStore) rollbackStoredPayload(ctx context.Context, key string, data []byte, indexErr error) error {
	wrapped := fmt.Errorf("failed to index response in Redis: %w", indexErr)

	deleted, rollbackErr := s.compareDeleteResponsePayload(ctx, key, data)
	if rollbackErr != nil {
		return fmt.Errorf("%w (rollback failed: %v)", wrapped, rollbackErr)
	}
	if !deleted {
		return fmt.Errorf("%w (payload changed before rollback, left in place)", wrapped)
	}
	return wrapped
}

// compareDeleteResponsePayload deletes key only if its current value equals
// expected. Used to roll back a response payload after its index write
// fails, without risking a blind DEL removing a value a concurrent writer
// stored after the original payload expired on its TTL; also reused by
// ensureConversationIndex to release the migration lock without releasing
// one it doesn't hold (e.g. one that expired and was re-acquired).
//
// Single-key Lua script — touches only KEYS[1] — so it stays legal in Redis
// Cluster; a two-key script spanning the response and index keys would not.
func (s *RedisStore) compareDeleteResponsePayload(ctx context.Context, key string, expected []byte) (bool, error) {
	res, err := compareDeleteScript.Run(ctx, s.client, []string{key}, expected).Result()
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
	stored, err := s.GetResponse(ctx, attempted.ID)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			// SETNX reported existence, but the payload is gone now (raced
			// with a delete, or expired). Nothing to repair from; keep the
			// duplicate contract.
			return nil
		}
		return fmt.Errorf("failed to read stored response %s for index repair: %w", attempted.ID, err)
	}

	if stored.ConversationID == "" || stored.ConversationID != attempted.ConversationID {
		// Either no index is expected, or the stored payload proves this
		// duplicate belongs to a different conversation than attempted.
		// Repairing the attempted conversation's index here would be
		// indexing a response that conversation does not actually own.
		return nil
	}

	if err := s.indexResponse(ctx, stored.ConversationID, stored.ID, stored.CreatedAt); err != nil {
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

	var response responseapi.StoredResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("failed to deserialize response: %w", err)
	}

	return &response, nil
}

// UpdateResponse replaces a response's payload and keeps its conversation
// index in step: indexes under the new ConversationID (if any), then
// best-effort unindexes the previous one if it changed. If the new index
// write fails, restores the previous payload bytes and best-effort
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

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	snapshot, err := s.replaceResponseAndSnapshot(ctx, key, response.ID, data)
	if err != nil {
		return err
	}

	if response.ConversationID != "" {
		if err := s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt); err != nil {
			return s.rollbackUpdatePayload(ctx, key, response.ID, data, snapshot, err)
		}
	}

	// The new index write (if any) is now confirmed in place, so the
	// previous entry — if the conversation actually changed — is safe to
	// drop. Best-effort: a stale leftover here is pruned by the next listing.
	if snapshot.conversationID != "" && snapshot.conversationID != response.ConversationID {
		if err := s.unindexResponse(ctx, snapshot.conversationID, response.ID); err != nil {
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
	ttlMillis := s.ttl.Milliseconds()
	if s.ttl > 0 && ttlMillis == 0 {
		ttlMillis = 1
	}

	result, err := replaceAndSnapshotResponseScript.Run(ctx, s.client, []string{key}, data, ttlMillis).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return responseUpdateSnapshot{}, ErrNotFound
		}
		return responseUpdateSnapshot{}, fmt.Errorf("failed to replace response %s: %w", responseID, err)
	}

	return decodeResponseUpdateSnapshot(result, capturedAt, responseID)
}

// readPreviousResponseForUpdate atomically snapshots a response's current
// payload bytes and remaining PTTL — doubling as the existence check — and
// best-effort parses its previous ConversationID/CreatedAt. All of this is
// needed if the update's own index write later fails and the payload must
// be restored with an equivalent remaining lifetime and reindexed. A parse
// failure is not fatal to the update: the snapshot's bytes are still usable
// for restore, it just has nothing to unindex or reindex.
func (s *RedisStore) readPreviousResponseForUpdate(ctx context.Context, key, responseID string) (responseUpdateSnapshot, error) {
	capturedAt := time.Now()

	result, err := snapshotResponseScript.Run(ctx, s.client, []string{key}).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return responseUpdateSnapshot{}, ErrNotFound
		}
		return responseUpdateSnapshot{}, fmt.Errorf("failed to snapshot response %s for update: %w", responseID, err)
	}

	return decodeResponseUpdateSnapshot(result, capturedAt, responseID)
}

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

	var previous responseapi.StoredResponse
	if err := json.Unmarshal(snapshot.data, &previous); err != nil {
		logging.Warnf("RedisStore: failed to parse previous stored response %s during update: %v", responseID, err)
		return snapshot, nil
	}
	snapshot.conversationID = previous.ConversationID
	snapshot.createdAt = previous.CreatedAt

	return snapshot, nil
}

// compareRestoreResponsePayload restores previous with the snapshot's
// projected remaining TTL, but only if key's current value still equals
// expectedCurrent — the exact bytes the failed update wrote — so a rollback
// can never clobber a newer concurrent write. remainingTTLMillis follows
// responseUpdateSnapshot.remainingTTLMillis's convention: -1 persistent, 0
// means the snapshot's own TTL has since elapsed (delete rather than
// resurrect), positive is milliseconds remaining.
//
// Single-key Lua script — touches only KEYS[1] — so it stays legal in Redis
// Cluster, matching compareDeleteResponsePayload.
func (s *RedisStore) compareRestoreResponsePayload(ctx context.Context, key string, expectedCurrent, previous []byte, remainingTTLMillis int64) (compareRestoreResult, error) {
	res, err := compareRestoreScript.Run(ctx, s.client, []string{key}, expectedCurrent, previous, remainingTTLMillis).Result()
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
// fails. Restores the pre-update snapshot via compare-and-swap against the
// exact bytes this call just wrote (failedData) — never a blind SET — so a
// newer concurrent update landing between this update's write and its
// failed rollback is never overwritten: compareRestoreConflict means
// exactly that happened, and this leaves the newer payload alone rather
// than reindexing the stale snapshot it was about to restore.
// compareRestoreExpired means the CAS matched but the snapshot's own TTL
// had elapsed by the time of the restore, so the key was deleted instead —
// also not reindexed, since there is nothing left to point an index at.
// Only compareRestoreRestored reindexes the previous conversation, using
// its own stored CreatedAt, never a fabricated timestamp.
func (s *RedisStore) rollbackUpdatePayload(ctx context.Context, key, responseID string, failedData []byte, snapshot responseUpdateSnapshot, indexErr error) error {
	wrapped := fmt.Errorf("failed to index updated response in Redis: %w", indexErr)

	result, restoreErr := s.compareRestoreResponsePayload(ctx, key, failedData, snapshot.data, snapshot.remainingTTLMillis())
	if restoreErr != nil {
		return fmt.Errorf("%w (rollback failed: %v)", wrapped, restoreErr)
	}

	switch result {
	case compareRestoreRestored:
		if snapshot.conversationID != "" {
			if reindexErr := s.indexResponse(ctx, snapshot.conversationID, responseID, snapshot.createdAt); reindexErr != nil {
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

	// Needed to drop the index entry; also the existence check.
	conversationID, err := s.storedConversationID(ctx, responseID)
	if err != nil {
		return err
	}

	deleted, err := s.client.Del(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to delete response from Redis: %w", err)
	}
	if deleted == 0 {
		return ErrNotFound
	}

	// Best-effort: the payload delete above is the user-visible operation, and
	// a stale index entry is pruned by the next listing that finds it missing.
	if err := s.unindexResponse(ctx, conversationID, responseID); err != nil {
		logging.Warnf("RedisStore: failed to remove response %s from conversation %s index: %v",
			responseID, conversationID, err)
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
