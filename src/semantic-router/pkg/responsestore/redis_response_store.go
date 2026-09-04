package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

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
		// The payload already here is the source of truth for whether this
		// retry needs an index repair, not the caller's attempted request:
		// verified inside repairExistingResponseIndex.
		if response.ConversationID != "" {
			if repairErr := s.repairExistingResponseIndex(ctx, response); repairErr != nil {
				return repairErr
			}
		}
		return ErrAlreadyExists
	}

	if response.ConversationID == "" {
		return nil
	}

	if err := s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt); err != nil {
		indexErr := fmt.Errorf("failed to index response in Redis: %w", err)

		// Compare-delete rollback: never a blind DEL. Only removes the
		// payload if it is still exactly what this call wrote, so a
		// concurrent writer that stored a new value after this payload's TTL
		// expired is never clobbered (the ABA race the blueprint calls out).
		deleted, rollbackErr := s.compareDeleteResponsePayload(ctx, key, data)
		if rollbackErr != nil {
			return fmt.Errorf("%w (rollback failed: %v)", indexErr, rollbackErr)
		}
		if !deleted {
			return fmt.Errorf("%w (payload changed before rollback, left in place)", indexErr)
		}
		return indexErr
	}

	return nil
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

	// Doubles as the existence check. Kept as raw bytes (not just the
	// previously-parsed ConversationID storedConversationID would give) so a
	// failed index write below can restore the exact previous payload.
	previousData, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return ErrNotFound
		}
		return fmt.Errorf("failed to check response existence: %w", err)
	}

	var previous responseapi.StoredResponse
	previousConversationID := ""
	if err := json.Unmarshal(previousData, &previous); err != nil {
		// Not fatal: the update can still proceed, it just has nothing to
		// unindex or restore from — same posture as storedConversationID.
		logging.Warnf("RedisStore: failed to parse previous stored response %s during update: %v", response.ID, err)
	} else {
		previousConversationID = previous.ConversationID
	}

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to update response in Redis: %w", err)
	}

	if response.ConversationID != "" {
		if err := s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt); err != nil {
			indexErr := fmt.Errorf("failed to index updated response in Redis: %w", err)

			if restoreErr := s.client.Set(ctx, key, previousData, s.ttl).Err(); restoreErr != nil {
				return fmt.Errorf("%w (failed to restore previous payload: %v)", indexErr, restoreErr)
			}
			if previousConversationID != "" {
				if reindexErr := s.indexResponse(ctx, previousConversationID, response.ID, previous.CreatedAt); reindexErr != nil {
					logging.Warnf("RedisStore: failed to reindex restored response %s under previous conversation %s after update rollback: %v",
						response.ID, previousConversationID, reindexErr)
				}
			}
			return indexErr
		}
	}

	// The new index write (if any) is now confirmed in place, so the
	// previous entry — if the conversation actually changed — is safe to
	// drop. Best-effort: a stale leftover here is pruned by the next listing.
	if previousConversationID != "" && previousConversationID != response.ConversationID {
		if err := s.unindexResponse(ctx, previousConversationID, response.ID); err != nil {
			logging.Warnf("RedisStore: failed to remove response %s from previous conversation %s index: %v",
				response.ID, previousConversationID, err)
		}
	}

	return nil
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
