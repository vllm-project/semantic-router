package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// deleteConversationResponseBatch deletes one bounded batch of a
// conversation's indexed responses, verifying ownership before deleting any
// payload — never a GET-then-blind-DEL. A response's CURRENT stored
// ConversationID is the only thing that decides whether its payload is
// deleted here: the index that named it is not proof enough, since
// UpdateResponse's move to a different conversation unindexes the old entry
// only best-effort (see UpdateResponse's doc comment) — a concurrent cascade
// delete of the old conversation must never destroy a payload that has
// already moved to a new owner.
//
// Per response ID in the batch:
//   - missing payload (already deleted, or expired): a stale index member,
//     scheduled for ZREM; not an error.
//   - GET failed for a reason other than "missing", or the payload doesn't
//     parse as JSON: left alone entirely — payload and index member both
//     preserved. This is a real, retryable failure, not a decidable
//     outcome, so it must never be silently skipped.
//   - stored ConversationID differs from conversationID: the response moved
//     to a different conversation since being indexed here. Its payload is
//     preserved untouched; only the stale membership in *this*
//     conversation's index is scheduled for ZREM.
//   - stored ConversationID matches: deleted via compareDeleteResponsePayload
//     against the exact bytes just read, never a blind DEL, so a concurrent
//     write landing between this read and the delete can never be destroyed
//     out from under its writer. A CAS conflict here is treated like a GET
//     failure: preserved, retryable, reported.
//
// Every independent failure across the batch is collected with errors.Join
// and returned together, after every response ID has had its own
// independent chance to resolve — one bad payload never blocks the rest of
// the batch from making progress. ZREM only ever removes members proven
// deleted, missing, or moved.
func (s *RedisStore) deleteConversationResponseBatch(ctx context.Context, conversationID string, candidates []redis.Z) error {
	if len(candidates) == 0 {
		return nil
	}

	responseIDs, err := responseIDsFromCandidates(candidates)
	if err != nil {
		return err
	}

	// Remove the old membership before resolving payload ownership. This
	// ordering is essential: after a successful payload delete, another
	// writer may recreate the same response ID and ZADD it. There must be no
	// later unconditional ZREM that can erase that fresh membership.
	if err := s.unindexResponse(ctx, conversationID, responseIDs...); err != nil {
		return err
	}

	keys := make([]string, len(responseIDs))
	for i, responseID := range responseIDs {
		keys[i] = s.buildKey(ResponseKeyPrefix + responseID)
	}
	results := fetchResponsePayloadsPipelined(ctx, s.client, keys)

	toRestore, errs := s.resolveCascadeBatchResults(ctx, conversationID, responseIDs, candidates, results)
	if len(toRestore) > 0 {
		if err := s.restoreConversationIndexMembers(ctx, conversationID, toRestore); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

func responseIDsFromCandidates(candidates []redis.Z) ([]string, error) {
	responseIDs := make([]string, len(candidates))
	for i, candidate := range candidates {
		responseID, ok := candidate.Member.(string)
		if !ok {
			return nil, fmt.Errorf("unexpected conversation index member type %T", candidate.Member)
		}
		responseIDs[i] = responseID
	}
	return responseIDs, nil
}

func (s *RedisStore) resolveCascadeBatchResults(
	ctx context.Context,
	conversationID string,
	responseIDs []string,
	candidates []redis.Z,
	results []responsePayloadResult,
) ([]redis.Z, []error) {
	var toRestore []redis.Z
	var errs []error
	for i, result := range results {
		responseID := responseIDs[i]
		switch {
		case errors.Is(result.err, redis.Nil):
			// The stale member was already removed above.
		case result.err != nil:
			errs = append(errs, fmt.Errorf("failed to read response %s for cascade delete: %w", responseID, result.err))
			toRestore = append(toRestore, candidates[i])
		default:
			resolved, err := s.resolveCascadeDeleteOutcome(ctx, conversationID, responseID, result)
			if err != nil {
				errs = append(errs, err)
				toRestore = append(toRestore, candidates[i])
				continue
			}
			if !resolved {
				toRestore = append(toRestore, candidates[i])
			}
		}
	}
	return toRestore, errs
}

// restoreConversationIndexMembers makes an interrupted cascade retryable.
// NX is deliberate: if a concurrent writer already re-added a member with a
// newer score, restoring the old candidate must not overwrite that score.
func (s *RedisStore) restoreConversationIndexMembers(ctx context.Context, conversationID string, members []redis.Z) error {
	if len(members) == 0 {
		return nil
	}
	if err := s.client.ZAddArgs(ctx, s.conversationIndexKey(conversationID), redis.ZAddArgs{
		NX:      true,
		Members: members,
	}).Err(); err != nil {
		return fmt.Errorf("failed to restore %d unresolved response(s) to conversation %s index: %w",
			len(members), conversationID, err)
	}
	return nil
}

// resolveCascadeDeleteOutcome decides and executes one response's cascade
// outcome once its payload has been successfully read: prune-only if it has
// moved to a different conversation, or compare-delete if conversationID
// still owns it. Returns whether the index member is now safe to remove.
func (s *RedisStore) resolveCascadeDeleteOutcome(ctx context.Context, conversationID, responseID string, result responsePayloadResult) (bool, error) {
	var stored responseapi.StoredResponse
	if err := json.Unmarshal(result.raw, &stored); err != nil {
		return false, fmt.Errorf("failed to parse response %s during cascade delete: %w", responseID, err)
	}
	if stored.ID != responseID {
		return false, fmt.Errorf("response %s payload identity mismatch during cascade delete", responseID)
	}

	if stored.ConversationID != conversationID {
		// Moved to a different conversation since being indexed here (an
		// UpdateResponse best-effort unindex of the old entry hadn't run
		// yet, or failed): the payload belongs to someone else now, so only
		// the stale membership in *this* conversation's index is pruned.
		return true, nil
	}

	deleted, err := s.compareDeleteResponsePayload(ctx, result.key, result.raw)
	if err != nil {
		return false, fmt.Errorf("failed to delete response %s during cascade delete: %w", responseID, err)
	}
	if !deleted {
		// A concurrent write landed between the read above and this
		// compare-delete: the payload is no longer the bytes just read, so
		// it was left alone rather than risk destroying that newer write.
		return false, fmt.Errorf("response %s changed concurrently during cascade delete; left in place for retry", responseID)
	}

	return true, nil
}
