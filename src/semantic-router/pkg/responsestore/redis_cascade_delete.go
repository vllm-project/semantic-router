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

// responsePayloadResult is one response ID's raw GET outcome from
// fetchResponsePayloadsPipelined: exactly the bytes stored at key at the
// moment of the read, for a later exact-match compare-delete, or the error
// GET returned instead. err wraps redis.Nil (checked with errors.Is) for a
// missing payload — deliberately distinguished from any other failure, since
// callers must treat the two very differently: missing is safe to prune from
// an index, any other GET failure is not.
type responsePayloadResult struct {
	responseID string
	key        string
	raw        []byte
	err        error
}

// fetchResponsePayloadsPipelined GETs a batch of response payloads from
// client in one round trip, returning one result per input ID in the same
// order. Every GET is independent and single-key — never MGET — pipelined
// only for round-trip efficiency: in Redis Cluster mode, the response IDs
// belonging to one conversation's index can hash to any slot, and a
// multi-key command spanning different slots fails with CROSSSLOT, whereas
// go-redis's ClusterClient.Pipeline still routes each independent
// single-key command to its own slot's node correctly.
func (s *RedisStore) fetchResponsePayloadsPipelined(ctx context.Context, client redis.UniversalClient, responseIDs []string) []responsePayloadResult {
	results := make([]responsePayloadResult, len(responseIDs))
	if len(responseIDs) == 0 {
		return results
	}

	pipe := client.Pipeline()
	cmds := make([]*redis.StringCmd, len(responseIDs))
	for i, id := range responseIDs {
		key := s.buildKey(ResponseKeyPrefix + id)
		results[i] = responsePayloadResult{responseID: id, key: key}
		cmds[i] = pipe.Get(ctx, key)
	}

	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		logging.Debugf("RedisStore: cascade-delete fetch pipeline execution completed with some errors: %v", err)
	}

	for i, cmd := range cmds {
		data, err := cmd.Bytes()
		if err != nil {
			results[i].err = err
			continue
		}
		results[i].raw = data
	}

	return results
}

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
func (s *RedisStore) deleteConversationResponseBatch(ctx context.Context, conversationID string, responseIDs []string) error {
	if len(responseIDs) == 0 {
		return nil
	}

	results := s.fetchResponsePayloadsPipelined(ctx, s.client, responseIDs)

	var (
		toUnindex []string
		errs      []error
	)
	for _, result := range results {
		switch {
		case errors.Is(result.err, redis.Nil):
			toUnindex = append(toUnindex, result.responseID)
		case result.err != nil:
			errs = append(errs, fmt.Errorf("failed to read response %s for cascade delete: %w", result.responseID, result.err))
		default:
			unindex, err := s.resolveCascadeDeleteOutcome(ctx, conversationID, result)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			if unindex {
				toUnindex = append(toUnindex, result.responseID)
			}
		}
	}

	if len(toUnindex) > 0 {
		if err := s.unindexResponse(ctx, conversationID, toUnindex...); err != nil {
			errs = append(errs, fmt.Errorf("failed to remove %d resolved response(s) from conversation %s index: %w",
				len(toUnindex), conversationID, err))
		}
	}

	return errors.Join(errs...)
}

// resolveCascadeDeleteOutcome decides and executes one response's cascade
// outcome once its payload has been successfully read: prune-only if it has
// moved to a different conversation, or compare-delete if conversationID
// still owns it. Returns whether the index member is now safe to remove.
func (s *RedisStore) resolveCascadeDeleteOutcome(ctx context.Context, conversationID string, result responsePayloadResult) (bool, error) {
	var stored responseapi.StoredResponse
	if err := json.Unmarshal(result.raw, &stored); err != nil {
		return false, fmt.Errorf("failed to parse response %s during cascade delete: %w", result.responseID, err)
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
		return false, fmt.Errorf("failed to delete response %s during cascade delete: %w", result.responseID, err)
	}
	if !deleted {
		// A concurrent write landed between the read above and this
		// compare-delete: the payload is no longer the bytes just read, so
		// it was left alone rather than risk destroying that newer write.
		return false, fmt.Errorf("response %s changed concurrently during cascade delete; left in place for retry", result.responseID)
	}

	return true, nil
}
