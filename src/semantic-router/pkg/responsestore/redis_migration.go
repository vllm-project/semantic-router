package responsestore

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// isMigrationComplete reports whether FinalizeConversationIndexMigration has
// completed a cluster-wide sweep. Once true, ListResponsesByConversation and
// the cascade-delete path both trust the secondary index unconditionally —
// including a missing index meaning "empty," not "not yet migrated" — and
// never fall back to a per-conversation scan again, not even for a
// conversation ID no write has ever indexed.
func (s *RedisStore) isMigrationComplete(ctx context.Context) (bool, error) {
	value, err := s.client.Get(ctx, s.buildKey(ConversationIndexMigrationStatusKey)).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return false, nil
		}
		return false, fmt.Errorf("failed to check conversation index migration status: %w", err)
	}

	return value == conversationIndexMigrationCompleteValue, nil
}

// FinalizeConversationIndexMigration performs a one-time, cluster-wide sweep
// of every response payload, batch-indexing each by its ConversationID, and
// then marks the migration complete so every future read and cascade delete
// trusts the secondary index unconditionally without ever scanning again —
// see ConversationIndexMigrationStatusKey for exactly what that guarantees
// and what it risks if run too early.
//
// This is an explicit, operator-triggered step: nothing in this package's
// request paths calls it. It must only be run once every pod that predates
// this index has finished rolling out, so no more unindexed writes can land
// after the sweep's cursor has already passed their shard — once
// ConversationIndexMigrationStatusKey is set, the per-conversation lazy
// fallback that would otherwise self-heal a missed write (ensureConversationIndex)
// is bypassed entirely, for every conversation, forever.
//
// Idempotent: a no-op if migration is already marked complete. Safe to run
// concurrently with ordinary traffic — the sweep only ZADDs, so a response
// indexed by its own StoreResponse call mid-sweep is simply rediscovered and
// harmlessly re-added with the same score.
//
// Holds the entire sweep's discovered (conversation ID, response ID,
// created_at) triples in memory before writing any of them, grouped by
// conversation — proportional to the total number of responses across the
// whole keyspace, not any one conversation. For a very large deployment
// this is a real memory cost; this issue does not attempt to bound it
// further (e.g. by streaming per-conversation batches to Redis as they are
// discovered), since doing so would need to buffer per-conversation state
// across an unbounded, cursor-ordered scan for comparatively little benefit
// at the scale this store otherwise targets.
func (s *RedisStore) FinalizeConversationIndexMigration(ctx context.Context) error {
	if complete, err := s.isMigrationComplete(ctx); err != nil {
		return err
	} else if complete {
		return nil
	}

	lockKey := s.buildKey(ConversationIndexMigrationLockKey)
	token := []byte(fmt.Sprintf("%d:%d", time.Now().UnixNano(), os.Getpid()))

	acquired, err := s.client.SetNX(ctx, lockKey, token, conversationIndexGlobalMigrationLockTTL).Result()
	if err != nil {
		return fmt.Errorf("failed to acquire conversation index migration lock: %w", err)
	}
	if !acquired {
		// Deliberately not a wait-then-run-anyway loop like the per-
		// conversation ensureConversationIndex: this is an explicit,
		// operator-triggered admin operation with exactly one expected
		// caller at a time, not a request path with many concurrent
		// callers that all need a result right now.
		return fmt.Errorf("conversation index migration is already in progress (lock held)")
	}
	defer func() {
		if _, releaseErr := s.compareDeleteResponsePayload(ctx, lockKey, token); releaseErr != nil {
			logging.Debugf("RedisStore: failed to release conversation index migration lock: %v", releaseErr)
		}
	}()

	// Recheck under the lock: a previous run may have completed between
	// this call's first check above and acquiring the lock.
	if complete, err := s.isMigrationComplete(ctx); err != nil {
		return err
	} else if complete {
		return nil
	}

	byConversation := make(map[string][]discoveredConversationResponse)
	if err := s.scanResponsePayloads(ctx, func(batch []*responseapi.StoredResponse) error {
		for _, response := range batch {
			if response.ConversationID == "" {
				continue
			}
			byConversation[response.ConversationID] = append(byConversation[response.ConversationID], discoveredConversationResponse{
				id:        response.ID,
				createdAt: response.CreatedAt,
			})
		}
		return nil
	}); err != nil {
		return fmt.Errorf("failed to sweep response payloads for migration finalization: %w", err)
	}

	for conversationID, found := range byConversation {
		sort.Slice(found, func(i, j int) bool {
			if found[i].createdAt != found[j].createdAt {
				return found[i].createdAt < found[j].createdAt
			}
			return found[i].id < found[j].id
		})
		if err := s.indexBackfilledResponses(ctx, conversationID, found); err != nil {
			return fmt.Errorf("failed to index conversation %s during migration finalization: %w", conversationID, err)
		}
	}

	// Persistent: this status, unlike the per-conversation migrated marker,
	// is never meant to expire or be re-derived — it is the operator's
	// durable record that the sweep ran to completion.
	if err := s.client.Set(ctx, s.buildKey(ConversationIndexMigrationStatusKey), conversationIndexMigrationCompleteValue, 0).Err(); err != nil {
		return fmt.Errorf("failed to mark conversation index migration complete: %w", err)
	}

	return nil
}
