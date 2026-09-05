package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// indexResponse adds a response to its conversation index, scored by
// created_at, and refreshes the TTL of both the index and — if one already
// exists — the migrated marker.
//
// Deliberately does not set or clear the migrated marker itself, only
// refreshes an existing one's TTL: this call proves nothing about whether a
// legacy-scan backfill has ever run for conversationID (a fresh
// conversation's very first write reaches this exact path), so it must
// never be mistaken for the signal that makes the index trustworthy as
// exhaustive. If a migrated marker already exists, though, a completed
// backfill's "exhaustive as of scan time" guarantee stays true across this
// correctly-indexed incremental write — the two states are additive, not
// competing — so keeping the marker alive as long as the data it now also
// describes (rather than letting it lapse on its own separate, possibly
// shorter TTL) avoids forcing a needless re-scan on an actively-written
// conversation. EXPIRE on a marker that was never set is a documented no-op
// (returns 0), not an error, so this is safe to call unconditionally.
//
// Returns an error instead of swallowing it: the payload this indexes is
// already durable by the time this runs (StoreResponse writes it first), so
// the caller — not this helper — must decide what an index failure means:
// StoreResponse rolls the payload back, UpdateResponse restores the previous
// payload, DeleteResponse and lazy backfill may choose to log and continue.
func (s *RedisStore) indexResponse(ctx context.Context, conversationID, responseID string, createdAt int64) error {
	if conversationID == "" || responseID == "" {
		return nil
	}
	if s.indexResponseOverride != nil {
		return s.indexResponseOverride(ctx, conversationID, responseID, createdAt)
	}

	indexKey := s.conversationIndexKey(conversationID)

	pipe := s.client.Pipeline()
	pipe.ZAdd(ctx, indexKey, redis.Z{Score: float64(createdAt), Member: responseID})
	if s.ttl > 0 {
		// Outlive the newest member. Guarded: EXPIRE with 0 deletes the key.
		pipe.Expire(ctx, indexKey, s.ttl)
		pipe.Expire(ctx, s.conversationIndexMigratedKey(conversationID), s.ttl)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to index response %s in conversation %s: %w", responseID, conversationID, err)
	}

	return nil
}

// unindexResponse drops response IDs from a conversation index. ZREM is
// variadic but touches only one key (the zset), all members belong to the
// same conversation index, so it stays Cluster safe.
func (s *RedisStore) unindexResponse(ctx context.Context, conversationID string, responseIDs ...string) error {
	if conversationID == "" || len(responseIDs) == 0 {
		return nil
	}

	members := make([]interface{}, len(responseIDs))
	for i, responseID := range responseIDs {
		members[i] = responseID
	}

	if err := s.client.ZRem(ctx, s.conversationIndexKey(conversationID), members...).Err(); err != nil {
		return fmt.Errorf("failed to remove %d response(s) from conversation %s index: %w", len(responseIDs), conversationID, err)
	}

	return nil
}

// legacyBackfillResult reports how many responses a lazy backfill scan found
// and indexed for a conversation, mainly for logging.
type legacyBackfillResult struct {
	// Found is the number of responses discovered belonging to the scanned
	// conversation. Zero means the scan confirmed the conversation empty.
	Found int
}

// conversationMigrated reports whether a legacy-scan backfill has ever
// completed for conversationID — the single signal that makes the index's
// current state (populated, or absent entirely) trustworthy as exhaustive.
//
// Deliberately not "does the index exist, or the marker": a conversation
// can have real indexed members from ordinary post-upgrade writes with no
// backfill ever having run for it, so index-existence alone must never be
// read as "migration complete" (that conflation is exactly the bug this
// marker exists to prevent — see ConversationIndexMigratedKeyPrefix). A
// single-key EXISTS check, Cluster safe.
func (s *RedisStore) conversationMigrated(ctx context.Context, conversationID string) (bool, error) {
	migrated, err := s.client.Exists(ctx, s.conversationIndexMigratedKey(conversationID)).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check conversation migrated marker: %w", err)
	}

	return migrated > 0, nil
}

// ensureConversationIndex guarantees that, barring a concurrent delete of
// the marker immediately afterward, the conversation is marked migrated
// once this returns without error — meaning its index (populated or
// absent) may now be trusted as exhaustive. It runs the O(N) legacy scan
// (lazyBackfillConversationIndex) at most once per conversation per
// marker lifetime, and runs it unconditionally when not yet migrated, even
// if the index already has some members from earlier post-upgrade writes:
// those members alone do not prove nothing legacy is left to discover.
//
// A short migration lock (SET NX PX conversationIndexMigrationLockTTL)
// keeps concurrent readers of the same missing index from all scanning at
// once, but holding it is an optimization, not a correctness dependency: a
// reader that cannot acquire it backs off briefly, rechecks, and — if the
// holder appears to have died mid-scan (pod restart, deadline) rather than
// finished — runs the scan itself anyway. Every path converges on the same
// additive, idempotent backfill.
func (s *RedisStore) ensureConversationIndex(ctx context.Context, conversationID string) error {
	lockKey := s.conversationIndexLockKey(conversationID)
	token := []byte(fmt.Sprintf("%d:%d", time.Now().UnixNano(), os.Getpid()))

	acquired, err := s.client.SetNX(ctx, lockKey, token, conversationIndexMigrationLockTTL).Result()
	if err != nil {
		return fmt.Errorf("failed to acquire conversation index migration lock: %w", err)
	}

	if acquired {
		return s.backfillUnderLock(ctx, conversationID, lockKey, token)
	}

	return s.awaitOrBackfillConversationIndex(ctx, conversationID)
}

// backfillUnderLock holds the migration lock this call just acquired,
// rechecks whether a concurrent writer or backfill already resolved the
// conversation, and otherwise runs the legacy scan. Always releases the
// lock via compare-delete on return, which never releases a lock this call
// doesn't currently hold (e.g. one that already expired and was re-acquired
// by someone else).
func (s *RedisStore) backfillUnderLock(ctx context.Context, conversationID, lockKey string, token []byte) error {
	defer func() {
		if _, releaseErr := s.compareDeleteResponsePayload(ctx, lockKey, token); releaseErr != nil {
			logging.Debugf("RedisStore: failed to release conversation index migration lock for %s: %v",
				conversationID, releaseErr)
		}
	}()

	// Recheck under the lock: a backfill that started just before this one
	// acquired it may have already migrated this conversation.
	migrated, err := s.conversationMigrated(ctx, conversationID)
	if err != nil {
		return err
	}
	if migrated {
		return nil
	}

	result, err := s.lazyBackfillConversationIndex(ctx, conversationID)
	if err != nil {
		return err
	}
	logging.Debugf("RedisStore: lazy-backfilled conversation %s index with %d response(s)",
		conversationID, result.Found)

	return nil
}

// awaitOrBackfillConversationIndex backs off briefly waiting for another
// reader's lock-held backfill to finish, then runs the scan itself if the
// conversation still isn't marked migrated — the holder may have died
// mid-scan (pod restart, deadline). Correctness wins over avoiding
// duplicate work, which is the lock's only job.
func (s *RedisStore) awaitOrBackfillConversationIndex(ctx context.Context, conversationID string) error {
	const (
		lockWaitAttempts = 5
		lockWaitDelay    = 100 * time.Millisecond
	)

	for range lockWaitAttempts {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(lockWaitDelay):
		}

		migrated, err := s.conversationMigrated(ctx, conversationID)
		if err != nil {
			return err
		}
		if migrated {
			return nil
		}
	}

	result, err := s.lazyBackfillConversationIndex(ctx, conversationID)
	if err != nil {
		return err
	}
	logging.Debugf("RedisStore: lazy-backfilled conversation %s index with %d response(s) after lock wait",
		conversationID, result.Found)

	return nil
}

// lazyBackfillConversationIndex performs the one-time O(N) scan that makes a
// conversation's full response set discoverable: it walks every response
// payload once, keeps the ones matching conversationID — whether or not
// they were already indexed by an ordinary post-upgrade write — indexes
// them, and marks the conversation migrated so future reads trust the
// index without re-scanning.
//
// Idempotent and additive: it only ZADDs discovered members (including ones
// already present, harmlessly re-adding the same score) and never deletes,
// so a concurrently indexed write racing this scan is never undone by it,
// no matter which finishes first.
func (s *RedisStore) lazyBackfillConversationIndex(ctx context.Context, conversationID string) (legacyBackfillResult, error) {
	if s.lazyBackfillPreScanHook != nil {
		s.lazyBackfillPreScanHook()
	}

	found, err := s.scanConversationResponses(ctx, conversationID)
	if err != nil {
		return legacyBackfillResult{}, fmt.Errorf("failed to backfill conversation index: %w", err)
	}

	if len(found) == 0 {
		s.markConversationMigrated(ctx, conversationID, false)
		return legacyBackfillResult{Found: 0}, nil
	}

	if err := s.indexBackfilledResponses(ctx, conversationID, found); err != nil {
		return legacyBackfillResult{}, err
	}
	s.markConversationMigrated(ctx, conversationID, true)

	return legacyBackfillResult{Found: len(found)}, nil
}

// discoveredConversationResponse is one response scanConversationResponses
// found belonging to a specific conversation.
type discoveredConversationResponse struct {
	id        string
	createdAt int64
}

// scanConversationResponses walks every response payload once via
// scanResponsePayloads, collecting the ones belonging to conversationID in
// deterministic order: primary by created_at, tie-broken by response ID
// (blueprint §3.6) — never fabricating sub-second ordering from wall clock
// time.
func (s *RedisStore) scanConversationResponses(ctx context.Context, conversationID string) ([]discoveredConversationResponse, error) {
	var found []discoveredConversationResponse

	err := s.scanResponsePayloads(ctx, func(batch []*responseapi.StoredResponse) error {
		for _, response := range batch {
			if response.ConversationID == conversationID {
				found = append(found, discoveredConversationResponse{id: response.ID, createdAt: response.CreatedAt})
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	sort.Slice(found, func(i, j int) bool {
		if found[i].createdAt != found[j].createdAt {
			return found[i].createdAt < found[j].createdAt
		}
		return found[i].id < found[j].id
	})

	return found, nil
}

// markConversationMigrated records that a legacy-scan backfill has
// completed for conversationID — the signal ListResponsesByConversation and
// cascade delete both check before trusting the index's current state as
// exhaustive (ConversationIndexMigratedKeyPrefix), independent of whether
// the index happens to already have members from earlier ordinary writes.
//
// TTL: when the scan found responses to index, the full store TTL — the
// marker should live exactly as long as the data it now describes, and
// indexResponse refreshes it further on every subsequent write to the same
// conversation. When the scan found nothing, capped at
// emptyConversationIndexMarkerMaxTTL (or the store's own TTL if that is
// shorter) instead: an empty result is a more perishable claim, since an
// indexing-unaware writer could still land a response later (the
// rolling-upgrade blind spot that cap exists to bound).
//
// Best-effort: correctness is preserved either way, since the next read
// simply re-scans rather than trust a marker that failed to write.
func (s *RedisStore) markConversationMigrated(ctx context.Context, conversationID string, found bool) {
	ttl := emptyConversationIndexMarkerMaxTTL
	switch {
	case found:
		ttl = s.ttl
	case s.ttl > 0 && s.ttl < ttl:
		ttl = s.ttl
	}

	if err := s.client.Set(ctx, s.conversationIndexMigratedKey(conversationID), "v1", ttl).Err(); err != nil {
		logging.Debugf("RedisStore: failed to mark conversation %s migrated: %v", conversationID, err)
	}
}

// indexBackfilledResponses ZADDs discovered responses into the conversation
// index in redisBackfillBatchSize batches and refreshes the index TTL.
// Idempotent for members already indexed by an earlier ordinary write —
// re-adding one with the same score is a no-op.
func (s *RedisStore) indexBackfilledResponses(ctx context.Context, conversationID string, found []discoveredConversationResponse) error {
	indexKey := s.conversationIndexKey(conversationID)

	for start := 0; start < len(found); start += redisBackfillBatchSize {
		end := min(start+redisBackfillBatchSize, len(found))

		members := make([]redis.Z, end-start)
		for i, d := range found[start:end] {
			members[i] = redis.Z{Score: float64(d.createdAt), Member: d.id}
		}
		if err := s.client.ZAdd(ctx, indexKey, members...).Err(); err != nil {
			return fmt.Errorf("failed to backfill conversation index: %w", err)
		}
	}

	if s.ttl > 0 {
		if err := s.client.Expire(ctx, indexKey, s.ttl).Err(); err != nil {
			logging.Warnf("RedisStore: failed to refresh TTL on backfilled conversation index %s: %v",
				conversationID, err)
		}
	}

	return nil
}

// scanResponsePayloads walks every response payload key exactly once,
// decoding each into a StoredResponse and delivering them to visit in
// bounded batches (redisBackfillBatchSize).
//
// Cluster-aware: a single Redis Cluster node's keyspace only holds the slots
// assigned to it, so in Cluster mode this scans every master via
// ForEachMaster. Standalone mode scans the one client directly.
//
// Used only by lazy legacy backfill — this is the O(N) operation the index
// exists to avoid on the hot read path.
func (s *RedisStore) scanResponsePayloads(ctx context.Context, visit func(batch []*responseapi.StoredResponse) error) error {
	s.scanInvocations.Add(1)

	pattern := s.buildKey(ResponseKeyPrefix + "*")

	scanNode := func(ctx context.Context, client redis.UniversalClient) error {
		var keys []string
		flush := func() error {
			if len(keys) == 0 {
				return nil
			}
			batch := s.getResponsesPipelined(ctx, client, keys)
			keys = keys[:0]
			if len(batch) == 0 {
				return nil
			}
			return visit(batch)
		}

		iter := client.Scan(ctx, 0, pattern, redisScanCount).Iterator()
		for iter.Next(ctx) {
			keys = append(keys, iter.Val())
			if len(keys) >= redisBackfillBatchSize {
				if err := flush(); err != nil {
					return err
				}
			}
		}
		if err := iter.Err(); err != nil {
			return fmt.Errorf("failed to scan response keys: %w", err)
		}

		return flush()
	}

	if clusterClient, ok := s.client.(*redis.ClusterClient); ok {
		return clusterClient.ForEachMaster(ctx, func(ctx context.Context, master *redis.Client) error {
			return scanNode(ctx, master)
		})
	}

	return scanNode(ctx, s.client)
}

// getResponsesPipelined GETs and decodes payloads for a batch of already-
// prefixed keys in one round trip against the given client. Malformed or
// missing payloads are skipped with a log line rather than failing the
// batch: a legacy scan must make forward progress even if one record is
// corrupt or expired mid-scan.
func (s *RedisStore) getResponsesPipelined(ctx context.Context, client redis.UniversalClient, keys []string) []*responseapi.StoredResponse {
	if len(keys) == 0 {
		return nil
	}

	pipe := client.Pipeline()
	cmds := make([]*redis.StringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.Get(ctx, key)
	}
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		logging.Debugf("RedisStore: scan pipeline execution completed with some errors: %v", err)
	}

	responses := make([]*responseapi.StoredResponse, 0, len(keys))
	for i, cmd := range cmds {
		data, err := cmd.Bytes()
		if err != nil {
			if !errors.Is(err, redis.Nil) {
				logging.Warnf("RedisStore: failed to get response at key %s during scan: %v", keys[i], err)
			}
			continue
		}

		var response responseapi.StoredResponse
		if err := json.Unmarshal(data, &response); err != nil {
			logging.Warnf("RedisStore: failed to parse response at key %s during scan: %v", keys[i], err)
			continue
		}
		if response.ID == "" {
			continue
		}

		responses = append(responses, &response)
	}

	return responses
}
