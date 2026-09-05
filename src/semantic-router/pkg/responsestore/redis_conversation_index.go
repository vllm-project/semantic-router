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
// created_at; refreshes the index TTL; and clears the empty-conversation
// marker so a newly indexed write is never hidden behind a stale "scanned,
// found nothing" marker (checked here best-effort — a marker left behind by
// this call failing is harmless, because every read checks the index before
// the marker).
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
	}
	// Single-key DEL, never combined with another key: Cluster safe. A no-op
	// (returns 0, not an error) when no marker exists.
	pipe.Del(ctx, s.emptyConversationIndexMarkerKey(conversationID))

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
	// conversation. Zero means the empty marker was set instead.
	Found int
}

// indexOrMarkerExists reports whether a conversation's index or its empty
// marker exists, via two single-key EXISTS calls. Deliberately not a
// variadic Exists(ctx, indexKey, markerKey): the two keys have different
// prefixes and are not guaranteed to share a Cluster hash slot.
func (s *RedisStore) indexOrMarkerExists(ctx context.Context, conversationID string) (bool, error) {
	indexExists, err := s.client.Exists(ctx, s.conversationIndexKey(conversationID)).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check conversation index: %w", err)
	}
	if indexExists > 0 {
		return true, nil
	}

	emptyExists, err := s.client.Exists(ctx, s.emptyConversationIndexMarkerKey(conversationID)).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check conversation index empty marker: %w", err)
	}

	return emptyExists > 0, nil
}

// ensureConversationIndex guarantees that, barring a concurrent delete of
// both immediately afterward, either the conversation's index or its empty
// marker exists once this returns without error. It runs the O(N) legacy
// scan (lazyBackfillConversationIndex) at most once per conversation per
// marker/index lifetime.
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

	// Recheck under the lock: a writer, or a backfill that started just
	// before this one acquired it, may have already resolved this conversation.
	exists, err := s.indexOrMarkerExists(ctx, conversationID)
	if err != nil {
		return err
	}
	if exists {
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
// index/marker still hasn't appeared — the holder may have died mid-scan
// (pod restart, deadline). Correctness wins over avoiding duplicate work,
// which is the lock's only job.
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

		exists, err := s.indexOrMarkerExists(ctx, conversationID)
		if err != nil {
			return err
		}
		if exists {
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
// pre-index conversation's responses discoverable: it walks every response
// payload once, keeps the ones matching conversationID, and either indexes
// them or — if none match — sets the empty marker so the next read does not
// scan again.
//
// Idempotent and additive: it only ZADDs discovered members and never
// deletes, so a concurrently indexed write (e.g. a StoreResponse racing this
// scan) is never undone by it, no matter which finishes first.
func (s *RedisStore) lazyBackfillConversationIndex(ctx context.Context, conversationID string) (legacyBackfillResult, error) {
	if s.lazyBackfillPreScanHook != nil {
		s.lazyBackfillPreScanHook()
	}

	found, err := s.scanConversationResponses(ctx, conversationID)
	if err != nil {
		return legacyBackfillResult{}, fmt.Errorf("failed to backfill conversation index: %w", err)
	}

	if len(found) == 0 {
		s.setEmptyConversationIndexMarker(ctx, conversationID)
		return legacyBackfillResult{Found: 0}, nil
	}

	if err := s.indexBackfilledResponses(ctx, conversationID, found); err != nil {
		return legacyBackfillResult{}, err
	}

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

// setEmptyConversationIndexMarker records that a completed scan found no
// live responses for conversationID. TTL is capped at
// emptyConversationIndexMarkerMaxTTL regardless of the store's data TTL —
// including when s.ttl == 0 (a persistent store, no data TTL at all): the
// marker's job is stampede protection, not data retention, and letting it
// outlive that bounded purpose is exactly the rolling-upgrade blind spot
// the cap exists to close (see its doc comment). When s.ttl is positive but
// shorter than the cap, use s.ttl instead — there is no point marking a
// conversation empty for longer than its own data would live.
//
// Best-effort: correctness is preserved either way, since the next read
// simply scans again rather than trust a marker that failed to write.
func (s *RedisStore) setEmptyConversationIndexMarker(ctx context.Context, conversationID string) {
	markerTTL := emptyConversationIndexMarkerMaxTTL
	if s.ttl > 0 && s.ttl < markerTTL {
		markerTTL = s.ttl
	}

	if err := s.client.Set(ctx, s.emptyConversationIndexMarkerKey(conversationID), "v1", markerTTL).Err(); err != nil {
		logging.Debugf("RedisStore: failed to set empty conversation index marker for %s: %v", conversationID, err)
	}
}

// indexBackfilledResponses ZADDs discovered responses into the conversation
// index in redisBackfillBatchSize batches, refreshes the index TTL, and
// best-effort clears the empty marker — the index now exists, and every
// read checks it before the marker, so a marker left behind by a failed
// clear here is harmless.
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

	if err := s.client.Del(ctx, s.emptyConversationIndexMarkerKey(conversationID)).Err(); err != nil {
		logging.Debugf("RedisStore: failed to clear empty conversation index marker for %s: %v",
			conversationID, err)
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
