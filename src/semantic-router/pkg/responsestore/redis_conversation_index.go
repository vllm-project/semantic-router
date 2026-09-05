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

// conversationIndexProof is the typed value stored at a conversation's
// migrated marker key. Its presence and value (not the index key's mere
// existence) are what let a read or cascade delete trust the index as
// exhaustive — see ConversationIndexMigratedKeyPrefix and
// conversationIndexProof (the method).
type conversationIndexProof string

const (
	// conversationIndexProofEmpty means a completed legacy scan found no
	// live responses for the conversation. A weaker, more perishable claim
	// than populated — see markConversationMigrated's TTL policy.
	conversationIndexProofEmpty conversationIndexProof = "v1:empty"
	// conversationIndexProofPopulated means a completed legacy scan (or,
	// via refreshPopulatedConversationProof, a later ordinary write to an
	// already-populated conversation) found live responses now reflected
	// in the index.
	conversationIndexProofPopulated conversationIndexProof = "v1:populated"
)

// refreshPopulatedProofScript extends the migrated marker's TTL only if its
// current value is still exactly ARGV[1] (conversationIndexProofPopulated).
// Deliberately conditional rather than a blind EXPIRE: indexResponse must
// never extend a "confirmed empty" proof's life just because an ordinary
// write happened to land on that conversation afterward — see
// refreshPopulatedConversationProof. Single-key: KEYS[1] only.
var refreshPopulatedProofScript = redis.NewScript(`
if redis.call("GET", KEYS[1]) ~= ARGV[1] then
	return 0
end
return redis.call("PEXPIRE", KEYS[1], ARGV[2])
`)

// indexResponse adds a response to its conversation index, scored by
// created_at, and refreshes the index's own TTL.
//
// Deliberately does not set the migrated proof itself: this call proves
// nothing about whether a legacy-scan backfill has ever run for
// conversationID (a fresh conversation's very first write reaches this
// exact path), so it must never be mistaken for the signal that makes the
// index trustworthy as exhaustive. It does best-effort refresh an
// *existing* conversationIndexProofPopulated proof's TTL via
// refreshPopulatedConversationProof, so an actively-written conversation's
// proof doesn't need to outlive its own separate TTL and force a needless
// re-scan — but it will never refresh, let alone set, a
// conversationIndexProofEmpty proof: that value is stale the moment a real
// write lands, and extending its life would be exactly backwards.
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

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to index response %s in conversation %s: %w", responseID, conversationID, err)
	}

	if err := s.refreshPopulatedConversationProof(ctx, conversationID); err != nil {
		// Best-effort, by design: the ZADD above is what every read
		// actually consults for correctness. A missed refresh here only
		// risks the proof lapsing early and forcing one avoidable rescan.
		logging.Debugf("RedisStore: failed to refresh conversation %s migrated proof: %v", conversationID, err)
	}

	return nil
}

// refreshPopulatedConversationProof best-effort extends a conversation's
// migrated marker TTL, but only when its current value is exactly
// conversationIndexProofPopulated — see refreshPopulatedProofScript and
// indexResponse's doc comment for why an empty proof must never be
// refreshed this way. A no-op against a persistent store (s.ttl <= 0),
// since a populated proof there was set with no TTL to refresh.
func (s *RedisStore) refreshPopulatedConversationProof(ctx context.Context, conversationID string) error {
	if s.ttl <= 0 {
		return nil
	}

	key := s.conversationIndexMigratedKey(conversationID)
	if _, err := refreshPopulatedProofScript.Run(ctx, s.client, []string{key}, string(conversationIndexProofPopulated), s.ttl.Milliseconds()).Result(); err != nil {
		return fmt.Errorf("failed to refresh conversation %s migrated proof: %w", conversationID, err)
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

// conversationIndexProof reads a conversation's migrated marker with GET,
// not EXISTS, and reports its typed value along with whether that value is
// actually resolved. resolved=false covers both "no marker at all" and "a
// marker exists but holds a value this code doesn't recognize" (e.g. a
// future proof format, or corruption) — both fail safely into needing
// migration, rather than trusting a value never proven correct. This is
// deliberately not "does the index exist, or the marker": a conversation
// can have real indexed members from ordinary post-upgrade writes with no
// backfill ever having run for it, so index-existence alone must never be
// read as "migration complete" (that conflation is exactly the bug this
// marker exists to prevent — see ConversationIndexMigratedKeyPrefix). A
// single-key GET, Cluster safe.
func (s *RedisStore) conversationIndexProof(ctx context.Context, conversationID string) (conversationIndexProof, bool, error) {
	value, err := s.client.Get(ctx, s.conversationIndexMigratedKey(conversationID)).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return "", false, nil
		}
		return "", false, fmt.Errorf("failed to read conversation migrated proof: %w", err)
	}

	switch proof := conversationIndexProof(value); proof {
	case conversationIndexProofEmpty, conversationIndexProofPopulated:
		return proof, true, nil
	default:
		return "", false, nil
	}
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
	if _, resolved, err := s.conversationIndexProof(ctx, conversationID); err != nil {
		return err
	} else if resolved {
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

		if _, resolved, err := s.conversationIndexProof(ctx, conversationID); err != nil {
			return err
		} else if resolved {
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
		if err := s.markConversationMigrated(ctx, conversationID, conversationIndexProofEmpty); err != nil {
			// Best-effort: correctness is preserved either way, since the
			// next read simply re-scans rather than trust a proof that
			// failed to write.
			logging.Debugf("RedisStore: failed to mark conversation %s migrated (empty): %v", conversationID, err)
		}
		return legacyBackfillResult{Found: 0}, nil
	}

	if err := s.indexBackfilledResponses(ctx, conversationID, found); err != nil {
		return legacyBackfillResult{}, err
	}
	if err := s.markConversationMigrated(ctx, conversationID, conversationIndexProofPopulated); err != nil {
		logging.Debugf("RedisStore: failed to mark conversation %s migrated (populated): %v", conversationID, err)
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

// markConversationMigrated records that a legacy-scan backfill has
// completed for conversationID with the given typed proof — the signal
// ListResponsesByConversation and cascade delete both check before
// trusting the index's current state as exhaustive
// (ConversationIndexMigratedKeyPrefix), independent of whether the index
// happens to already have members from earlier ordinary writes.
//
// TTL: conversationIndexProofPopulated gets the full store TTL — the proof
// should live exactly as long as the data it now describes, and
// indexResponse's refreshPopulatedConversationProof extends it further on
// every subsequent write to the same conversation.
// conversationIndexProofEmpty is capped at emptyConversationIndexMarkerMaxTTL
// (or the store's own TTL if that is shorter) instead: an empty result is a
// more perishable claim, since an indexing-unaware writer could still land
// a response later (the rolling-upgrade blind spot that cap exists to
// bound) — and, per indexResponse, an empty proof is never refreshed by an
// ordinary write, so it must expire and force a re-scan on its own.
//
// Returns the write error rather than swallowing it, so a caller that
// wants to know can (e.g. Phase 4's streaming backfill, which must not
// publish a proof at all on partial failure); callers for whom this
// remains best-effort (this file's own lazyBackfillConversationIndex) log
// and continue, since the next read simply re-scans rather than trust a
// proof that failed to write.
func (s *RedisStore) markConversationMigrated(ctx context.Context, conversationID string, proof conversationIndexProof) error {
	ttl := emptyConversationIndexMarkerMaxTTL
	switch {
	case proof == conversationIndexProofPopulated:
		ttl = s.ttl
	case s.ttl > 0 && s.ttl < ttl:
		ttl = s.ttl
	}

	if err := s.client.Set(ctx, s.conversationIndexMigratedKey(conversationID), string(proof), ttl).Err(); err != nil {
		return fmt.Errorf("failed to mark conversation %s migrated: %w", conversationID, err)
	}

	return nil
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
