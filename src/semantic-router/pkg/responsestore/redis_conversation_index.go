package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync/atomic"

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
// the global scan lease immediately afterward, the conversation is marked
// migrated once this returns without error — meaning its index (populated
// or absent) may now be trusted as exhaustive. It runs the O(N) legacy
// scan (lazyBackfillConversationIndex) at most once per conversation per
// proof lifetime, and runs it unconditionally when not yet migrated, even
// if the index already has some members from earlier post-upgrade writes:
// those members alone do not prove nothing legacy is left to discover.
//
// Waiter state machine: check whether the whole store is already
// migration-complete or this conversation already has a resolved proof
// (conversationIndexResolved) before ever contending for the lease; if
// neither, block (via withConversationIndexScanLease, respecting request
// cancellation) until this call holds the single global scan lease, then
// recheck resolution once more — a concurrent scan may have just resolved
// this same conversation while this call was waiting — before actually
// running the legacy scan. Never falls back to scanning without the lease,
// at any timeout: unlike the superseded per-conversation lock, there is no
// duplicate-scan risk to bound here, since the lease is what makes "at
// most one full-keyspace scan running at a time" true in the first place.
func (s *RedisStore) ensureConversationIndex(ctx context.Context, conversationID string) error {
	if resolved, err := s.conversationIndexResolved(ctx, conversationID); err != nil {
		return err
	} else if resolved {
		return nil
	}

	return s.withConversationIndexScanLease(ctx, func(leaseCtx context.Context) error {
		if resolved, err := s.conversationIndexResolved(leaseCtx, conversationID); err != nil {
			return err
		} else if resolved {
			return nil
		}

		found, err := s.lazyBackfillConversationIndex(leaseCtx, conversationID)
		if err != nil {
			return err
		}
		logging.Debugf("RedisStore: lazy-backfilled conversation %s index with %d response(s)",
			conversationID, found)

		return nil
	})
}

// conversationIndexResolved reports whether a scan for conversationID would
// be redundant: either the whole store is already marked migration-complete
// (ConversationIndexMigrationStatusKey), or this specific conversation
// already carries a resolved typed proof.
func (s *RedisStore) conversationIndexResolved(ctx context.Context, conversationID string) (bool, error) {
	if complete, err := s.isMigrationComplete(ctx); err != nil {
		return false, err
	} else if complete {
		return true, nil
	}

	_, resolved, err := s.conversationIndexProof(ctx, conversationID)
	return resolved, err
}

// lazyBackfillConversationIndex performs the one-time O(N) scan that makes a
// conversation's full response set discoverable: it walks every response
// payload once (scanResponsePayloads, Cluster-aware via ForEachMaster) and,
// for each decoded batch, streams the matching members straight into the
// index via indexBackfillBatch — never accumulating the scan's findings
// into one shared slice first. That matters specifically because
// ForEachMaster invokes its per-master callback concurrently in Cluster
// mode: any shared, unsynchronized state written from inside the visit
// callback (as a single accumulated slice would be) is a data race,
// whereas a callback-local batch flushed immediately, plus only an
// atomic.Int64 running total, has no shared mutable state to race on.
//
// Idempotent and additive: every ZADD (including one re-adding a member an
// ordinary write already indexed, harmlessly, with the same score) only
// ever adds, so a concurrently indexed write racing this scan is never
// undone by it, no matter which finishes first, and concurrent per-master
// ZADDs from different callback invocations are independent, idempotent
// operations that need no coordination between themselves.
//
// The typed proof is set only after the scan and every ZADD succeed —
// on any error, this returns without marking migrated, matching blueprint
// §5 Phase 3's "no proof on partial success"; the next call is a safe,
// fully idempotent retry.
func (s *RedisStore) lazyBackfillConversationIndex(ctx context.Context, conversationID string) (int64, error) {
	if s.lazyBackfillPreScanHook != nil {
		s.lazyBackfillPreScanHook()
	}

	var total atomic.Int64
	err := s.scanResponsePayloads(ctx, func(batch []*responseapi.StoredResponse) error {
		return s.indexBackfillMatches(ctx, conversationID, batch, &total)
	})
	if err != nil {
		return 0, fmt.Errorf("failed to backfill conversation index: %w", err)
	}

	found := total.Load()
	if found == 0 {
		s.finishEmptyBackfill(ctx, conversationID)
		return 0, nil
	}

	s.finishPopulatedBackfill(ctx, conversationID)
	return found, nil
}

// indexBackfillMatches filters one scanned batch down to the members
// belonging to conversationID and flushes them to the index in chunks
// bounded by redisBackfillBatchSize, adding each flushed chunk's size to
// total. The members slice is callback-local: safe even when ForEachMaster
// invokes this concurrently across masters, since each invocation gets its
// own slice, and total is the only state shared between them — updated
// exclusively through atomic.Int64.
func (s *RedisStore) indexBackfillMatches(ctx context.Context, conversationID string, batch []*responseapi.StoredResponse, total *atomic.Int64) error {
	members := make([]redis.Z, 0, min(len(batch), redisBackfillBatchSize))
	for _, response := range batch {
		if response.ConversationID != conversationID {
			continue
		}
		members = append(members, redis.Z{Score: float64(response.CreatedAt), Member: response.ID})
		if len(members) >= redisBackfillBatchSize {
			if err := s.indexBackfillBatch(ctx, conversationID, members); err != nil {
				return err
			}
			total.Add(int64(len(members)))
			members = members[:0]
		}
	}
	if len(members) == 0 {
		return nil
	}
	if err := s.indexBackfillBatch(ctx, conversationID, members); err != nil {
		return err
	}
	total.Add(int64(len(members)))
	return nil
}

// finishEmptyBackfill marks conversationID migrated with the empty proof
// after a completed scan found no live responses for it. Best-effort: see
// markConversationMigrated's own doc comment for why a failed write here is
// logged and swallowed rather than returned.
func (s *RedisStore) finishEmptyBackfill(ctx context.Context, conversationID string) {
	if err := s.markConversationMigrated(ctx, conversationID, conversationIndexProofEmpty); err != nil {
		logging.Debugf("RedisStore: failed to mark conversation %s migrated (empty): %v", conversationID, err)
	}
}

// finishPopulatedBackfill refreshes the backfilled index's TTL once — after
// every batch across every master has already succeeded, not per batch,
// which would be redundant work for no additional safety — and marks
// conversationID migrated with the populated proof. Both steps are
// best-effort; see markConversationMigrated.
func (s *RedisStore) finishPopulatedBackfill(ctx context.Context, conversationID string) {
	if s.ttl > 0 {
		if err := s.client.Expire(ctx, s.conversationIndexKey(conversationID), s.ttl).Err(); err != nil {
			logging.Warnf("RedisStore: failed to refresh TTL on backfilled conversation index %s: %v",
				conversationID, err)
		}
	}
	if err := s.markConversationMigrated(ctx, conversationID, conversationIndexProofPopulated); err != nil {
		logging.Debugf("RedisStore: failed to mark conversation %s migrated (populated): %v", conversationID, err)
	}
}

// indexBackfillBatch ZADDs one bounded batch (at most redisBackfillBatchSize
// members, enforced by lazyBackfillConversationIndex's caller-side
// allocation) into a conversation's index. A thin wrapper — its only job is
// giving this one Redis call its own name and error context, since
// lazyBackfillConversationIndex now calls it once per flushed batch per
// callback invocation, potentially from several concurrent goroutines (one
// per Cluster master) at once; each call is independent and idempotent, so
// no coordination between concurrent callers is needed.
func (s *RedisStore) indexBackfillBatch(ctx context.Context, conversationID string, members []redis.Z) error {
	if len(members) == 0 {
		return nil
	}
	if err := s.client.ZAdd(ctx, s.conversationIndexKey(conversationID), members...).Err(); err != nil {
		return fmt.Errorf("failed to backfill conversation index: %w", err)
	}
	return nil
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
