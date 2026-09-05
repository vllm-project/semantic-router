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
	// live responses for the conversation.
	conversationIndexProofEmpty conversationIndexProof = "v1:empty"
	// conversationIndexProofPopulated means a completed legacy scan found
	// live responses, now reflected in the index.
	//
	// Both values are equally perishable, and for the same reason: either
	// claim can be invalidated by an index-unaware writer landing a
	// response after the scan that produced it. See markConversationMigrated
	// for the TTL policy that bounds how long either may be trusted.
	conversationIndexProofPopulated conversationIndexProof = "v1:populated"
)

// indexResponse adds a response to its conversation index, scored by
// created_at, and refreshes the index's own TTL.
//
// Deliberately touches no migrated proof at all — neither setting one nor
// extending one. Setting: this call proves nothing about whether a
// legacy-scan backfill has ever run for conversationID (a fresh
// conversation's very first write reaches this exact path), so it must
// never be mistaken for the signal that makes the index trustworthy as
// exhaustive. Extending: a pre-finalization proof is a *time-bounded*
// claim (see markConversationMigrated), and letting ordinary writes push
// its deadline out would defeat that bound entirely — a conversation
// written to more often than once every conversationIndexProofMaxTTL
// would keep its proof alive indefinitely, so a response an index-unaware
// writer landed after the scan's cursor passed would never be rediscovered.
// Every proof must expire on its own schedule until FinalizeConversationIndex
// seals the store; after that, proofs are not consulted at all.
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
// neither, block (via withConversationIndexScanLeaseUntil, respecting
// request cancellation) until this call holds the single global scan lease,
// rechecking resolution on every acquisition attempt so a waiter returns as
// soon as a concurrent scan resolves this same conversation, and once more
// under the lease before actually running the legacy scan. Never falls back
// to scanning without the lease, at any timeout: unlike the superseded
// per-conversation lock, there is no
// duplicate-scan risk to bound here, since the lease is what makes "at
// most one full-keyspace scan running at a time" true in the first place.
func (s *RedisStore) ensureConversationIndex(ctx context.Context, conversationID string) error {
	if resolved, err := s.conversationIndexResolved(ctx, conversationID); err != nil {
		return err
	} else if resolved {
		return nil
	}

	return s.withConversationIndexScanLeaseUntil(ctx, func(checkCtx context.Context) (bool, error) {
		return s.conversationIndexResolved(checkCtx, conversationID)
	}, func(leaseCtx context.Context) error {
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
// be redundant: either the whole store is already marked finalized
// (ConversationIndexCompletionKeySuffix), or this specific conversation
// already carries a resolved typed proof.
func (s *RedisStore) conversationIndexResolved(ctx context.Context, conversationID string) (bool, error) {
	if complete, err := s.conversationIndexFinalized(ctx); err != nil {
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
// TTL: every proof this writes — populated as well as empty — is capped at
// conversationIndexProofMaxTTL (or the store's own TTL if that is
// shorter), and nothing ever extends it afterwards.
//
// A populated proof used to get the full store TTL, on the reasoning that
// there is no blind spot once real data has been discovered and indexed.
// That reasoning does not survive a rolling upgrade. An index-unaware
// writer can land an unindexed response into an *already populated*
// conversation just as easily as into an empty one — and if it lands after
// this scan's cursor has passed its shard, only the proof expiring can
// force the re-scan that discovers it. With a full-TTL proof that
// indexResponse refreshed on every subsequent write, a conversation written
// to more often than the store TTL would hold a proof that never expires,
// hiding that response permanently. So every pre-finalization proof is a
// deliberately short-lived, self-revalidating claim: the store re-scans a
// conversation at most once per cap until FinalizeConversationIndex seals
// the whole keyspace, after which proofs stop being consulted entirely and
// the re-scan cost disappears with them.
//
// Returns the write error rather than swallowing it, so a caller that
// wants to know can (e.g. Phase 4's streaming backfill, which must not
// publish a proof at all on partial failure); callers for whom this
// remains best-effort (this file's own lazyBackfillConversationIndex) log
// and continue, since the next read simply re-scans rather than trust a
// proof that failed to write.
func (s *RedisStore) markConversationMigrated(ctx context.Context, conversationID string, proof conversationIndexProof) error {
	ttl := conversationIndexProofMaxTTL
	if s.ttl > 0 && s.ttl < ttl {
		ttl = s.ttl
	}

	if err := s.client.Set(ctx, s.conversationIndexMigratedKey(conversationID), string(proof), ttl).Err(); err != nil {
		return fmt.Errorf("failed to mark conversation %s migrated: %w", conversationID, err)
	}

	return nil
}

// scanResponsePayloads walks every response payload key exactly once and
// delivers strictly decoded records in bounded batches. A key expiring
// between SCAN and GET is benign; any other GET, decode, or key-identity
// failure aborts the scan so no completeness proof is published from an
// incomplete observation.
//
// Shared by the per-conversation lazy legacy backfill
// (lazyBackfillConversationIndex) and the whole-keyspace finalization sweep
// (sweepAndIndexAllConversations) — this is the O(N) operation the index
// exists to avoid on the hot read path. See scanResponseKeys for the
// Cluster-aware key walking.
func (s *RedisStore) scanResponsePayloads(ctx context.Context, visit func(batch []*responseapi.StoredResponse) error) error {
	return s.scanResponseKeys(ctx, func(ctx context.Context, client redis.UniversalClient, keys []string) ([]*responseapi.StoredResponse, error) {
		return s.getResponsesPipelined(ctx, client, keys)
	}, visit)
}

// scanResponseKeys walks every response payload key exactly once via SCAN,
// fetching each bounded batch of keys (redisBackfillBatchSize) through
// fetch and delivering the result to visit.
//
// Cluster-aware: a single Redis Cluster node's keyspace only holds the slots
// assigned to it, so in Cluster mode this scans every master via
// ForEachMaster (which invokes fetch/visit concurrently across masters, so
// callers must not share mutable callback state except through atomics).
// Standalone mode scans the one client directly.
func (s *RedisStore) scanResponseKeys(
	ctx context.Context,
	fetch func(ctx context.Context, client redis.UniversalClient, keys []string) ([]*responseapi.StoredResponse, error),
	visit func(batch []*responseapi.StoredResponse) error,
) error {
	s.scanInvocations.Add(1)

	pattern := s.buildKey(ResponseKeyPrefix + "*")

	if clusterClient, ok := s.client.(*redis.ClusterClient); ok {
		return clusterClient.ForEachMaster(ctx, func(ctx context.Context, master *redis.Client) error {
			return scanResponseNode(ctx, master, pattern, fetch, visit)
		})
	}

	return scanResponseNode(ctx, s.client, pattern, fetch, visit)
}

func scanResponseNode(
	ctx context.Context,
	client redis.UniversalClient,
	pattern string,
	fetch func(context.Context, redis.UniversalClient, []string) ([]*responseapi.StoredResponse, error),
	visit func([]*responseapi.StoredResponse) error,
) error {
	keys := make([]string, 0, redisBackfillBatchSize)
	flush := func() error {
		if len(keys) == 0 {
			return nil
		}
		batch, err := fetch(ctx, client, keys)
		keys = keys[:0]
		if err != nil || len(batch) == 0 {
			return err
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

// getResponsesPipelined decodes the shared raw pipelined GET results used by
// both lazy backfill and finalization. A key that expired between SCAN and
// GET is a benign TTL race and is skipped; every other read, decode, or
// key-identity failure aborts the whole scan rather than being logged and
// skipped, so neither a per-conversation proof nor the global completion
// record is ever published from an observation known to be incomplete.
func (s *RedisStore) getResponsesPipelined(ctx context.Context, client redis.UniversalClient, keys []string) ([]*responseapi.StoredResponse, error) {
	if len(keys) == 0 {
		return nil, nil
	}

	results := fetchResponsePayloadsPipelined(ctx, client, keys)
	responses := make([]*responseapi.StoredResponse, 0, len(keys))
	for i, result := range results {
		response, err := s.decodeScannedResponse(keys[i], result)
		if err != nil {
			return nil, err
		}
		if response != nil {
			responses = append(responses, response)
		}
	}

	return responses, nil
}

func (s *RedisStore) decodeScannedResponse(key string, result responsePayloadResult) (*responseapi.StoredResponse, error) {
	if result.err != nil {
		if errors.Is(result.err, redis.Nil) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read response at key %s during scan: %w", key, result.err)
	}

	var response responseapi.StoredResponse
	if err := json.Unmarshal(result.raw, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response at key %s during scan: %w", key, err)
	}
	if response.ID == "" || s.buildKey(ResponseKeyPrefix+response.ID) != key {
		return nil, fmt.Errorf("response payload identity does not match key %s", key)
	}
	return &response, nil
}
