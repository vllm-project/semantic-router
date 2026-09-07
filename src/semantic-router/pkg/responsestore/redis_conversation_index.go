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

// conversationIndexAddScript adds members to KEYS[1] and, in the same atomic
// step, makes sure the key's expiry covers at least ARGV[1] milliseconds.
// ARGV[1] <= 0 means "this index must never expire"; ARGV[2..] are score,
// member pairs, and may be empty to ask only about the lifetime.
//
// The expiry is only ever raised, never lowered. A conversation index must
// outlive every live payload it names, or a read finds no index and — once the
// store is finalized (ConversationIndexCompletionKeySuffix) — reports the
// conversation empty while its payloads are still sitting there, permanently
// and silently. Different writers know different lower bounds for how long
// that is: an ordinary write knows only the store TTL it just wrote its own
// payload with, while a backfill has seen a legacy payload's real remaining
// lifetime, which can be weeks longer. Whichever writer happens to run last
// must not be the one that decides.
//
// Reading PTTL *before* the ZADD is what makes that decision possible at all.
// Afterwards, a key with no expiry is ambiguous: it may be an index this
// script deliberately persisted to cover a payload that never expires, or one
// the ZADD just created, which Redis leaves without a TTL. Beforehand, the two
// are distinct — -1 is the former and must be left alone, -2 is the latter and
// takes the requested lifetime. That ambiguity is also why EXPIRE's GT flag
// cannot express this (it reads a missing TTL as infinite), quite apart from
// making Redis 7.0 a hard runtime requirement.
//
// Single-key: KEYS[1] only, so it stays legal in Redis Cluster. Callers keep
// members within redisBackfillBatchSize, so the unpack below stays far inside
// Lua's argument limit.
var conversationIndexAddScript = redis.NewScript(`
local existing = redis.call("PTTL", KEYS[1])
if #ARGV > 1 then
	redis.call("ZADD", KEYS[1], unpack(ARGV, 2))
end
if redis.call("EXISTS", KEYS[1]) == 0 then
	return 0
end
local want = tonumber(ARGV[1])
if want <= 0 then
	if existing >= 0 then
		redis.call("PERSIST", KEYS[1])
	end
	return 1
end
if existing == -1 or (existing >= 0 and existing >= want) then
	return 1
end
redis.call("PEXPIRE", KEYS[1], want)
return 1
`)

// conversationIndexAddArgs builds conversationIndexAddScript's ARGV: the
// requested lifetime, then each member's score and value.
func conversationIndexAddArgs(lifetimeMillis int64, members []redis.Z) []interface{} {
	args := make([]interface{}, 0, 1+2*len(members))
	args = append(args, lifetimeMillis)
	for _, member := range members {
		args = append(args, member.Score, member.Member)
	}
	return args
}

// addConversationIndexMembers runs conversationIndexAddScript as a standalone
// command. EVALSHA with go-redis's NOSCRIPT fallback, which is only available
// outside a pipeline.
func (s *RedisStore) addConversationIndexMembers(ctx context.Context, indexKey string, lifetimeMillis int64, members []redis.Z) error {
	return conversationIndexAddScript.Run(ctx, s.client, []string{indexKey}, conversationIndexAddArgs(lifetimeMillis, members)...).Err()
}

// queueConversationIndexMembers queues the same script onto an existing
// pipeline, for a caller writing to several conversations at once.
//
// EVAL rather than EVALSHA: inside a pipeline go-redis cannot run Script.Run's
// NOSCRIPT-to-EVAL fallback, because a queued command's error is not readable
// until Exec, so a NOSCRIPT against a Redis that has never seen this script
// would fail the whole batch. Redis caches by SHA on EVAL too, so the only
// cost is shipping the script body once per conversation in the batch.
func queueConversationIndexMembers(ctx context.Context, pipe redis.Pipeliner, indexKey string, lifetimeMillis int64, members []redis.Z) {
	conversationIndexAddScript.Eval(ctx, pipe, []string{indexKey}, conversationIndexAddArgs(lifetimeMillis, members)...)
}

// longerIndexLifetime folds one member payload's remaining lifetime into the
// lifetime its conversation index is being asked to cover, keeping whichever
// is longer.
//
// Both arguments use this package's index-lifetime convention: a non-positive
// value means "never expires", and therefore dominates any finite one. A
// payload Redis reported as already gone (unknownPayloadTTL) is not a
// lifetime at all and must be filtered out by the caller rather than passed
// in — it would otherwise read as "unbounded".
func longerIndexLifetime(want, candidate int64) int64 {
	if want <= 0 || candidate <= 0 {
		return 0
	}
	if candidate > want {
		return candidate
	}
	return want
}

// scannedResponse is one strictly decoded payload together with the remaining
// lifetime Redis reported for it in the same round trip.
//
// The lifetime travels with the payload because only the scan that read it
// knows how long it actually has left: a payload written before the store's
// TTL was shortened, or by an older deployment, routinely outlives s.ttl by
// weeks, and the index built from it must outlive it in turn. ttlMillis
// follows Redis's PTTL convention — -1 never expires, unknownPayloadTTL means
// there was nothing left to measure.
type scannedResponse struct {
	response  *responseapi.StoredResponse
	ttlMillis int64
}

// indexLifetime is how long this response's index membership must remain
// readable, in milliseconds, under the convention longerIndexLifetime uses.
// A payload with no measurable lifetime contributes nothing, so it reports
// the store's own TTL instead of pretending the index may expire immediately.
func (r scannedResponse) indexLifetime(storeTTLMillis int64) int64 {
	if r.ttlMillis == unknownPayloadTTL {
		return storeTTLMillis
	}
	return longerIndexLifetime(storeTTLMillis, r.ttlMillis)
}

// indexResponse adds a response to its conversation index, scored by
// created_at, and makes the index outlive the payload it just named.
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
//
// memberLifetimeMillis is how long the payload being indexed has left, which
// only the caller knows: a fresh write just stored it with the store TTL, a
// rollback is restoring a snapshot with whatever was left of the original,
// and a repair is re-indexing a payload written at some earlier time. The
// index is extended to cover it and never shortened — see
// extendConversationIndexLifetimeScript for why that matters.
func (s *RedisStore) indexResponse(ctx context.Context, conversationID, responseID string, createdAt, memberLifetimeMillis int64) error {
	if conversationID == "" || responseID == "" {
		return nil
	}
	indexKey := s.conversationIndexKey(conversationID)
	lifetime := longerIndexLifetime(s.ttlMillis(), memberLifetimeMillis)
	members := []redis.Z{{Score: float64(createdAt), Member: responseID}}

	if err := s.addConversationIndexMembers(ctx, indexKey, lifetime, members); err != nil {
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
// Idempotent and additive: every index write (including one re-adding a
// member an ordinary write already indexed, harmlessly, with the same score)
// only ever adds, so a concurrently indexed write racing this scan is never
// undone by it, no matter which finishes first, and concurrent per-master
// writes from different callback invocations are independent, idempotent
// operations that need no coordination between themselves.
//
// The typed proof is set only after the scan and every index write succeed —
// on any error, this returns without marking migrated, matching blueprint
// §5 Phase 3's "no proof on partial success"; the next call is a safe,
// fully idempotent retry.
func (s *RedisStore) lazyBackfillConversationIndex(ctx context.Context, conversationID string) (int64, error) {
	var total atomic.Int64
	err := s.scanResponsePayloads(ctx, func(batch []scannedResponse) error {
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
//
// Each chunk carries the longest lifetime among its own members, and nothing
// accumulates a maximum across chunks: because the index's expiry is only
// ever extended, applying each chunk's own bound converges on the longest
// across all of them, in any order and from any number of concurrent
// callback invocations. That is what keeps this free of shared mutable state
// beyond total.
func (s *RedisStore) indexBackfillMatches(ctx context.Context, conversationID string, batch []scannedResponse, total *atomic.Int64) error {
	storeTTL := s.ttlMillis()
	members := make([]redis.Z, 0, min(len(batch), redisBackfillBatchSize))
	lifetime := storeTTL
	flush := func() error {
		if err := s.indexBackfillBatch(ctx, conversationID, members, lifetime); err != nil {
			return err
		}
		total.Add(int64(len(members)))
		members = members[:0]
		lifetime = storeTTL
		return nil
	}

	for _, scanned := range batch {
		if scanned.response.ConversationID != conversationID {
			continue
		}
		members = append(members, redis.Z{Score: float64(scanned.response.CreatedAt), Member: scanned.response.ID})
		lifetime = longerIndexLifetime(lifetime, scanned.indexLifetime(storeTTL))
		if len(members) >= redisBackfillBatchSize {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	if len(members) == 0 {
		return nil
	}
	return flush()
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

// finishPopulatedBackfill makes sure the backfilled index covers at least the
// store's own TTL — once, after every batch across every master has already
// succeeded — and marks conversationID migrated with the populated proof. The
// floor is all this adds: each batch has already extended the index to cover
// its own longest-lived member, and this can only ever raise that, never
// bring it back down to s.ttl. Both steps are best-effort; see
// markConversationMigrated.
func (s *RedisStore) finishPopulatedBackfill(ctx context.Context, conversationID string) {
	if err := s.addConversationIndexMembers(ctx, s.conversationIndexKey(conversationID), s.ttlMillis(), nil); err != nil {
		logging.Warnf("RedisStore: failed to refresh TTL on backfilled conversation index %s: %v",
			conversationID, err)
	}
	if err := s.markConversationMigrated(ctx, conversationID, conversationIndexProofPopulated); err != nil {
		logging.Debugf("RedisStore: failed to mark conversation %s migrated (populated): %v", conversationID, err)
	}
}

// indexBackfillBatch adds one bounded batch (at most redisBackfillBatchSize
// members, enforced by lazyBackfillConversationIndex's caller-side
// allocation) to a conversation's index, covering lifetimeMillis. A thin
// wrapper — its only job is giving this one Redis call its own name and error
// context, since
// lazyBackfillConversationIndex now calls it once per flushed batch per
// callback invocation, potentially from several concurrent goroutines (one
// per Cluster master) at once; each call is independent and idempotent, so
// no coordination between concurrent callers is needed.
func (s *RedisStore) indexBackfillBatch(ctx context.Context, conversationID string, members []redis.Z, lifetimeMillis int64) error {
	if len(members) == 0 {
		return nil
	}
	if err := s.addConversationIndexMembers(ctx, s.conversationIndexKey(conversationID), lifetimeMillis, members); err != nil {
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
func (s *RedisStore) scanResponsePayloads(ctx context.Context, visit func(batch []scannedResponse) error) error {
	return s.scanResponseKeys(ctx, func(ctx context.Context, client redis.UniversalClient, keys []string) ([]scannedResponse, error) {
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
	fetch func(ctx context.Context, client redis.UniversalClient, keys []string) ([]scannedResponse, error),
	visit func(batch []scannedResponse) error,
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
	fetch func(context.Context, redis.UniversalClient, []string) ([]scannedResponse, error),
	visit func([]scannedResponse) error,
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

// getResponsesPipelined decodes the shared raw pipelined results used by both
// lazy backfill and finalization. A key that expired between SCAN and GET is a
// benign TTL race and is skipped; every other read, decode, or key-identity
// failure aborts the whole scan rather than being logged and skipped, so
// neither a per-conversation proof nor the global completion record is ever
// published from an observation known to be incomplete.
//
// Each payload's remaining lifetime is read in the same pipeline, because the
// index these scans build has to outlive the payloads they found — and a
// legacy payload's real lifetime is knowable here and nowhere later.
func (s *RedisStore) getResponsesPipelined(ctx context.Context, client redis.UniversalClient, keys []string) ([]scannedResponse, error) {
	if len(keys) == 0 {
		return nil, nil
	}

	results := fetchResponsePayloadsAndTTLsPipelined(ctx, client, keys)
	responses := make([]scannedResponse, 0, len(keys))
	for i, result := range results {
		response, err := s.decodeScannedResponse(keys[i], result)
		if err != nil {
			return nil, err
		}
		if response != nil {
			responses = append(responses, scannedResponse{response: response, ttlMillis: result.ttlMillis})
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
