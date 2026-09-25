package responsestore

import (
	"context"
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

// conversationIndexAddScript atomically updates a conversation's response
// ZSET (KEYS[1]) and generation-witness HASH (KEYS[2]). The keys carry the
// same escaped conversation hash tag, so this remains legal in Redis Cluster.
// ARGV[1] is the required lifetime; the remaining arguments are score,
// response ID, generation triples.
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
// A legacy member has no generation. It may be added only with ZADD NX and
// never receives a witness: generation-less records cannot authorize a later
// prune or CAS delete. The two keys are extended to the same monotonic
// lifetime, so the witness cannot expire before the membership it protects.
//
// ARGV[2] selects who is allowed to overwrite an existing witness, and the
// distinction is a correctness boundary, not a convenience:
//
//   - "owned": the caller wrote this exact payload generation moments ago
//     (StoreResponse, UpdateResponse, a rollback restore), so it is by
//     definition the newest and overwrites unconditionally.
//   - "repair": the caller only *read* the generation, so its observation may
//     already be stale. The witness is installed compare-and-set, only while
//     it still equals the value ARGV[i+3] says the caller observed. A scan
//     passes the empty string here, which is the maintainer's "install only
//     while absent" rule; cascade passes the witness its own snapshot read.
//
// Without that split, a delayed scan could stamp the generation it read weeks
// of keyspace ago over a witness a live update had just installed — after
// which a conditional prune, quite correctly, removes a membership whose
// payload is alive, and the response is left indexed nowhere.
//
// Returns how many witnesses were actually written, so a repair caller can
// tell a real install from a compare-and-set that legitimately lost.
var conversationIndexAddScript = redis.NewScript(`
local zset_ttl = redis.call("PTTL", KEYS[1])
local generation_ttl = redis.call("PTTL", KEYS[2])
local owned = ARGV[2] == "owned"
local installed = 0

for i = 3, #ARGV, 4 do
	local score = ARGV[i]
	local response_id = ARGV[i + 1]
	local generation = ARGV[i + 2]
	local expected = ARGV[i + 3]
	if generation == "" then
		if redis.call("HEXISTS", KEYS[2], response_id) == 0 then
			redis.call("ZADD", KEYS[1], "NX", score, response_id)
		end
	elseif owned then
		redis.call("ZADD", KEYS[1], score, response_id)
		redis.call("HSET", KEYS[2], response_id, generation)
		installed = installed + 1
	else
		local current = redis.call("HGET", KEYS[2], response_id)
		if current == false then
			current = ""
		end
		if current == expected then
			redis.call("ZADD", KEYS[1], score, response_id)
			redis.call("HSET", KEYS[2], response_id, generation)
			installed = installed + 1
		end
	end
end

if redis.call("EXISTS", KEYS[1]) == 0 then
	redis.call("DEL", KEYS[2])
	return 0
end

local want = tonumber(ARGV[1])
local generation_exists = redis.call("EXISTS", KEYS[2]) == 1
if generation_exists then
	if want <= 0 or zset_ttl == -1 or generation_ttl == -1 then
		redis.call("PERSIST", KEYS[1])
		redis.call("PERSIST", KEYS[2])
		return installed
	end
	local shared_ttl = want
	if zset_ttl > shared_ttl then shared_ttl = zset_ttl end
	if generation_ttl > shared_ttl then shared_ttl = generation_ttl end
	redis.call("PEXPIRE", KEYS[1], shared_ttl)
	redis.call("PEXPIRE", KEYS[2], shared_ttl)
	return installed
end

if want <= 0 or zset_ttl == -1 then
	redis.call("PERSIST", KEYS[1])
	return installed
end
if zset_ttl < want then redis.call("PEXPIRE", KEYS[1], want) end
return installed
`)

// conditionalUnindexScript removes response IDs only while their sidecar
// witness still equals the generation observed by the caller. ZREM and HDEL
// are one same-slot atomic operation.
//
// A blank expected generation is a legitimate observation, not a missing
// one: it is exactly what a member backfilled from a legacy payload carries
// (see conversationIndexAddScript), and what a DeleteResponse or
// UpdateResponse that displaced a legacy payload owns. Refusing to compare
// it made those members immortal — a payload that expires or is deleted
// leaves a tombstone no path can ever remove, which past finalization
// repeatedly underfills every page that reads it.
//
// Comparing it is still generation-safe, and for the same reason a non-blank
// comparison is: the removal fires only while the sidecar field is *still*
// absent. Any generation-aware writer that recreates the response, or moves
// it back, installs its own witness through conversationIndexAddScript
// first, so a stale blank-expected cleanup finds a mismatch and no-ops. The
// only writer that can defeat it is an index-unaware one, which exists only
// before finalization — and there conversationIndexProofMaxTTL forces the
// re-scan that rediscovers the member.
//
// Redis Lua answers a missing hash field with false rather than the empty
// string, so absence is normalized before the comparison; without that,
// blank could never match anything.
var conditionalUnindexScript = redis.NewScript(`
local removed = 0
for i = 1, #ARGV, 2 do
	local response_id = ARGV[i]
	local expected = ARGV[i + 1]
	local current = redis.call("HGET", KEYS[2], response_id)
	if current == false then
		current = ""
	end
	if current == expected then
		removed = removed + redis.call("ZREM", KEYS[1], response_id)
		redis.call("HDEL", KEYS[2], response_id)
	end
end
return removed
`)

// indexWitnessMode says whether a caller is entitled to overwrite an existing
// generation witness. See conversationIndexAddScript.
type indexWitnessMode string

const (
	// witnessOwned is for a caller that just wrote the payload carrying this
	// generation, and therefore holds the newest one by construction.
	witnessOwned indexWitnessMode = "owned"
	// witnessRepair is for a caller that only read the generation. Its write
	// is compare-and-set against conversationIndexMember.expected.
	witnessRepair indexWitnessMode = "repair"
)

type conversationIndexMember struct {
	responseID string
	generation string
	// expected is the witness value the caller observed, and is honored only
	// in witnessRepair mode. The empty string means "observed absent", which
	// is what every scan passes.
	expected string
	score    float64
}

type responseGenerationWitness struct {
	responseID string
	generation string
}

// conversationIndexAddArgs builds conversationIndexAddScript's ARGV: the
// requested lifetime, then each member's score and value.
func conversationIndexAddArgs(lifetimeMillis int64, mode indexWitnessMode, members []conversationIndexMember) []interface{} {
	args := make([]interface{}, 0, 2+4*len(members))
	args = append(args, lifetimeMillis, string(mode))
	for _, member := range members {
		args = append(args, member.score, member.responseID, member.generation, member.expected)
	}
	return args
}

// addConversationIndexMembers runs conversationIndexAddScript as a standalone
// command. EVALSHA with go-redis's NOSCRIPT fallback, which is only available
// outside a pipeline.
func (s *RedisStore) addConversationIndexMembers(ctx context.Context, conversationID string, mode indexWitnessMode, lifetimeMillis int64, members []conversationIndexMember) (int, error) {
	keys := []string{s.conversationIndexKey(conversationID), s.conversationIndexGenerationKey(conversationID)}
	res, err := conversationIndexAddScript.Run(ctx, s.client, keys, conversationIndexAddArgs(lifetimeMillis, mode, members)...).Result()
	if err != nil {
		return 0, err
	}
	installed, ok := res.(int64)
	if !ok {
		return 0, fmt.Errorf("unexpected conversation index add result type %T for %s", res, conversationID)
	}
	return int(installed), nil
}

// queueConversationIndexMembers queues the same script onto an existing
// pipeline, for a caller writing to several conversations at once.
//
// EVAL rather than EVALSHA: inside a pipeline go-redis cannot run Script.Run's
// NOSCRIPT-to-EVAL fallback, because a queued command's error is not readable
// until Exec, so a NOSCRIPT against a Redis that has never seen this script
// would fail the whole batch. Redis caches by SHA on EVAL too, so the only
// cost is shipping the script body once per conversation in the batch.
func (s *RedisStore) queueConversationIndexMembers(ctx context.Context, pipe redis.Pipeliner, conversationID string, mode indexWitnessMode, lifetimeMillis int64, members []conversationIndexMember) {
	keys := []string{s.conversationIndexKey(conversationID), s.conversationIndexGenerationKey(conversationID)}
	conversationIndexAddScript.Eval(ctx, pipe, keys, conversationIndexAddArgs(lifetimeMillis, mode, members)...)
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
	response   *responseapi.StoredResponse
	generation string
	ttlMillis  int64
}

// scanLegacyPolicy makes promotion an explicit capability of a scan. Lazy
// backfill runs while index-unaware writers may still exist and must therefore
// preserve legacy payloads. Only FinalizeConversationIndex, whose operator
// contract requires those writers to be drained first, may promote them.
type scanLegacyPolicy uint8

const (
	preserveLegacyPayloads scanLegacyPolicy = iota
	promoteLegacyForFinalization
)

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
// conversationIndexAddScript for why that matters.
func (s *RedisStore) indexResponse(ctx context.Context, conversationID, responseID, generation string, createdAt, memberLifetimeMillis int64) error {
	if conversationID == "" || responseID == "" {
		return nil
	}
	lifetime := longerIndexLifetime(s.ttlMillis(), memberLifetimeMillis)
	members := []conversationIndexMember{{responseID: responseID, generation: generation, score: float64(createdAt)}}

	if _, err := s.addConversationIndexMembers(ctx, conversationID, witnessOwned, lifetime, members); err != nil {
		return fmt.Errorf("failed to index response %s in conversation %s: %w", responseID, conversationID, err)
	}

	return nil
}

// repairResponseWitness installs a witness a caller only *read*, and reports
// whether it actually landed.
//
// Compare-and-set against the witness the caller observed, never a blind
// overwrite, because a reader's observation can already be stale by the time
// it writes: a live update that installed its own witness in between owns the
// membership, and stamping a read-somewhere-earlier generation over it would
// leave the next conditional prune correctly removing a membership whose
// payload is alive. Callers that just wrote the payload use indexResponse
// instead — they hold the newest generation by construction.
//
// expected == "" means "I observed no witness", which is both what a scan
// asserts and what a cascade batch reads for a legacy member. A blank
// generation still routes through the script's legacy branch (ZADD NX, no
// witness), so a caller repairing a pre-upgrade payload's membership keeps
// working; it simply installs nothing and reports zero.
func (s *RedisStore) repairResponseWitness(ctx context.Context, conversationID, responseID, generation, expected string, createdAt, memberLifetimeMillis int64) (int, error) {
	if conversationID == "" || responseID == "" {
		return 0, nil
	}
	lifetime := longerIndexLifetime(s.ttlMillis(), memberLifetimeMillis)
	members := []conversationIndexMember{{
		responseID: responseID, generation: generation, expected: expected, score: float64(createdAt),
	}}

	installed, err := s.addConversationIndexMembers(ctx, conversationID, witnessRepair, lifetime, members)
	if err != nil {
		return 0, fmt.Errorf("failed to repair witness for response %s in conversation %s: %w", responseID, conversationID, err)
	}
	return installed, nil
}

// unindexResponseGenerations removes only generation witnesses the caller
// actually observed, and reports how many memberships that actually removed.
// The ZSET and sidecar HASH are co-located and changed by one script,
// preventing a stale reader from deleting a recreated member.
//
// The count matters to cascade delete, which must not mistake "I issued the
// right conditional command" for "the member is gone": a writer that keeps
// refreshing a witness makes every conditional removal a legal no-op, and a
// drain loop counting attempts rather than removals would spin on it forever.
func (s *RedisStore) unindexResponseGenerations(ctx context.Context, conversationID string, witnesses ...responseGenerationWitness) (int, error) {
	if conversationID == "" || len(witnesses) == 0 {
		return 0, nil
	}

	args := make([]interface{}, 0, 2*len(witnesses))
	for _, witness := range witnesses {
		args = append(args, witness.responseID, witness.generation)
	}

	keys := []string{s.conversationIndexKey(conversationID), s.conversationIndexGenerationKey(conversationID)}
	res, err := conditionalUnindexScript.Run(ctx, s.client, keys, args...).Result()
	if err != nil {
		return 0, fmt.Errorf("failed to remove %d response(s) from conversation %s index: %w", len(witnesses), conversationID, err)
	}
	removed, ok := res.(int64)
	if !ok {
		return 0, fmt.Errorf("unexpected conditional unindex result type %T for conversation %s", res, conversationID)
	}

	return int(removed), nil
}

// dropObservedMembership removes a membership a caller has just proven stale
// by displacing or deleting the payload that owned it, honoring the same
// finalization gate as every other blank-witness cleanup.
//
// A generational witness is dropped unconditionally: the caller observed that
// exact generation and only that generation can be removed. A blank one is
// dropped only once the store is finalized, because until then an
// index-unaware writer can put a payload back under the same ID without
// touching the sidecar — reachable in particular after DeleteResponse, which
// leaves the key free for a SETNX to win. Before finalization the membership
// is left in place; it is a tombstone the finalization sweep resolves, which
// is the trade the migration window is explicitly willing to make.
//
// Best-effort, like both of its callers: a skipped or failed cleanup costs a
// stale member, never a lost payload.
func (s *RedisStore) dropObservedMembership(ctx context.Context, conversationID string, witness responseGenerationWitness) error {
	if conversationID == "" {
		return nil
	}
	if witness.generation == "" {
		finalized, err := s.conversationIndexFinalized(ctx)
		if err != nil {
			return err
		}
		if !finalized {
			logging.Debugf("RedisStore: leaving response %s indexed in conversation %s: its payload carried no generation and the store is not finalized",
				witness.responseID, conversationID)
			return nil
		}
	}

	_, err := s.unindexResponseGenerations(ctx, conversationID, witness)
	return err
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
	err := s.scanResponsePayloads(ctx, preserveLegacyPayloads, func(batch []scannedResponse) error {
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
	members := make([]conversationIndexMember, 0, min(len(batch), redisBackfillBatchSize))
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
		members = append(members, conversationIndexMember{
			responseID: scanned.response.ID,
			generation: scanned.generation,
			score:      float64(scanned.response.CreatedAt),
		})
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
	if _, err := s.addConversationIndexMembers(ctx, conversationID, witnessRepair, s.ttlMillis(), nil); err != nil {
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
func (s *RedisStore) indexBackfillBatch(ctx context.Context, conversationID string, members []conversationIndexMember, lifetimeMillis int64) error {
	if len(members) == 0 {
		return nil
	}
	// witnessRepair: a scan only ever *read* these generations, and a live
	// writer that has since claimed a member owns its witness.
	if _, err := s.addConversationIndexMembers(ctx, conversationID, witnessRepair, lifetimeMillis, members); err != nil {
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
