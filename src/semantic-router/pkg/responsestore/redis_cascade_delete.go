package responsestore

import (
	"context"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"
)

// deleteEmptyConversationIndexScript removes an empty ZSET and its sidecar
// HASH atomically. A member arriving before the ZCARD keeps both keys alive;
// a writer arriving afterwards recreates both through conversationIndexAddScript.
var deleteEmptyConversationIndexScript = redis.NewScript(`
if redis.call("ZCARD", KEYS[1]) == 0 then
	redis.call("DEL", KEYS[1])
	redis.call("DEL", KEYS[2])
	return 1
end
return 0
`)

// readCascadeCandidatesScript snapshots ZSET candidates with their generation
// witnesses. Both keys are co-located by their escaped conversation hash tag.
var readCascadeCandidatesScript = redis.NewScript(`
local members = redis.call("ZRANGE", KEYS[1], 0, ARGV[1])
local result = {}
for _, response_id in ipairs(members) do
	table.insert(result, response_id)
	table.insert(result, redis.call("HGET", KEYS[2], response_id) or "")
end
return result
`)

type cascadeCandidate struct {
	responseID string
	generation string
}

// cascadeBatchProgress separates actual index drainage from the one other
// state transition that is safe to treat as monotone: promotion after global
// finalization. Witness repairs are intentionally absent; installing a repair
// does not shrink the index and a concurrent writer can invalidate it again.
type cascadeBatchProgress struct {
	membershipsRemoved int
	payloadsPromoted   int
}

func (s *RedisStore) deleteEmptyConversationIndex(ctx context.Context, conversationID string) (bool, error) {
	keys := []string{s.conversationIndexKey(conversationID), s.conversationIndexGenerationKey(conversationID)}
	res, err := deleteEmptyConversationIndexScript.Run(ctx, s.client, keys).Result()
	if err != nil {
		return false, fmt.Errorf("failed to delete conversation index %s: %w", keys[0], err)
	}
	deleted, ok := res.(int64)
	if !ok {
		return false, fmt.Errorf("unexpected conversation index delete result type %T for %s", res, keys[0])
	}
	return deleted > 0, nil
}

func (s *RedisStore) readCascadeCandidates(ctx context.Context, conversationID string) ([]cascadeCandidate, error) {
	keys := []string{s.conversationIndexKey(conversationID), s.conversationIndexGenerationKey(conversationID)}
	result, err := readCascadeCandidatesScript.Run(ctx, s.client, keys, redisDeleteBatchSize-1).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to list responses for deletion: %w", err)
	}
	items, ok := result.([]interface{})
	if !ok || len(items)%2 != 0 {
		return nil, fmt.Errorf("unexpected cascade candidate result: %#v", result)
	}

	candidates := make([]cascadeCandidate, 0, len(items)/2)
	for i := 0; i < len(items); i += 2 {
		responseID, idOK := items[i].(string)
		generation, generationOK := items[i+1].(string)
		if !idOK || !generationOK {
			return nil, fmt.Errorf("unexpected cascade candidate types %T/%T", items[i], items[i+1])
		}
		candidates = append(candidates, cascadeCandidate{responseID: responseID, generation: generation})
	}
	return candidates, nil
}

// deleteConversationResponseBatch resolves one bounded, generation-snapshotted
// batch and reports memberships actually removed separately from safe,
// post-finalization promotions. That distinction keeps witness repairs from
// masquerading as drainage and defeating the caller's bounded race loop.
//
// allowLegacyCleanup is the finalization gate. It guards removing a membership
// whose witness is blank and promoting a live legacy payload. Before
// finalization an index-unaware writer can recreate an absent payload without
// touching the sidecar, so the blank-expected removal still matches and the
// member goes — and unlike the read path, cascade destroys its own recovery
// on the way out. It deletes the migrated marker and the
// conversation record once the index drains, so no later scan will ever
// rediscover the recreated payload. The atomic ZCARD==0 check in
// deleteEmptyConversationIndexScript, and the whole race-round loop, exist to
// stop precisely that orphaning; an ungated blank ZREM would walk straight
// through both while reporting success. After finalization every writer is
// generation-aware, installs a witness before it can matter, and the blank
// removal is safe again.
//
// A live legacy payload is gated too. Promoting it before finalization could
// install a nonblank sidecar which an index-unaware writer then leaves stale
// when replacing the payload. That stale sidecar would bypass the blank gate
// on a later cleanup. Once finalized, legacy writers are gone, promotion is
// monotone, and a later round can delete the payload under the ordinary
// generation CAS.
func (s *RedisStore) deleteConversationResponseBatch(
	ctx context.Context,
	conversationID string,
	candidates []cascadeCandidate,
	allowLegacyCleanup bool,
) (cascadeBatchProgress, error) {
	if len(candidates) == 0 {
		return cascadeBatchProgress{}, nil
	}

	responseIDs := make([]string, len(candidates))
	for i, candidate := range candidates {
		responseIDs[i] = candidate.responseID
	}
	results := fetchResponsePayloadsPipelined(ctx, s.client, responseKeys(s, responseIDs))

	var progress cascadeBatchProgress
	var errs []error
	for i, result := range results {
		candidate := candidates[i]

		switch {
		case errors.Is(result.err, redis.Nil):
			removed, err := s.dropCascadeMembership(ctx, conversationID, candidate, allowLegacyCleanup)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			progress.membershipsRemoved += removed
		case result.err != nil:
			errs = append(errs, fmt.Errorf("failed to read response %s for cascade delete: %w", candidate.responseID, result.err))
		default:
			candidateProgress, err := s.resolveCascadeDeleteOutcome(ctx, conversationID, candidate, result, allowLegacyCleanup)
			if err != nil {
				errs = append(errs, err)
			}
			progress.membershipsRemoved += candidateProgress.membershipsRemoved
			progress.payloadsPromoted += candidateProgress.payloadsPromoted
		}
	}
	return progress, errors.Join(errs...)
}

// dropCascadeMembership removes a membership this batch has proven stale,
// conditional on the sidecar still holding exactly the witness it observed.
// A blank witness is refused before finalization: see
// deleteConversationResponseBatch for why cascade cannot safely absorb that
// race and still remove the conversation's recovery anchors.
func (s *RedisStore) dropCascadeMembership(
	ctx context.Context,
	conversationID string,
	candidate cascadeCandidate,
	allowLegacyCleanup bool,
) (int, error) {
	if candidate.generation == "" && !allowLegacyCleanup {
		return 0, fmt.Errorf(
			"response %s has no generation witness and the store is not finalized; refusing cascade delete",
			candidate.responseID)
	}

	return s.unindexResponseGenerations(ctx, conversationID,
		responseGenerationWitness{responseID: candidate.responseID, generation: candidate.generation})
}

// resolveCascadeDeleteOutcome decides and executes one candidate's outcome
// once its payload has been read. Only memberships genuinely removed and
// post-finalization payloads genuinely promoted count as progress; witness
// repair is deliberately a zero-drain result.
//
// Deliberately not "did I issue the right command". Every mutation here is
// conditional, so a writer that keeps changing a payload turns each one into a
// legal no-op — and a drain loop that counted those as progress would never
// terminate against a conversation being rewritten faster than it drains.
//
// A live payload that still belongs to this conversation is never unindexed,
// whatever its witness says. If the two disagree, the witness is repaired from
// the payload and the next round deletes it; removing the membership instead
// would leave a live response indexed nowhere, which past finalization nothing
// rediscovers.
func (s *RedisStore) resolveCascadeDeleteOutcome(
	ctx context.Context,
	conversationID string,
	candidate cascadeCandidate,
	result responsePayloadResult,
	allowLegacyCleanup bool,
) (cascadeBatchProgress, error) {
	record, err := decodeResponseRecord(result.raw)
	if err != nil {
		return cascadeBatchProgress{}, fmt.Errorf("failed to parse response %s during cascade delete: %w", candidate.responseID, err)
	}
	if record.response.ID != candidate.responseID {
		return cascadeBatchProgress{}, fmt.Errorf("response %s payload identity mismatch during cascade delete", candidate.responseID)
	}

	if record.response.ConversationID != conversationID {
		// Moved on since being indexed here. The payload belongs to another
		// conversation now, so only this conversation's stale membership goes
		// — under the same blank-witness gate, since an index-unaware writer
		// could just as well have moved it back after this read.
		removed, err := s.dropCascadeMembership(ctx, conversationID, candidate, allowLegacyCleanup)
		return cascadeBatchProgress{membershipsRemoved: removed}, err
	}

	if record.generation == "" {
		if !allowLegacyCleanup {
			return cascadeBatchProgress{}, fmt.Errorf(
				"response %s has a legacy payload and the store is not finalized; refusing cascade delete",
				candidate.responseID)
		}
		// Finalization guarantees every remaining writer is generation-aware.
		// Upgrade the residual legacy payload, install its witness, and stop
		// there; a later round directly observes the generated payload before
		// deleting it.
		return s.upgradeCascadeCandidate(ctx, conversationID, candidate, result, record)
	}

	if record.generation != candidate.generation {
		// The snapshot's witness has fallen behind the payload — an index
		// write still in flight, or an upgrade whose witness install lost its
		// race. Repair it from the payload rather than remove a live member.
		_, err := s.repairResponseWitness(ctx, conversationID, record.response.ID, record.generation,
			candidate.generation, record.response.CreatedAt, s.ttlMillis())
		return cascadeBatchProgress{}, err
	}

	deleted, err := s.compareDeleteResponsePayload(ctx, result.key, record.generation)
	if err != nil {
		return cascadeBatchProgress{}, fmt.Errorf("failed to delete response %s during cascade delete: %w", candidate.responseID, err)
	}
	if !deleted {
		return cascadeBatchProgress{}, fmt.Errorf("response %s changed concurrently during cascade delete; left in place for retry", candidate.responseID)
	}

	removed, err := s.unindexResponseGenerations(ctx, conversationID,
		responseGenerationWitness{responseID: candidate.responseID, generation: candidate.generation})
	return cascadeBatchProgress{membershipsRemoved: removed}, err
}

// upgradeCascadeCandidate turns a legacy payload into an ordinary
// generation-bearing one after finalization and records the witness, leaving
// deletion to a later round. With index-unaware writers operationally drained,
// both steps are monotone: a payload is upgraded at most once, and the witness
// install is compare-and-set against what the batch observed.
func (s *RedisStore) upgradeCascadeCandidate(
	ctx context.Context,
	conversationID string,
	candidate cascadeCandidate,
	result responsePayloadResult,
	record responseRecord,
) (cascadeBatchProgress, error) {
	generation, ttlMillis, upgraded, err := promoteLegacyResponsePayload(ctx, s.client, result.key, record)
	if err != nil {
		return cascadeBatchProgress{}, fmt.Errorf("failed to upgrade legacy response %s during cascade delete: %w", candidate.responseID, err)
	}
	if !upgraded {
		return cascadeBatchProgress{}, fmt.Errorf("response %s changed while being upgraded during cascade delete; left in place for retry",
			candidate.responseID)
	}

	if _, err := s.repairResponseWitness(ctx, conversationID, record.response.ID, generation,
		candidate.generation, record.response.CreatedAt, ttlMillis); err != nil {
		return cascadeBatchProgress{}, err
	}

	// Promotion is tracked separately from membership removal. It is safe to
	// exempt this one transition from a race round only because the caller has
	// already observed the permanent finalization record: no legacy writer can
	// turn the payload generation-less again.
	return cascadeBatchProgress{payloadsPromoted: 1}, nil
}
