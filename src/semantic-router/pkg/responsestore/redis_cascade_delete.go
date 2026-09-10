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
// batch and reports how much the cascade actually advanced, so the caller can
// tell real progress from a conversation being rewritten faster than it drains.
//
// allowBlankCleanup is the finalization gate, and it guards exactly one thing:
// removing a membership whose witness is blank *and* whose payload this batch
// found absent. Before finalization an index-unaware writer can recreate that
// payload without touching the sidecar, so the blank-expected removal still
// matches and the member goes — and unlike the read path, cascade destroys its
// own recovery on the way out. It deletes the migrated marker and the
// conversation record once the index drains, so no later scan will ever
// rediscover the recreated payload. The atomic ZCARD==0 check in
// deleteEmptyConversationIndexScript, and the whole race-round loop, exist to
// stop precisely that orphaning; an ungated blank ZREM would walk straight
// through both while reporting success. After finalization every writer is
// generation-aware, installs a witness before it can matter, and the blank
// removal is safe again.
//
// Legacy payloads that still exist need no gate, because they are not removed
// on the strength of their blankness — they are upgraded
// (promoteLegacyResponsePayload), their witness is installed, and the next
// round deletes them under the ordinary generation CAS. Promotion never
// authorizes a delete in the same step: byte equality is what matched it, and
// byte equality must not reach a DEL even transitively.
func (s *RedisStore) deleteConversationResponseBatch(
	ctx context.Context,
	conversationID string,
	candidates []cascadeCandidate,
	allowBlankCleanup bool,
) (int, error) {
	if len(candidates) == 0 {
		return 0, nil
	}

	responseIDs := make([]string, len(candidates))
	for i, candidate := range candidates {
		responseIDs[i] = candidate.responseID
	}
	results := fetchResponsePayloadsPipelined(ctx, s.client, responseKeys(s, responseIDs))

	resolved := 0
	var errs []error
	for i, result := range results {
		candidate := candidates[i]

		switch {
		case errors.Is(result.err, redis.Nil):
			removed, err := s.dropCascadeMembership(ctx, conversationID, candidate, allowBlankCleanup)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			resolved += removed
		case result.err != nil:
			errs = append(errs, fmt.Errorf("failed to read response %s for cascade delete: %w", candidate.responseID, result.err))
		default:
			progressed, err := s.resolveCascadeDeleteOutcome(ctx, conversationID, candidate, result, allowBlankCleanup)
			if err != nil {
				errs = append(errs, err)
			}
			resolved += progressed
		}
	}
	return resolved, errors.Join(errs...)
}

// dropCascadeMembership removes a membership this batch has proven stale,
// conditional on the sidecar still holding exactly the witness it observed.
// A blank witness is refused before finalization: see
// deleteConversationResponseBatch for why cascade cannot absorb that race the
// way a listing can.
func (s *RedisStore) dropCascadeMembership(
	ctx context.Context,
	conversationID string,
	candidate cascadeCandidate,
	allowBlankCleanup bool,
) (int, error) {
	if candidate.generation == "" && !allowBlankCleanup {
		return 0, fmt.Errorf(
			"response %s has no generation witness and the store is not finalized; refusing cascade delete",
			candidate.responseID)
	}

	return s.unindexResponseGenerations(ctx, conversationID,
		responseGenerationWitness{responseID: candidate.responseID, generation: candidate.generation})
}

// resolveCascadeDeleteOutcome decides and executes one candidate's outcome
// once its payload has been read, reporting how much the cascade actually
// advanced: memberships genuinely removed, witnesses genuinely installed, or a
// legacy payload genuinely upgraded.
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
	allowBlankCleanup bool,
) (int, error) {
	record, err := decodeResponseRecord(result.raw)
	if err != nil {
		return 0, fmt.Errorf("failed to parse response %s during cascade delete: %w", candidate.responseID, err)
	}
	if record.response.ID != candidate.responseID {
		return 0, fmt.Errorf("response %s payload identity mismatch during cascade delete", candidate.responseID)
	}

	if record.response.ConversationID != conversationID {
		// Moved on since being indexed here. The payload belongs to another
		// conversation now, so only this conversation's stale membership goes
		// — under the same blank-witness gate, since an index-unaware writer
		// could just as well have moved it back after this read.
		return s.dropCascadeMembership(ctx, conversationID, candidate, allowBlankCleanup)
	}

	if record.generation == "" {
		// A legacy payload this conversation still owns. Upgrade it, install
		// the witness the upgrade minted, and stop there: the delete happens
		// on a later round, authorized by a direct observation of a
		// generational payload rather than by the byte equality that matched
		// the upgrade.
		return s.upgradeCascadeCandidate(ctx, conversationID, candidate, result, record)
	}

	if record.generation != candidate.generation {
		// The snapshot's witness has fallen behind the payload — an index
		// write still in flight, or an upgrade whose witness install lost its
		// race. Repair it from the payload rather than remove a live member.
		return s.repairResponseWitness(ctx, conversationID, record.response.ID, record.generation,
			candidate.generation, record.response.CreatedAt, s.ttlMillis())
	}

	deleted, err := s.compareDeleteResponsePayload(ctx, result.key, record.generation)
	if err != nil {
		return 0, fmt.Errorf("failed to delete response %s during cascade delete: %w", candidate.responseID, err)
	}
	if !deleted {
		return 0, fmt.Errorf("response %s changed concurrently during cascade delete; left in place for retry", candidate.responseID)
	}

	return s.unindexResponseGenerations(ctx, conversationID,
		responseGenerationWitness{responseID: candidate.responseID, generation: candidate.generation})
}

// upgradeCascadeCandidate turns a legacy payload into an ordinary
// generation-bearing one and records the witness, leaving the deletion to a
// later round. Both steps are monotone: a payload is upgraded at most once,
// and the witness install is compare-and-set against what the batch observed.
func (s *RedisStore) upgradeCascadeCandidate(
	ctx context.Context,
	conversationID string,
	candidate cascadeCandidate,
	result responsePayloadResult,
	record responseRecord,
) (int, error) {
	generation, ttlMillis, upgraded, err := promoteLegacyResponsePayload(ctx, s.client, result.key, record)
	if err != nil {
		return 0, fmt.Errorf("failed to upgrade legacy response %s during cascade delete: %w", candidate.responseID, err)
	}
	if !upgraded {
		return 0, fmt.Errorf("response %s changed while being upgraded during cascade delete; left in place for retry",
			candidate.responseID)
	}

	if _, err := s.repairResponseWitness(ctx, conversationID, record.response.ID, generation,
		candidate.generation, record.response.CreatedAt, ttlMillis); err != nil {
		return 0, err
	}

	// The upgrade itself is the progress, whether or not the witness install
	// won its race: the payload can never be legacy again, so the next round
	// resolves this candidate through the ordinary generational path.
	return 1, nil
}
