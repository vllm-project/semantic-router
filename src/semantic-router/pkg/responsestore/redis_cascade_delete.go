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
// batch. Legacy candidates fail closed. Every prune is conditional on the
// sidecar still carrying the observed generation, and every payload delete is
// conditional on cjson-decoded payload generation equality.
func (s *RedisStore) deleteConversationResponseBatch(ctx context.Context, conversationID string, candidates []cascadeCandidate) error {
	if len(candidates) == 0 {
		return nil
	}

	responseIDs := make([]string, len(candidates))
	for i, candidate := range candidates {
		responseIDs[i] = candidate.responseID
	}
	results := fetchResponsePayloadsPipelined(ctx, s.client, responseKeys(s, responseIDs))

	var errs []error
	for i, result := range results {
		candidate := candidates[i]
		if candidate.generation == "" {
			errs = append(errs, fmt.Errorf("response %s has no generation witness; refusing cascade delete", candidate.responseID))
			continue
		}

		witness := responseGenerationWitness{responseID: candidate.responseID, generation: candidate.generation}
		switch {
		case errors.Is(result.err, redis.Nil):
			if err := s.unindexResponseGenerations(ctx, conversationID, witness); err != nil {
				errs = append(errs, err)
			}
		case result.err != nil:
			errs = append(errs, fmt.Errorf("failed to read response %s for cascade delete: %w", candidate.responseID, result.err))
		default:
			if err := s.resolveCascadeDeleteOutcome(ctx, conversationID, candidate, result, witness); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return errors.Join(errs...)
}

func (s *RedisStore) resolveCascadeDeleteOutcome(
	ctx context.Context,
	conversationID string,
	candidate cascadeCandidate,
	result responsePayloadResult,
	witness responseGenerationWitness,
) error {
	record, err := decodeResponseRecord(result.raw)
	if err != nil {
		return fmt.Errorf("failed to parse response %s during cascade delete: %w", candidate.responseID, err)
	}
	if record.response.ID != candidate.responseID {
		return fmt.Errorf("response %s payload identity mismatch during cascade delete", candidate.responseID)
	}
	if record.generation == "" {
		return fmt.Errorf("response %s has no payload generation; refusing cascade delete", candidate.responseID)
	}

	if record.generation != candidate.generation || record.response.ConversationID != conversationID {
		return s.unindexResponseGenerations(ctx, conversationID, witness)
	}

	deleted, err := s.compareDeleteResponsePayload(ctx, result.key, candidate.generation)
	if err != nil {
		return fmt.Errorf("failed to delete response %s during cascade delete: %w", candidate.responseID, err)
	}
	if !deleted {
		return fmt.Errorf("response %s changed concurrently during cascade delete; left in place for retry", candidate.responseID)
	}

	return s.unindexResponseGenerations(ctx, conversationID, witness)
}
