package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func (s *RedisStore) CreateConversation(ctx context.Context, conversation *responseapi.StoredConversation) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversation == nil || conversation.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ConversationKeyPrefix + conversation.ID)

	exists, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to check conversation existence: %w", err)
	}
	if exists > 0 {
		return ErrAlreadyExists
	}

	data, err := json.Marshal(conversation)
	if err != nil {
		return fmt.Errorf("failed to serialize conversation: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to store conversation in Redis: %w", err)
	}

	return nil
}

func (s *RedisStore) GetConversation(ctx context.Context, conversationID string) (*responseapi.StoredConversation, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if conversationID == "" {
		return nil, ErrInvalidInput
	}

	key := s.buildKey(ConversationKeyPrefix + conversationID)

	data, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to get conversation from Redis: %w", err)
	}

	var conversation responseapi.StoredConversation
	if err := json.Unmarshal(data, &conversation); err != nil {
		return nil, fmt.Errorf("failed to deserialize conversation: %w", err)
	}

	return &conversation, nil
}

func (s *RedisStore) UpdateConversation(ctx context.Context, conversation *responseapi.StoredConversation) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversation == nil || conversation.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ConversationKeyPrefix + conversation.ID)

	exists, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to check conversation existence: %w", err)
	}
	if exists == 0 {
		return ErrNotFound
	}

	data, err := json.Marshal(conversation)
	if err != nil {
		return fmt.Errorf("failed to serialize conversation: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to update conversation in Redis: %w", err)
	}

	return nil
}

// DeleteConversation deletes a conversation and, when deleteResponses is
// set, cascades to its indexed responses first. The conversation record is
// deleted last on purpose: if the cascade fails partway, the conversation
// key survives as the retry anchor instead of leaving orphaned response
// payloads with no record pointing at them (blueprint §5 Phase 5).
func (s *RedisStore) DeleteConversation(ctx context.Context, conversationID string, deleteResponses bool) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversationID == "" {
		return ErrInvalidInput
	}

	convKey := s.buildKey(ConversationKeyPrefix + conversationID)

	exists, err := s.client.Exists(ctx, convKey).Result()
	if err != nil {
		return fmt.Errorf("failed to check conversation existence: %w", err)
	}
	if exists == 0 {
		return ErrNotFound
	}

	if deleteResponses {
		if err := s.deleteConversationResponses(ctx, conversationID); err != nil {
			return err
		}
	}

	if err := s.client.Del(ctx, convKey).Err(); err != nil {
		return fmt.Errorf("failed to delete conversation from Redis: %w", err)
	}

	return nil
}

// deleteConversationResponses removes a conversation's responses and its
// index/marker, in bounded batches of redisDeleteBatchSize read from the
// front of the index — never one ZRANGE 0 -1 or one pipeline sized to the
// whole conversation. Reads the index directly rather than through
// ListResponsesByConversation, which would additionally cap the cascade at
// a single page.
//
// Ensures the index is resolved before cascading: if neither the index nor
// the empty marker exists yet, this conversation's responses may still be
// unindexed legacy payloads (pre-#2814 data, or a write from an
// indexing-unaware pod mid rolling upgrade). Without this,
// ZRange below would just see a missing index, the loop would exit on its
// first iteration having deleted nothing, and those payloads would be
// orphaned forever — a silent regression from the pre-#2814 scan-based
// cascade delete. ensureConversationIndex resolves the ambiguity exactly
// as a read would: backfill from a legacy scan, or confirm the conversation
// is genuinely empty.
//
// Each iteration reads rank 0..redisDeleteBatchSize-1 again (not an offsetting
// range), atomically pairing each member with its sidecar generation. Payload
// deletion and later ZSET/HASH cleanup are both conditional on that observed
// generation. Before finalization, a live legacy candidate fails closed: an
// index-unaware writer could otherwise replace its promoted payload while
// leaving the minted sidecar stale. Once finalized, such writers are gone, so
// a residual legacy payload is upgraded in place and then deleted through the
// ordinary generation CAS on a later round. A legacy membership whose payload
// is already gone is governed by the same finalization gate.
// If a batch reports an unresolved response, this stops instead of silently
// reporting success. Already-resolved members are gone; a stale witness left
// by a cleanup failure remains a safe retry anchor.
//
// An empty read is not on its own permission to delete the index. A
// StoreResponse landing between that read and the delete has committed a real
// member, and deleting the key unconditionally would erase it — leaving a live
// payload nothing points at, which past finalization no rescan will ever
// recover. So the delete is conditional on the index still being empty at the
// instant it runs (deleteEmptyConversationIndex), and a member that beat it
// there sends this back around to resolve that member like any other. Only a
// conversation being written to faster than it can be drained gives up, and it
// says so rather than reporting a cascade that silently left responses behind.
func (s *RedisStore) deleteConversationResponses(ctx context.Context, conversationID string) error {
	if err := s.ensureConversationIndexResolved(ctx, conversationID); err != nil {
		return err
	}

	// Read once, before the loop: whether legacy payloads and blank-witness
	// memberships may be cleaned up at all. The completion record only ever
	// goes from absent to permanently present (and is process-cached), so one
	// observation is as good as re-reading it every round — and a cascade that
	// changed its mind halfway would be harder to reason about than one that
	// does not.
	allowLegacyCleanup, err := s.conversationIndexFinalized(ctx)
	if err != nil {
		return err
	}

	raced := 0
	for {
		state, err := s.runConversationCascadeIteration(ctx, conversationID, allowLegacyCleanup)
		if err != nil {
			return err
		}
		if state == cascadeIterationComplete {
			break
		}
		if state == cascadeIterationAdvanced {
			continue
		}

		// Either a non-removing batch (for example, a witness repair) or a
		// write committed after the empty candidate read. Both consume the
		// same bounded race budget.
		raced++
		if raced > conversationIndexCascadeMaxRaceRounds {
			return fmt.Errorf("conversation %s kept receiving responses during cascade delete after %d attempts; retry",
				conversationID, raced)
		}
	}

	// Single-key delete, never combined with the index or the response keys
	// above.
	if err := s.client.Del(ctx, s.conversationIndexMigratedKey(conversationID)).Err(); err != nil {
		return fmt.Errorf("failed to delete conversation migrated marker for %s: %w", conversationID, err)
	}

	return nil
}

type cascadeIterationState uint8

const (
	cascadeIterationAdvanced cascadeIterationState = iota
	cascadeIterationStalled
	cascadeIterationComplete
)

// runConversationCascadeIteration executes one bounded state-machine step:
// drain a non-empty candidate batch, or prove and remove an empty index. It
// keeps the public orchestrator small while preserving the distinction between
// irreversible progress and a race that must consume the retry budget.
func (s *RedisStore) runConversationCascadeIteration(
	ctx context.Context,
	conversationID string,
	allowLegacyCleanup bool,
) (cascadeIterationState, error) {
	candidates, err := s.readCascadeCandidates(ctx, conversationID)
	if err != nil {
		return cascadeIterationStalled, fmt.Errorf("failed to list responses for deletion: %w", err)
	}
	if len(candidates) > 0 {
		stalled, err := s.drainConversationResponseBatch(ctx, conversationID, candidates, allowLegacyCleanup)
		if err != nil {
			return cascadeIterationStalled, err
		}
		if stalled {
			return cascadeIterationStalled, nil
		}
		return cascadeIterationAdvanced, nil
	}

	emptied, err := s.deleteEmptyConversationIndex(ctx, conversationID)
	if err != nil {
		return cascadeIterationStalled, err
	}
	if emptied {
		return cascadeIterationComplete, nil
	}
	return cascadeIterationStalled, nil
}

// drainConversationResponseBatch reports whether a non-empty candidate batch
// failed to make irreversible progress. Membership removal is real drainage;
// a post-finalization promotion is also monotone because old writers have been
// drained. Witness repair is deliberately stalled: a hot writer can invalidate
// it repeatedly, so it must consume the outer loop's bounded race budget.
func (s *RedisStore) drainConversationResponseBatch(
	ctx context.Context,
	conversationID string,
	candidates []cascadeCandidate,
	allowLegacyCleanup bool,
) (bool, error) {
	progress, err := s.deleteConversationResponseBatch(ctx, conversationID, candidates, allowLegacyCleanup)
	if err != nil {
		return false, err
	}
	return progress.membershipsRemoved == 0 && progress.payloadsPromoted == 0, nil
}

// ensureConversationIndexResolved backfills a conversation's index before a
// cascade delete if it isn't marked migrated yet. Without this, a
// conversation whose index exists only because of an ordinary post-upgrade
// write — with older, still-unindexed legacy responses sitting alongside it
// — would have deleteConversationResponses' batch loop delete only what the
// index happens to already list, permanently orphaning the rest once the
// conversation record itself is gone. Resolves the ambiguity exactly as a
// read would: backfill from a legacy scan (additive — never removes what
// the index already has), or confirm the conversation is genuinely empty.
//
// Once the whole store is marked migration-complete
// (ConversationIndexCompletionKeySuffix), this returns immediately without
// even checking the per-conversation marker: the index is trusted
// unconditionally, so cascade delete never scans, matching the read path in
// ListResponsesByConversation.
func (s *RedisStore) ensureConversationIndexResolved(ctx context.Context, conversationID string) error {
	if resolved, err := s.conversationIndexResolved(ctx, conversationID); err != nil {
		return err
	} else if resolved {
		return nil
	}
	return s.ensureConversationIndex(ctx, conversationID)
}

func (s *RedisStore) ListConversations(ctx context.Context, opts ListOptions) ([]*responseapi.StoredConversation, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}

	pattern := s.buildKey(ConversationKeyPrefix + "*")
	var conversations []*responseapi.StoredConversation

	iter := s.client.Scan(ctx, 0, pattern, 0).Iterator()
	for iter.Next(ctx) {
		key := iter.Val()

		data, err := s.client.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var conversation responseapi.StoredConversation
		if err := json.Unmarshal(data, &conversation); err != nil {
			continue
		}

		conversations = append(conversations, &conversation)
	}

	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan conversations: %w", err)
	}

	// Apply list options (limit, pagination)
	conversations = ApplyConvListOptions(conversations, opts)

	return conversations, nil
}

// AddResponseToConversation indexes an already-stored response under a
// conversation, verified against the response's own stored ConversationID —
// never against the caller's say-so alone, so this can never create an
// index entry for a conversation the response doesn't actually belong to.
//
// Deliberately narrow (blueprint §3.7/§7): a response with no stored
// ConversationID, or one that belongs to a different conversation than
// conversationID, returns ErrInvalidInput rather than silently adopting the
// caller's conversationID or rewriting the stored payload's membership.
// Widening this to actually reassign a response's conversation belongs to
// the broader Conversations API semantics in #2999, not this lookup fix —
// today's only real caller path is StoreResponse indexing directly from
// StoredResponse.ConversationID, so this method mainly exists to satisfy
// the ConversationStore interface usefully rather than as a no-op.
func (s *RedisStore) AddResponseToConversation(ctx context.Context, conversationID, responseID string) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversationID == "" || responseID == "" {
		return ErrInvalidInput
	}

	stored, lifetimeMillis, err := s.getResponseWithLifetime(ctx, responseID)
	if err != nil {
		return err
	}

	if stored.response.ConversationID == "" || stored.response.ConversationID != conversationID {
		return ErrInvalidInput
	}

	// witnessRepair: this only read the generation, so it must not overwrite a
	// witness a live writer already owns.
	if _, err := s.repairResponseWitness(ctx, conversationID, responseID, stored.generation, "",
		stored.response.CreatedAt, lifetimeMillis); err != nil {
		return fmt.Errorf("failed to index response in Redis: %w", err)
	}

	return nil
}
