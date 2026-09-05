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
// Each iteration reads rank 0..redisDeleteBatchSize-1 again (not an
// offsetting range): the ZREM at the end of the previous iteration already
// removed those members, so the next read naturally advances. If a batch's
// payload deletes or index removal fails, this returns the error and stops
// rather than silently logging and reporting success — per blueprint §5
// Phase 5, the caller needs to know the cascade was only partially applied.
// The failure is safely retryable: undeleted response keys are unaffected,
// and a Del of an already-deleted key is a no-op, not an error.
func (s *RedisStore) deleteConversationResponses(ctx context.Context, conversationID string) error {
	indexKey := s.conversationIndexKey(conversationID)

	if err := s.ensureConversationIndexResolved(ctx, conversationID); err != nil {
		return err
	}

	for {
		responseIDs, err := s.client.ZRange(ctx, indexKey, 0, redisDeleteBatchSize-1).Result()
		if err != nil {
			return fmt.Errorf("failed to list responses for deletion: %w", err)
		}
		if len(responseIDs) == 0 {
			break
		}

		if err := s.deleteResponsePayloadBatch(ctx, conversationID, responseIDs); err != nil {
			return err
		}

		if err := s.unindexResponse(ctx, conversationID, responseIDs...); err != nil {
			return fmt.Errorf("failed to remove deleted responses from conversation %s index: %w", conversationID, err)
		}
	}

	// Single-key deletes, never combined with each other or with the
	// response keys above.
	if err := s.client.Del(ctx, indexKey).Err(); err != nil {
		return fmt.Errorf("failed to delete conversation index for %s: %w", conversationID, err)
	}
	if err := s.client.Del(ctx, s.emptyConversationIndexMarkerKey(conversationID)).Err(); err != nil {
		return fmt.Errorf("failed to delete conversation index empty marker for %s: %w", conversationID, err)
	}

	return nil
}

// ensureConversationIndexResolved backfills a conversation's index before a
// cascade delete if neither it nor the empty marker exists yet. Without
// this, deleteConversationResponses' batch loop would see a missing index,
// exit on its first iteration having deleted nothing, and silently orphan
// any still-unindexed legacy payload — a regression from the pre-#2814
// scan-based cascade. Resolves the ambiguity exactly as a read would:
// backfill from a legacy scan, or confirm the conversation is genuinely
// empty.
func (s *RedisStore) ensureConversationIndexResolved(ctx context.Context, conversationID string) error {
	resolved, err := s.indexOrMarkerExists(ctx, conversationID)
	if err != nil {
		return err
	}
	if resolved {
		return nil
	}
	return s.ensureConversationIndex(ctx, conversationID)
}

// deleteResponsePayloadBatch deletes one batch of response payload keys,
// pipelined, unless a test override replaces the deletion entirely.
func (s *RedisStore) deleteResponsePayloadBatch(ctx context.Context, conversationID string, responseIDs []string) error {
	if s.deleteResponseBatchOverride != nil {
		if err := s.deleteResponseBatchOverride(ctx, responseIDs); err != nil {
			return fmt.Errorf("failed to delete %d response(s) of conversation %s: %w", len(responseIDs), conversationID, err)
		}
		return nil
	}

	pipe := s.client.Pipeline()
	for _, responseID := range responseIDs {
		pipe.Del(ctx, s.buildKey(ResponseKeyPrefix+responseID))
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to delete %d response(s) of conversation %s: %w", len(responseIDs), conversationID, err)
	}
	return nil
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

	stored, err := s.GetResponse(ctx, responseID)
	if err != nil {
		return err
	}

	if stored.ConversationID == "" || stored.ConversationID != conversationID {
		return ErrInvalidInput
	}

	if err := s.indexResponse(ctx, conversationID, responseID, stored.CreatedAt); err != nil {
		return fmt.Errorf("failed to index response in Redis: %w", err)
	}

	return nil
}
