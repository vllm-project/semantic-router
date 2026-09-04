package responsestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// storedConversationID reports a response's current conversation so update and
// delete can repair the index, and returns ErrNotFound when it is absent. An
// unreadable payload is not fatal, it only costs the old index entry.
func (s *RedisStore) storedConversationID(ctx context.Context, responseID string) (string, error) {
	data, err := s.client.Get(ctx, s.buildKey(ResponseKeyPrefix+responseID)).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return "", ErrNotFound
		}
		return "", fmt.Errorf("failed to check response existence: %w", err)
	}

	var stored responseapi.StoredResponse
	if err := json.Unmarshal(data, &stored); err != nil {
		logging.Warnf("RedisStore: failed to parse stored response %s while updating its conversation index: %v",
			responseID, err)
		return "", nil
	}

	return stored.ConversationID, nil
}
func (s *RedisStore) collectChainIDs(ctx context.Context, startID string) ([]string, error) {
	var responseIDs []string
	currentID := startID
	visited := make(map[string]bool)

	// Maximum chain length to prevent infinite loops
	const maxChainLength = 1000

	for currentID != "" && len(responseIDs) < maxChainLength {
		// Prevent circular references
		if visited[currentID] {
			logging.Warnf("RedisStore: circular reference detected at %s", currentID)
			break
		}
		visited[currentID] = true

		responseIDs = append(responseIDs, currentID)

		response, err := s.GetResponse(ctx, currentID)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				// If this is the first response (start of chain), return error
				if len(responseIDs) == 1 {
					return nil, ErrNotFound
				}
				// Otherwise, just break - the chain ended early
				logging.Warnf("RedisStore: response %s not found in chain", currentID)
				break
			}
			return nil, fmt.Errorf("failed to fetch response %s: %w", currentID, err)
		}

		currentID = response.PreviousResponseID
	}

	return responseIDs, nil
}

// fetchResponsesPipelined loads response IDs in one round trip, also returning
// the IDs whose payload is gone so index-driven callers can prune them.
func (s *RedisStore) fetchResponsesPipelined(ctx context.Context, responseIDs []string) ([]*responseapi.StoredResponse, []string, error) {
	if len(responseIDs) == 0 {
		return []*responseapi.StoredResponse{}, nil, nil
	}

	pipe := s.client.Pipeline()

	cmds := make([]*redis.StringCmd, len(responseIDs))
	for i, id := range responseIDs {
		key := s.buildKey(ResponseKeyPrefix + id)
		cmds[i] = pipe.Get(ctx, key)
	}

	_, err := pipe.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		// Some commands might fail, but we continue to process successful ones
		logging.Debugf("RedisStore: pipeline execution completed with some errors: %v", err)
	}

	// Process results
	var (
		found      []*responseapi.StoredResponse
		missingIDs []string
	)
	for i, cmd := range cmds {
		data, err := cmd.Bytes()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				logging.Warnf("RedisStore: response %s not found (may have expired)", responseIDs[i])
				missingIDs = append(missingIDs, responseIDs[i])
				continue
			}
			logging.Warnf("RedisStore: failed to get response %s: %v", responseIDs[i], err)
			continue
		}

		var response responseapi.StoredResponse
		if err := json.Unmarshal(data, &response); err != nil {
			logging.Warnf("RedisStore: failed to parse response %s: %v", responseIDs[i], err)
			continue
		}

		found = append(found, &response)
	}

	return found, missingIDs, nil
}
