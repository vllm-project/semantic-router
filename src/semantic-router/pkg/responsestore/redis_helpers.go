package responsestore

import (
	"context"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// responsePayloadResult is one independent GET outcome, optionally with the
// key's remaining lifetime read in the same round trip. The helpers below
// pipeline single-key commands, allowing go-redis to route each one to its
// owning Cluster node without issuing a cross-slot MGET.
//
// ttlMillis follows Redis's PTTL convention (-1 never expires, -2 no such
// key) and is only meaningful when the result came from
// fetchResponsePayloadsAndTTLsPipelined; the GET-only helper leaves it at
// unknownPayloadTTL.
type responsePayloadResult struct {
	key       string
	raw       []byte
	ttlMillis int64
	err       error
}

// unknownPayloadTTL is responsePayloadResult.ttlMillis when no PTTL was
// asked for, or when Redis answered that the key is already gone. Both are
// "this payload contributes nothing to how long its conversation index must
// live", and Redis's own -2 is the natural spelling of that.
const unknownPayloadTTL int64 = -2

// fetchResponsePayloadsPipelined reads each key's payload. Used by the read
// and cascade paths, which act on the payload alone and have no reason to
// pay for a second command per key.
func fetchResponsePayloadsPipelined(ctx context.Context, client redis.UniversalClient, keys []string) []responsePayloadResult {
	return fetchResponsePayloads(ctx, client, keys, false)
}

// fetchResponsePayloadsAndTTLsPipelined additionally reads how long each
// payload has left, in the same pipeline. Used by the scan paths (lazy
// backfill and the finalization sweep), which must make a conversation's
// index outlive payloads they did not write and whose real lifetimes can be
// far longer than the store's currently configured TTL — see
// extendConversationIndexLifetimeScript.
func fetchResponsePayloadsAndTTLsPipelined(ctx context.Context, client redis.UniversalClient, keys []string) []responsePayloadResult {
	return fetchResponsePayloads(ctx, client, keys, true)
}

func fetchResponsePayloads(ctx context.Context, client redis.UniversalClient, keys []string, withTTL bool) []responsePayloadResult {
	results := make([]responsePayloadResult, len(keys))
	if len(keys) == 0 {
		return results
	}

	pipe := client.Pipeline()
	cmds := make([]*redis.StringCmd, len(keys))
	var ttlCmds []*redis.DurationCmd
	if withTTL {
		ttlCmds = make([]*redis.DurationCmd, len(keys))
	}
	for i, key := range keys {
		results[i].key = key
		results[i].ttlMillis = unknownPayloadTTL
		cmds[i] = pipe.Get(ctx, key)
		if withTTL {
			ttlCmds[i] = pipe.PTTL(ctx, key)
		}
	}
	_, _ = pipe.Exec(ctx)

	for i, cmd := range cmds {
		results[i].raw, results[i].err = cmd.Bytes()
		if withTTL {
			results[i].ttlMillis = decodePayloadTTL(ttlCmds[i])
		}
	}
	return results
}

// decodePayloadTTL turns go-redis's PTTL reply into raw milliseconds. go-redis
// reports the two sentinels as negative durations, which Duration.Milliseconds
// would otherwise round to 0 — indistinguishable from "expires right now" —
// so they are mapped back explicitly. A failed PTTL is treated as unknown
// rather than fatal: it only costs a shorter index lifetime, and the payload
// read itself is what the scan is really after.
func decodePayloadTTL(cmd *redis.DurationCmd) int64 {
	ttl, err := cmd.Result()
	if err != nil {
		return unknownPayloadTTL
	}
	switch {
	case ttl == -1:
		return -1
	case ttl < 0:
		return unknownPayloadTTL
	default:
		return ttl.Milliseconds()
	}
}

// getResponseWithLifetime reads a response and how long its payload has left,
// in one pipeline of single-key commands.
//
// Used by the two paths that index a payload they did not just write — a
// duplicate-ID repair and an explicit AddResponseToConversation. Both would
// otherwise have to assume the payload has the store's current TTL, which is
// exactly the assumption that lets a conversation index retire ahead of a
// longer-lived payload it names. Neither is a hot path, so the extra command
// costs nothing that matters.
func (s *RedisStore) getResponseWithLifetime(ctx context.Context, responseID string) (responseRecord, int64, error) {
	if !s.enabled {
		return responseRecord{}, 0, ErrStoreDisabled
	}
	if responseID == "" {
		return responseRecord{}, 0, ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + responseID)
	result := fetchResponsePayloadsAndTTLsPipelined(ctx, s.client, []string{key})[0]
	if result.err != nil {
		if errors.Is(result.err, redis.Nil) {
			return responseRecord{}, 0, ErrNotFound
		}
		return responseRecord{}, 0, fmt.Errorf("failed to get response from Redis: %w", result.err)
	}

	record, err := decodeResponseRecord(result.raw)
	if err != nil {
		return responseRecord{}, 0, err
	}

	return record, result.ttlMillis, nil
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

	keys := make([]string, len(responseIDs))
	for i, id := range responseIDs {
		keys[i] = s.buildKey(ResponseKeyPrefix + id)
	}
	results := fetchResponsePayloadsPipelined(ctx, s.client, keys)

	// Process results
	var (
		found      []*responseapi.StoredResponse
		missingIDs []string
	)
	for i, result := range results {
		data, err := result.raw, result.err
		if err != nil {
			if errors.Is(err, redis.Nil) {
				logging.Warnf("RedisStore: response %s not found (may have expired)", responseIDs[i])
				missingIDs = append(missingIDs, responseIDs[i])
				continue
			}
			logging.Warnf("RedisStore: failed to get response %s: %v", responseIDs[i], err)
			continue
		}

		record, err := decodeResponseRecord(data)
		if err != nil {
			logging.Warnf("RedisStore: failed to parse response %s: %v", responseIDs[i], err)
			continue
		}

		found = append(found, record.response)
	}

	return found, missingIDs, nil
}
