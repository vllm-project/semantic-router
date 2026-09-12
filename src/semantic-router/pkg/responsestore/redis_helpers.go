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
// unknownPayloadTTL. ttlErr is deliberately independent of err: a payload GET
// may succeed while its PTTL fails, and callers that build a durable index
// must reject that incomplete lifetime observation.
type responsePayloadResult struct {
	key       string
	raw       []byte
	ttlMillis int64
	ttlErr    error
	err       error
}

const (
	// unknownPayloadTTL is responsePayloadResult.ttlMillis when no PTTL was
	// asked for. Redis's own -2 is borrowed for it because that reply means
	// the same thing to a lifetime consumer — nothing to measure — but a
	// successful result never carries it: a PTTL of -2 behind a GET that
	// succeeded is reported as redis.Nil instead, see fetchResponsePayloads.
	unknownPayloadTTL int64 = -2

	// persistentPayloadTTL is the lifetime of a payload Redis reported as
	// never expiring. It is the only successful PTTL reply permitted to reach
	// the index-lifetime convention's "never expires" branch, where
	// conversationIndexAddScript PERSISTs a conversation's ZSET and its
	// generation sidecar.
	persistentPayloadTTL int64 = -1

	// minPayloadLifetimeMillis is the shortest finite lifetime this package
	// will report for a payload that still exists.
	//
	// Redis answers PTTL 0 for a key inside its final millisecond, and 0 is
	// exactly how longerIndexLifetime and conversationIndexAddScript spell
	// "never expires". Returning it verbatim would persist a conversation
	// index and its witness sidecar on behalf of a payload about to vanish,
	// and the script only ever raises an expiry, so no later finite write
	// could restore one. Clamping keeps the reply finite and lets the store
	// TTL — always at least this long — decide the index's real lifetime.
	minPayloadLifetimeMillis int64 = 1
)

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
		if !withTTL {
			continue
		}
		results[i].ttlMillis, results[i].ttlErr = decodePayloadTTL(ttlCmds[i])
		// GET and PTTL are separate commands even inside one pipeline, so a
		// payload can expire between them: GET returns the bytes and PTTL
		// answers -2. That -2 is Redis's definitive statement that the key
		// no longer exists, and a payload that can no longer be measured is
		// one this store no longer has. Reporting it as redis.Nil lets every
		// consumer take its existing not-found path. Passing the sentinel
		// through instead let a witness repair index the vanished payload
		// with a non-positive lifetime, which longerIndexLifetime and
		// conversationIndexAddScript read as "never expires" — an immortal
		// index for a payload that was already gone.
		if results[i].err == nil && results[i].ttlErr == nil && results[i].ttlMillis == unknownPayloadTTL {
			results[i].raw, results[i].err = nil, redis.Nil
		}
	}
	return results
}

// decodePayloadTTL turns go-redis's PTTL reply into raw milliseconds. go-redis
// reports the two sentinels as negative durations, which Duration.Milliseconds
// would otherwise round to 0 — indistinguishable from "expires right now" —
// so they are mapped back explicitly. A failed PTTL is not a lifetime: callers
// must keep its error distinct from Redis's successful "key is gone" reply so
// they cannot certify an index that may expire before a retained payload.
//
// Only PTTL -1 may report persistence. A successful finite reply is clamped to
// minPayloadLifetimeMillis, because Redis answers 0 for a key in its last
// millisecond and every consumer of this value reads a non-positive lifetime as
// "never expires". The same rule is already enforced on the package's other two
// lifetime paths — promoteLegacyPayloadScript refuses a pttl <= 0 outright, and
// an elapsed update snapshot deletes rather than restores — so a live payload's
// remaining milliseconds can never be mistaken for an unbounded one.
func decodePayloadTTL(cmd *redis.DurationCmd) (int64, error) {
	ttl, err := cmd.Result()
	if err != nil {
		return unknownPayloadTTL, err
	}
	switch {
	case ttl == -1:
		return persistentPayloadTTL, nil
	case ttl < 0:
		return unknownPayloadTTL, nil
	case ttl.Milliseconds() < minPayloadLifetimeMillis:
		return minPayloadLifetimeMillis, nil
	default:
		return ttl.Milliseconds(), nil
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
	if result.ttlErr != nil {
		return responseRecord{}, 0, fmt.Errorf("failed to get response lifetime from Redis: %w", result.ttlErr)
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
