package responsestore

import (
	"context"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// readIndexWindowScript snapshots each ZSET member together with its
// generation witness. The ZSET and HASH share a Redis Cluster hash tag, so a
// later conditional prune can prove it is still removing the same generation
// this window observed.
var readIndexWindowScript = redis.NewScript(`
local members
if ARGV[3] == "1" then
	members = redis.call("ZRANGE", KEYS[1], ARGV[1], ARGV[2])
else
	members = redis.call("ZREVRANGE", KEYS[1], ARGV[1], ARGV[2])
end
local result = {}
for _, response_id in ipairs(members) do
	table.insert(result, response_id)
	table.insert(result, redis.call("HGET", KEYS[2], response_id) or "")
end
return result
`)

// ListResponsesByConversation lists a conversation's responses via the
// secondary index, at a cost proportional to the requested page rather than
// the keyspace or even the conversation's full history (see
// listIndexedResponseIDs).
//
// Read path: if the whole store is marked migration-complete
// (ConversationIndexCompletionKeySuffix, set once by an operator-triggered
// FinalizeConversationIndex sweep), skip straight to reading the
// index — no per-conversation check, and never a scan, not even for a
// conversation ID nothing has ever indexed. Otherwise: not yet migrated for
// this conversation specifically → run ensureConversationIndex, which
// backfills from a legacy scan (additively — it never removes members an
// ordinary write already indexed) or confirms the conversation empty, then
// marks it migrated either way. Once migrated (whether just now, already
// from an earlier read, or the whole store is finalized), the index's
// current state is trustworthy: read it if it exists, otherwise there is
// nothing to return.
//
// Checking a migrated signal rather than index-existence first is the fix
// for a state index-existence alone cannot distinguish: a conversation with
// unindexed legacy responses that then receives an ordinary post-upgrade
// write has an index — created by that one write's indexResponse call —
// containing only the new response. Trusting that index as complete would
// silently and permanently hide the older ones. See
// ConversationIndexMigratedKeyPrefix and ConversationIndexCompletionKeySuffix.
//
// Order/After/Before parity note: this implementation honors ListOptions.Order
// (default "desc", newest first) and After/Before cursors, per the contract
// documented on ListOptions in interface.go. MemoryStore does not — it
// always returns insertion order regardless of these fields (see its own
// doc comment). Bringing MemoryStore into line is out of scope for #2814;
// callers that need a specific order from either backend today should not
// assume Redis and MemoryStore agree on default order.
func (s *RedisStore) ListResponsesByConversation(ctx context.Context, conversationID string, opts ListOptions) ([]*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if conversationID == "" {
		return nil, ErrInvalidInput
	}

	normalized, err := normalizeResponseListOptions(opts)
	if err != nil {
		return nil, err
	}

	if err := s.ensureConversationResolvedForRead(ctx, conversationID); err != nil {
		return nil, err
	}

	// Whichever path resolved it — store-wide finalization, an existing
	// per-conversation marker, or ensureConversationIndex just now — the
	// index's current membership is now the source of truth: read it
	// directly rather than re-check the migrated marker, since a
	// best-effort marker-write failure inside the backfill must not block
	// returning what was actually just discovered.
	return s.listIndexedResponses(ctx, conversationID, normalized)
}

// ensureConversationResolvedForRead guarantees that, once it returns
// without error, conversationID's index may be trusted as exhaustive for a
// read: either the whole store is marked migration-complete
// (ConversationIndexCompletionKeySuffix), or this specific conversation
// already carries a resolved proof (conversationIndexResolved), or
// ensureConversationIndex has just made it so.
func (s *RedisStore) ensureConversationResolvedForRead(ctx context.Context, conversationID string) error {
	if resolved, err := s.conversationIndexResolved(ctx, conversationID); err != nil {
		return err
	} else if resolved {
		return nil
	}

	return s.ensureConversationIndex(ctx, conversationID)
}

// listIndexedResponses reads one page of a conversation's responses through
// its already-confirmed-existing index: a bounded rank-window read
// (listIndexedResponseIDs), not a full-index scan, so cost is proportional
// to the page requested rather than the conversation's full history.
//
// Because only one page of IDs is read, pruning a stale or moved entry can
// make this return fewer than the requested Limit even when more matching
// responses exist further in the index. That is an accepted Phase 4
// trade-off (blueprint §5 Phase 4): topping up short pages by re-reading
// further windows would turn a bug fix into a pagination redesign.
func (s *RedisStore) listIndexedResponses(ctx context.Context, conversationID string, opts normalizedListOptions) ([]*responseapi.StoredResponse, error) {
	witnesses, err := s.listIndexedResponseIDs(ctx, conversationID, opts)
	if err != nil {
		return nil, err
	}
	if len(witnesses) == 0 {
		return nil, nil
	}

	responseIDs := make([]string, len(witnesses))
	for i, witness := range witnesses {
		responseIDs[i] = witness.responseID
	}
	results := fetchResponsePayloadsPipelined(ctx, s.client, responseKeys(s, responseIDs))

	responses := make([]*responseapi.StoredResponse, 0, len(results))
	toPrune := make([]responseGenerationWitness, 0, len(results))
	for i, result := range results {
		witness := witnesses[i]
		response, prune := evaluateIndexedResponse(conversationID, witness, result)
		if prune {
			toPrune = append(toPrune, witness)
		}
		if response != nil {
			responses = append(responses, response)
		}
	}

	if err := s.unindexResponseGenerations(ctx, conversationID, toPrune...); err != nil {
		logging.Warnf("RedisStore: failed to prune %d stale index entr(y/ies) from conversation %s: %v",
			len(toPrune), conversationID, err)
	}

	return responses, nil
}

// evaluateIndexedResponse classifies one generation-snapshotted membership.
// Legacy payloads remain readable but can never authorize pruning.
func evaluateIndexedResponse(
	conversationID string,
	witness responseGenerationWitness,
	result responsePayloadResult,
) (*responseapi.StoredResponse, bool) {
	if errors.Is(result.err, redis.Nil) {
		return nil, witness.generation != ""
	}
	if result.err != nil {
		logging.Warnf("RedisStore: failed to get response %s: %v", witness.responseID, result.err)
		return nil, false
	}

	record, err := decodeResponseRecord(result.raw)
	if err != nil {
		logging.Warnf("RedisStore: failed to parse response %s: %v", witness.responseID, err)
		return nil, false
	}
	if record.generation == "" {
		if record.response.ConversationID == conversationID {
			return record.response, false
		}
		return nil, false
	}
	if witness.generation == "" {
		return nil, false
	}
	if record.generation != witness.generation || record.response.ConversationID != conversationID {
		return nil, true
	}
	return record.response, false
}

func responseKeys(store *RedisStore, responseIDs []string) []string {
	keys := make([]string, len(responseIDs))
	for i, responseID := range responseIDs {
		keys[i] = store.buildKey(ResponseKeyPrefix + responseID)
	}
	return keys
}

// normalizedListOptions is ListOptions after validation and defaulting:
// Limit is always in [1, MaxListLimit], and Order is always exactly "asc"
// or "desc".
type normalizedListOptions struct {
	Limit  int
	Order  string
	After  string
	Before string
}

// normalizeResponseListOptions validates and defaults a caller's ListOptions
// for indexed reads. Order defaults to "desc" (newest first), matching the
// documented ListOptions.Order contract in interface.go and OpenAI's list
// default — a contract neither store implementation actually honored before
// this issue (both simply returned index/insertion order regardless of
// Order). Rejects an unrecognized Order and rejects After and Before set
// together, rather than silently picking one and ignoring the other.
func normalizeResponseListOptions(opts ListOptions) (normalizedListOptions, error) {
	if opts.After != "" && opts.Before != "" {
		return normalizedListOptions{}, ErrInvalidInput
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	if limit > MaxListLimit {
		limit = MaxListLimit
	}

	order := opts.Order
	switch order {
	case "":
		order = "desc"
	case "asc", "desc":
		// already valid
	default:
		return normalizedListOptions{}, ErrInvalidInput
	}

	return normalizedListOptions{Limit: limit, Order: order, After: opts.After, Before: opts.Before}, nil
}

// listIndexedResponseIDs reads one bounded window of response IDs from a
// conversation's index: at most normalizeResponseListOptions(opts).Limit
// IDs, in the requested order, optionally positioned after/before a cursor
// response ID — never a full ZRANGE 0 -1.
//
// Cursors are resolved via ZRANK (ascending order) or ZREVRANK (descending
// order), i.e. rank in the order actually being read, and the window is
// then read with the matching ZRANGE/ZREVRANGE. A cursor naming a response
// ID that is not currently a member of the index (evicted, wrong
// conversation, typo'd by the caller) yields an empty page rather than an
// error: the same behavior as an ordinary page with nothing left to return.
func (s *RedisStore) listIndexedResponseIDs(ctx context.Context, conversationID string, normalized normalizedListOptions) ([]responseGenerationWitness, error) {
	indexKey := s.conversationIndexKey(conversationID)
	ascending := normalized.Order == "asc"

	start, end, ok, err := s.resolveListWindow(ctx, indexKey, ascending, normalized)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}

	return s.readIndexRange(ctx, conversationID, ascending, start, end)
}

// rankInIndex resolves a cursor response ID's rank in the given order (asc:
// ZRANK, desc: ZREVRANK). ok=false, not an error, means the cursor is not a
// current index member — the documented behavior for both After and Before.
func (s *RedisStore) rankInIndex(ctx context.Context, indexKey string, ascending bool, member string) (rank int64, ok bool, err error) {
	var cmd *redis.IntCmd
	if ascending {
		cmd = s.client.ZRank(ctx, indexKey, member)
	} else {
		cmd = s.client.ZRevRank(ctx, indexKey, member)
	}

	r, err := cmd.Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return 0, false, nil
		}
		return 0, false, fmt.Errorf("failed to rank conversation index cursor %s: %w", member, err)
	}

	return r, true, nil
}

// resolveListWindow computes the inclusive [start, end] rank window to read
// for one page, honoring an After or Before cursor. ok=false means the page
// is empty (cursor not found, or the window has nothing in it) without that
// being an error.
func (s *RedisStore) resolveListWindow(ctx context.Context, indexKey string, ascending bool, normalized normalizedListOptions) (start, end int64, ok bool, err error) {
	limit := int64(normalized.Limit)

	switch {
	case normalized.After != "":
		r, found, rankErr := s.rankInIndex(ctx, indexKey, ascending, normalized.After)
		if rankErr != nil || !found {
			return 0, 0, false, rankErr
		}
		start, end = r+1, r+limit
	case normalized.Before != "":
		r, found, rankErr := s.rankInIndex(ctx, indexKey, ascending, normalized.Before)
		if rankErr != nil || !found {
			return 0, 0, false, rankErr
		}
		end = r - 1
		start = max(0, end-limit+1)
	default:
		start, end = 0, limit-1
	}

	if end < start {
		return 0, 0, false, nil
	}

	return start, end, true, nil
}

// readIndexRange reads one inclusive rank window [start, end] in the given
// order (asc: ZRANGE, desc: ZREVRANGE).
func (s *RedisStore) readIndexRange(ctx context.Context, conversationID string, ascending bool, start, end int64) ([]responseGenerationWitness, error) {
	direction := 0
	if ascending {
		direction = 1
	}
	keys := []string{s.conversationIndexKey(conversationID), s.conversationIndexGenerationKey(conversationID)}
	result, err := readIndexWindowScript.Run(ctx, s.client, keys, start, end, direction).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to read conversation index window: %w", err)
	}
	items, ok := result.([]interface{})
	if !ok || len(items)%2 != 0 {
		return nil, fmt.Errorf("unexpected conversation index window result: %#v", result)
	}
	witnesses := make([]responseGenerationWitness, 0, len(items)/2)
	for i := 0; i < len(items); i += 2 {
		responseID, idOK := items[i].(string)
		generation, generationOK := items[i+1].(string)
		if !idOK || !generationOK {
			return nil, fmt.Errorf("unexpected conversation index window member types %T/%T", items[i], items[i+1])
		}
		witnesses = append(witnesses, responseGenerationWitness{responseID: responseID, generation: generation})
	}
	return witnesses, nil
}
