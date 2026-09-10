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
// its already-confirmed-existing index. Every individual rank-window read and
// payload pipeline is bounded by listIndexScanMaxStride; ordinary live pages
// still use exactly the caller's Limit.
//
// A stale-only window cannot be returned as an empty terminal page while live
// responses remain behind it. After a successful conditional prune this
// re-resolves the original After/Before window so rank shifts — including ones
// caused by concurrent cleaners — can never make the reader skip members. A
// zero-live, successfully pruned run grows its bounded window exponentially
// up to listIndexScanMaxStride to amortize round trips through mass expiry.
//
// Total work in a cleanup-heavy call is proportional to the clearable stale
// prefix, but each successful removal is permanent and therefore amortized.
// If candidates repeatedly survive their conditional cleanup, this returns
// ErrIndexContended rather than spinning or presenting an ambiguous short
// page. Blank-witness tombstones remain deliberately non-prunable before
// finalization: proof expiry does not remove them, so they can continue to
// underfill a page until the operator completes finalization. Walking past
// that migration-boundary state needs an API-visible continuation and is out
// of scope for this generation-safe cleanup path.
func (s *RedisStore) listIndexedResponses(ctx context.Context, conversationID string, opts normalizedListOptions) ([]*responseapi.StoredResponse, error) {
	// Blank-witness memberships may only be cleaned up once the store is
	// finalized — the same rule cascade delete follows, for the same reason:
	// before then an index-unaware writer can recreate a payload without
	// touching the sidecar, and the blank-expected removal would still match.
	// The completion record is process-cached and ensureConversationResolvedForRead
	// has already consulted it, so this costs nothing in the steady state.
	allowBlankCleanup, err := s.conversationIndexFinalized(ctx)
	if err != nil {
		return nil, err
	}

	stride := opts.Limit
	contendedRounds := 0
	for {
		page, collectErr := s.collectIndexedPage(ctx, conversationID, opts, stride, allowBlankCleanup)

		if responses, complete := selectCompleteIndexedPage(conversationID, opts, page, collectErr); complete {
			return responses, nil
		}
		if collectErr != nil {
			return nil, collectErr
		}
		if page.pruneCandidates == 0 {
			return page.responses, nil
		}

		if page.membershipsRemoved == 0 {
			contendedRounds++
			if contendedRounds >= listIndexMaxContentionRounds {
				return nil, fmt.Errorf("%w: conversation %s did not advance after %d cleanup attempts",
					ErrIndexContended, conversationID, contendedRounds)
			}
			continue
		}

		contendedRounds = 0
		if len(page.responses) == 0 && stride < listIndexScanMaxStride {
			stride = min(stride*2, listIndexScanMaxStride)
		}
	}
}

// selectCompleteIndexedPage limits an oversized cleanup snapshot to the
// caller's logical page. Returning a complete page is safe even if the
// best-effort cleanup of stale members later in the same snapshot failed: the
// response page neither depends on that cleanup nor claims exhaustion.
func selectCompleteIndexedPage(
	conversationID string,
	opts normalizedListOptions,
	page collectedIndexedPage,
	collectErr error,
) ([]*responseapi.StoredResponse, bool) {
	if len(page.responses) < opts.Limit {
		return nil, false
	}
	if collectErr != nil {
		logging.Warnf("RedisStore: failed to prune stale index entries after filling conversation %s page: %v",
			conversationID, collectErr)
	}

	// Before windows grow away from their cursor by moving their starting
	// rank backwards. Keep the responses nearest that fixed cursor; taking
	// the first Limit would return older entries that only entered through
	// the oversized cleanup window. Default and After windows grow at the
	// far end, so their first Limit remain the logical page.
	start := 0
	if opts.Before != "" {
		start = len(page.responses) - opts.Limit
	}
	return page.responses[start : start+opts.Limit], true
}

// collectedIndexedPage is one bounded index snapshot after its payloads have
// been resolved and its stale memberships conditionally pruned. Candidate and
// actual-removal counts stay distinct because a concurrent writer may claim a
// generation after the snapshot, making cleanup a correct no-op.
type collectedIndexedPage struct {
	responses          []*responseapi.StoredResponse
	pruneCandidates    int
	membershipsRemoved int
}

// collectIndexedPage reads one windowSize-bounded window, resolves its
// payloads, and conditionally prunes the memberships that window proved
// stale. windowSize is independent of opts.Limit: the former may grow to
// listIndexScanMaxStride while the latter remains the caller's return limit.
// It returns the page together with the cleanup error so a full oversized
// window can still be served, while an underfilled caller never mistakes
// failed cleanup for end-of-list.
func (s *RedisStore) collectIndexedPage(
	ctx context.Context,
	conversationID string,
	opts normalizedListOptions,
	windowSize int,
	allowBlankCleanup bool,
) (collectedIndexedPage, error) {
	witnesses, err := s.listIndexedResponseIDs(ctx, conversationID, opts, windowSize)
	if err != nil {
		return collectedIndexedPage{}, err
	}
	if len(witnesses) == 0 {
		return collectedIndexedPage{}, nil
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
		response, prune := evaluateIndexedResponse(conversationID, witness, result, allowBlankCleanup)
		if prune {
			toPrune = append(toPrune, witness)
		}
		if response != nil {
			responses = append(responses, response)
		}
	}

	page := collectedIndexedPage{
		responses:       responses,
		pruneCandidates: len(toPrune),
	}
	page.membershipsRemoved, err = s.unindexResponseGenerations(ctx, conversationID, toPrune...)
	if err != nil {
		return page, fmt.Errorf("failed to prune %d stale index entr(y/ies) from conversation %s: %w",
			len(toPrune), conversationID, err)
	}

	return page, nil
}

// evaluateIndexedResponse classifies one generation-snapshotted membership,
// reporting the response this page should return and whether the membership
// should be pruned.
//
// The two are deliberately independent. Readability follows only from the
// payload: a response whose stored ConversationID names this conversation is
// returned whether or not its sidecar witness agrees, because a witness that
// has fallen behind (a legacy member not yet upgraded, an index write still in
// flight) says nothing about whether the response belongs here. Dropping those
// from the page silently hid live data.
//
// Pruning is authorized only by the two conditions that are stable rather than
// transient — the payload is proven absent, or it names a different
// conversation — and even then only through the conditional unindex, which
// fires solely while the sidecar still holds exactly this witness.
//
// A blank witness authorizes a prune only once the store is finalized. Against
// a generation-aware writer the blank comparison is already safe, since any
// recreate or move-back installs a witness first and the stale cleanup finds a
// mismatch. An index-unaware writer installs nothing, and while such writers
// can still exist — which is to say, before finalization — a blank-expected
// prune can unindex a payload one of them just recreated. On this path that
// costs at most one conversationIndexProofMaxTTL of invisibility before a
// re-scan repairs it, but cascade delete cannot absorb the same race and the
// rule is kept uniform rather than split by caller. Until finalization a blank
// member remains readable but neither removable nor promotable; the
// operator-authorized finalization sweep resolves it after old writers drain.
func evaluateIndexedResponse(
	conversationID string,
	witness responseGenerationWitness,
	result responsePayloadResult,
	allowBlankCleanup bool,
) (*responseapi.StoredResponse, bool) {
	prunable := witness.generation != "" || allowBlankCleanup

	if errors.Is(result.err, redis.Nil) {
		return nil, prunable
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
	if record.response.ConversationID != conversationID {
		return nil, prunable
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
// conversation's index: at most windowSize IDs, in the requested order,
// optionally positioned after/before a cursor response ID — never a full
// ZRANGE 0 -1. normalized.Limit remains the caller's return limit; the
// cleanup loop passes its independently capped internal stride as windowSize.
//
// Cursors are resolved via ZRANK (ascending order) or ZREVRANK (descending
// order), i.e. rank in the order actually being read, and the window is
// then read with the matching ZRANGE/ZREVRANGE. A cursor naming a response
// ID that is not currently a member of the index (evicted, wrong
// conversation, typo'd by the caller) yields an empty page rather than an
// error: the same behavior as an ordinary page with nothing left to return.
func (s *RedisStore) listIndexedResponseIDs(
	ctx context.Context,
	conversationID string,
	normalized normalizedListOptions,
	windowSize int,
) ([]responseGenerationWitness, error) {
	indexKey := s.conversationIndexKey(conversationID)
	ascending := normalized.Order == "asc"

	start, end, ok, err := s.resolveListWindow(ctx, indexKey, ascending, normalized, windowSize)
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
func (s *RedisStore) resolveListWindow(
	ctx context.Context,
	indexKey string,
	ascending bool,
	normalized normalizedListOptions,
	windowSize int,
) (start, end int64, ok bool, err error) {
	limit := int64(windowSize)

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
