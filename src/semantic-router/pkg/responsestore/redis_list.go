package responsestore

import (
	"context"
	"errors"
	"fmt"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// readIndexWindowScript resolves a page's cursor, reads its rank window, and
// snapshots each member's generation witness in one atomic step.
//
// All three happen inside a single script because a cursor's rank is only
// meaningful against the membership it was measured in. Resolving ZRANK in one
// command and reading ZRANGE in the next let any member inserted or removed
// *before* the cursor in between shift every later rank: a removal made the
// saved window start one past the cursor's true successor and skipped a live
// response, and an insertion made it start on the cursor itself and returned
// the response the caller already held. Redis executes a script without
// interleaving other commands, so the rank, the range, and the witnesses here
// all describe one membership.
//
// KEYS[1] is the conversation ZSET and KEYS[2] its witness HASH; they share a
// Redis Cluster hash tag. ARGV: [1] "1" for ascending rank order, [2] cursor
// mode — "none", "after", or "before" — [3] the cursor member, [4] the window
// width in ranks.
//
// Replies with a status word, an exhaustion flag, then (member, witness)
// pairs. "missing" means the cursor is not a current member; "empty" means
// nothing lies in the requested direction at all.
//
// The exhaustion flag is computed here rather than inferred in Go, because
// the two directions prove it differently and only the script sees the rank
// geometry. A forward window never clamps: it is exhausted when ZRANGE hands
// back fewer members than asked for. A Before window is the reverse: every
// rank in [0, cursor-1] exists, so it always comes back full, and the only
// evidence that nothing precedes it is that its start clamped at rank 0.
// Reporting the clamped span and comparing counts against it read that clamp
// as a full window with more behind it, and a cursor near the start could
// never return its few preceding responses.
var readIndexWindowScript = redis.NewScript(`
local ascending = ARGV[1] == "1"
local mode = ARGV[2]
local cursor = ARGV[3]
local width = tonumber(ARGV[4])

local start, stop
if mode == "none" then
	start, stop = 0, width - 1
else
	local rank
	if ascending then
		rank = redis.call("ZRANK", KEYS[1], cursor)
	else
		rank = redis.call("ZREVRANK", KEYS[1], cursor)
	end
	if not rank then
		return {"missing", 0}
	end
	if mode == "after" then
		start, stop = rank + 1, rank + width
	else
		stop = rank - 1
		start = math.max(0, stop - width + 1)
	end
end
if stop < start then
	return {"empty", 0}
end

local members
if ascending then
	members = redis.call("ZRANGE", KEYS[1], start, stop)
else
	members = redis.call("ZREVRANGE", KEYS[1], start, stop)
end
local exhausted
if mode == "before" then
	exhausted = start == 0
else
	exhausted = #members < width
end
local result = {"ready", exhausted and 1 or 0}
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
// page.
//
// A window may also hold members that are unreadable yet must be left alone:
// before finalization a blank-witness membership whose payload is gone is
// deliberately non-prunable, and proof expiry never clears it, because a
// rescan adds live members but removes nothing. Such a window yields no
// responses and no prune candidates, which reported as an empty terminal page
// and hid every retained response behind it.
//
// Those members are reached by *widening* the window from the caller's own
// anchor, never by re-anchoring onto one of them. A response ID is not a
// stable index position: an ordinary StoreResponse of a response whose payload
// had expired re-scores its membership through conversationIndexAddScript's
// unqualified ZADD, so an anchor placed on a blocked member follows that member
// to its new rank and can carry the next window past live responses that never
// moved. Re-resolving the anchor server-side does not help — ZRANK faithfully
// returns the new incarnation's position. Only the caller's own anchor, which
// this loop never rewrites, is immune.
//
// Widening is capped at listIndexScanMaxStride, so the walk costs at most
// log2(listIndexScanMaxStride/Limit) bounded reads. A run of blocked members
// too long to see past reports ErrIndexTraversalBlocked — never exhaustion.
//
// An underfilled page may only be returned once it is *proven* complete: the
// window script must report the index exhausted in the direction being read —
// a forward window that came back short, or a Before window that clamped at
// rank 0. A full, unclamped window that underfills the page is evidence of the
// opposite, and widens instead.
//
// No page is returned at all — full or proven complete — if any member of the
// window could not be read. A Redis failure says nothing about whether that
// response is live, and every success path hands the caller a cursor that has
// already advanced past it. Because each round re-reads a superset of the last
// window, the final window's failures are exactly the ones the returned page
// would have stepped over, so that one check fails closed for the whole walk.
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

	walk := indexWalk{stride: opts.Limit}
	for {
		// opts, never a rewritten anchor: the window origin is fixed for the
		// life of the call and only its width changes.
		page, collectErr := s.collectIndexedPage(ctx, conversationID, opts, walk.stride, allowBlankCleanup)

		// Checked before either success path: a full page and a proven-complete
		// short page would both advance the caller's cursor past the member
		// that could not be read.
		if page.readFailures > 0 {
			return nil, fmt.Errorf("%w: %d payload(s) in conversation %s; first failure: %w",
				ErrPayloadReadFailed, page.readFailures, conversationID, page.firstReadErr)
		}
		if responses, complete := selectCompleteIndexedPage(conversationID, opts, page, collectErr); complete {
			return responses, nil
		}
		if collectErr != nil {
			return nil, collectErr
		}

		// Either the caller's cursor is not a current member or nothing lies
		// in the requested direction. Both are the documented empty page.
		if page.window.resolution != listWindowReady {
			return nil, nil
		}

		// Stale memberships are resolved before exhaustion is consulted. A
		// window can be short and still owe the caller another look: a
		// membership recreated between this page's payload read and its
		// conditional prune is only observed by re-reading the same window.
		if page.pruneCandidates > 0 {
			if err := walk.notePruneRound(conversationID, page); err != nil {
				return nil, err
			}
			continue
		}

		// The script proved nothing lies beyond this window, so whatever it
		// yielded is the whole remainder. This is the only condition under
		// which an underfilled page is returned rather than widened.
		if page.window.exhausted {
			return page.responses, nil
		}

		// A full window that still underfills the page is holding members this
		// read may not remove and cannot return. Only a wider one reaches past
		// them, and the width is bounded.
		if err := walk.widen(conversationID); err != nil {
			return nil, err
		}
	}
}

// indexWalk is the state one listIndexedResponses call carries between
// rounds: how wide the next window is, and how many consecutive rounds have
// identified stale members without managing to remove any.
type indexWalk struct {
	stride          int
	contendedRounds int
}

// notePruneRound records the outcome of a round that found prunable members.
// A round that removed nothing is contention — a writer claimed the generation
// after the snapshot — and is re-read; too many in a row become
// ErrIndexContended rather than an underfilled page a paginator would read as
// terminal. A round that removed something is durable progress; if it also
// yielded no responses the run is dead and the next window grows so a mass
// expiry costs round trips proportional to listIndexScanMaxStride rather than
// to the caller's Limit.
func (w *indexWalk) notePruneRound(conversationID string, page collectedIndexedPage) error {
	if page.membershipsRemoved == 0 {
		w.contendedRounds++
		if w.contendedRounds >= listIndexMaxContentionRounds {
			return fmt.Errorf("%w: conversation %s did not advance after %d cleanup attempts",
				ErrIndexContended, conversationID, w.contendedRounds)
		}
		return nil
	}
	w.contendedRounds = 0
	if len(page.responses) == 0 {
		w.stride = growListStride(w.stride)
	}
	return nil
}

// widen grows the next window past members this read may not remove, or
// reports that the widest bounded window has already been tried.
func (w *indexWalk) widen(conversationID string) error {
	if w.stride >= listIndexScanMaxStride {
		return blockedTraversalError(conversationID, w.stride)
	}
	w.stride = growListStride(w.stride)
	return nil
}

// growListStride widens the next bounded window after one that returned no
// responses, so a long dead or blocked run costs round trips proportional to
// listIndexScanMaxStride rather than to the caller's Limit.
func growListStride(stride int) int {
	if stride >= listIndexScanMaxStride {
		return stride
	}
	return min(stride*2, listIndexScanMaxStride)
}

// blockedTraversalError reports a run of unreadable-but-unremovable members
// longer than the widest bounded window may see past. Read failures never
// reach here: they fail the page before any widening is attempted.
func blockedTraversalError(conversationID string, width int) error {
	return fmt.Errorf(
		"%w: conversation %s still holds unreadable memberships this read may not remove across a %d-member window; run FinalizeConversationIndex to clear them",
		ErrIndexTraversalBlocked, conversationID, width)
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

	// membersRead is how many index members the window actually returned.
	membersRead int

	// readFailures counts members whose payload could not be read at all, as
	// distinct from being proven absent. Any such member fails the page: it
	// may be live, and every success path would hand back a cursor already
	// past it. firstReadErr carries the first underlying failure for the
	// error the caller sees.
	readFailures int
	firstReadErr error

	window listWindowRead
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
	witnesses, window, err := s.listIndexedResponseIDs(ctx, conversationID, opts, windowSize)
	if err != nil {
		return collectedIndexedPage{window: window}, err
	}
	if len(witnesses) == 0 {
		return collectedIndexedPage{window: window}, nil
	}

	responseIDs := make([]string, len(witnesses))
	for i, witness := range witnesses {
		responseIDs[i] = witness.responseID
	}
	results := fetchResponsePayloadsPipelined(ctx, s.client, responseKeys(s, responseIDs))

	responses := make([]*responseapi.StoredResponse, 0, len(results))
	toPrune := make([]responseGenerationWitness, 0, len(results))
	readFailures := 0
	var firstReadErr error
	for i, result := range results {
		witness := witnesses[i]
		response, prune, readFailed := evaluateIndexedResponse(conversationID, witness, result, allowBlankCleanup)
		if prune {
			toPrune = append(toPrune, witness)
		}
		if readFailed {
			readFailures++
			if firstReadErr == nil {
				firstReadErr = result.err
			}
		}
		if response != nil {
			responses = append(responses, response)
		}
	}

	page := collectedIndexedPage{
		responses:       responses,
		pruneCandidates: len(toPrune),
		membersRead:     len(witnesses),
		readFailures:    readFailures,
		firstReadErr:    firstReadErr,
		window:          window,
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
// readFailed distinguishes a payload this page could not read from one it read
// and found unusable. A Redis failure says nothing about whether the response
// is still there, so the page fails rather than return past it; a payload that
// decoded badly was genuinely observed and can never be returned, so skipping
// it costs nothing a retry would recover.
func evaluateIndexedResponse(
	conversationID string,
	witness responseGenerationWitness,
	result responsePayloadResult,
	allowBlankCleanup bool,
) (response *responseapi.StoredResponse, prune bool, readFailed bool) {
	prunable := witness.generation != "" || allowBlankCleanup

	if errors.Is(result.err, redis.Nil) {
		return nil, prunable, false
	}
	if result.err != nil {
		logging.Warnf("RedisStore: failed to get response %s: %v", witness.responseID, result.err)
		return nil, false, true
	}

	record, err := decodeResponseRecord(result.raw)
	if err != nil {
		logging.Warnf("RedisStore: failed to parse response %s: %v", witness.responseID, err)
		return nil, false, false
	}
	if record.response.ConversationID != conversationID {
		return nil, prunable, false
	}
	return record.response, false, false
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
// The cursor's rank, the window, and every member's witness come from one
// readIndexWindowScript call, so they describe a single membership — see the
// script for why resolving the rank separately could skip or repeat a live
// response. A cursor naming a response ID that is not currently a member
// (evicted, wrong conversation, typo'd by the caller) yields an empty page
// rather than an error: the same behavior as an ordinary page with nothing
// left to return.
func (s *RedisStore) listIndexedResponseIDs(
	ctx context.Context,
	conversationID string,
	normalized normalizedListOptions,
	windowSize int,
) ([]responseGenerationWitness, listWindowRead, error) {
	direction := 0
	if normalized.Order == "asc" {
		direction = 1
	}
	mode, cursor := "none", ""
	switch {
	case normalized.After != "":
		mode, cursor = "after", normalized.After
	case normalized.Before != "":
		mode, cursor = "before", normalized.Before
	}

	keys := []string{s.conversationIndexKey(conversationID), s.conversationIndexGenerationKey(conversationID)}
	result, err := readIndexWindowScript.Run(ctx, s.client, keys, direction, mode, cursor, windowSize).Result()
	if err != nil {
		return nil, listWindowRead{}, fmt.Errorf("failed to read conversation index window: %w", err)
	}
	return decodeIndexWindow(result)
}

// decodeIndexWindow unpacks readIndexWindowScript's reply: a status word, an
// exhaustion flag, then (member, witness) pairs.
func decodeIndexWindow(result interface{}) ([]responseGenerationWitness, listWindowRead, error) {
	items, ok := result.([]interface{})
	if !ok || len(items) < 2 || len(items)%2 != 0 {
		return nil, listWindowRead{}, fmt.Errorf("unexpected conversation index window result: %#v", result)
	}
	status, statusOK := items[0].(string)
	exhausted, exhaustedOK := items[1].(int64)
	if !statusOK || !exhaustedOK {
		return nil, listWindowRead{}, fmt.Errorf("unexpected conversation index window header types %T/%T", items[0], items[1])
	}

	resolution, err := parseWindowResolution(status)
	if err != nil {
		return nil, listWindowRead{}, err
	}
	window := listWindowRead{resolution: resolution, exhausted: exhausted == 1}

	witnesses, err := decodeWindowMembers(items[2:])
	return witnesses, window, err
}

func parseWindowResolution(status string) (listWindowResolution, error) {
	switch status {
	case "ready":
		return listWindowReady, nil
	case "empty":
		return listWindowEmpty, nil
	case "missing":
		return listWindowCursorMissing, nil
	default:
		return 0, fmt.Errorf("unexpected conversation index window status %q", status)
	}
}

func decodeWindowMembers(pairs []interface{}) ([]responseGenerationWitness, error) {
	witnesses := make([]responseGenerationWitness, 0, len(pairs)/2)
	for i := 0; i < len(pairs); i += 2 {
		responseID, idOK := pairs[i].(string)
		generation, generationOK := pairs[i+1].(string)
		if !idOK || !generationOK {
			return nil, fmt.Errorf("unexpected conversation index window member types %T/%T", pairs[i], pairs[i+1])
		}
		witnesses = append(witnesses, responseGenerationWitness{responseID: responseID, generation: generation})
	}
	return witnesses, nil
}

// listWindowRead describes the bounded window one read actually covered: how
// it resolved, and whether the script proved that nothing lies beyond it in
// the direction being read.
//
// Exhaustion is a fact about rank geometry that the script establishes and
// Go only consumes — see readIndexWindowScript for why the two directions
// prove it differently. It is the only condition under which an underfilled
// page may be returned rather than widened.
type listWindowRead struct {
	resolution listWindowResolution
	exhausted  bool
}

// listWindowResolution says why readIndexWindowScript did or did not produce
// a rank window. The states are kept distinct because an exhausted index and an
// unresolvable cursor both used to arrive as "no members", and the listing
// loop has to tell them apart before an empty result may be reported as
// end-of-list.
type listWindowResolution uint8

const (
	// listWindowReady means a real window was read.
	listWindowReady listWindowResolution = iota
	// listWindowEmpty means the index holds nothing further in this direction.
	listWindowEmpty
	// listWindowCursorMissing means the cursor response ID is not a current
	// member. The listing loop only ever passes the caller's own After/Before,
	// so this is the documented empty page rather than a lost traversal
	// anchor — it is kept distinct from listWindowEmpty because the two are
	// different facts about the index, not because they are handled apart.
	listWindowCursorMissing
)
