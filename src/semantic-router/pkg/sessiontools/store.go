package sessiontools

import (
	"context"
	"errors"
	"time"
)

// Sentinel errors returned by Store implementations. Callers should use
// errors.Is against these rather than matching on error strings.
var (
	// ErrRevisionExhausted fails writes before changing state or admission
	// bookkeeping. Revisions must never wrap or be reset in a live store.
	ErrRevisionExhausted = errors.New("sessiontools: revision space exhausted")

	// ErrRevisionMismatch is returned by CompareAndSwap (alongside applied
	// = false) when expectedRevision no longer matches the stored
	// revision. This is an expected, retryable outcome of concurrent
	// writers racing, not a store failure — callers should reload and
	// retry within their own bounded retry budget, not treat it as fatal.
	ErrRevisionMismatch = errors.New("sessiontools: revision mismatch")

	// ErrStoreClosed is returned by any operation on a Store after Close
	// has completed.
	ErrStoreClosed = errors.New("sessiontools: store is closed")

	// ErrStateCorrupted is returned by Load when a stored value exists but
	// cannot be decoded safely. A storage adapter may remove corrupt bytes
	// only conditionally on the observed value; delayed deletion by key is
	// unsafe. Decodable but invalid snapshots retain their revision so the
	// manager can conditionally reset them, never partially reuse them.
	ErrStateCorrupted = errors.New("sessiontools: stored state is corrupted")
)

// VersionedState wraps a State with whether it was actually found. The
// zero State returned when Found is false must not be used.
type VersionedState struct {
	State State
	Found bool
}

// QuotaKey identifies the cardinality bucket a session belongs to for
// per-principal quota enforcement (config.ToolSessionStoreConfig's
// max_sessions_per_identity). Both fields together are the bucket identity
// — Namespace is the recipe partition, Principal the opaque
// HMAC-derived authenticated-principal key (see PL-0042 section 2.3); a
// caller must never place a raw user/tenant/session ID here.
type QuotaKey struct {
	Principal string
	Namespace string
}

// Store is the transport-storage contract for session-scoped sticky
// tool-set state. Implementations must be safe for concurrent use.
//
// Store owns key-value mechanics and the actual encoded write-size bound.
// State validation, merge, retries and receipts belong to Manager. Runtime
// eligibility and fallback belong to its caller. Implementations must honor
// context cancellation; callers must not mutate arguments during a call.
//
// Deviation from the original interface sketch: Load returns
// (VersionedState, error), not (VersionedState, bool, error). The sketch's
// third bool return value would have carried no information beyond
// VersionedState.Found — two independent signals for the same fact
// invites them to disagree. VersionedState.Found is the single source of
// truth for presence; error is reserved for genuine store failures
// (ErrStoreClosed, ErrStateCorrupted).
type Store interface {
	// Load retrieves the state stored under key. A missing key is not an
	// error: it returns (VersionedState{Found: false}, nil).
	Load(ctx context.Context, key string) (VersionedState, error)

	// CompareAndSwap atomically writes next under key, succeeding only if
	// the store's current revision for key equals expectedRevision
	// (expectedRevision == 0 means "key must not already exist" — the
	// creation case). ttl sets/refreshes the key's expiry on success.
	// quota identifies the cardinality bucket this key counts against for
	// admission/eviction purposes.
	// The store stamps LastSeenAt/ExpiresAt in UTC and checks the final
	// encoded size, including its allocated revision, before publication or
	// eviction. Failed preparation may consume a revision; gaps are valid.
	//
	// Implementations must make revisions incarnation-safe: a revision
	// value must never be reused at a given key across that key's full
	// lifetime, including across expiry and later recreation. A per-key
	// counter that resets on every fresh admission does not satisfy this —
	// a writer holding a revision captured from an earlier, since-expired
	// incarnation could then have its stale CompareAndSwap coincidentally
	// match a completely different, later incarnation and silently
	// overwrite it.
	//
	// Returns (true, nil) on success. Returns (false, ErrRevisionMismatch)
	// when expectedRevision does not match — an expected outcome under
	// concurrent writers, not treated as a failure by this signature
	// itself; callers decide their own retry policy. Returns (false, err)
	// for any other failure.
	CompareAndSwap(
		ctx context.Context,
		key string,
		expectedRevision uint64,
		next State,
		ttl time.Duration,
		quota QuotaKey,
	) (bool, error)

	// Delete unconditionally removes the current key. It is for explicit
	// deletion, not invalidation based on an earlier Load; Manager resets
	// those snapshots with CompareAndSwap instead. Missing keys are a no-op.
	Delete(ctx context.Context, key string) error

	// Close releases resources held by the store. After Close returns,
	// every other method returns ErrStoreClosed. Close itself is
	// idempotent.
	Close() error
}
