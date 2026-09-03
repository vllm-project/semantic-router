package sessiontools

import (
	"context"
	"errors"
	"time"
)

// Sentinel errors returned by Store implementations. Callers should use
// errors.Is against these rather than matching on error strings.
var (
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
	// cannot be decoded or fails State.Validate. Corrupted values are
	// deleted best-effort by the store before returning this error, never
	// returned as a partially-trusted State.
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
// Store is deliberately narrow: it owns key-value mechanics (load, atomic
// compare-and-swap, delete, lifecycle) only. Decode validation,
// deterministic merge, retry policy, L1 hydration, fallback, and receipts
// belong to the manager built on top of Store (a later task) — see PL-0042
// section 7.1, "Manager.Update owns decode validation... Extproc never
// calls Redis directly."
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

	// Delete removes key. Idempotent: deleting an absent key is not an
	// error.
	Delete(ctx context.Context, key string) error

	// Close releases resources held by the store. After Close returns,
	// every other method returns ErrStoreClosed. Close itself is
	// idempotent.
	Close() error
}
