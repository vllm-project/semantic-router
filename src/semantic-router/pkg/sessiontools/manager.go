package sessiontools

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// MaxUpdateAttempts includes the initial CAS. It limits contention work even
// when a store returns conflicts faster than the overall deadline elapses.
const MaxUpdateAttempts = 3

var (
	ErrRetryExhausted = errors.New("sessiontools: CAS retry budget exhausted")
	ErrStoreContract  = errors.New("sessiontools: inconsistent CAS result")
)

// ManagerOptions uses resolved global store TTL/timeout values. Timeout covers
// the entire update, including planning and all retries, under the caller's
// deadline. Store implementations must honor cancellation; the manager does
// not detach timed-out store calls into background goroutines.
type ManagerOptions struct {
	TTL     time.Duration
	Timeout time.Duration
	Clock   func() time.Time
}

// Manager owns bounded selection updates. Its immutable options and a
// concurrency-safe Store allow concurrent calls. It does not own Store.Close:
// the runtime owner must drain borrowers before closing generation resources.
// No runtime path constructs this library while sticky enablement is gated.
type Manager struct {
	store   Store
	options ManagerOptions
}

type UpdateRequest struct {
	Key       string
	Quota     QuotaKey
	Selection SelectionInput
}

// UpdateResult reports a confirmed write at its CAS linearization point, not
// an enduring latest snapshot. It deliberately exposes no guessed Revision:
// Store.CompareAndSwap assigns that token and does not return it.
type UpdateResult struct {
	Tools    []ToolIdentity
	Receipt  SelectionReceipt
	Attempts int
}

func NewManager(store Store, options ManagerOptions) (*Manager, error) {
	if store == nil || options.TTL <= 0 || options.Timeout <= 0 {
		return nil, fmt.Errorf("%w: store, TTL and timeout are required", ErrInvalidSelection)
	}
	if options.Clock == nil {
		options.Clock = time.Now
	}
	return &Manager{store: store, options: options}, nil
}

// Update freezes current eligible evidence before loading. Invalid snapshots
// are reset through their observed token, never by unconditional deletion.
// Only a definite (false, ErrRevisionMismatch) is retryable. Other failures,
// including an ambiguous write timeout, return no proposed tools or receipt;
// the runtime adapter must use a currently authorized stateless fallback or
// reject when explicit requirements cannot be honored.
func (m *Manager) Update(ctx context.Context, request UpdateRequest) (UpdateResult, error) {
	ctx, cancel := context.WithTimeout(ctx, m.options.Timeout)
	defer cancel()
	if err := ctx.Err(); err != nil {
		return UpdateResult{}, err
	}
	if request.Key == "" || request.Quota.Principal == "" || request.Quota.Namespace == "" {
		return UpdateResult{}, fmt.Errorf("%w: trusted storage and quota keys are required", ErrInvalidSelection)
	}
	selection, err := prepareSelection(request.Selection)
	if err != nil {
		return UpdateResult{}, err
	}
	for attempt := 1; attempt <= MaxUpdateAttempts; attempt++ {
		result, retry, err := m.attempt(ctx, request, selection)
		if err == nil {
			return UpdateResult{Tools: result.Tools, Receipt: result.Receipt, Attempts: attempt}, nil
		}
		if !retry {
			return UpdateResult{Attempts: attempt}, err
		}
	}
	return UpdateResult{Attempts: MaxUpdateAttempts}, ErrRetryExhausted
}

func (m *Manager) attempt(ctx context.Context, request UpdateRequest, selection preparedSelection) (MergeResult, bool, error) {
	if err := ctx.Err(); err != nil {
		return MergeResult{}, false, err
	}
	loaded, err := m.store.Load(ctx, request.Key)
	if err != nil {
		return MergeResult{}, false, err
	}
	var prior *State
	var revision uint64
	if loaded.Found {
		if loaded.State.Revision == 0 {
			return MergeResult{}, false, ErrStateCorrupted
		}
		copy := loaded.State.Clone()
		prior, revision = &copy, copy.Revision
	}
	result, err := selection.merge(prior, m.options.Clock(), m.options.TTL)
	if err != nil {
		return MergeResult{}, false, err
	}
	if err = ctx.Err(); err != nil {
		return MergeResult{}, false, err
	}
	applied, err := m.store.CompareAndSwap(ctx, request.Key, revision, result.State.Clone(), m.options.TTL, request.Quota)
	if !applied && err == nil || applied && err != nil {
		return MergeResult{}, false, ErrStoreContract
	}
	return result, !applied && errors.Is(err, ErrRevisionMismatch), err
}
