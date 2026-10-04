package sessiontools

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type scriptedStore struct {
	Store
	load func(context.Context, string) (VersionedState, error)
	cas  func(context.Context, string, uint64, State, time.Duration, QuotaKey) (bool, error)
}

func (s scriptedStore) Load(ctx context.Context, key string) (VersionedState, error) {
	return s.load(ctx, key)
}

func (s scriptedStore) CompareAndSwap(ctx context.Context, key string, revision uint64, state State, ttl time.Duration, quota QuotaKey) (bool, error) {
	return s.cas(ctx, key, revision, state, ttl, quota)
}

func managerFixture(t *testing.T, store Store) (*Manager, UpdateRequest) {
	t.Helper()
	input := selectionFixture()
	manager, err := NewManager(store, ManagerOptions{TTL: input.TTL, Timeout: time.Second, Clock: func() time.Time { return input.Now }})
	require.NoError(t, err)
	return manager, UpdateRequest{
		Key: "opaque-key", Quota: QuotaKey{Principal: "opaque-principal", Namespace: "recipe"}, Selection: input.SelectionInput,
	}
}

func TestManagerConflictRecomputesWithFrozenInputsAndOneDeadline(t *testing.T) {
	input := selectionFixture()
	input.Ranked = input.Ranked[:1]
	successor := mergeOK(t, input).State
	successor.Revision = 17
	loads, writes := 0, 0
	var deadline time.Time
	var request UpdateRequest
	store := scriptedStore{
		load: func(ctx context.Context, _ string) (VersionedState, error) {
			current, ok := ctx.Deadline()
			require.True(t, ok)
			loads++
			if loads == 1 {
				deadline = current
				return VersionedState{}, nil
			}
			require.Equal(t, deadline, current)
			return VersionedState{Found: true, State: successor.Clone()}, nil
		},
		cas: func(ctx context.Context, _ string, revision uint64, state State, _ time.Duration, _ QuotaKey) (bool, error) {
			current, _ := ctx.Deadline()
			require.Equal(t, deadline, current)
			writes++
			if writes == 1 {
				require.Zero(t, revision)
				// Callback mutation occurs after preparation, without a data race.
				// Neither the caller nor a store retaining a proposal can alter retry evidence.
				request.Selection.Eligible[1].Name = "mutated"
				request.Selection.Ranked[1].Name = "mutated"
				state.Tools[1].Name = "mutated-store-proposal"
				return false, fmt.Errorf("conflict: %w", ErrRevisionMismatch)
			}
			require.Equal(t, uint64(17), revision)
			require.Equal(t, uint64(2), state.Turn)
			require.Len(t, state.Tools, 2, "growth must be recomputed from the successor")
			require.Equal(t, "b", state.Tools[1].Name)
			return true, nil
		},
	}
	manager, initial := managerFixture(t, store)
	request = initial
	result, err := manager.Update(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, 2, result.Attempts)
	require.Equal(t, 1, result.Receipt.Added)
	require.Equal(t, 2, loads)
}

func TestManagerOnlyRetriesDefiniteWriteConflicts(t *testing.T) {
	tests := []struct {
		name    string
		applied bool
		err     error
		want    error
		calls   int
	}{
		{"conflict cap", false, ErrRevisionMismatch, ErrRetryExhausted, MaxUpdateAttempts},
		{"ambiguous timeout", false, context.DeadlineExceeded, context.DeadlineExceeded, 1},
		{"closed", false, ErrStoreClosed, ErrStoreClosed, 1},
		{"allocator exhausted", false, ErrRevisionExhausted, ErrRevisionExhausted, 1},
		{"invalid false nil", false, nil, ErrStoreContract, 1},
		{"invalid true conflict", true, ErrRevisionMismatch, ErrStoreContract, 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			calls := 0
			store := scriptedStore{
				load: func(context.Context, string) (VersionedState, error) { return VersionedState{}, nil },
				cas: func(context.Context, string, uint64, State, time.Duration, QuotaKey) (bool, error) {
					calls++
					return test.applied, test.err
				},
			}
			manager, request := managerFixture(t, store)
			result, err := manager.Update(context.Background(), request)
			require.ErrorIs(t, err, test.want)
			require.Equal(t, test.calls, calls)
			require.Empty(t, result.Tools)
			require.Empty(t, result.Receipt)
		})
	}
}

func TestManagerLoadFailuresAndCancellationNeverRetry(t *testing.T) {
	for _, loadErr := range []error{ErrStateCorrupted, ErrRevisionMismatch, context.Canceled} {
		calls := 0
		store := scriptedStore{load: func(context.Context, string) (VersionedState, error) {
			calls++
			return VersionedState{}, loadErr
		}}
		manager, request := managerFixture(t, store)
		_, err := manager.Update(context.Background(), request)
		require.ErrorIs(t, err, loadErr)
		require.Equal(t, 1, calls)
	}
	ctx, cancel := context.WithCancel(context.Background())
	loads := 0
	store := scriptedStore{
		load: func(context.Context, string) (VersionedState, error) {
			loads++
			return VersionedState{}, nil
		},
		cas: func(context.Context, string, uint64, State, time.Duration, QuotaKey) (bool, error) {
			cancel()
			return false, ErrRevisionMismatch
		},
	}
	defer cancel()
	manager, request := managerFixture(t, store)
	_, err := manager.Update(ctx, request)
	require.ErrorIs(t, err, context.Canceled)
	require.Equal(t, 1, loads)
}

func TestManagerStaleInvalidationPreservesSuccessor(t *testing.T) {
	input := selectionFixture()
	input.Ranked = input.Ranked[4:5]
	successor := mergeOK(t, input).State
	successor.Revision = 99
	invalid := successor.Clone()
	invalid.SchemaVersion, invalid.Revision = SchemaVersion-1, 98
	loads := 0
	store := scriptedStore{
		load: func(context.Context, string) (VersionedState, error) {
			loads++
			if loads == 1 {
				return VersionedState{Found: true, State: invalid.Clone()}, nil
			}
			return VersionedState{Found: true, State: successor.Clone()}, nil
		},
		cas: func(_ context.Context, _ string, revision uint64, state State, _ time.Duration, _ QuotaKey) (bool, error) {
			if revision == 98 {
				return false, ErrRevisionMismatch
			}
			require.Equal(t, uint64(99), revision)
			require.Equal(t, "e", state.Tools[0].Name, "reset must not erase the successor's retained prefix")
			successor = state.Clone()
			return true, nil
		},
	}
	// Delete is deliberately unavailable: any unconditional cleanup panics.
	manager, request := managerFixture(t, store)
	result, err := manager.Update(context.Background(), request)
	require.NoError(t, err)
	require.Equal(t, ReasonNone, result.Receipt.Reason)
	result.Tools[0].Name = "caller-mutation"
	require.Equal(t, "e", successor.Tools[0].Name)
}

func TestManagerCallerDeadlineAndInvalidRequest(t *testing.T) {
	manager, request := managerFixture(t, scriptedStore{}) // Store calls would panic.
	ctx, cancel := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer cancel()
	_, err := manager.Update(ctx, request)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	request.Selection.Required = []string{"missing"}
	_, err = manager.Update(context.Background(), request)
	var required *RequiredToolsError
	require.ErrorAs(t, err, &required)
	request.Key = ""
	_, err = manager.Update(context.Background(), request)
	require.ErrorIs(t, err, ErrInvalidSelection)
}

func TestManagerFoundStateRequiresObservedToken(t *testing.T) {
	store := scriptedStore{load: func(context.Context, string) (VersionedState, error) {
		return VersionedState{Found: true, State: State{Revision: 0}}, nil
	}}
	manager, request := managerFixture(t, store)
	_, err := manager.Update(context.Background(), request)
	require.ErrorIs(t, err, ErrStateCorrupted) // Neither CAS(0) nor Delete is valid cleanup here.
}
