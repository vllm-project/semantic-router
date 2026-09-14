package sessiontools

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

type managerTestStore struct {
	state       State
	found       bool
	loadErr     error
	deleteErr   error
	casErr      error
	casApplied  *bool
	loadCalls   int
	deleteCalls int
	casCalls    int
}

func (s *managerTestStore) Load(context.Context, string) (VersionedState, error) {
	s.loadCalls++
	if s.loadErr != nil {
		return VersionedState{}, s.loadErr
	}
	if !s.found {
		return VersionedState{}, nil
	}
	return VersionedState{State: s.state.Clone(), Found: true}, nil
}

func (s *managerTestStore) CompareAndSwap(
	_ context.Context,
	_ string,
	expectedRevision uint64,
	next State,
	_ time.Duration,
	_ QuotaKey,
) (bool, error) {
	s.casCalls++
	if s.casErr != nil {
		return false, s.casErr
	}
	if s.casApplied != nil {
		if !*s.casApplied {
			return false, ErrRevisionMismatch
		}
	}
	if s.found && expectedRevision != s.state.Revision {
		return false, ErrRevisionMismatch
	}
	if !s.found && expectedRevision != 0 {
		return false, ErrRevisionMismatch
	}
	s.state = next.Clone()
	s.found = true
	return true, nil
}

func (s *managerTestStore) Delete(context.Context, string) error {
	s.deleteCalls++
	if s.deleteErr != nil {
		return s.deleteErr
	}
	s.found = false
	s.state = State{}
	return nil
}

func (s *managerTestStore) DeleteIfToken(ctx context.Context, key string, token StateToken) (bool, error) {
	if !s.found || s.state.Revision != token.Revision {
		return false, nil
	}
	if err := s.Delete(ctx, key); err != nil {
		return false, err
	}
	return true, nil
}

func (s *managerTestStore) Close() error { return nil }

func managerTestInput() SelectionInput {
	return SelectionInput{
		Enabled:               true,
		Trusted:               true,
		Key:                   "opaque-session-key",
		Quota:                 QuotaKey{Principal: "opaque-principal", Namespace: "recipe-a"},
		PolicyFingerprint:     "policy-v1",
		CatalogFingerprint:    "catalog-v1",
		CapabilityFingerprint: "capability-v1",
		Authorized: []ToolCandidate{
			{Name: "search", DefinitionFingerprint: "fp-search"},
			{Name: "lookup", DefinitionFingerprint: "fp-lookup"},
			{Name: "math", DefinitionFingerprint: "fp-math"},
		},
		Selected:           []ToolCandidate{{Name: "search", DefinitionFingerprint: "fp-search"}},
		Turn:               0,
		MaxTools:           3,
		MaxNewToolsPerTurn: 1,
		PinCalledTools:     true,
		StrategyID:         "semantic",
	}
}

func managerTestOptions(clock func() time.Time) ManagerOptions {
	return ManagerOptions{
		TTL:              time.Minute,
		MaxStateBytes:    16384,
		OperationTimeout: time.Second,
		MaxCASRetries:    2,
		Clock:            clock,
	}
}

func newManagerForTest(t *testing.T, store Store, clock func() time.Time) *Manager {
	t.Helper()
	manager, err := NewManager(store, managerTestOptions(clock))
	if err != nil {
		t.Fatal(err)
	}
	return manager
}

func managerToolNames(candidates []ToolCandidate) []string {
	names := make([]string, len(candidates))
	for index, candidate := range candidates {
		names[index] = candidate.Name
	}
	return names
}

func TestManagerSelect_DisabledAndUntrustedBypassStore(t *testing.T) {
	store := &managerTestStore{}
	clock := func() time.Time { return time.Unix(100, 0) }
	manager := newManagerForTest(t, store, clock)

	input := managerTestInput()
	input.Enabled = false
	got := manager.Select(context.Background(), input)
	if !reflect.DeepEqual(got.Selected, input.Selected) {
		t.Fatalf("disabled selection = %+v, want %+v", got.Selected, input.Selected)
	}
	if got.Receipt.Reason != SelectionReasonDisabled || got.Receipt.Fallback || store.loadCalls != 0 {
		t.Fatalf("disabled receipt/store calls = %+v/%d", got.Receipt, store.loadCalls)
	}

	input.Enabled = true
	input.Trusted = false
	got = manager.Select(context.Background(), input)
	if !reflect.DeepEqual(got.Selected, input.Selected) {
		t.Fatalf("untrusted selection = %+v, want %+v", got.Selected, input.Selected)
	}
	if got.Receipt.Reason != SelectionReasonUntrusted || !got.Receipt.Fallback || store.loadCalls != 0 {
		t.Fatalf("untrusted receipt/store calls = %+v/%d", got.Receipt, store.loadCalls)
	}
}

func TestManagerSelect_SeedsReusesAndGrowsDeterministically(t *testing.T) {
	store := &managerTestStore{}
	clock := func() time.Time { return time.Unix(100, 0) }
	manager := newManagerForTest(t, store, clock)
	input := managerTestInput()

	first := manager.Select(context.Background(), input)
	if !first.Receipt.Committed || first.Receipt.Reused {
		t.Fatalf("initial receipt = %+v", first.Receipt)
	}
	if want := []string{"search"}; !reflect.DeepEqual(managerToolNames(first.Selected), want) {
		t.Fatalf("initial tools = %v, want %v", managerToolNames(first.Selected), want)
	}

	input.Turn = 1
	input.Selected = []ToolCandidate{{Name: "lookup", DefinitionFingerprint: "fp-lookup"}}
	second := manager.Select(context.Background(), input)
	if !second.Receipt.Committed || !second.Receipt.Reused {
		t.Fatalf("reuse receipt = %+v", second.Receipt)
	}
	if want := []string{"search", "lookup"}; !reflect.DeepEqual(managerToolNames(second.Selected), want) {
		t.Fatalf("grown tools = %v, want %v", managerToolNames(second.Selected), want)
	}

	input.Turn = 2
	input.Selected = []ToolCandidate{{Name: "math", DefinitionFingerprint: "fp-math"}}
	third := manager.Select(context.Background(), input)
	if want := []string{"search", "lookup", "math"}; !reflect.DeepEqual(managerToolNames(third.Selected), want) {
		t.Fatalf("second growth = %v, want %v", managerToolNames(third.Selected), want)
	}
}

func TestManagerSelect_FingerprintMismatchInvalidatesAndSeedsFresh(t *testing.T) {
	store := &managerTestStore{}
	clock := func() time.Time { return time.Unix(100, 0) }
	manager := newManagerForTest(t, store, clock)
	input := managerTestInput()
	_ = manager.Select(context.Background(), input)

	input.Turn = 1
	input.CatalogFingerprint = "catalog-v2"
	input.Selected = []ToolCandidate{{Name: "lookup", DefinitionFingerprint: "fp-lookup"}}
	got := manager.Select(context.Background(), input)
	if !got.Receipt.Committed || !got.Receipt.Invalidated || got.Receipt.Reused {
		t.Fatalf("mismatch receipt = %+v", got.Receipt)
	}
	if got.Receipt.Reason != SelectionReasonCatalogChanged {
		t.Fatalf("mismatch reason = %q, want %q", got.Receipt.Reason, SelectionReasonCatalogChanged)
	}
	if want := []string{"lookup"}; !reflect.DeepEqual(managerToolNames(got.Selected), want) {
		t.Fatalf("fresh tools = %v, want %v", managerToolNames(got.Selected), want)
	}
	if store.deleteCalls != 1 {
		t.Fatalf("delete calls = %d, want 1", store.deleteCalls)
	}
}

func TestManagerSelect_ExpiredStateIsInvalidated(t *testing.T) {
	now := time.Unix(100, 0)
	store := &managerTestStore{
		found: true,
		state: State{
			SchemaVersion:         SchemaVersion,
			Revision:              1,
			PolicyFingerprint:     "policy-v1",
			CatalogFingerprint:    "catalog-v1",
			CapabilityFingerprint: "capability-v1",
			Tools:                 []ToolState{{Name: "search", DefinitionFingerprint: "fp-search"}},
			CreatedAt:             now.Add(-time.Minute),
			LastSeenAt:            now.Add(-time.Second),
			ExpiresAt:             now.Add(-time.Second),
		},
	}
	manager := newManagerForTest(t, store, func() time.Time { return now })
	got := manager.Select(context.Background(), managerTestInput())
	if !got.Receipt.Committed || !got.Receipt.Invalidated || got.Receipt.Reused {
		t.Fatalf("expired receipt = %+v", got.Receipt)
	}
	if got.Receipt.Reason != SelectionReasonStateExpired || store.deleteCalls != 1 {
		t.Fatalf("expired reason/delete = %q/%d", got.Receipt.Reason, store.deleteCalls)
	}
}

func TestManagerSelect_MemoryStoreExpiryReceipt(t *testing.T) {
	now := newSyntheticClock(time.Unix(100, 0))
	store := newTestStore(t, now, 100, 10, 1)
	options := managerTestOptions(now.Now)
	options.TTL = time.Second
	manager, err := NewManager(store, options)
	if err != nil {
		t.Fatal(err)
	}
	input := managerTestInput()
	if first := manager.Select(context.Background(), input); !first.Receipt.Committed {
		t.Fatalf("initial selection = %+v", first.Receipt)
	}

	now.Advance(2 * time.Second)
	input.Turn = 1
	got := manager.Select(context.Background(), input)
	if !got.Receipt.Committed || !got.Receipt.Invalidated || got.Receipt.Reason != SelectionReasonStateExpired {
		t.Fatalf("expired memory-store receipt = %+v", got.Receipt)
	}
}

func TestManagerSelect_CorruptAndUnavailableStateFailSafely(t *testing.T) {
	t.Run("corrupt state is deleted and reseeded", func(t *testing.T) {
		store := &managerTestStore{loadErr: ErrStateCorrupted}
		manager := newManagerForTest(t, store, func() time.Time { return time.Unix(100, 0) })
		got := manager.Select(context.Background(), managerTestInput())
		if !got.Receipt.Committed || !got.Receipt.Invalidated || got.Receipt.Reason != SelectionReasonStateCorrupted {
			t.Fatalf("corrupt receipt = %+v", got.Receipt)
		}
		if store.deleteCalls != 0 || store.casCalls != 1 {
			t.Fatalf("corrupt store calls = delete:%d cas:%d", store.deleteCalls, store.casCalls)
		}
	})

	t.Run("load error returns ordinary selection", func(t *testing.T) {
		store := &managerTestStore{loadErr: errors.New("backend unavailable")}
		manager := newManagerForTest(t, store, func() time.Time { return time.Unix(100, 0) })
		input := managerTestInput()
		got := manager.Select(context.Background(), input)
		if !reflect.DeepEqual(got.Selected, input.Selected) {
			t.Fatalf("fallback selection = %+v, want %+v", got.Selected, input.Selected)
		}
		if !got.Receipt.Fallback || got.Receipt.Reason != SelectionReasonStoreUnavailable || store.casCalls != 0 {
			t.Fatalf("load fallback = %+v, cas calls=%d", got.Receipt, store.casCalls)
		}
	})

	t.Run("delete error returns ordinary selection", func(t *testing.T) {
		store := &managerTestStore{
			found:     true,
			deleteErr: errors.New("delete unavailable"),
			state: State{
				SchemaVersion:         SchemaVersion,
				Revision:              1,
				PolicyFingerprint:     "old-policy",
				CatalogFingerprint:    "catalog-v1",
				CapabilityFingerprint: "capability-v1",
				CreatedAt:             time.Unix(1, 0),
				LastSeenAt:            time.Unix(2, 0),
				ExpiresAt:             time.Unix(1000, 0),
			},
		}
		manager := newManagerForTest(t, store, func() time.Time { return time.Unix(100, 0) })
		input := managerTestInput()
		got := manager.Select(context.Background(), input)
		if !got.Receipt.Fallback || got.Receipt.Reason != SelectionReasonStoreUnavailable {
			t.Fatalf("delete fallback = %+v", got.Receipt)
		}
	})
}

func TestManagerSelect_CASConflictRetriesAndEventuallyFallsBack(t *testing.T) {
	t.Run("reloads and remerges after conflict", func(t *testing.T) {
		input := managerTestInput()
		input.Turn = 1
		input.Selected = []ToolCandidate{{Name: "lookup", DefinitionFingerprint: "fp-lookup"}}
		// The first CAS conflicts. The scripted store then exposes a valid
		// state as if another request won the race before the retry's Load.
		scripted := &scriptedManagerStore{
			loads: []VersionedState{
				{Found: false},
				{Found: true, State: managerStoredState(input)},
			},
			casResults: []scriptedCASResult{{Err: ErrRevisionMismatch}, {Applied: true}},
		}
		manager := newManagerForTest(t, scripted, func() time.Time { return time.Unix(100, 0) })
		got := manager.Select(context.Background(), input)
		if !got.Receipt.Committed || got.Receipt.CASRetries != 1 || !got.Receipt.Reused {
			t.Fatalf("retry receipt = %+v", got.Receipt)
		}
		if want := []string{"search", "lookup"}; !reflect.DeepEqual(managerToolNames(got.Selected), want) {
			t.Fatalf("retry selection = %v, want %v", managerToolNames(got.Selected), want)
		}
	})

	t.Run("bounded exhaustion falls back", func(t *testing.T) {
		scripted := &scriptedManagerStore{
			loads: []VersionedState{{Found: false}, {Found: false}},
			casResults: []scriptedCASResult{
				{Err: ErrRevisionMismatch},
				{Err: ErrRevisionMismatch},
			},
		}
		options := managerTestOptions(func() time.Time { return time.Unix(100, 0) })
		options.MaxCASRetries = 1
		manager, err := NewManager(scripted, options)
		if err != nil {
			t.Fatal(err)
		}
		input := managerTestInput()
		got := manager.Select(context.Background(), input)
		if !got.Receipt.Fallback || got.Receipt.Reason != SelectionReasonCASConflict || got.Receipt.CASRetries != 1 {
			t.Fatalf("exhaustion receipt = %+v", got.Receipt)
		}
		if !reflect.DeepEqual(got.Selected, input.Selected) {
			t.Fatalf("exhaustion selection = %+v, want %+v", got.Selected, input.Selected)
		}
	})
}

func managerStoredState(input SelectionInput) State {
	now := time.Unix(100, 0)
	return State{
		SchemaVersion:         SchemaVersion,
		Revision:              1,
		PolicyFingerprint:     input.PolicyFingerprint,
		CatalogFingerprint:    input.CatalogFingerprint,
		CapabilityFingerprint: input.CapabilityFingerprint,
		Tools:                 []ToolState{{Name: "search", DefinitionFingerprint: "fp-search", FirstSeenTurn: 0}},
		CreatedAt:             now.Add(-time.Minute),
		LastSeenAt:            now,
		ExpiresAt:             now.Add(time.Minute),
	}
}

type scriptedCASResult struct {
	Applied bool
	Err     error
}

type scriptedManagerStore struct {
	loads      []VersionedState
	loadIndex  int
	casResults []scriptedCASResult
	casIndex   int
}

func (s *scriptedManagerStore) Load(context.Context, string) (VersionedState, error) {
	if s.loadIndex >= len(s.loads) {
		return VersionedState{}, nil
	}
	result := s.loads[s.loadIndex]
	s.loadIndex++
	return VersionedState{State: result.State.Clone(), Found: result.Found}, nil
}

func (s *scriptedManagerStore) CompareAndSwap(
	context.Context,
	string,
	uint64,
	State,
	time.Duration,
	QuotaKey,
) (bool, error) {
	if s.casIndex >= len(s.casResults) {
		return false, ErrRevisionMismatch
	}
	result := s.casResults[s.casIndex]
	s.casIndex++
	return result.Applied, result.Err
}

func (s *scriptedManagerStore) Delete(context.Context, string) error { return nil }

func (s *scriptedManagerStore) Close() error { return nil }
