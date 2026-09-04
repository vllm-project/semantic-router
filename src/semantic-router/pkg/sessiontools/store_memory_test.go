package sessiontools

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// syntheticClock lets tests advance time deterministically instead of
// relying on real sleeps.
type syntheticClock struct {
	mu  sync.Mutex
	now time.Time
}

func newSyntheticClock(start time.Time) *syntheticClock {
	return &syntheticClock{now: start}
}

func (c *syntheticClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *syntheticClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

func intPtr(v int) *int { return &v }

func newTestState(revision uint64) State {
	return State{
		SchemaVersion:      SchemaVersion,
		Revision:           revision,
		PolicyFingerprint:  "policy",
		CatalogFingerprint: "catalog",
		Tools: []ToolState{
			{Name: "search", DefinitionFingerprint: "fp", FirstSeenTurn: 0},
		},
	}
}

func newTestStore(t *testing.T, clock *syntheticClock, maxSessions, maxSessionsByIdentity, ttlSeconds int) *MemoryStore {
	t.Helper()
	cfg := config.ToolSessionStoreConfig{
		TTLSeconds:            intPtr(ttlSeconds),
		MaxSessions:           intPtr(maxSessions),
		MaxSessionsByIdentity: intPtr(maxSessionsByIdentity),
	}
	return NewMemoryStore(cfg, clock.Now)
}

func TestMemoryStore_Load_MissingKey(t *testing.T) {
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	got, err := store.Load(context.Background(), "missing")
	if err != nil {
		t.Fatal(err)
	}
	if got.Found {
		t.Fatalf("expected not found, got %+v", got)
	}
}

func TestMemoryStore_CompareAndSwap_CreateThenLoad(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	applied, err := store.CompareAndSwap(ctx, "sess-1", 0, newTestState(0), time.Minute, quota)
	if err != nil {
		t.Fatal(err)
	}
	if !applied {
		t.Fatal("expected creation to apply")
	}

	got, err := store.Load(ctx, "sess-1")
	if err != nil {
		t.Fatal(err)
	}
	if !got.Found {
		t.Fatal("expected the created session to be found")
	}
	if got.State.Revision != 1 {
		t.Fatalf("revision after creation = %d, want 1", got.State.Revision)
	}
}

func TestMemoryStore_CompareAndSwap_RevisionIncrementsOnUpdate(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	if _, err := store.CompareAndSwap(ctx, "sess-1", 0, newTestState(0), time.Minute, quota); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load(ctx, "sess-1")
	if err != nil || !loaded.Found {
		t.Fatalf("load after create: found=%v err=%v", loaded.Found, err)
	}
	if loaded.State.Revision != 1 {
		t.Fatalf("revision = %d, want 1", loaded.State.Revision)
	}

	applied, err := store.CompareAndSwap(ctx, "sess-1", loaded.State.Revision, newTestState(0), time.Minute, quota)
	if err != nil {
		t.Fatal(err)
	}
	if !applied {
		t.Fatal("expected the second CAS to apply")
	}
	loaded2, err := store.Load(ctx, "sess-1")
	if err != nil || !loaded2.Found {
		t.Fatalf("load after update: found=%v err=%v", loaded2.Found, err)
	}
	if loaded2.State.Revision != 2 {
		t.Fatalf("revision after update = %d, want 2", loaded2.State.Revision)
	}
}

func TestMemoryStore_CompareAndSwap_RevisionMismatch(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	if _, err := store.CompareAndSwap(ctx, "sess-1", 0, newTestState(0), time.Minute, quota); err != nil {
		t.Fatal(err)
	}

	applied, err := store.CompareAndSwap(ctx, "sess-1", 99, newTestState(0), time.Minute, quota)
	if applied {
		t.Fatal("expected the mismatched CAS not to apply")
	}
	if !errors.Is(err, ErrRevisionMismatch) {
		t.Fatalf("err = %v, want ErrRevisionMismatch", err)
	}
}

func TestMemoryStore_CompareAndSwap_CreateWithNonZeroRevisionIsMismatch(t *testing.T) {
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	applied, err := store.CompareAndSwap(context.Background(), "sess-new", 5, newTestState(0), time.Minute, QuotaKey{})
	if applied {
		t.Fatal("expected creation with a non-zero expectedRevision to be rejected")
	}
	if !errors.Is(err, ErrRevisionMismatch) {
		t.Fatalf("err = %v, want ErrRevisionMismatch", err)
	}
}

func TestMemoryStore_CopyOnReadAndWrite(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	original := newTestState(0)
	if _, err := store.CompareAndSwap(ctx, "sess-1", 0, original, time.Minute, quota); err != nil {
		t.Fatal(err)
	}
	// Mutating the caller's own copy after the call must not affect what
	// is stored.
	original.Tools[0].Name = "mutated-after-write"

	loadedOnce, err := store.Load(ctx, "sess-1")
	if err != nil || !loadedOnce.Found {
		t.Fatalf("load: found=%v err=%v", loadedOnce.Found, err)
	}
	if loadedOnce.State.Tools[0].Name == "mutated-after-write" {
		t.Fatal("CompareAndSwap must copy next, not alias the caller's slice")
	}

	// Mutating what one Load call returned must not affect a second Load.
	loadedOnce.State.Tools[0].Name = "mutated-after-read"
	loadedTwice, err := store.Load(ctx, "sess-1")
	if err != nil || !loadedTwice.Found {
		t.Fatalf("second load: found=%v err=%v", loadedTwice.Found, err)
	}
	if loadedTwice.State.Tools[0].Name == "mutated-after-read" {
		t.Fatal("Load must return a copy, not a pointer into the stored state")
	}
}

func TestMemoryStore_TTLExpiration_SyntheticClock(t *testing.T) {
	ctx := context.Background()
	clock := newSyntheticClock(time.Now())
	store := newTestStore(t, clock, 100, 10, 60) // 60s TTL
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	if _, err := store.CompareAndSwap(ctx, "sess-1", 0, newTestState(0), 60*time.Second, quota); err != nil {
		t.Fatal(err)
	}
	if got, err := store.Load(ctx, "sess-1"); err != nil || !got.Found {
		t.Fatalf("expected found before expiry: found=%v err=%v", got.Found, err)
	}

	clock.Advance(61 * time.Second)

	got, err := store.Load(ctx, "sess-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.Found {
		t.Fatal("expected the session to be expired after the TTL elapsed")
	}
}

func TestMemoryStore_TTLSlidesOnLoad(t *testing.T) {
	ctx := context.Background()
	clock := newSyntheticClock(time.Now())
	store := newTestStore(t, clock, 100, 10, 60)
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	if _, err := store.CompareAndSwap(ctx, "sess-1", 0, newTestState(0), 60*time.Second, quota); err != nil {
		t.Fatal(err)
	}

	// Touch the session just before it would otherwise expire; sliding
	// expiry should push the deadline out again.
	clock.Advance(50 * time.Second)
	if got, err := store.Load(ctx, "sess-1"); err != nil || !got.Found {
		t.Fatalf("expected found at 50s: found=%v err=%v", got.Found, err)
	}

	clock.Advance(50 * time.Second) // total 100s since creation, but only 50s since the Load refresh
	got, err := store.Load(ctx, "sess-1")
	if err != nil {
		t.Fatal(err)
	}
	if !got.Found {
		t.Fatal("expected the session to still be alive: Load should have slid its expiry")
	}
}

// TestMemoryStore_ExpiredReadmission_ABARace guards against the ABA race
// deleteIfExpiredLocked/compareAndSwapCreate exist to prevent: a stale
// "this entry looked expired" observation (from Load or a losing
// CompareAndSwap(0) racer) must never delete a fresh, live entry that a
// concurrent CompareAndSwap(0) admitted at the same key in the meantime.
// Repeated across many attempts with real goroutine concurrency (not just
// -race's happens-before tracking) because the race window is narrow and
// timing-dependent.
func TestMemoryStore_ExpiredReadmission_ABARace(t *testing.T) {
	const attempts = 200
	const casWorkers = 64
	const loadWorkers = 64

	ctx := context.Background()
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	for attempt := 0; attempt < attempts; attempt++ {
		clock := newSyntheticClock(time.Unix(int64(attempt+1), 0))
		store := newTestStore(t, clock, 1000, 1000, 60)
		key := fmt.Sprintf("aba-race-%d", attempt)

		initial := newTestState(0)
		initial.PolicyFingerprint = "expired"
		if applied, err := store.CompareAndSwap(ctx, key, 0, initial, time.Second, quota); err != nil || !applied {
			t.Fatalf("attempt %d: initial create applied=%v err=%v", attempt, applied, err)
		}

		clock.Advance(2 * time.Second)

		start := make(chan struct{})
		var wg sync.WaitGroup
		var mu sync.Mutex
		appliedCAS := 0
		errs := make(chan error, casWorkers+loadWorkers)

		for i := 0; i < casWorkers; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				<-start

				next := newTestState(0)
				next.PolicyFingerprint = fmt.Sprintf("fresh-%d-%d", attempt, i)
				ok, err := store.CompareAndSwap(ctx, key, 0, next, time.Minute, quota)
				if err != nil && !errors.Is(err, ErrRevisionMismatch) {
					errs <- fmt.Errorf("cas worker %d: unexpected err: %w", i, err)
					return
				}
				if ok {
					mu.Lock()
					appliedCAS++
					mu.Unlock()
				}
			}(i)
		}

		for i := 0; i < loadWorkers; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				<-start

				if _, err := store.Load(ctx, key); err != nil {
					errs <- fmt.Errorf("load worker %d: unexpected err: %w", i, err)
				}
			}(i)
		}

		close(start)
		wg.Wait()
		close(errs)

		for err := range errs {
			t.Fatal(err)
		}

		if appliedCAS != 1 {
			t.Fatalf("attempt %d: CAS(0) successes = %d, want exactly 1", attempt, appliedCAS)
		}

		final, err := store.Load(ctx, key)
		if err != nil {
			t.Fatalf("attempt %d: final load err: %v", attempt, err)
		}
		if !final.Found {
			t.Fatalf("attempt %d: fresh state was pruned by expired cleanup", attempt)
		}
		if final.State.Revision != 1 {
			t.Fatalf("attempt %d: final revision = %d, want 1", attempt, final.State.Revision)
		}
		if final.State.PolicyFingerprint == "expired" {
			t.Fatalf("attempt %d: expired state survived readmission", attempt)
		}
	}
}

func TestMemoryStore_GlobalCapacityEviction(t *testing.T) {
	ctx := context.Background()
	clock := newSyntheticClock(time.Now())
	store := newTestStore(t, clock, 3, 100, 1800) // global cap of 3, generous per-identity cap

	for i := 0; i < 3; i++ {
		key := fmt.Sprintf("sess-%d", i)
		quota := QuotaKey{Principal: fmt.Sprintf("user-%d", i), Namespace: "recipe-a"}
		if _, err := store.CompareAndSwap(ctx, key, 0, newTestState(0), time.Hour, quota); err != nil {
			t.Fatal(err)
		}
		clock.Advance(time.Second) // distinct LastSeenAt per session
	}

	// Store is now at the global cap of 3. Admitting a 4th must evict the
	// least-recently-seen (sess-0) to make room.
	quota3 := QuotaKey{Principal: "user-3", Namespace: "recipe-a"}
	if _, err := store.CompareAndSwap(ctx, "sess-3", 0, newTestState(0), time.Hour, quota3); err != nil {
		t.Fatal(err)
	}

	if got, err := store.Load(ctx, "sess-0"); err != nil || got.Found {
		t.Fatalf("expected sess-0 (oldest) to be evicted: found=%v err=%v", got.Found, err)
	}
	if got, err := store.Load(ctx, "sess-3"); err != nil || !got.Found {
		t.Fatalf("expected the newly admitted sess-3 to be present: found=%v err=%v", got.Found, err)
	}
}

func TestMemoryStore_PerIdentityCapacityEviction(t *testing.T) {
	ctx := context.Background()
	clock := newSyntheticClock(time.Now())
	store := newTestStore(t, clock, 100, 2, 1800) // per-identity cap of 2, generous global cap
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}
	other := QuotaKey{Principal: "user-2", Namespace: "recipe-a"}

	if _, err := store.CompareAndSwap(ctx, "u1-sess-a", 0, newTestState(0), time.Hour, quota); err != nil {
		t.Fatal(err)
	}
	clock.Advance(time.Second)
	if _, err := store.CompareAndSwap(ctx, "u1-sess-b", 0, newTestState(0), time.Hour, quota); err != nil {
		t.Fatal(err)
	}
	clock.Advance(time.Second)
	// A different principal's own session must not count against user-1's
	// quota, and must not be evicted by user-1's admissions.
	if _, err := store.CompareAndSwap(ctx, "u2-sess-a", 0, newTestState(0), time.Hour, other); err != nil {
		t.Fatal(err)
	}
	clock.Advance(time.Second)

	// user-1 is now at their cap of 2; a third session for user-1 must
	// evict user-1's oldest (u1-sess-a), not touch user-2's session.
	if _, err := store.CompareAndSwap(ctx, "u1-sess-c", 0, newTestState(0), time.Hour, quota); err != nil {
		t.Fatal(err)
	}

	if got, err := store.Load(ctx, "u1-sess-a"); err != nil || got.Found {
		t.Fatalf("expected user-1's oldest session to be evicted: found=%v err=%v", got.Found, err)
	}
	if got, err := store.Load(ctx, "u1-sess-b"); err != nil || !got.Found {
		t.Fatalf("expected user-1's newer session to survive: found=%v err=%v", got.Found, err)
	}
	if got, err := store.Load(ctx, "u2-sess-a"); err != nil || !got.Found {
		t.Fatalf("expected user-2's session to be unaffected by user-1's quota: found=%v err=%v", got.Found, err)
	}
}

func TestMemoryStore_ConcurrentCAS_DistinctKeys(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 1000, 1000, 1800)

	const goroutines = 50
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("sess-%d", i)
			quota := QuotaKey{Principal: fmt.Sprintf("user-%d", i), Namespace: "recipe-a"}
			applied, err := store.CompareAndSwap(ctx, key, 0, newTestState(0), time.Hour, quota)
			if err != nil {
				errs <- err
				return
			}
			if !applied {
				errs <- fmt.Errorf("key %s: creation did not apply", key)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	for i := 0; i < goroutines; i++ {
		key := fmt.Sprintf("sess-%d", i)
		got, err := store.Load(ctx, key)
		if err != nil || !got.Found {
			t.Fatalf("%s: found=%v err=%v", key, got.Found, err)
		}
	}
}

func TestMemoryStore_ConcurrentCAS_SameKeyExactlyOneWinner(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 100, 1800)
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	const goroutines = 20
	var wg sync.WaitGroup
	var applied int32
	var mu sync.Mutex
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ok, err := store.CompareAndSwap(ctx, "contended", 0, newTestState(0), time.Hour, quota)
			if err != nil && !errors.Is(err, ErrRevisionMismatch) {
				t.Errorf("unexpected error: %v", err)
			}
			if ok {
				mu.Lock()
				applied++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if applied != 1 {
		t.Fatalf("exactly one concurrent creation should apply, got %d", applied)
	}
}

func TestMemoryStore_Delete(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	quota := QuotaKey{Principal: "user-1", Namespace: "recipe-a"}

	if _, err := store.CompareAndSwap(ctx, "sess-1", 0, newTestState(0), time.Hour, quota); err != nil {
		t.Fatal(err)
	}
	if err := store.Delete(ctx, "sess-1"); err != nil {
		t.Fatal(err)
	}
	if got, err := store.Load(ctx, "sess-1"); err != nil || got.Found {
		t.Fatalf("expected deleted session to be gone: found=%v err=%v", got.Found, err)
	}
	// Deleting an already-absent key is idempotent, not an error.
	if err := store.Delete(ctx, "sess-1"); err != nil {
		t.Fatal(err)
	}

	// Deleting must free the identity's quota slot for a fresh admission.
	if _, err := store.CompareAndSwap(ctx, "sess-2", 0, newTestState(0), time.Hour, quota); err != nil {
		t.Fatal(err)
	}
}

func TestMemoryStore_ClosedStoreRejectsOperations(t *testing.T) {
	ctx := context.Background()
	store := newTestStore(t, newSyntheticClock(time.Now()), 100, 10, 1800)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	// Close is idempotent.
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := store.Load(ctx, "sess-1"); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("Load after Close: err = %v, want ErrStoreClosed", err)
	}
	if _, err := store.CompareAndSwap(ctx, "sess-1", 0, newTestState(0), time.Hour, QuotaKey{}); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("CompareAndSwap after Close: err = %v, want ErrStoreClosed", err)
	}
	if err := store.Delete(ctx, "sess-1"); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("Delete after Close: err = %v, want ErrStoreClosed", err)
	}
}
