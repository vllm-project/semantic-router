package sessiontelemetry

import (
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
)

type fakeRouterSessionStateStore struct {
	snapshot RouterSessionSnapshot
	found    bool
	saved    int
}

func (f *fakeRouterSessionStateStore) Load(string) (RouterSessionSnapshot, bool, error) {
	return f.snapshot, f.found, nil
}

func (f *fakeRouterSessionStateStore) SaveIfVersion(
	snapshot RouterSessionSnapshot,
	expectedVersion uint64,
	_ time.Duration,
) (bool, error) {
	if f.found && f.snapshot.Version != expectedVersion {
		return false, nil
	}

	f.found = true
	f.saved++
	next := snapshot
	next.Version = expectedVersion + 1

	f.snapshot = next
	return true, nil
}

func (f *fakeRouterSessionStateStore) Close() error { return nil }

type blockingRouterSessionStateStore struct {
	saveStarted     chan struct{}
	allowSave       chan struct{}
	startOnce       sync.Once
	activeSave      atomic.Bool
	closeDuringSave atomic.Bool
	closeCalls      atomic.Int32
}

func (s *blockingRouterSessionStateStore) Load(string) (RouterSessionSnapshot, bool, error) {
	return RouterSessionSnapshot{}, false, nil
}

func (s *blockingRouterSessionStateStore) SaveIfVersion(
	snapshot RouterSessionSnapshot,
	expectedVersion uint64,
	ttl time.Duration,
) (bool, error) {
	s.activeSave.Store(true)
	defer s.activeSave.Store(false)

	s.startOnce.Do(func() {
		close(s.saveStarted)
	})

	<-s.allowSave

	return true, nil
}

func (s *blockingRouterSessionStateStore) Close() error {
	if s.activeSave.Load() {
		s.closeDuringSave.Store(true)
	}
	s.closeCalls.Add(1)
	return nil
}

func TestRouterSessionStateStoreRestoresAfterLocalReset(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	store := &fakeRouterSessionStateStore{}
	SetRouterSessionStateStore(store)
	t.Cleanup(func() {
		SetRouterSessionStateStore(nil)
		ResetRouterSessionMemoryForTesting()
	})

	RecordSessionDecision(SessionDecisionParams{
		SessionID:     "session-a",
		SelectedModel: "model-a",
		Timestamp:     time.Now(),
	})
	if store.saved == 0 {
		t.Fatal("shared store did not receive session snapshot")
	}

	ResetRouterSessionMemoryForTesting()
	snapshot, ok := GetRouterSessionSnapshot("session-a", time.Now())
	if !ok {
		t.Fatal("shared session snapshot was not restored")
	}
	if snapshot.CurrentModel != "model-a" {
		t.Fatalf("restored model = %q", snapshot.CurrentModel)
	}
}

func TestPublishedStoreOperationLeaseSurvivesDoubleSwap(t *testing.T) {
	ResetRouterSessionMemoryForTesting()
	firstSlot := NewRouterSessionStateStoreSlot(&fakeRouterSessionStateStore{})
	secondStore := &blockingRouterSessionStateStore{
		saveStarted: make(chan struct{}),
		allowSave:   make(chan struct{}),
	}
	secondSlot := NewRouterSessionStateStoreSlot(secondStore)
	thirdSlot := NewRouterSessionStateStoreSlot(&fakeRouterSessionStateStore{})
	var releaseSave sync.Once
	t.Cleanup(func() {
		releaseSave.Do(func() {
			close(secondStore.allowSave)
		})
		PublishRouterSessionStateStore(nil)
		_ = firstSlot.RetireAndClose()
		_ = secondSlot.RetireAndClose()
		_ = thirdSlot.RetireAndClose()
		ResetRouterSessionMemoryForTesting()
	})

	// G1 is still active when G1 -> G2 publishes S2. Its next persistence
	// operation therefore leases S2 and deliberately blocks inside Save.
	PublishRouterSessionStateStore(firstSlot)
	PublishRouterSessionStateStore(secondSlot)
	saveDone := make(chan struct{})
	go func() {
		RecordSessionDecision(SessionDecisionParams{
			SessionID:     "g1-long-stream",
			SelectedModel: "model-a",
			Timestamp:     time.Now(),
		})
		close(saveDone)
	}()
	waitForSignal(t, secondStore.saveStarted, "S2 Save did not start")

	// G2 -> G3 retires S2 even though G2 itself has no request leases. S2's
	// operation lease must keep Close blocked until the G1 operation returns.
	PublishRouterSessionStateStore(thirdSlot)
	retireDone := make(chan error, 1)
	go func() {
		retireDone <- secondSlot.RetireAndClose()
	}()
	waitForSignal(t, secondSlot.retiredCh, "S2 retirement did not start")
	if got := secondStore.closeCalls.Load(); got != 0 {
		t.Fatalf("S2 close calls while Save is leased = %d, want 0", got)
	}
	if _, release, acquired := secondSlot.acquire(); acquired {
		release()
		t.Fatal("retired S2 accepted a new operation lease")
	}

	releaseSave.Do(func() {
		close(secondStore.allowSave)
	})
	waitForSignal(t, saveDone, "S2 Save did not finish")
	select {
	case err := <-retireDone:
		if err != nil {
			t.Fatalf("S2 retirement error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("S2 retirement did not finish")
	}
	if got := secondStore.closeCalls.Load(); got != 1 {
		t.Fatalf("S2 close calls after Save release = %d, want 1", got)
	}
	if secondStore.closeDuringSave.Load() {
		t.Fatal("S2 was closed while its Save operation was still active")
	}
}

func waitForSignal(t *testing.T, signal <-chan struct{}, failure string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatal(failure)
	}
}

func requireSnapshot(
	t *testing.T,
	store RouterSessionStateStore,
	sessionID string,
) RouterSessionSnapshot {
	t.Helper()

	snapshot, found, err := store.Load(sessionID)
	if err != nil {
		t.Fatalf("loading snapshot: %v", err)
	}
	if !found {
		t.Fatal("snapshot was not stored")
	}

	return snapshot
}

func requireCASSuccess(
	t *testing.T,
	store RouterSessionStateStore,
	snapshot RouterSessionSnapshot,
	expectedVersion uint64,
) {
	t.Helper()

	ok, err := store.SaveIfVersion(snapshot, expectedVersion, time.Minute)
	if err != nil {
		t.Fatalf("CAS failed: %v", err)
	}
	if !ok {
		t.Fatal("CAS was rejected")
	}
}

func requireCASRejected(
	t *testing.T,
	store RouterSessionStateStore,
	snapshot RouterSessionSnapshot,
	expectedVersion uint64,
) {
	t.Helper()

	ok, err := store.SaveIfVersion(snapshot, expectedVersion, time.Minute)
	if err != nil {
		t.Fatalf("CAS returned error: %v", err)
	}
	if ok {
		t.Fatal("stale CAS was accepted")
	}
}

func TestRouterSessionStateStoreRejectsStaleVersion(t *testing.T) {
	store := &fakeRouterSessionStateStore{}

	initial := RouterSessionSnapshot{
		SessionID:    "session-a",
		Version:      0,
		CurrentModel: "model-a",
		TurnCount:    1,
	}

	requireCASSuccess(t, store, initial, 0)

	first := requireSnapshot(t, store, "session-a")
	if first.Version != 1 {
		t.Fatalf("version = %d, want 1", first.Version)
	}

	// Simulate another router writing the next version.
	second := first
	second.CurrentModel = "model-b"
	second.TurnCount = 2

	requireCASSuccess(t, store, second, 1)

	// This writer is stale: it still believes Redis is at version 1.
	stale := first
	stale.CurrentModel = "model-stale"
	stale.TurnCount = 99

	requireCASRejected(t, store, stale, 1)

	final := requireSnapshot(t, store, "session-a")

	if final.Version != 2 {
		t.Fatalf("final version = %d, want 2", final.Version)
	}
	if final.CurrentModel != "model-b" {
		t.Fatalf(
			"stale writer overwrote current model: got %q, want %q",
			final.CurrentModel,
			"model-b",
		)
	}
	if final.TurnCount != 2 {
		t.Fatalf("stale writer overwrote turn count: got %d, want 2", final.TurnCount)
	}
}

func newRedisRouterSessionTestStore(t *testing.T) (
	RouterSessionStateStore,
	*redis.Client,
	context.Context,
	RouterSessionSnapshot,
) {
	t.Helper()

	address := os.Getenv("REDIS_ROUTER_SESSION_TEST_ADDR")
	if address == "" {
		t.Skip("REDIS_ROUTER_SESSION_TEST_ADDR is not set")
	}

	client := redis.NewClient(&redis.Options{
		Addr: address,
	})
	t.Cleanup(func() {
		_ = client.Close()
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	t.Cleanup(cancel)

	if err := client.Ping(ctx).Err(); err != nil {
		t.Skipf("Redis is not available at %s: %v", address, err)
	}

	prefix := fmt.Sprintf(
		"test:router-session-cas:%d:",
		time.Now().UnixNano(),
	)

	store, err := NewRedisRouterSessionStateStore(
		RedisRouterSessionStoreConfig{
			Address:   address,
			Timeout:   time.Second,
			KeyPrefix: prefix,
		},
	)
	if err != nil {
		t.Fatalf("creating Redis store: %v", err)
	}
	t.Cleanup(func() {
		_ = store.Close()
	})

	sessionID := "concurrent-session"
	ctxKey := prefix + sessionID

	if err := client.Del(ctx, ctxKey).Err(); err != nil {
		t.Fatalf("clearing test key: %v", err)
	}

	initial := RouterSessionSnapshot{
		SessionID:    sessionID,
		Version:      0,
		CurrentModel: "initial",
		TurnCount:    1,
	}

	return store, client, ctx, initial
}

func TestRedisRouterSessionStateStoreConcurrentCAS(t *testing.T) {
	store, _, _, initial := newRedisRouterSessionTestStore(t)

	requireCASSuccess(t, store, initial, 0)

	first := requireSnapshot(t, store, initial.SessionID)
	if first.Version != 1 {
		t.Fatalf("initial version = %d, want 1", first.Version)
	}

	var wg sync.WaitGroup

	type result struct {
		ok  bool
		err error
	}

	results := make(chan result, 2)

	write := func(model string) {
		defer wg.Done()

		snapshot := first
		snapshot.CurrentModel = model
		snapshot.TurnCount++

		saveOK, saveErr := store.SaveIfVersion(snapshot, 1, time.Minute)
		results <- result{ok: saveOK, err: saveErr}
	}

	wg.Add(2)
	go write("model-a")
	go write("model-b")
	wg.Wait()
	close(results)

	successes := 0
	for result := range results {
		if result.err != nil {
			t.Fatalf("concurrent CAS returned error: %v", result.err)
		}
		if result.ok {
			successes++
		}
	}

	if successes != 1 {
		t.Fatalf("successful CAS operations = %d, want exactly 1", successes)
	}

	final := requireSnapshot(t, store, first.SessionID)

	if final.Version != 2 {
		t.Fatalf("final version = %d, want 2", final.Version)
	}

	if final.CurrentModel != "model-a" &&
		final.CurrentModel != "model-b" {
		t.Fatalf("unexpected final model = %q", final.CurrentModel)
	}

	if final.TurnCount != 2 {
		t.Fatalf("final turn count = %d, want 2", final.TurnCount)
	}
}

func TestRouterSessionStateStoreReconcileStaleWriter(t *testing.T) {
	store := &fakeRouterSessionStateStore{}

	// Both replicas begin from the same snapshot.
	initial := RouterSessionSnapshot{
		SessionID:    "reconcile-session",
		Version:      0,
		CurrentModel: "model-a",
		TurnCount:    1,
	}

	requireCASSuccess(t, store, initial, 0)

	base := requireSnapshot(t, store, initial.SessionID)
	if base.Version != 1 {
		t.Fatalf("initial version = %d, want 1", base.Version)
	}

	// Replica A and Replica B both observe version 1.
	replicaA := base
	replicaB := base

	// Each replica processes one independent turn.
	replicaA.TurnCount++
	replicaA.CurrentModel = "model-a"

	replicaB.TurnCount++
	replicaB.CurrentModel = "model-b"

	// Replica A wins the race and advances Redis to version 2.
	requireCASSuccess(t, store, replicaA, 1)

	// Replica B is stale because it still expects version 1.
	requireCASRejected(t, store, replicaB, 1)

	// Reconcile:
	// reload the latest shared snapshot and reapply only Replica B's
	// local event rather than overwriting the newer shared state.
	latest := requireSnapshot(t, store, "reconcile-session")

	if latest.Version != 2 {
		t.Fatalf("latest version = %d, want 2", latest.Version)
	}
	if latest.TurnCount != 2 {
		t.Fatalf("latest turn count = %d, want 2", latest.TurnCount)
	}

	reconciled := latest
	reconciled.TurnCount++
	reconciled.CurrentModel = replicaB.CurrentModel

	requireCASSuccess(t, store, reconciled, latest.Version)

	final := requireSnapshot(t, store, "reconcile-session")

	if final.Version != 3 {
		t.Fatalf("final version = %d, want 3", final.Version)
	}

	if final.TurnCount != 3 {
		t.Fatalf(
			"final turn count = %d, want 3; one replica's event was lost",
			final.TurnCount,
		)
	}

	if final.CurrentModel != "model-b" {
		t.Fatalf(
			"final current model = %q, want %q",
			final.CurrentModel,
			"model-b",
		)
	}
}
