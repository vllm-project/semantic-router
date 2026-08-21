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

	snapshot.Version = expectedVersion + 1
	f.snapshot = snapshot
	f.found = true
	f.saved++
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

	snapshot.Version = expectedVersion + 1
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

func TestRouterSessionStateStoreRejectsStaleVersion(t *testing.T) {
	store := &fakeRouterSessionStateStore{}

	initial := RouterSessionSnapshot{
		SessionID:    "session-a",
		Version:      0,
		CurrentModel: "model-a",
		TurnCount:    1,
	}

	ok, err := store.SaveIfVersion(initial, 0, time.Hour)
	if err != nil {
		t.Fatalf("initial save failed: %v", err)
	}
	if !ok {
		t.Fatal("initial save was rejected")
	}

	first, found, err := store.Load("session-a")
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	if !found {
		t.Fatal("initial snapshot was not stored")
	}
	if first.Version != 1 {
		t.Fatalf("version = %d, want 1", first.Version)
	}

	// Simulate another router writing the next version.
	second := first
	second.CurrentModel = "model-b"
	second.TurnCount = 2

	ok, err = store.SaveIfVersion(second, 1, time.Hour)
	if err != nil {
		t.Fatalf("second save failed: %v", err)
	}
	if !ok {
		t.Fatal("second save was unexpectedly rejected")
	}

	// This writer is stale: it still believes Redis is at version 1.
	stale := first
	stale.CurrentModel = "model-stale"
	stale.TurnCount = 99

	ok, err = store.SaveIfVersion(stale, 1, time.Hour)
	if err != nil {
		t.Fatalf("stale save returned error: %v", err)
	}
	if ok {
		t.Fatal("stale save was accepted")
	}

	final, found, err := store.Load("session-a")
	if err != nil {
		t.Fatalf("final load failed: %v", err)
	}
	if !found {
		t.Fatal("final snapshot disappeared")
	}

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

func TestRedisRouterSessionStateStoreConcurrentCAS(t *testing.T) {
	address := os.Getenv("REDIS_ROUTER_SESSION_TEST_ADDR")
	if address == "" {
		t.Skip("REDIS_ROUTER_SESSION_TEST_ADDR is not set")
	}

	client := redis.NewClient(&redis.Options{
		Addr: address,
	})
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

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
	defer store.Close()

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

	ok, err := store.SaveIfVersion(initial, 0, time.Minute)
	if err != nil {
		t.Fatalf("initial CAS failed: %v", err)
	}
	if !ok {
		t.Fatal("initial CAS was rejected")
	}

	first, found, err := store.Load(sessionID)
	if err != nil {
		t.Fatalf("loading initial snapshot: %v", err)
	}
	if !found {
		t.Fatal("initial snapshot was not stored")
	}
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

		ok, err := store.SaveIfVersion(snapshot, 1, time.Minute)
		results <- result{ok: ok, err: err}
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
		t.Fatalf(
			"successful CAS operations = %d, want exactly 1",
			successes,
		)
	}

	final, found, err := store.Load(sessionID)
	if err != nil {
		t.Fatalf("loading final snapshot: %v", err)
	}
	if !found {
		t.Fatal("final snapshot disappeared")
	}

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

	ok, err := store.SaveIfVersion(initial, 0, time.Minute)
	if err != nil {
		t.Fatalf("initial CAS failed: %v", err)
	}
	if !ok {
		t.Fatal("initial CAS was rejected")
	}

	base, found, err := store.Load("reconcile-session")
	if err != nil {
		t.Fatalf("loading initial snapshot: %v", err)
	}
	if !found {
		t.Fatal("initial snapshot was not stored")
	}
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
	ok, err = store.SaveIfVersion(replicaA, 1, time.Minute)
	if err != nil {
		t.Fatalf("replica A CAS failed: %v", err)
	}
	if !ok {
		t.Fatal("replica A CAS was unexpectedly rejected")
	}

	// Replica B is stale because it still expects version 1.
	ok, err = store.SaveIfVersion(replicaB, 1, time.Minute)
	if err != nil {
		t.Fatalf("replica B stale CAS returned error: %v", err)
	}
	if ok {
		t.Fatal("replica B stale CAS was unexpectedly accepted")
	}

	// Reconcile:
	// reload the latest shared snapshot and reapply only Replica B's
	// local event rather than overwriting the newer shared state.
	latest, found, err := store.Load("reconcile-session")
	if err != nil {
		t.Fatalf("loading latest snapshot for reconciliation: %v", err)
	}
	if !found {
		t.Fatal("latest snapshot disappeared during reconciliation")
	}

	if latest.Version != 2 {
		t.Fatalf("latest version = %d, want 2", latest.Version)
	}
	if latest.TurnCount != 2 {
		t.Fatalf("latest turn count = %d, want 2", latest.TurnCount)
	}

	reconciled := latest
	reconciled.TurnCount++
	reconciled.CurrentModel = replicaB.CurrentModel

	ok, err = store.SaveIfVersion(reconciled, latest.Version, time.Minute)
	if err != nil {
		t.Fatalf("reconciled CAS failed: %v", err)
	}
	if !ok {
		t.Fatal("reconciled CAS was unexpectedly rejected")
	}

	final, found, err := store.Load("reconcile-session")
	if err != nil {
		t.Fatalf("loading final snapshot: %v", err)
	}
	if !found {
		t.Fatal("final snapshot disappeared")
	}

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
