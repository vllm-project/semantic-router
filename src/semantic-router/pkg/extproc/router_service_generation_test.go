package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestRouterServiceSwapPublishesBeforeRetiredGenerationDrains(t *testing.T) {
	storage := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	oldRouter := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage)}
	service := NewRouterService(oldRouter)
	generation := service.current.Load()
	generation.refs.Add(1)

	swapped := make(chan struct{})
	go func() {
		_ = service.Swap(&OpenAIRouter{}, nil)
		close(swapped)
	}()

	select {
	case <-swapped:
	case <-time.After(time.Second):
		t.Fatal("Swap() blocked management publication on a retired stream")
	}
	if storage.closeCalls.Load() != 0 {
		t.Fatal("retired replay store closed while a stream still held the router")
	}

	generation.refs.Done()
	deadline := time.Now().Add(time.Second)
	for storage.closeCalls.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if storage.closeCalls.Load() != 1 {
		t.Fatalf("retired replay store close calls = %d, want 1", storage.closeCalls.Load())
	}
}

func TestRouterServiceCloseWaitsForAllRetiredGenerationsAndRejectsReload(t *testing.T) {
	firstStore := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	secondStore := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	first := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(firstStore)}
	second := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(secondStore)}
	service := NewRouterService(first)
	firstGeneration := service.current.Load()
	firstGeneration.refs.Add(1)

	if err := service.Swap(second, nil); err != nil {
		t.Fatalf("Swap() error = %v", err)
	}
	closed := make(chan struct{})
	go func() {
		_ = service.Close()
		close(closed)
	}()
	select {
	case <-closed:
		t.Fatal("Close() returned before an older generation drained")
	case <-time.After(25 * time.Millisecond):
	}

	rejectedStore := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	rejected := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(rejectedStore)}
	if err := service.Swap(rejected, nil); err == nil {
		t.Fatal("Swap() accepted a router after shutdown began")
	}
	if rejectedStore.closeCalls.Load() != 1 {
		t.Fatalf("rejected router close calls = %d, want 1", rejectedStore.closeCalls.Load())
	}

	firstGeneration.refs.Done()
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("Close() did not wait for every retired generation")
	}
	if firstStore.closeCalls.Load() != 1 || secondStore.closeCalls.Load() != 1 {
		t.Fatalf("generation close calls = %d / %d, want 1 / 1", firstStore.closeCalls.Load(), secondStore.closeCalls.Load())
	}
}

func TestRouterServiceSwapWaitsForLeasedManagementRuntimeBeforeClosingStore(t *testing.T) {
	storage := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	oldRouter := &OpenAIRouter{ReplayRecorder: routerreplay.NewRecorder(storage)}
	oldRuntime := oldRouter.routerLearningRuntimeState()
	registry := routerruntime.NewRegistry(nil)
	registry.SetLearningRuntime(oldRuntime)
	service := NewRouterService(oldRouter)

	leased, release := registry.AcquireLearningRuntime()
	if leased != oldRuntime {
		t.Fatalf("acquired runtime = %T, want old router runtime", leased)
	}
	newRouter := &OpenAIRouter{}
	newRuntime := newRouter.routerLearningRuntimeState()
	if err := service.Swap(newRouter, func() {
		registry.SetLearningRuntime(newRuntime)
	}); err != nil {
		t.Fatalf("Swap() error = %v", err)
	}
	if storage.closeCalls.Load() != 0 {
		t.Fatal("old replay store closed while a management request held its runtime lease")
	}

	current, currentRelease := registry.AcquireLearningRuntime()
	if current != newRuntime {
		t.Fatalf("new management request acquired %T, want new router runtime", current)
	}
	currentRelease()
	release()
	deadline := time.Now().Add(time.Second)
	for storage.closeCalls.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if storage.closeCalls.Load() != 1 {
		t.Fatalf("old replay store close calls = %d, want 1 after lease release", storage.closeCalls.Load())
	}
	if err := service.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestRouterServiceSwapRetiresPublishedSessionStoreAfterGenerationDrains(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	oldStore := &trackingRouterSessionStateStore{closed: make(chan struct{})}
	newStore := &trackingRouterSessionStateStore{}
	oldStoreSlot := sessiontelemetry.NewRouterSessionStateStoreSlot(oldStore)
	newStoreSlot := sessiontelemetry.NewRouterSessionStateStoreSlot(newStore)
	sessiontelemetry.PublishRouterSessionStateStore(oldStoreSlot)
	t.Cleanup(func() {
		sessiontelemetry.SetRouterSessionStateStore(nil)
		sessiontelemetry.ResetRouterSessionMemoryForTesting()
	})

	oldRouter := &OpenAIRouter{routerSessionStateStore: oldStoreSlot}
	newRouter := &OpenAIRouter{routerSessionStateStore: newStoreSlot}
	service := NewRouterService(oldRouter)
	oldGeneration := service.current.Load()
	oldGeneration.refs.Add(1)

	if err := service.Swap(newRouter, func() {
		publishRouterLearningStateStore(newRouter)
	}); err != nil {
		t.Fatalf("Swap() error = %v", err)
	}
	if got := oldStore.closeCalls.Load(); got != 0 {
		t.Fatalf("old store close calls = %d while generation is leased, want 0", got)
	}

	sessiontelemetry.RecordSessionDecision(sessiontelemetry.SessionDecisionParams{
		SessionID:     "new-generation-store",
		SelectedModel: "model-b",
		Timestamp:     time.Now(),
	})
	if got := newStore.saveCalls.Load(); got != 1 {
		t.Fatalf("new store save calls = %d after publish, want 1", got)
	}
	if got := oldStore.saveCalls.Load(); got != 0 {
		t.Fatalf("old store save calls = %d after publish, want 0", got)
	}

	oldGeneration.refs.Done()
	waitForRouterSessionStoreClose(t, oldStore)
	if got := newStore.closeCalls.Load(); got != 0 {
		t.Fatalf("new store close calls = %d before service close, want 0", got)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if got := newStore.closeCalls.Load(); got != 1 {
		t.Fatalf("new store close calls = %d after service close, want 1", got)
	}
}

func waitForRouterSessionStoreClose(t *testing.T, stateStore *trackingRouterSessionStateStore) {
	t.Helper()
	select {
	case <-stateStore.closed:
	case <-time.After(time.Second):
		t.Fatal("retired store did not close")
	}
	if got := stateStore.closeCalls.Load(); got != 1 {
		t.Fatalf("retired store close calls = %d, want 1", got)
	}
}
