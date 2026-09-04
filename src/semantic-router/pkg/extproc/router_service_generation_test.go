package extproc

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

func TestRouterServiceShutdownBoundsStuckGenerationWithoutClosingUnderLease(t *testing.T) {
	storage := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	recorder := routerreplay.NewRecorder(storage)
	resources := newResourceScope()
	resources.add(recorder.Close)
	router := (&routerComponents{resources: resources, replayRecorder: recorder}).buildRouter()
	service := NewRouterService(router)
	generation := service.current.Load()
	release, acquired := generation.acquire()
	if !acquired {
		t.Fatal("failed to acquire generation")
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	err := service.Shutdown(shutdownCtx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Shutdown() error = %v, want deadline exceeded", err)
	}
	if storage.closeCalls.Load() != 0 {
		t.Fatal("generation resources closed while a lease was still active")
	}
	release()
	deadline := time.Now().Add(time.Second)
	for storage.closeCalls.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if storage.closeCalls.Load() != 1 {
		t.Fatalf("generation close calls = %d, want 1 after lease release", storage.closeCalls.Load())
	}
}

func TestRouterServiceShutdownIncludesCompletedGenerationErrorsOnTimeout(t *testing.T) {
	closeErr := errors.New("retired generation close failed")
	closed := make(chan struct{})
	firstResources := newResourceScope()
	firstResources.add(func() error {
		close(closed)
		return closeErr
	})
	first := (&routerComponents{resources: firstResources}).buildRouter()
	second := (&routerComponents{resources: newResourceScope()}).buildRouter()
	service := NewRouterService(first)
	if err := service.Swap(second, nil); err != nil {
		t.Fatalf("Swap() error = %v", err)
	}
	<-closed
	waitForRetiredGenerationError(t, service)

	release, acquired := service.current.Load().acquire()
	if !acquired {
		t.Fatal("failed to acquire current generation")
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	err := service.Shutdown(shutdownCtx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Shutdown() error = %v, want deadline exceeded", err)
	}
	if !errors.Is(err, closeErr) {
		t.Fatalf("Shutdown() error = %v, want retired generation close error", err)
	}
	release()
}

func TestRouterLearningRuntimeGenerationIsStableDuringConcurrentLeases(t *testing.T) {
	router := (&routerComponents{resources: newResourceScope()}).buildRouter()
	service := NewRouterService(router)
	runtime := router.routerLearningRuntimeState()

	var wg sync.WaitGroup
	for range 100 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			if got := router.routerLearningRuntimeState(); got != runtime {
				t.Errorf("routerLearningRuntimeState() returned a different runtime")
			}
		}()
		go func() {
			defer wg.Done()
			release, acquired := runtime.AcquireLease()
			if !acquired {
				t.Errorf("AcquireLease() rejected an active generation")
				return
			}
			release()
		}()
	}
	wg.Wait()
	if err := service.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func waitForRetiredGenerationError(t *testing.T, service *RouterService) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		service.errMu.Lock()
		hasError := len(service.errors) > 0
		service.errMu.Unlock()
		if hasError {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("retired generation close error was not recorded")
}

func TestRouterServiceShutdownPreservesAcknowledgedLearningOutcome(t *testing.T) {
	memoryStore := store.NewMemoryStore(10, 0)
	countingStore := &countingCloseStore{Storage: memoryStore}
	appendStarted := make(chan struct{})
	releaseAppend := make(chan struct{})
	storage := &controlledOutcomeStore{
		Storage:       countingStore,
		appendStarted: appendStarted,
		releaseAppend: releaseAppend,
	}
	recorder := routerreplay.NewRecorder(storage)
	if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:            "acknowledged-outcome",
		SelectedModel: "model-a",
	}); err != nil {
		t.Fatalf("add replay record: %v", err)
	}
	resources := newResourceScope()
	resources.add(recorder.Close)
	router := (&routerComponents{resources: resources, replayRecorder: recorder}).buildRouter()
	service := NewRouterService(router)
	runtime := router.routerLearningRuntimeState()
	releaseLease, acquired := runtime.AcquireLease()
	if !acquired {
		t.Fatal("failed to acquire learning runtime")
	}

	outcomeDone := make(chan routerruntime.RouterOutcomeResult, 1)
	go func() {
		defer releaseLease()
		outcomeDone <- runtime.UpdateOutcome(context.Background(), &routerruntime.RouterOutcome{
			ReplayID:  "acknowledged-outcome",
			Source:    routerruntime.RouterOutcomeSourceEval,
			Target:    routerruntime.RouterOutcomeTargetModel,
			TargetRef: "model-a",
			Verdict:   routerruntime.RouterOutcomeVerdictGoodFit,
		})
	}()
	<-appendStarted

	shutdownDone := make(chan error, 1)
	shutdownCtx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	go func() { shutdownDone <- service.Shutdown(shutdownCtx) }()
	requireShutdownPending(t, shutdownDone)
	if countingStore.closeCalls.Load() != 0 {
		t.Fatal("replay store closed before the accepted outcome persisted")
	}

	close(releaseAppend)
	result := <-outcomeDone
	if !result.Recorded || result.Updated != 1 {
		t.Fatalf("UpdateOutcome() result = %#v, want acknowledged mutation", result)
	}
	if err := <-shutdownDone; err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}
	if countingStore.closeCalls.Load() != 1 {
		t.Fatalf("replay store close calls = %d, want 1", countingStore.closeCalls.Load())
	}
	record, found, err := memoryStore.Get(context.Background(), "acknowledged-outcome")
	if err != nil || !found || len(record.Outcomes) != 1 {
		t.Fatalf("persisted outcome = found:%v err:%v record:%#v", found, err, record)
	}
}

func requireShutdownPending(t *testing.T, shutdownDone <-chan error) {
	t.Helper()
	select {
	case err := <-shutdownDone:
		t.Fatalf("Shutdown() returned before the accepted outcome persisted: %v", err)
	case <-time.After(25 * time.Millisecond):
	}
}

func TestRouterServiceSwapPublishesBeforeRetiredGenerationDrains(t *testing.T) {
	storage := &countingCloseStore{Storage: store.NewMemoryStore(10, 0)}
	recorder := routerreplay.NewRecorder(storage)
	resources := newResourceScope()
	resources.add(recorder.Close)
	oldRouter := (&routerComponents{resources: resources, replayRecorder: recorder}).buildRouter()
	service := NewRouterService(oldRouter)
	generation := service.current.Load()
	releaseGeneration, acquired := generation.acquire()
	if !acquired {
		t.Fatal("failed to acquire initial generation")
	}

	swapped := make(chan struct{})
	go func() {
		_ = service.Swap((&routerComponents{resources: newResourceScope()}).buildRouter(), nil)
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

	releaseGeneration()
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
	firstRecorder := routerreplay.NewRecorder(firstStore)
	firstResources := newResourceScope()
	firstResources.add(firstRecorder.Close)
	first := (&routerComponents{resources: firstResources, replayRecorder: firstRecorder}).buildRouter()
	secondRecorder := routerreplay.NewRecorder(secondStore)
	secondResources := newResourceScope()
	secondResources.add(secondRecorder.Close)
	second := (&routerComponents{resources: secondResources, replayRecorder: secondRecorder}).buildRouter()
	service := NewRouterService(first)
	firstGeneration := service.current.Load()
	releaseFirstGeneration, acquired := firstGeneration.acquire()
	if !acquired {
		t.Fatal("failed to acquire first generation")
	}

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
	rejectedRecorder := routerreplay.NewRecorder(rejectedStore)
	rejectedResources := newResourceScope()
	rejectedResources.add(rejectedRecorder.Close)
	rejected := (&routerComponents{resources: rejectedResources, replayRecorder: rejectedRecorder}).buildRouter()
	if err := service.Swap(rejected, nil); err == nil {
		t.Fatal("Swap() accepted a router after shutdown began")
	}
	if rejectedStore.closeCalls.Load() != 1 {
		t.Fatalf("rejected router close calls = %d, want 1", rejectedStore.closeCalls.Load())
	}

	releaseFirstGeneration()
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
	recorder := routerreplay.NewRecorder(storage)
	resources := newResourceScope()
	resources.add(recorder.Close)
	oldRouter := (&routerComponents{resources: resources, replayRecorder: recorder}).buildRouter()
	oldRuntime := oldRouter.routerLearningRuntimeState()
	registry := routerruntime.NewRegistry(nil)
	registry.SetLearningRuntime(oldRuntime)
	service := NewRouterService(oldRouter)

	leased, release := registry.AcquireLearningRuntime()
	if leased != oldRuntime {
		t.Fatalf("acquired runtime = %T, want old router runtime", leased)
	}
	newRouter := (&routerComponents{resources: newResourceScope()}).buildRouter()
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

	oldResources := newResourceScope()
	registerRouterSessionStore(oldResources, oldStoreSlot)
	oldRouter := (&routerComponents{resources: oldResources, routerSessionStore: oldStoreSlot}).buildRouter()
	newResources := newResourceScope()
	registerRouterSessionStore(newResources, newStoreSlot)
	newRouter := (&routerComponents{resources: newResources, routerSessionStore: newStoreSlot}).buildRouter()
	service := NewRouterService(oldRouter)
	oldGeneration := service.current.Load()
	releaseOldGeneration, acquired := oldGeneration.acquire()
	if !acquired {
		t.Fatal("failed to acquire old generation")
	}

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

	releaseOldGeneration()
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
