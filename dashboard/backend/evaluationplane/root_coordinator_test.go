package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestServicesShareOneRootCoordinatorAndLastCloseEvictsIt(t *testing.T) {
	first, root := newTestService(t, &controlledProcess{}, 1)
	t.Cleanup(func() { _ = first.Close() })
	second := newSubscriberPeerService(t, first)
	t.Cleanup(func() { _ = second.Close() })
	if first.store.lifecycle != second.store.lifecycle || first.store.runIndex != second.store.runIndex ||
		first.activity != second.activity || first.ownership.coordinator != second.ownership.coordinator {
		t.Fatal("same-root Services did not share one root coordinator")
	}
	evaluationRootCoordinators.Lock()
	coordinator := evaluationRootCoordinators.byRoot[root]
	refs := coordinator.serviceRefs
	evaluationRootCoordinators.Unlock()
	if coordinator != first.store.lifecycle || refs != 2 {
		t.Fatalf("root registry coordinator=%p refs=%d, want %p refs=2", coordinator, refs, first.store.lifecycle)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("close peer Service: %v", err)
	}
	evaluationRootCoordinators.Lock()
	retained := evaluationRootCoordinators.byRoot[root]
	if retained != nil {
		refs = retained.serviceRefs
	} else {
		refs = 0
	}
	evaluationRootCoordinators.Unlock()
	if retained != coordinator || refs != 1 {
		t.Fatalf("root registry after peer close coordinator=%p refs=%d", retained, refs)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("close last Service: %v", err)
	}
	evaluationRootCoordinators.Lock()
	_, exists := evaluationRootCoordinators.byRoot[root]
	evaluationRootCoordinators.Unlock()
	if exists {
		t.Fatal("last Service close retained the root coordinator")
	}
}

func TestDifferentRootsUseDifferentCoordinators(t *testing.T) {
	first, _ := newTestService(t, &controlledProcess{}, 1)
	second, _ := newTestService(t, &controlledProcess{}, 1)
	t.Cleanup(func() { _ = first.Close() })
	t.Cleanup(func() { _ = second.Close() })
	if first.store.lifecycle == second.store.lifecycle || first.store.runIndex == second.store.runIndex ||
		first.activity.eventSubscribers == second.activity.eventSubscribers {
		t.Fatal("different evaluation roots shared process-local coordinator state")
	}
}

func TestPeerStoreLeaseKeepsCoordinatorAliveAfterLastServiceClose(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	coordinator := service.store.lifecycle
	peer := newTestPeerStore(t, service.store)

	if err := service.Close(); err != nil {
		t.Fatalf("close last Service while peer Store is live: %v", err)
	}
	evaluationRootCoordinators.Lock()
	retained := evaluationRootCoordinators.byRoot[root]
	refs := 0
	if retained != nil {
		refs = retained.serviceRefs
	}
	evaluationRootCoordinators.Unlock()
	if retained != coordinator || refs != 1 || peer.lifecycle != coordinator {
		t.Fatalf(
			"peer Store lease did not retain its coordinator: retained=%p peer=%p want=%p refs=%d",
			retained, peer.lifecycle, coordinator, refs,
		)
	}
	if _, err := peer.ListRuns(); err != nil {
		t.Fatalf("peer Store became unusable after Service close: %v", err)
	}
}

func TestRootCoordinatorInitializationFailureTransfersStartupAuthority(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create private evaluation root: %v", err)
	}
	first, err := acquireEvaluationStoreOwnership(root)
	if err != nil {
		t.Fatalf("acquire first root ownership: %v", err)
	}
	second, err := acquireEvaluationStoreOwnership(root)
	if err != nil {
		_ = first.release()
		t.Fatalf("acquire second root ownership: %v", err)
	}
	t.Cleanup(func() {
		_ = first.release()
		_ = second.release()
	})
	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	firstDone := make(chan error, 1)
	injected := errors.New("injected first startup failure")
	go func() {
		firstDone <- first.initialize(func(startupAuthority bool) error {
			if !startupAuthority {
				return errors.New("first opener did not receive startup authority")
			}
			close(firstEntered)
			<-releaseFirst
			return injected
		})
	}()
	<-firstEntered

	secondAttempted := make(chan struct{})
	secondEntered := make(chan bool, 1)
	secondDone := make(chan error, 1)
	go func() {
		close(secondAttempted)
		secondDone <- second.initialize(func(startupAuthority bool) error {
			secondEntered <- startupAuthority
			return nil
		})
	}()
	<-secondAttempted
	select {
	case authority := <-secondEntered:
		t.Fatalf("second opener initialized before the first result: startup_authority=%t", authority)
	default:
	}

	close(releaseFirst)
	select {
	case err := <-firstDone:
		if !errors.Is(err, injected) {
			t.Fatalf("first initialization error=%v, want injected failure", err)
		}
	case <-time.After(time.Second):
		t.Fatal("first initialization did not return")
	}
	select {
	case authority := <-secondEntered:
		if !authority {
			t.Fatal("startup authority was not transferred after first initialization failed")
		}
	case <-time.After(time.Second):
		t.Fatal("second initialization did not take over")
	}
	select {
	case err := <-secondDone:
		if err != nil {
			t.Fatalf("second initialization: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("second initialization did not finish")
	}
}

func TestEvidencePublicationIsRootScoped(t *testing.T) {
	first, _ := newTestService(t, &controlledProcess{}, 1)
	peer := newTestPeerStore(t, first.store)
	other, _ := newTestService(t, &controlledProcess{}, 1)

	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	firstDone := make(chan struct{})
	go func() {
		_ = first.store.withEvidenceSerialization(func() error {
			close(firstEntered)
			<-releaseFirst
			return nil
		})
		close(firstDone)
	}()
	<-firstEntered

	peerAttempted := make(chan struct{})
	peerEntered := make(chan struct{})
	peerDone := make(chan struct{})
	go func() {
		close(peerAttempted)
		_ = peer.withEvidenceSerialization(func() error {
			close(peerEntered)
			return nil
		})
		close(peerDone)
	}()
	<-peerAttempted
	select {
	case <-peerEntered:
		t.Fatal("same-root evidence publication escaped the shared coordinator")
	default:
	}

	otherEntered := make(chan struct{})
	go func() {
		_ = other.store.withEvidenceSerialization(func() error {
			close(otherEntered)
			return nil
		})
	}()
	select {
	case <-otherEntered:
	case <-time.After(time.Second):
		t.Fatal("different-root evidence publication was serialized globally")
	}

	close(releaseFirst)
	for description, done := range map[string]<-chan struct{}{
		"first publication": firstDone,
		"peer publication":  peerDone,
	} {
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatalf("%s did not finish", description)
		}
	}
}

func TestServiceCloseDrainsOperationsBeforeEvictingRootCoordinator(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	coordinator := service.store.lifecycle
	releaseOperation, operationErr := service.beginOperation()
	if operationErr != nil {
		t.Fatalf("begin in-flight operation: %v", operationErr)
	}
	closeAttempted := make(chan struct{})
	closeDone := make(chan error, 1)
	go func() {
		close(closeAttempted)
		closeDone <- service.Close()
	}()
	<-closeAttempted
	select {
	case err := <-closeDone:
		t.Fatalf("Close returned before the in-flight operation drained: %v", err)
	default:
	}
	evaluationRootCoordinators.Lock()
	retained := evaluationRootCoordinators.byRoot[root]
	evaluationRootCoordinators.Unlock()
	if retained != coordinator {
		t.Fatal("Close evicted the root coordinator while an operation still held its lease")
	}

	releaseOperation()
	select {
	case err := <-closeDone:
		if err != nil {
			t.Fatalf("Close after operation drain: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not finish after the in-flight operation drained")
	}
	evaluationRootCoordinators.Lock()
	_, exists := evaluationRootCoordinators.byRoot[root]
	evaluationRootCoordinators.Unlock()
	if exists {
		t.Fatal("last Service Close retained the drained root coordinator")
	}
	if _, err := service.Catalog(); !errors.Is(err, ErrConflict) {
		t.Fatalf("closed Service operation error=%v, want ErrConflict", err)
	}

	reopened, reopenErr := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		RouterAPIURL: service.registrySource.routerAPIURL, EnvoyURL: service.registrySource.envoyURL,
		CodeRevision: service.codeRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if reopenErr != nil {
		t.Fatalf("reopen Service after drained close: %v", reopenErr)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	if reopened.store.lifecycle == coordinator {
		t.Fatal("reopened Service reused an evicted coordinator")
	}
}
