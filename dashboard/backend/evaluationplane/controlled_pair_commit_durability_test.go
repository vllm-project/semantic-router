package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestControlledPairDeletionIntentMakesPagedLedgerFailClosed(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish controlled pair: %v", err)
	}
	failAfterControlledPairManifestState(service.store, controlledPairStateDeleting)
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err == nil {
		t.Fatal("controlled pair deletion intent persistence failure was accepted")
	}
	if _, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 100}); !errors.Is(err, ErrConflict) {
		t.Fatalf("deleting pair paged ledger error=%v, want ErrConflict", err)
	}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err != nil {
		t.Fatalf("controlled pair deletion retry: %v", err)
	}
	page, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 100})
	if err != nil || !page.LedgerComplete {
		t.Fatalf("committed pair deletion page=%+v err=%v", page, err)
	}
	for _, run := range page.Runs {
		if run.ID == pair.BaselineRunID || run.ID == pair.CandidateRunID {
			t.Fatalf("committed pair member remained in paged ledger: %s", run.ID)
		}
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before controlled pair deletion restart: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("reopen after controlled pair deletion retry: %v", err)
	}
}

func TestControlledPairDeletionRetryRecoversPartialMemberCleanup(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish controlled pair: %v", err)
	}
	baselineDir := filepath.Join(service.store.runsRoot, pair.BaselineRunID)
	service.store.pairPersistence = &recordingControlledPairPersistence{
		delegate: atomicControlledPairPersistence{},
		removeAll: func(path string) error {
			if path != baselineDir {
				return os.ErrInvalid
			}
			if err := os.Remove(filepath.Join(path, controlledPairMembershipFile)); err != nil {
				return err
			}
			return errors.New("injected partial controlled pair member cleanup")
		},
	}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err == nil {
		t.Fatal("partial controlled pair cleanup was accepted")
	}
	service.store.pairPersistence = atomicControlledPairPersistence{}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err != nil {
		t.Fatalf("controlled pair partial cleanup retry: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before partial controlled pair cleanup restart: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("reopen after partial controlled pair cleanup: %v", err)
	}
}

func TestControlledPairRunningRetryClosesAggregateSyncFailure(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish controlled pair durability fixture: %v", err)
	}
	recorder := &recordingControlledPairPersistence{
		delegate:                    service.store.pairPersistence,
		failManifestDirectorySyncAt: 2,
	}
	service.store.pairPersistence = recorder

	service.store.lifecycle.mu.Lock()
	first, firstErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if firstErr == nil {
		t.Fatal("controlled pair start succeeded after an uncertain visible running manifest")
	}
	if first.LaunchOwner {
		t.Fatalf("uncertain visible running manifest transferred launch ownership: %+v", first)
	}
	visible, visibleErr := service.store.readControlledPair(pair.PairID)
	if visibleErr != nil || visible.State != controlledPairStateRunning {
		t.Fatalf("failed running publication is not retryable: state=%s err=%v", visible.State, visibleErr)
	}

	recorder.fail = "sync_parent"
	service.store.lifecycle.mu.Lock()
	_, secondErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if secondErr == nil {
		t.Fatal("controlled pair running retry bypassed persistent aggregate sync failure")
	}
	service.store.lifecycle.mu.Lock()
	started, startErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if startErr != nil || started.Pair.State != controlledPairStateRunning {
		t.Fatalf("controlled pair retry did not close aggregate durability: state=%s err=%v", started.Pair.State, startErr)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close controlled pair durability service: %v", err)
	}
	reopened, reopenErr := newStandaloneStore(root)
	if reopenErr != nil {
		t.Fatalf("reopen durable controlled pair: %v", reopenErr)
	}
	durable, durableErr := reopened.readControlledPair(pair.PairID)
	if durableErr != nil || durable.State != controlledPairStateRunning {
		t.Fatalf("controlled pair state changed after restart: state=%s err=%v", durable.State, durableErr)
	}
}

func TestControlledPairTerminalRetryClosesMemberAndAggregateSyncFailure(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish controlled pair terminal fixture: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	_, startErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if startErr != nil {
		t.Fatalf("start controlled pair terminal fixture: %v", startErr)
	}
	recorder := &recordingControlledPairPersistence{
		delegate:                    service.store.pairPersistence,
		failManifestDirectorySyncAt: 2,
	}
	service.store.pairPersistence = recorder

	service.store.lifecycle.mu.Lock()
	_, firstErr := service.store.cancelControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if firstErr == nil {
		t.Fatal("controlled pair cancellation succeeded after an uncertain visible terminal manifest")
	}
	visible, visibleErr := service.store.readControlledPair(pair.PairID)
	if visibleErr != nil || visible.State != controlledPairStateTerminal {
		t.Fatalf("failed terminal publication is not retryable: state=%s err=%v", visible.State, visibleErr)
	}

	recorder.fail = "sync_parent"
	service.store.lifecycle.mu.Lock()
	_, secondErr := service.store.cancelControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if secondErr == nil {
		t.Fatal("controlled pair terminal retry bypassed persistent aggregate sync failure")
	}
	service.store.lifecycle.mu.Lock()
	terminal, terminalErr := service.store.cancelControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if terminalErr != nil || terminal.State != controlledPairStateTerminal ||
		terminal.BaselineRun.Status != StatusCancelled || terminal.CandidateRun.Status != StatusCancelled {
		t.Fatalf("controlled pair retry did not close terminal durability: pair=%+v err=%v", terminal, terminalErr)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close terminal controlled pair service: %v", err)
	}
	reopened, reopenErr := newStandaloneStore(root)
	if reopenErr != nil {
		t.Fatalf("reopen terminal controlled pair: %v", reopenErr)
	}
	durable, durableErr := reopened.readControlledPair(pair.PairID)
	if durableErr != nil || durable.State != controlledPairStateTerminal ||
		durable.BaselineRun.Status != StatusCancelled || durable.CandidateRun.Status != StatusCancelled {
		t.Fatalf("terminal controlled pair changed after restart: pair=%+v err=%v", durable, durableErr)
	}
}
