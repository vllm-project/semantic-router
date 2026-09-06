package evaluationplane

import (
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

type pausingRunStatusPersistence struct {
	delegate runStatusPersistence
	runID    string
	status   RunStatus
	entered  chan struct{}
	release  chan struct{}
	once     sync.Once
}

func (p *pausingRunStatusPersistence) Write(path string, run Run) error {
	if filepath.Base(filepath.Dir(path)) == p.runID && run.Status == p.status {
		p.once.Do(func() {
			close(p.entered)
			<-p.release
		})
	}
	return p.delegate.Write(path, run)
}

func (p *pausingRunStatusPersistence) SyncDirectory(path, description string) error {
	return p.delegate.SyncDirectory(path, description)
}

func startedControlledPairForMutationTest(
	t *testing.T,
	service *Service,
) (controlledPairManifest, controlledPairStartResult) {
	t.Helper()
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish controlled pair: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	started, err := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if err != nil {
		t.Fatalf("start controlled pair: %v", err)
	}
	return pair, started
}

func assertControlledPairMutationBlocksCancel(
	t *testing.T,
	service *Service,
	pairID string,
	entered, release chan struct{},
	mutation func() error,
) (ControlledPairExecution, error) {
	t.Helper()
	mutationDone := make(chan error, 1)
	go func() { mutationDone <- mutation() }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("controlled pair mutation did not reach persistence seam")
	}
	type cancelResult struct {
		execution ControlledPairExecution
		err       error
	}
	cancelDone := make(chan cancelResult, 1)
	go func() {
		execution, err := service.CancelControlledPairExecutionAs(SystemActor(), pairID)
		cancelDone <- cancelResult{execution: execution, err: err}
	}()
	select {
	case result := <-cancelDone:
		t.Fatalf("pair cancellation crossed an in-flight member mutation: %+v err=%v", result.execution, result.err)
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	if err := <-mutationDone; err != nil {
		t.Fatalf("controlled pair mutation: %v", err)
	}
	result := <-cancelDone
	return result.execution, result.err
}

func TestControlledPairProgressMutationCannotRestoreRunningAfterCancel(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, started := startedControlledPairForMutationTest(t, service)
	stale := started.Baseline
	stale.Progress.Message = "deterministic in-flight progress"
	persistence := &pausingRunStatusPersistence{
		delegate: service.store.statusPersistence, runID: stale.ID, status: StatusRunning,
		entered: make(chan struct{}), release: make(chan struct{}),
	}
	service.store.statusPersistence = persistence
	execution, err := assertControlledPairMutationBlocksCancel(
		t, service, pair.PairID, persistence.entered, persistence.release,
		func() error { return service.store.updateRunFixture(stale) },
	)
	if err != nil || execution.State != controlledPairStateTerminal {
		t.Fatalf("cancel after progress mutation=%+v err=%v", execution, err)
	}
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		run, readErr := service.store.GetRun(runID)
		if readErr != nil || run.Status != StatusCancelled {
			t.Fatalf("cancelled member=%+v err=%v", run, readErr)
		}
	}
	if err := service.store.updateRunFixture(stale); !errors.Is(err, ErrConflict) {
		t.Fatalf("stale progress retry error=%v, want ErrConflict", err)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); err != nil {
		t.Fatalf("restart after ordered progress/cancel: %v", err)
	}
}

func TestControlledPairSealingMutationWinsBarrierBeforeCancel(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, started := startedControlledPairForMutationTest(t, service)
	persistence := &pausingRunStatusPersistence{
		delegate: service.store.statusPersistence, runID: started.Baseline.ID, status: StatusSealing,
		entered: make(chan struct{}), release: make(chan struct{}),
	}
	service.store.statusPersistence = persistence
	_, err := assertControlledPairMutationBlocksCancel(
		t, service, pair.PairID, persistence.entered, persistence.release,
		func() error {
			_, commitErr := service.store.commitRunSealing(started.Baseline.ID)
			return commitErr
		},
	)
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("cancel after sealing mutation error=%v, want ErrConflict", err)
	}
	baseline, readErr := service.store.GetRun(started.Baseline.ID)
	if readErr != nil || baseline.Status != StatusSealing {
		t.Fatalf("sealing member=%+v err=%v", baseline, readErr)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); err != nil {
		t.Fatalf("restart after ordered sealing/cancel: %v", err)
	}
}

func TestControlledPairTerminalMutationIsPreservedByCancel(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, started := startedControlledPairForMutationTest(t, service)
	terminal, _ := service.buildTerminalRun(started.Baseline, errors.New("deterministic member failure"))
	persistence := &pausingRunStatusPersistence{
		delegate: service.store.statusPersistence, runID: terminal.ID, status: terminal.Status,
		entered: make(chan struct{}), release: make(chan struct{}),
	}
	service.store.statusPersistence = persistence
	execution, err := assertControlledPairMutationBlocksCancel(
		t, service, pair.PairID, persistence.entered, persistence.release,
		func() error {
			_, commitErr := service.store.commitTerminalRun(terminal)
			return commitErr
		},
	)
	if err != nil {
		t.Fatalf("cancel after terminal mutation: %v", err)
	}
	if execution.BaselineRun.Status != StatusFailed || execution.CandidateRun.Status != StatusCancelled {
		t.Fatalf("cancel overwrote terminal member: %+v", execution)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); err != nil {
		t.Fatalf("restart after ordered terminal/cancel: %v", err)
	}
}
