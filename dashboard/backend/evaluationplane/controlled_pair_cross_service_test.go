package evaluationplane

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type controlledPairTwoServiceBarrierProcess struct {
	*controlledPairStoreTestProcess
	freezeOnce    sync.Once
	freezeEntered chan struct{}
	freezeRelease chan struct{}
}

func (p *controlledPairTwoServiceBarrierProcess) freezeControlledPairCredentials(
	ctx context.Context,
	_ RunManifest,
) (workerBrokerCredentials, error) {
	p.freezeOnce.Do(func() {
		close(p.freezeEntered)
		select {
		case <-ctx.Done():
		case <-p.freezeRelease:
		}
	})
	return workerBrokerCredentials{}, ctx.Err()
}

type controlledPairCreateResult struct {
	execution ControlledPairExecution
	err       error
}

func TestControlledPairConcurrentCreateIsSingleFlightAcrossServices(t *testing.T) {
	process := &controlledPairTwoServiceBarrierProcess{
		controlledPairStoreTestProcess: &controlledPairStoreTestProcess{
			controlledProcess: controlledProcess{started: make(chan ProcessSpec, 4)},
		},
		freezeEntered: make(chan struct{}),
		freezeRelease: make(chan struct{}),
	}
	first, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = first.Close() })
	baselineSource := createSealedControlledPairSource(t, first, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, first, candidateTargetID)

	second, err := newControlledPairTestService(Options{
		DataDir: first.store.Root(), PythonPath: "python3", ConfigPath: first.registrySource.configPath,
		DeploymentsDir: first.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: process,
	})
	if err != nil {
		t.Fatalf("open second controlled-pair service: %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })

	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	firstResult := make(chan controlledPairCreateResult, 1)
	secondResult := make(chan controlledPairCreateResult, 1)
	go func() {
		execution, createErr := first.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
		firstResult <- controlledPairCreateResult{execution: execution, err: createErr}
	}()

	select {
	case <-process.freezeEntered:
	case <-time.After(time.Second):
		t.Fatal("first service did not reach the controlled-pair publication barrier")
	}

	// Keep the first Service immediately before its worker-slot reservation.
	// In the old per-Service single-flight implementation this allowed the
	// second Service to publish a different server timestamp for the same key.
	first.mu.Lock()
	secondStarted := make(chan struct{})
	go func() {
		close(secondStarted)
		execution, createErr := second.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
		secondResult <- controlledPairCreateResult{execution: execution, err: createErr}
	}()
	<-secondStarted
	close(process.freezeRelease)

	var premature *controlledPairCreateResult
	select {
	case result := <-secondResult:
		premature = &result
	case <-time.After(100 * time.Millisecond):
	}
	first.mu.Unlock()
	if premature != nil {
		t.Fatalf(
			"second Service escaped root-shared single-flight before the owner published: execution=%+v err=%v",
			premature.execution,
			premature.err,
		)
	}

	var firstCreate, secondCreate controlledPairCreateResult
	select {
	case firstCreate = <-firstResult:
	case <-time.After(3 * time.Second):
		t.Fatal("first controlled-pair create did not finish")
	}
	select {
	case secondCreate = <-secondResult:
	case <-time.After(3 * time.Second):
		t.Fatal("second controlled-pair create did not finish")
	}
	assertControlledPairSingleFlightResult(t, process, request, second, firstCreate, secondCreate)
}

func assertControlledPairSingleFlightResult(
	t *testing.T,
	process *controlledPairTwoServiceBarrierProcess,
	request CreateControlledPairRequest,
	second *Service,
	firstCreate, secondCreate controlledPairCreateResult,
) {
	t.Helper()
	if firstCreate.err != nil || secondCreate.err != nil {
		t.Fatalf("concurrent creates failed: first=%v second=%v", firstCreate.err, secondCreate.err)
	}
	if firstCreate.execution.ID != secondCreate.execution.ID ||
		firstCreate.execution.BaselineRun.ID != secondCreate.execution.BaselineRun.ID ||
		firstCreate.execution.CandidateRun.ID != secondCreate.execution.CandidateRun.ID ||
		!firstCreate.execution.BaselineRun.CreatedAt.Equal(secondCreate.execution.BaselineRun.CreatedAt) ||
		!firstCreate.execution.CandidateRun.CreatedAt.Equal(secondCreate.execution.CandidateRun.CreatedAt) {
		t.Fatalf(
			"concurrent creates diverged: first=%+v second=%+v",
			firstCreate.execution,
			secondCreate.execution,
		)
	}

	deadline := time.Now().Add(time.Second)
	for process.calls.Load() != 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if calls := process.calls.Load(); calls != 2 {
		t.Fatalf("worker calls=%d, want exactly one two-member launch", calls)
	}

	conflict := request
	conflict.CandidateRunID = newTestClientRequestID()
	if _, err := second.CreateControlledPairExecutionAs(context.Background(), SystemActor(), conflict); !errors.Is(err, ErrConflict) {
		t.Fatalf("same key with different immutable request error=%v, want ErrConflict", err)
	}
	if calls := process.calls.Load(); calls != 2 {
		t.Fatalf("conflicting retry launched workers: calls=%d", calls)
	}
}

func TestControlledPairOwnerCancellationReleasesLifecycleAndSharedFlight(t *testing.T) {
	process := &controlledPairTwoServiceBarrierProcess{
		controlledPairStoreTestProcess: &controlledPairStoreTestProcess{
			controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)},
		},
		freezeEntered: make(chan struct{}), freezeRelease: make(chan struct{}),
	}
	first, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = first.Close() })
	baselineSource := createSealedControlledPairSource(t, first, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, first, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result := make(chan controlledPairCreateResult, 1)
	go func() {
		execution, err := first.CreateControlledPairExecutionAs(ctx, SystemActor(), request)
		result <- controlledPairCreateResult{execution: execution, err: err}
	}()
	select {
	case <-process.freezeEntered:
	case <-time.After(time.Second):
		t.Fatal("controlled-pair owner did not reach credential freeze")
	}
	cancel()
	select {
	case created := <-result:
		if !errors.Is(created.err, context.Canceled) {
			t.Fatalf("cancelled controlled-pair owner error=%v, want context.Canceled", created.err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled controlled-pair owner did not leave prelaunch")
	}
	if _, err := first.store.readControlledPair(request.ClientRequestID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("cancelled controlled-pair owner published an aggregate: %v", err)
	}
	if calls := process.calls.Load(); calls != 0 {
		t.Fatalf("cancelled controlled-pair owner started workers: calls=%d", calls)
	}

	second, err := newControlledPairTestService(Options{
		DataDir: first.store.Root(), PythonPath: "python3", ConfigPath: first.registrySource.configPath,
		DeploymentsDir: first.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: process,
	})
	if err != nil {
		t.Fatalf("open second service after owner cancellation: %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })
	if execution, err := second.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err != nil ||
		execution.State != controlledPairStateRunning {
		t.Fatalf("second service retry after owner cancellation execution=%+v err=%v", execution, err)
	}
	deadline := time.Now().Add(time.Second)
	for process.calls.Load() != 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if calls := process.calls.Load(); calls != 2 {
		t.Fatalf("second service retry worker calls=%d, want 2", calls)
	}
}

func TestControlledPairPreCancelledOwnerDoesNotPublishOrReserve(t *testing.T) {
	process := &controlledPairStoreTestProcess{}
	service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = service.Close() })
	baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := service.CreateControlledPairExecutionAs(ctx, SystemActor(), request); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled owner error=%v, want context.Canceled", err)
	}
	if _, err := service.store.readControlledPair(request.ClientRequestID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("pre-cancelled owner published aggregate: %v", err)
	}
	service.mu.Lock()
	prelaunches := service.prelaunchCount
	service.mu.Unlock()
	if calls := process.calls.Load(); calls != 0 || service.activity.workerSlotsInUse() != 0 || prelaunches != 0 {
		t.Fatalf("pre-cancelled owner leaked work: calls=%d slots=%d prelaunches=%d", calls, service.activity.workerSlotsInUse(), prelaunches)
	}
}

func TestControlledPairPreCancelledFollowerDoesNotJoinSharedFlight(t *testing.T) {
	process := &controlledPairTwoServiceBarrierProcess{
		controlledPairStoreTestProcess: &controlledPairStoreTestProcess{
			controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)},
		},
		freezeEntered: make(chan struct{}), freezeRelease: make(chan struct{}),
	}
	owner, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = owner.Close() })
	baselineSource := createSealedControlledPairSource(t, owner, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, owner, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	ownerResult := make(chan controlledPairCreateResult, 1)
	go func() {
		execution, err := owner.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
		ownerResult <- controlledPairCreateResult{execution: execution, err: err}
	}()
	select {
	case <-process.freezeEntered:
	case <-time.After(time.Second):
		t.Fatal("owner did not enter shared flight")
	}
	follower, err := newControlledPairTestService(Options{
		DataDir: owner.store.Root(), PythonPath: "python3", ConfigPath: owner.registrySource.configPath,
		DeploymentsDir: owner.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: process,
	})
	if err != nil {
		t.Fatalf("open follower service: %v", err)
	}
	t.Cleanup(func() { _ = follower.Close() })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := follower.CreateControlledPairExecutionAs(ctx, SystemActor(), request); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled follower error=%v, want context.Canceled", err)
	}
	follower.mu.Lock()
	prelaunches := follower.prelaunchCount
	follower.mu.Unlock()
	if follower.activity.workerSlotsInUse() != 0 || prelaunches != 0 {
		t.Fatalf("pre-cancelled follower leaked work: slots=%d prelaunches=%d", follower.activity.workerSlotsInUse(), prelaunches)
	}
	select {
	case completed := <-ownerResult:
		t.Fatalf("pre-cancelled follower disturbed owner: %+v", completed)
	default:
	}
	close(process.freezeRelease)
	select {
	case completed := <-ownerResult:
		if completed.err != nil || completed.execution.State != controlledPairStateRunning {
			t.Fatalf("owner completion execution=%+v err=%v", completed.execution, completed.err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("owner did not complete after follower cancellation")
	}
}

func TestControlledPairCloseCancelsAndDrainsPrelaunch(t *testing.T) {
	process := &controlledPairTwoServiceBarrierProcess{
		controlledPairStoreTestProcess: &controlledPairStoreTestProcess{
			controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)},
		},
		freezeEntered: make(chan struct{}), freezeRelease: make(chan struct{}),
	}
	first, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	baselineSource := createSealedControlledPairSource(t, first, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, first, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	result := make(chan controlledPairCreateResult, 1)
	go func() {
		execution, err := first.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
		result <- controlledPairCreateResult{execution: execution, err: err}
	}()
	select {
	case <-process.freezeEntered:
	case <-time.After(time.Second):
		t.Fatal("controlled-pair owner did not reach credential freeze")
	}
	closed := make(chan error, 1)
	go func() { closed <- first.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("close blocked prelaunch: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not drain cancelled controlled-pair prelaunch")
	}
	select {
	case created := <-result:
		if !errors.Is(created.err, context.Canceled) {
			t.Fatalf("closed controlled-pair owner error=%v, want context.Canceled", created.err)
		}
	case <-time.After(time.Second):
		t.Fatal("closed controlled-pair owner did not exit")
	}
	if _, err := first.store.readControlledPair(request.ClientRequestID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Close allowed prelaunch publication: %v", err)
	}
	if calls := process.calls.Load(); calls != 0 {
		t.Fatalf("Close allowed prelaunch worker start: calls=%d", calls)
	}

	second, err := newControlledPairTestService(Options{
		DataDir: first.store.Root(), PythonPath: "python3", ConfigPath: first.registrySource.configPath,
		DeploymentsDir: first.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: process,
	})
	if err != nil {
		t.Fatalf("open second service after Close: %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })
	if execution, err := second.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err != nil ||
		execution.State != controlledPairStateRunning {
		t.Fatalf("second service retry after Close execution=%+v err=%v", execution, err)
	}
	deadline := time.Now().Add(time.Second)
	for process.calls.Load() != 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if calls := process.calls.Load(); calls != 2 {
		t.Fatalf("second service retry after Close worker calls=%d, want 2", calls)
	}
}

func TestControlledPairFollowerCloseCancelsSharedFlightWait(t *testing.T) {
	process := &controlledPairTwoServiceBarrierProcess{
		controlledPairStoreTestProcess: &controlledPairStoreTestProcess{
			controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)},
		},
		freezeEntered: make(chan struct{}), freezeRelease: make(chan struct{}),
	}
	owner, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = owner.Close() })
	baselineSource := createSealedControlledPairSource(t, owner, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, owner, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	ownerResult := make(chan controlledPairCreateResult, 1)
	go func() {
		execution, err := owner.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
		ownerResult <- controlledPairCreateResult{execution: execution, err: err}
	}()
	select {
	case <-process.freezeEntered:
	case <-time.After(time.Second):
		t.Fatal("owner did not reach credential freeze")
	}

	follower, err := newControlledPairTestService(Options{
		DataDir: owner.store.Root(), PythonPath: "python3", ConfigPath: owner.registrySource.configPath,
		DeploymentsDir: owner.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: process,
	})
	if err != nil {
		t.Fatalf("open follower service: %v", err)
	}
	followerResult := make(chan controlledPairCreateResult, 1)
	go func() {
		execution, createErr := follower.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
		followerResult <- controlledPairCreateResult{execution: execution, err: createErr}
	}()
	deadline := time.Now().Add(time.Second)
	for {
		follower.mu.Lock()
		registered := follower.prelaunchCount == 1
		follower.mu.Unlock()
		if registered {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("follower did not enter prelaunch before shared-flight wait")
		}
		time.Sleep(time.Millisecond)
	}

	closed := make(chan error, 1)
	go func() { closed <- follower.Close() }()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("close follower: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("follower Close did not drain its shared-flight wait")
	}
	select {
	case created := <-followerResult:
		if !errors.Is(created.err, context.Canceled) {
			t.Fatalf("closed follower error=%v, want context.Canceled", created.err)
		}
	case <-time.After(time.Second):
		t.Fatal("closed follower did not leave shared-flight wait")
	}
	select {
	case created := <-ownerResult:
		t.Fatalf("follower Close interrupted the independent owner: %+v", created)
	default:
	}

	close(process.freezeRelease)
	select {
	case created := <-ownerResult:
		if created.err != nil || created.execution.State != controlledPairStateRunning {
			t.Fatalf("owner completion execution=%+v err=%v", created.execution, created.err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("owner did not finish after credential freeze released")
	}
	deadline = time.Now().Add(time.Second)
	for process.calls.Load() != 2 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if calls := process.calls.Load(); calls != 2 {
		t.Fatalf("owner launch calls=%d, want exactly 2", calls)
	}
}
