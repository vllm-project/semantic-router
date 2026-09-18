package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestCreateControlledPairExecutionReservesCapacityBeforePublicationAndIsIdempotent(t *testing.T) {
	process := &controlledPairStoreTestProcess{controlledProcess: controlledProcess{started: make(chan ProcessSpec, 4)}}
	service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = service.Close() })
	baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}

	first, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateControlledPairExecution: %v", err)
	}
	second, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("idempotent CreateControlledPairExecution: %v", err)
	}
	if first.ID != second.ID || second.CandidateRun.BaselineRunID != second.BaselineRun.ID ||
		!second.BaselineRun.CreatedAt.Before(second.CandidateRun.CreatedAt) {
		t.Fatalf("idempotent controlled pair lost aggregate identity: first=%+v second=%+v", first, second)
	}
	if first.State != controlledPairStateRunning || !first.Capabilities.CanCancel || first.Capabilities.CanDelete ||
		!controlledPairRunMembershipMatches(first.BaselineRun, first.ID, controlledPairRoleBaseline) ||
		!controlledPairRunMembershipMatches(first.CandidateRun, first.ID, controlledPairRoleCandidate) {
		t.Fatalf("controlled pair public membership/capabilities are incomplete: %+v", first)
	}
	for range 2 {
		select {
		case <-process.started:
		case <-time.After(time.Second):
			t.Fatal("controlled pair worker did not start")
		}
	}
	select {
	case extra := <-process.started:
		t.Fatalf("idempotent request launched an extra worker: %+v", extra)
	case <-time.After(20 * time.Millisecond):
	}
}

func TestCreateControlledPairExecutionConcurrentIdempotencyHasOneLaunchOwner(t *testing.T) {
	for _, capacity := range []int{2, 4} {
		t.Run(fmt.Sprintf("capacity_%d", capacity), func(t *testing.T) {
			process := &controlledPairStoreTestProcess{
				controlledProcess: controlledProcess{started: make(chan ProcessSpec, 4)},
			}
			service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, capacity)
			t.Cleanup(func() { _ = service.Close() })
			baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
			candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
			request := CreateControlledPairRequest{
				ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
				CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
				CandidateRunID: newTestClientRequestID(),
			}

			start := make(chan struct{})
			results := make(chan ControlledPairExecution, 2)
			errorsSeen := make(chan error, 2)
			for range 2 {
				go func() {
					<-start
					execution, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
					results <- execution
					errorsSeen <- err
				}()
			}
			close(start)
			for range 2 {
				if err := <-errorsSeen; err != nil {
					t.Fatalf("concurrent controlled pair request: %v", err)
				}
			}
			first, second := <-results, <-results
			if first.ID != second.ID || first.BaselineRun.ID != second.BaselineRun.ID ||
				first.CandidateRun.ID != second.CandidateRun.ID {
				t.Fatalf("concurrent requests returned different aggregates: first=%+v second=%+v", first, second)
			}
			for range 2 {
				select {
				case <-process.started:
				case <-time.After(time.Second):
					t.Fatal("controlled pair launch owner did not start both workers")
				}
			}
			select {
			case extra := <-process.started:
				t.Fatalf("concurrent retry launched an extra worker: %+v", extra)
			case <-time.After(20 * time.Millisecond):
			}
			if calls := process.calls.Load(); calls != 2 {
				t.Fatalf("worker calls=%d, want exactly one two-member launch", calls)
			}
			service.mu.Lock()
			active := len(service.active)
			service.mu.Unlock()
			if active != 2 {
				t.Fatalf("active worker handles=%d, want exactly 2", active)
			}
		})
	}
}

func TestCreateControlledPairExecutionSameRequestResumesEveryPublicationAndStartFault(t *testing.T) {
	failures := append(controlledPairPublicationFailureCases(), controlledPairStartFailureCases()...)
	for _, failure := range failures {
		t.Run(failure.name, func(t *testing.T) {
			process := &controlledPairStoreTestProcess{
				controlledProcess: controlledProcess{started: make(chan ProcessSpec, 4)},
			}
			service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
			t.Cleanup(func() { _ = service.Close() })
			baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
			candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
			request := CreateControlledPairRequest{
				ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
				CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
				CandidateRunID: newTestClientRequestID(),
			}
			failure.install(service.store, controlledPairManifest{
				BaselineRunID: request.BaselineRunID, CandidateRunID: request.CandidateRunID,
			})
			if _, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err == nil {
				t.Fatalf("persistence failure %s did not interrupt first request", failure.name)
			}
			durable, err := service.store.readControlledPair(request.ClientRequestID)
			if err != nil {
				t.Fatalf("failure %s lost authoritative request identity: %v", failure.name, err)
			}
			createdAt := durable.BaselineRun.CreatedAt
			execution, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
			if err != nil {
				t.Fatalf("retry %s: %v", failure.name, err)
			}
			if execution.BaselineRun.CreatedAt != createdAt || execution.BaselineRun.ID != durable.BaselineRunID ||
				execution.CandidateRun.ID != durable.CandidateRunID {
				t.Fatalf("retry %s regenerated pair identity: before=%+v after=%+v", failure.name, durable, execution)
			}
			deadline := time.Now().Add(time.Second)
			for process.calls.Load() != 2 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			if calls := process.calls.Load(); calls != 2 {
				t.Fatalf("retry %s launched %d workers, want exactly 2", failure.name, calls)
			}
		})
	}
}

func TestCreateControlledPairExecutionDefersLaunchUntilRunningManifestSyncRetry(t *testing.T) {
	process := &controlledPairStoreTestProcess{
		controlledProcess: controlledProcess{started: make(chan ProcessSpec, 4)},
	}
	service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = service.Close() })
	baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	service.store.pairPersistence = &recordingControlledPairPersistence{
		delegate: atomicControlledPairPersistence{}, failManifestDirectorySyncAt: 4,
	}
	if _, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err == nil {
		t.Fatal("running manifest directory sync ambiguity was not propagated")
	}
	visible, pairReadErr := service.store.readControlledPair(request.ClientRequestID)
	if pairReadErr != nil || visible.State != controlledPairStateRunning {
		t.Fatalf("uncertain running manifest is not available for explicit retry: pair=%+v err=%v", visible, pairReadErr)
	}
	if calls := process.calls.Load(); calls != 0 {
		t.Fatalf("uncertain running manifest launched %d workers, want zero", calls)
	}
	if execution, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err != nil ||
		execution.State != controlledPairStateRunning {
		t.Fatalf("same-service retry execution=%+v err=%v", execution, err)
	}
	requireControlledPairWorkersStarted(t, process)
	if calls := process.calls.Load(); calls != 2 {
		t.Fatalf("same-service retry launched %d workers, want 2", calls)
	}

	secondProcess := &controlledPairStoreTestProcess{controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)}}
	second, err := newControlledPairTestService(Options{
		DataDir: service.store.Root(), PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		DeploymentsDir: service.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: secondProcess,
	})
	if err != nil {
		t.Fatalf("open second service after ambiguous running commit: %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })
	if execution, err := second.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err != nil ||
		execution.State != controlledPairStateRunning {
		t.Fatalf("second-service retry execution=%+v err=%v", execution, err)
	}
	if calls := secondProcess.calls.Load(); calls != 0 {
		t.Fatalf("second service relaunched %d workers", calls)
	}
}

func TestCreateControlledPairExecutionRetryAfterServiceRestartUsesDurableIdentity(t *testing.T) {
	for _, failure := range []controlledPairPersistenceFailureCase{
		{name: "baseline_published", install: func(store *Store, pair controlledPairManifest) {
			failAfterControlledPairRename(store, pair.BaselineRunID)
		}},
		{name: "start_intent", install: func(store *Store, _ controlledPairManifest) {
			failAfterControlledPairManifestState(store, controlledPairStateStarting)
		}},
	} {
		t.Run(failure.name, func(t *testing.T) {
			firstProcess := &controlledPairStoreTestProcess{}
			service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, firstProcess, 2)
			baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
			candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
			request := CreateControlledPairRequest{
				ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
				CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
				CandidateRunID: newTestClientRequestID(),
			}
			root, configPath, deploymentsDir := service.store.Root(), service.registrySource.configPath, service.registrySource.deploymentsDir
			failure.install(service.store, controlledPairManifest{BaselineRunID: request.BaselineRunID})
			if _, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err == nil {
				t.Fatalf("persistence failure %s did not interrupt first service", failure.name)
			}
			durable, err := service.store.readControlledPair(request.ClientRequestID)
			if err != nil {
				t.Fatalf("read durable retry identity: %v", err)
			}
			if closeErr := service.Close(); closeErr != nil {
				t.Fatalf("close first service: %v", closeErr)
			}
			secondProcess := &controlledPairStoreTestProcess{
				controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)},
			}
			restarted, err := newControlledPairTestService(Options{
				DataDir: root, PythonPath: "python3", ConfigPath: configPath, DeploymentsDir: deploymentsDir,
				CodeRevision: testSourceRevision, MaxConcurrent: 2, Process: secondProcess,
			})
			if err != nil {
				t.Fatalf("restart service after %s: %v", failure.name, err)
			}
			t.Cleanup(func() { _ = restarted.Close() })
			execution, err := restarted.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
			if err != nil {
				t.Fatalf("restart retry after %s: %v", failure.name, err)
			}
			if execution.BaselineRun.CreatedAt != durable.BaselineRun.CreatedAt ||
				execution.CandidateRun.CreatedAt != durable.CandidateRun.CreatedAt {
				t.Fatalf("restart retry regenerated identity: durable=%+v execution=%+v", durable, execution)
			}
		})
	}
}

func TestCreateControlledPairExecutionCapacityFailureLeavesNoAggregate(t *testing.T) {
	process := &controlledPairStoreTestProcess{}
	service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 1)
	baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	_, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("capacity error=%v, want ErrConflict", err)
	}
	if _, err := service.store.readControlledPair(request.ClientRequestID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("capacity rejection published pair aggregate: %v", err)
	}
	for _, runID := range []string{request.BaselineRunID, request.CandidateRunID} {
		if _, err := service.store.GetRun(runID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("capacity rejection published run %s: %v", runID, err)
		}
	}
}

func TestCreateControlledPairExecutionAuthorizesBothSourcesBeforeReadOrFreeze(t *testing.T) {
	process := &controlledPairStoreTestProcess{}
	service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	// Make the first source unreadable as sealed evidence. Authorization must
	// reject the unrelated actor before source loading can observe this damage.
	if err := os.Remove(filepath.Join(service.store.runsRoot, baselineSource.ID, reportFileName)); err != nil {
		t.Fatalf("remove controlled-pair source report: %v", err)
	}
	actor := testLifecycleActor(t, "unrelated-controlled-pair-owner", false)
	if _, err := service.CreateControlledPairExecutionAs(context.Background(), actor, request); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner source admission error=%v, want ErrForbidden", err)
	}
	if process.freezes != 0 {
		t.Fatalf("unauthorized source path touched freezer: freezes=%d", process.freezes)
	}
}
