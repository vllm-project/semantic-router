package evaluationplane

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestControlledPairRunningIntentExplicitRetryLaunchesBothWorkers(t *testing.T) {
	process := &controlledPairStoreTestProcess{
		controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)},
	}
	service, _, _ := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = service.Close() })
	_, request := stageControlledPairRunningIntent(t, service)

	execution, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
	if err != nil || execution.State != controlledPairStateRunning {
		t.Fatalf("resume controlled pair=%+v err=%v", execution, err)
	}
	requireControlledPairWorkersStarted(t, process)
	if process.calls.Load() != 2 {
		t.Fatalf("resumed controlled pair launched %d workers, want two", process.calls.Load())
	}
}

func TestConcurrentServicesResumeOneControlledPairWorkerSet(t *testing.T) {
	process := &controlledPairStoreTestProcess{
		controlledProcess: controlledProcess{started: make(chan ProcessSpec, 4)},
	}
	first, _, _ := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = first.Close() })
	_, request := stageControlledPairRunningIntent(t, first)
	second, err := newControlledPairTestService(Options{
		DataDir: first.store.Root(), PythonPath: "python3", ConfigPath: first.registrySource.configPath,
		DeploymentsDir: first.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: process,
	})
	if err != nil {
		t.Fatalf("peer NewService: %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })

	results := make(chan error, 2)
	for _, service := range []*Service{first, second} {
		go func(service *Service) {
			_, launchErr := service.CreateControlledPairExecutionAs(
				context.Background(), SystemActor(), request,
			)
			results <- launchErr
		}(service)
	}
	for range 2 {
		select {
		case launchErr := <-results:
			if launchErr != nil {
				t.Fatalf("concurrent controlled pair retry: %v", launchErr)
			}
		case <-time.After(time.Second):
			t.Fatal("concurrent controlled pair retry did not complete")
		}
	}
	requireControlledPairWorkersStarted(t, process)
	if process.calls.Load() != 2 {
		t.Fatalf("cross-service retry launched %d workers, want one pair", process.calls.Load())
	}
}

func TestControlledPairPartialWorkerOwnershipFailsClosed(t *testing.T) {
	process := &controlledPairStoreTestProcess{}
	service, _, _ := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = service.Close() })
	pair, request := stageControlledPairRunningIntent(t, service)
	_, cancel := context.WithCancel(context.Background())
	if !service.activity.claim([]string{pair.BaselineRunID}, []context.CancelFunc{cancel}) {
		t.Fatal("claim partial controlled pair activity")
	}
	defer func() {
		cancel()
		service.activity.release(pair.BaselineRunID)
	}()
	if _, err := service.CreateControlledPairExecutionAs(
		context.Background(), SystemActor(), request,
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("partial controlled pair retry error=%v, want ErrConflict", err)
	}
	if process.calls.Load() != 0 {
		t.Fatalf("partial ownership launched %d workers", process.calls.Load())
	}
}

func TestControlledPairStatusAndEventCommitFaultsRetryWithoutDuplicates(t *testing.T) {
	for _, test := range []struct {
		name   string
		inject func(*Service)
	}{
		{
			name: "member status",
			inject: func(service *Service) {
				service.store.statusPersistence = &faultingOrdinaryRunStatusPersistence{
					delegate: service.store.statusPersistence, writeFailure: true,
				}
			},
		},
		{
			name: "member start event",
			inject: func(service *Service) {
				service.store.eventPersistence = &faultingOrdinaryRunEventPersistence{
					delegate: service.store.eventPersistence, appendFailure: true,
				}
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			process := &controlledPairStoreTestProcess{
				controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)},
			}
			service, _, _ := newControlledPairExecutionTestService(t, process, 2)
			t.Cleanup(func() { _ = service.Close() })
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			); err != nil {
				t.Fatalf("publish pending controlled pair: %v", err)
			}
			test.inject(service)
			service.store.lifecycle.mu.Lock()
			_, startErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
			service.store.lifecycle.mu.Unlock()
			if startErr == nil {
				t.Fatal("faulted controlled pair start unexpectedly succeeded")
			}
			request := controlledPairRequestFromManifest(pair)
			if _, err := service.CreateControlledPairExecutionAs(
				context.Background(), SystemActor(), request,
			); err != nil {
				t.Fatalf("retry controlled pair start: %v", err)
			}
			requireControlledPairWorkersStarted(t, process)
			for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
				events, err := service.store.EventsAfter(runID, 0)
				if err != nil {
					t.Fatalf("read member events: %v", err)
				}
				startEvents := 0
				for _, event := range events {
					if event.ID == "2" && event.Message == "Controlled pair worker starting" {
						startEvents++
					}
				}
				if startEvents != 1 {
					t.Fatalf("run %s start events=%d, want one: %+v", runID, startEvents, events)
				}
			}
		})
	}
}

func TestRestartTerminalizesControlledPairRunningIntentWithoutReplay(t *testing.T) {
	process := &controlledPairStoreTestProcess{}
	service, _, _ := newControlledPairExecutionTestService(t, process, 2)
	pair, _ := stageControlledPairRunningIntent(t, service)
	root, configPath, deploymentsDir := service.store.Root(), service.registrySource.configPath, service.registrySource.deploymentsDir
	if err := service.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	restartedProcess := &controlledPairStoreTestProcess{}
	restarted, err := newControlledPairTestService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		DeploymentsDir: deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: restartedProcess,
	})
	if err != nil {
		t.Fatalf("restart NewService: %v", err)
	}
	t.Cleanup(func() { _ = restarted.Close() })
	execution, err := restarted.GetControlledPairExecutionAs(SystemActor(), pair.PairID)
	if err != nil || execution.State != controlledPairStateTerminal ||
		execution.BaselineRun.Status != StatusFailed || execution.CandidateRun.Status != StatusFailed {
		t.Fatalf("recovered controlled pair=%+v err=%v, want terminal failures", execution, err)
	}
	if restartedProcess.calls.Load() != 0 {
		t.Fatalf("restart replayed %d controlled pair workers", restartedProcess.calls.Load())
	}
}

func stageControlledPairRunningIntent(
	t *testing.T,
	service *Service,
) (controlledPairManifest, CreateControlledPairRequest) {
	t.Helper()
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish pending controlled pair: %v", err)
	}
	failAfterControlledPairManifestState(service.store, controlledPairStateRunning)
	service.store.lifecycle.mu.Lock()
	start, err := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if err == nil || start.LaunchOwner {
		t.Fatalf("stage Running launch intent=%+v err=%v", start, err)
	}
	durable, err := service.store.readControlledPair(pair.PairID)
	if err != nil || durable.State != controlledPairStateRunning {
		t.Fatalf("read Running launch intent=%+v err=%v", durable, err)
	}
	return durable, controlledPairRequestFromManifest(durable)
}

func controlledPairRequestFromManifest(pair controlledPairManifest) CreateControlledPairRequest {
	return CreateControlledPairRequest{
		ClientRequestID: pair.PairID, BaselineSourceRunID: pair.BaselineSourceRunID,
		CandidateSourceRunID: pair.CandidateSourceRunID, BaselineRunID: pair.BaselineRunID,
		CandidateRunID: pair.CandidateRunID,
	}
}

func requireControlledPairWorkersStarted(t *testing.T, process *controlledPairStoreTestProcess) {
	t.Helper()
	for range 2 {
		select {
		case <-process.started:
		case <-time.After(time.Second):
			t.Fatal("controlled pair worker did not start")
		}
	}
}
