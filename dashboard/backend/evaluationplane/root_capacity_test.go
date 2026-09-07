package evaluationplane

import (
	"context"
	"errors"
	"testing"
)

func (coordinator *evaluationRootCoordinator) workerSlotsInUse() int {
	coordinator.capacityMu.Lock()
	defer coordinator.capacityMu.Unlock()
	return len(coordinator.workerSlots)
}

func TestWorkerCapacityIsSharedAcrossServices(t *testing.T) {
	firstProcess := &controlledProcess{started: make(chan ProcessSpec, 1), release: make(chan struct{})}
	first, root := newTestService(t, firstProcess, 1)
	secondProcess := &controlledProcess{started: make(chan ProcessSpec, 1)}
	second, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: first.registrySource.configPath,
		RouterAPIURL: first.registrySource.routerAPIURL, EnvoyURL: first.registrySource.envoyURL,
		CodeRevision: first.codeRevision, MaxConcurrent: 1, Process: secondProcess,
	})
	if err != nil {
		t.Fatalf("open same-root capacity peer: %v", err)
	}
	t.Cleanup(func() { _ = second.Close() })

	firstRun, err := first.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create first capacity run: %v", err)
	}
	secondRun, err := second.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create second capacity run: %v", err)
	}
	if _, err := first.StartRunAs(context.Background(), SystemActor(), firstRun.ID); err != nil {
		t.Fatalf("start first capacity run: %v", err)
	}
	<-firstProcess.started
	if _, err := second.StartRunAs(context.Background(), SystemActor(), secondRun.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("same-root peer capacity error=%v, want ErrConflict", err)
	}
	if secondProcess.calls.Load() != 0 || first.activity.workerSlotsInUse() != 1 {
		t.Fatalf(
			"same-root capacity escaped global admission: second_calls=%d slots=%d",
			secondProcess.calls.Load(), first.activity.workerSlotsInUse(),
		)
	}
}

func TestControlledPairWorkerReservationIsAtomic(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 2)
	releaseOrdinary, reserved := service.activity.reserveWorkerSlots(1)
	if !reserved {
		t.Fatal("reserve ordinary worker slot")
	}
	if releasePair, err := service.reserveControlledPairWorkerSlots(context.Background()); !errors.Is(err, ErrConflict) || releasePair != nil {
		t.Fatalf("partial pair capacity reservation release=%v err=%v", releasePair != nil, err)
	}
	if used := service.activity.workerSlotsInUse(); used != 1 {
		t.Fatalf("failed pair reservation changed used slots=%d, want 1", used)
	}
	releaseOrdinary()
	releasePair, err := service.reserveControlledPairWorkerSlots(context.Background())
	if err != nil {
		t.Fatalf("reserve two controlled-pair slots: %v", err)
	}
	if used := service.activity.workerSlotsInUse(); used != 2 {
		t.Fatalf("controlled pair reserved slots=%d, want 2", used)
	}
	releasePair()
}

func TestWorkerAndEvidenceCapacityAreIsolatedByRoot(t *testing.T) {
	first, _ := newTestService(t, &controlledProcess{}, 1)
	peer := newSubscriberPeerService(t, first)
	other, _ := newTestService(t, &controlledProcess{}, 1)

	releaseFirstWorker, firstWorkerReserved := first.activity.reserveWorkerSlots(1)
	if !firstWorkerReserved {
		t.Fatal("reserve first-root worker slot")
	}
	defer releaseFirstWorker()
	if releasePeerWorker, reserved := peer.activity.reserveWorkerSlots(1); reserved || releasePeerWorker != nil {
		t.Fatal("same-root peer received capacity outside the shared budget")
	}
	releaseOtherWorker, otherWorkerReserved := other.activity.reserveWorkerSlots(1)
	if !otherWorkerReserved {
		t.Fatal("different root did not receive an independent worker budget")
	}
	releaseOtherWorker()

	evidenceReleases := make([]func(), 0, maxConcurrentEvidenceReads)
	for range maxConcurrentEvidenceReads {
		release, err := first.acquireEvidenceRead()
		if err != nil {
			t.Fatalf("reserve first-root evidence read: %v", err)
		}
		evidenceReleases = append(evidenceReleases, release)
	}
	if release, err := peer.acquireEvidenceRead(); !errors.Is(err, ErrConflict) || release != nil {
		t.Fatalf("same-root evidence capacity release=%v err=%v", release != nil, err)
	}
	releaseOtherEvidence, err := other.acquireEvidenceRead()
	if err != nil {
		t.Fatalf("different-root evidence read: %v", err)
	}
	releaseOtherEvidence()
	for _, release := range evidenceReleases {
		release()
	}
}

func TestPeerServiceRejectsWorkerCapacityDrift(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	peer, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		RouterAPIURL: service.registrySource.routerAPIURL, EnvoyURL: service.registrySource.envoyURL,
		CodeRevision: service.codeRevision, MaxConcurrent: 2, Process: &controlledProcess{},
	})
	if peer != nil {
		_ = peer.Close()
	}
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("same-root capacity drift error=%v, want ErrConflict", err)
	}
	evaluationRootCoordinators.Lock()
	refs := evaluationRootCoordinators.byRoot[root].serviceRefs
	evaluationRootCoordinators.Unlock()
	if refs != 1 || service.activity.workerCapacity != 1 {
		t.Fatalf("capacity drift changed root state: refs=%d capacity=%d", refs, service.activity.workerCapacity)
	}
}

func TestLastServiceReleaseFailsClosedOnCoordinatorLeaks(t *testing.T) {
	for _, test := range []struct {
		name string
		leak func(*Service) func()
	}{
		{
			name: "worker slot",
			leak: func(service *Service) func() {
				release, reserved := service.activity.reserveWorkerSlots(1)
				if !reserved {
					return func() {}
				}
				return release
			},
		},
		{
			name: "active run",
			leak: func(service *Service) func() {
				runID := newTestClientRequestID()
				if !service.activity.claim([]string{runID}, []context.CancelFunc{func() {}}) {
					return func() {}
				}
				return func() { service.activity.release(runID) }
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			coordinator := service.activity
			cleanupLeak := test.leak(service)
			if err := service.Close(); !errors.Is(err, ErrConflict) {
				t.Fatalf("Close with leaked coordinator state error=%v, want ErrConflict", err)
			}
			evaluationRootCoordinators.Lock()
			retained := evaluationRootCoordinators.byRoot[root]
			refs := 0
			if retained != nil {
				refs = retained.serviceRefs
			}
			evaluationRootCoordinators.Unlock()
			if retained != coordinator || refs != 1 || coordinator.releaseBlocked == nil {
				t.Fatalf("blocked release coordinator=%p refs=%d error=%v", retained, refs, coordinator.releaseBlocked)
			}
			peer, err := NewService(Options{DataDir: root})
			if peer != nil {
				_ = peer.Close()
			}
			if !errors.Is(err, ErrConflict) {
				t.Fatalf("new opener crossed blocked root release: %v", err)
			}

			cleanupLeak()
			_ = service.Close()
			evaluationRootCoordinators.Lock()
			_, exists := evaluationRootCoordinators.byRoot[root]
			evaluationRootCoordinators.Unlock()
			if exists {
				t.Fatal("release retry retained a quiescent root coordinator")
			}
		})
	}
}
