package evaluationplane

import (
	"errors"
	"testing"
)

func TestControlledPairCancelAndDeleteCrashRecoveryConverges(t *testing.T) {
	for _, failure := range controlledPairCancellationFailureCases() {
		t.Run(failure.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
				t.Fatalf("publish pair: %v", err)
			}
			service.store.lifecycle.mu.Lock()
			start, startErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
			service.store.lifecycle.mu.Unlock()
			if startErr != nil || !start.LaunchOwner {
				t.Fatalf("start pair: result=%+v err=%v", start, startErr)
			}
			failure.install(service.store, pair)
			service.store.lifecycle.mu.Lock()
			_, cancelErr := service.store.cancelControlledPairAs(SystemActor(), pair.PairID)
			service.store.lifecycle.mu.Unlock()
			if cancelErr == nil {
				t.Fatalf("persistence failure %s did not interrupt cancellation", failure.name)
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before cancellation recovery: %v", err)
			}
			reopened, err := newStandaloneStore(root)
			if err != nil {
				t.Fatalf("recover cancellation at %s: %v", failure.name, err)
			}
			baseline, baselineErr := reopened.GetRun(pair.BaselineRunID)
			candidate, candidateErr := reopened.GetRun(pair.CandidateRunID)
			if baselineErr != nil || candidateErr != nil || baseline.Status != StatusCancelled || candidate.Status != StatusCancelled {
				t.Fatalf("recovered cancellation split: baseline=%+v/%v candidate=%+v/%v", baseline, baselineErr, candidate, candidateErr)
			}
		})
	}

	for _, failure := range controlledPairDeletionFailureCases() {
		t.Run(failure.name, func(t *testing.T) {
			service, root := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
				t.Fatalf("publish pair: %v", err)
			}
			failure.install(service.store, pair)
			service.store.lifecycle.mu.Lock()
			deleteErr := service.store.deleteControlledPairAs(SystemActor(), pair.PairID)
			service.store.lifecycle.mu.Unlock()
			if deleteErr == nil {
				t.Fatalf("persistence failure %s did not interrupt deletion", failure.name)
			}
			if err := service.Close(); err != nil {
				t.Fatalf("close before deletion recovery: %v", err)
			}
			reopened, reopenErr := newStandaloneStore(root)
			if reopenErr != nil {
				t.Fatalf("recover deletion at %s: %v", failure.name, reopenErr)
			}
			for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
				if _, err := reopened.GetRun(runID); !errors.Is(err, ErrNotFound) {
					t.Fatalf("recovered deletion exposed %s: %v", runID, err)
				}
			}
			tombstone, tombstoneErr := reopened.readControlledPair(pair.PairID)
			if tombstoneErr != nil || tombstone.State != controlledPairStateDeleted {
				t.Fatalf("recovered deletion tombstone=%+v err=%v", tombstone, tombstoneErr)
			}
		})
	}
}
