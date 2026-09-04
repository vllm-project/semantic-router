package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestControlledPairAggregatePublishesOneStrictPendingPair(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	actor := testLifecycleActor(t, "controlled-pair-owner", false)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, actor)

	created, createErr := service.store.createControlledPairBundlesAs(actor, pair, baselineManifest, candidateManifest)
	if createErr != nil {
		t.Fatalf("create controlled pair aggregate: %v", createErr)
	}
	if created.State != controlledPairStatePending || created.PairID != pair.PairID ||
		created.OwnerPrincipalDigest != actor.principalDigest ||
		created.BaselineSourceRunID != pair.BaselineSourceRunID ||
		created.CandidateSourceRunID != pair.CandidateSourceRunID ||
		!digestPattern.MatchString(created.CohortDigest) || !digestPattern.MatchString(created.TreatmentDigest) {
		t.Fatalf("controlled pair aggregate lost durable identity: %+v", created)
	}
	assertStrictPendingPair(t, service.store, pair)
	assertInitialSnapshotEvent(t, service.store, pair.BaselineRunID)
	assertInitialSnapshotEvent(t, service.store, pair.CandidateRunID)
	if _, startErr := service.StartRunAs(context.Background(), actor, pair.BaselineRunID); !errors.Is(startErr, ErrConflict) {
		t.Fatalf("independent controlled pair member start error=%v, want ErrConflict", startErr)
	}
	if _, cancelErr := service.CancelRunAs(actor, pair.CandidateRunID); !errors.Is(cancelErr, ErrConflict) {
		t.Fatalf("independent controlled pair member cancellation error=%v, want ErrConflict", cancelErr)
	}
	if deleteErr := service.store.DeleteRunAs(actor, pair.CandidateRunID); !errors.Is(deleteErr, ErrConflict) {
		t.Fatalf("independent controlled pair member deletion error=%v, want ErrConflict", deleteErr)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close before controlled pair restart: %v", err)
	}
	reopened, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("reopen controlled pair store: %v", err)
	}
	assertStrictPendingPair(t, reopened, pair)
	durable, err := reopened.readControlledPair(pair.PairID)
	if err != nil || durable.State != controlledPairStatePending {
		t.Fatalf("reopened controlled pair aggregate=%+v err=%v", durable, err)
	}
}

func TestControlledPairDirectoryDurabilityPrecedesMemberPublication(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	recorder := &recordingControlledPairPersistence{delegate: atomicControlledPairPersistence{}}
	service.store.pairPersistence = recorder
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("create controlled pair with persistence recorder: %v", err)
	}
	assertOperationBefore(t, recorder.operations, "mkdir", "sync_parent")
	assertOperationBefore(t, recorder.operations, "sync_parent", "write_manifest")
	assertOperationBefore(t, recorder.operations, "write_manifest", "rename")

	for _, failure := range []string{"mkdir", "sync_parent", "write_manifest", "rename"} {
		t.Run("failure_"+failure, func(t *testing.T) {
			service, _ := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			service.store.pairPersistence = &recordingControlledPairPersistence{
				delegate: atomicControlledPairPersistence{}, fail: failure,
			}
			if _, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			); err == nil {
				t.Fatalf("persistence failure %s was accepted", failure)
			}
			if failure != "rename" {
				assertControlledPairAbsent(t, service.store, pair)
			}
		})
	}
}

func TestControlledPairManifestDirectorySyncFailureDoesNotPublishAndRetryResumesIdentity(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	recorder := &recordingControlledPairPersistence{
		delegate: atomicControlledPairPersistence{}, failManifestDirectorySync: true,
	}
	service.store.pairPersistence = recorder
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err == nil {
		t.Fatal("rename followed by manifest-directory sync failure was accepted")
	}
	if len(recorder.operations) == 0 || recorder.operations[len(recorder.operations)-1] != "manifest_sync_failure" {
		t.Fatalf("manifest failure did not occur after rename: %v", recorder.operations)
	}
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		if _, err := service.store.GetRun(runID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("durability uncertainty published member %s: %v", runID, err)
		}
	}
	durable, err := service.store.readControlledPair(pair.PairID)
	if err != nil || durable.State != controlledPairStatePublishing || !sameControlledPairIdentity(durable, pair) {
		t.Fatalf("durable retry identity=%+v err=%v", durable, err)
	}
	service.store.pairPersistence = atomicControlledPairPersistence{}
	resumed, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	)
	if err != nil || resumed.State != controlledPairStatePending {
		t.Fatalf("resume uncertain manifest publication: state=%s err=%v", resumed.State, err)
	}
	assertStrictPendingPair(t, service.store, pair)
}

func TestControlledPairAuditPostLinkSyncFailureReconcilesSequenceAndRetry(t *testing.T) {
	service, root := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	service.store.lifecycleAuditWriter = &postLinkFailingLifecycleAuditWriter{
		delegate: atomicLifecycleAuditWriter{}, fail: true,
	}
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err == nil {
		t.Fatal("post-link audit directory sync failure was accepted")
	}
	sequenceAfterFailure := service.store.lifecycle.sequence
	if sequenceAfterFailure == 0 {
		t.Fatal("committed audit record was not reconciled after sync failure")
	}
	service.store.lifecycleAuditWriter = atomicLifecycleAuditWriter{}
	created, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	)
	if err != nil || created.State != controlledPairStatePending {
		t.Fatalf("retry after audit uncertainty: state=%s err=%v", created.State, err)
	}
	if service.store.lifecycle.sequence <= sequenceAfterFailure {
		t.Fatalf("audit retry reused sequence %d", sequenceAfterFailure)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before reconciled audit restart: %v", err)
	}
	if _, err := newStandaloneStore(root); err != nil {
		t.Fatalf("restart after reconciled audit uncertainty: %v", err)
	}
}

func TestControlledPairIndexBatchNeverProjectsHalfPair(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	committed := make(chan struct{})
	release := make(chan struct{})
	basePersistence := service.store.pairPersistence
	pauseAfterControlledPairManifestState(
		service.store, controlledPairStatePending, committed, release,
	)
	result := make(chan error, 1)
	go func() {
		_, err := service.store.createControlledPairBundlesAs(
			SystemActor(), pair, baselineManifest, candidateManifest,
		)
		result <- err
	}()
	<-committed
	assertPairProjectionCount(t, service.store.runIndex.allRuns(), pair, 0)
	close(release)
	if err := <-result; err != nil {
		t.Fatalf("complete pair batch projection: %v", err)
	}
	assertPairProjectionCount(t, service.store.runIndex.allRuns(), pair, 2)

	startCommitted := make(chan struct{})
	startRelease := make(chan struct{})
	service.store.pairPersistence = basePersistence
	pauseAfterControlledPairManifestState(
		service.store, controlledPairStateRunning, startCommitted, startRelease,
	)
	go func() {
		service.store.lifecycle.mu.Lock()
		_, err := service.store.startControlledPairAs(SystemActor(), pair.PairID)
		service.store.lifecycle.mu.Unlock()
		result <- err
	}()
	<-startCommitted
	assertPairProjectionStatus(t, service.store.runIndex.allRuns(), pair, StatusPending)
	close(startRelease)
	if err := <-result; err != nil {
		t.Fatalf("complete pair start batch projection: %v", err)
	}
	assertPairProjectionStatus(t, service.store.runIndex.allRuns(), pair, StatusRunning)
}

func TestStoreRecoveryWaitsForActiveControlledPairPublication(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	intent := make(chan struct{})
	release := make(chan struct{})
	pauseAfterControlledPairManifestState(
		service.store, controlledPairStatePublishing, intent, release,
	)
	createResult := make(chan error, 1)
	go func() {
		_, err := service.store.createControlledPairBundlesAs(
			SystemActor(), pair, baselineManifest, candidateManifest,
		)
		createResult <- err
	}()
	<-intent
	reopenResult := make(chan struct {
		store *Store
		err   error
	}, 1)
	reopenAttempted := make(chan struct{})
	go func() {
		close(reopenAttempted)
		reopened, err := openTestPeerStore(t, service.store, LifecycleLimits{})
		reopenResult <- struct {
			store *Store
			err   error
		}{reopened, err}
	}()
	<-reopenAttempted
	select {
	case result := <-reopenResult:
		t.Fatalf("startup recovery crossed active publication barrier: %+v", result)
	default:
	}
	close(release)
	if err := <-createResult; err != nil {
		t.Fatalf("complete active pair publication: %v", err)
	}
	var result struct {
		store *Store
		err   error
	}
	select {
	case result = <-reopenResult:
		if result.err != nil {
			t.Fatalf("reopen after publication: %v", result.err)
		}
	case <-time.After(time.Second):
		t.Fatal("peer Store did not resume after controlled pair publication")
	}
	assertStrictPendingPair(t, result.store, pair)
}

func TestFreshEvaluationHierarchySyncsEveryParentAndStopsOnFailure(t *testing.T) {
	base := t.TempDir()
	target := filepath.Join(base, "evaluation", "controlled-pairs")
	var synced []string
	err := ensureDurablePrivateDirectoryTreeWithSync(target, func(path, _ string) error {
		synced = append(synced, path)
		return nil
	})
	if err != nil {
		t.Fatalf("create durable hierarchy: %v", err)
	}
	want := []string{base, filepath.Join(base, "evaluation"), filepath.Join(base, "evaluation")}
	if !reflect.DeepEqual(synced, want) {
		t.Fatalf("durable hierarchy sync order=%v, want %v", synced, want)
	}

	failureBase := t.TempDir()
	failureTarget := filepath.Join(failureBase, "evaluation", "controlled-pairs")
	err = ensureDurablePrivateDirectoryTreeWithSync(failureTarget, func(path, _ string) error {
		if path == failureBase {
			return errors.New("injected hierarchy parent sync failure")
		}
		return nil
	})
	if err == nil {
		t.Fatal("hierarchy parent sync failure was accepted")
	}
	if _, err := os.Lstat(filepath.Join(failureBase, "evaluation", "controlled-pairs")); !os.IsNotExist(err) {
		t.Fatalf("hierarchy advanced after failed parent sync: %v", err)
	}
	var retrySynced []string
	if err := ensureDurablePrivateDirectoryTreeWithSync(failureTarget, func(path, _ string) error {
		retrySynced = append(retrySynced, path)
		return nil
	}); err != nil {
		t.Fatalf("retry uncertain hierarchy: %v", err)
	}
	if len(retrySynced) == 0 || retrySynced[0] != failureBase {
		t.Fatalf("hierarchy retry did not resync uncertain parent chain: %v", retrySynced)
	}
}

func TestControlledPairAggregateRejectsInvalidContractWithoutResidualState(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*controlledPairManifest, *RunManifest, *RunManifest)
	}{
		{
			name: "non causal timestamp",
			mutate: func(pair *controlledPairManifest, _ *RunManifest, candidate *RunManifest) {
				pair.CandidateRun.CreatedAt = pair.BaselineRun.CreatedAt
				candidate.CreatedAt = pair.BaselineRun.CreatedAt
				refreshTestManifestDigest(t, candidate)
			},
		},
		{
			name: "candidate link",
			mutate: func(pair *controlledPairManifest, _ *RunManifest, candidate *RunManifest) {
				pair.CandidateRun.BaselineRunID = newTestClientRequestID()
				candidate.BaselineRunID = pair.CandidateRun.BaselineRunID
				refreshTestManifestDigest(t, candidate)
			},
		},
		{
			name: "candidate not pending",
			mutate: func(pair *controlledPairManifest, _ *RunManifest, _ *RunManifest) {
				startedAt := pair.CandidateRun.CreatedAt
				pair.CandidateRun.Status = StatusRunning
				pair.CandidateRun.StartedAt = &startedAt
			},
		},
		{
			name: "missing cohort digest",
			mutate: func(pair *controlledPairManifest, _ *RunManifest, _ *RunManifest) {
				pair.CohortDigest = ""
			},
		},
		{
			name: "source anchor binding",
			mutate: func(pair *controlledPairManifest, _ *RunManifest, _ *RunManifest) {
				pair.BaselineSourceAnchorDigest = digestString("different-source-anchor")
			},
		},
		{
			name: "member manifest binding",
			mutate: func(pair *controlledPairManifest, _ *RunManifest, _ *RunManifest) {
				pair.CandidateMemberManifestDigest = digestString("different-member-manifest")
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service, _ := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			test.mutate(&pair, &baselineManifest, &candidateManifest)
			_, err := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			)
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid controlled pair error=%v, want ErrInvalid", err)
			}
			assertControlledPairAbsent(t, service.store, pair)
		})
	}
}

func TestControlledPairAggregateEnforcesActorQuotaDestinationAndIdempotency(t *testing.T) {
	t.Run("actor", testControlledPairAggregateRejectsInvalidActor)

	t.Run("quota", func(t *testing.T) {
		service, _ := newControlledPairStoreTestService(t)
		pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
		service.store.lifecyclePolicy.Limits.MaxOwnerRuns = 1
		_, err := service.store.createControlledPairBundlesAs(
			SystemActor(), pair, baselineManifest, candidateManifest,
		)
		if !errors.Is(err, ErrQuota) {
			t.Fatalf("controlled pair quota error=%v, want ErrQuota", err)
		}
		assertControlledPairAbsent(t, service.store, pair)
	})

	t.Run("cross owner destination", func(t *testing.T) {
		service, _ := newControlledPairStoreTestService(t)
		ownerA := testLifecycleActor(t, "pair-owner-a", false)
		ownerB := testLifecycleActor(t, "pair-owner-b", false)
		pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, ownerB)
		occupant := pair.CandidateRun
		occupant.BaselineRunID = ""
		occupantManifest := candidateManifest
		occupantManifest.BaselineRunID = ""
		refreshTestManifestDigest(t, &occupantManifest)
		if _, err := service.store.CreateBundleAs(ownerA, occupant, occupantManifest); err != nil {
			t.Fatalf("occupy candidate destination: %v", err)
		}
		_, err := service.store.createControlledPairBundlesAs(ownerB, pair, baselineManifest, candidateManifest)
		if !errors.Is(err, ErrForbidden) {
			t.Fatalf("cross-owner destination error=%v, want ErrForbidden", err)
		}
		if _, err := service.store.GetRun(pair.BaselineRunID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("baseline remained after destination rejection: %v", err)
		}
	})

	t.Run("aggregate lifecycle ownership and retention", func(t *testing.T) {
		service, _ := newControlledPairStoreTestService(t)
		owner := testLifecycleActor(t, "pair-lifecycle-owner", false)
		other := testLifecycleActor(t, "pair-lifecycle-other", false)
		pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, owner)
		if _, err := service.store.createControlledPairBundlesAs(owner, pair, baselineManifest, candidateManifest); err != nil {
			t.Fatalf("publish controlled pair: %v", err)
		}
		if err := service.DeleteControlledPairExecutionAs(other, pair.PairID); !errors.Is(err, ErrForbidden) {
			t.Fatalf("cross-owner pair delete error=%v, want ErrForbidden", err)
		}
		hold := true
		if _, err := service.UpdateRunLifecycle(owner, pair.BaselineRunID, UpdateLifecycleRequest{EvidenceHold: &hold}); err != nil {
			t.Fatalf("hold pair member: %v", err)
		}
		if err := service.DeleteControlledPairExecutionAs(owner, pair.PairID); !errors.Is(err, ErrConflict) {
			t.Fatalf("held pair delete error=%v, want ErrConflict", err)
		}
		hold = false
		if _, err := service.UpdateRunLifecycle(owner, pair.BaselineRunID, UpdateLifecycleRequest{EvidenceHold: &hold}); err != nil {
			t.Fatalf("release pair member hold: %v", err)
		}
		protected := RetentionProtected
		if _, err := service.UpdateRunLifecycle(owner, pair.CandidateRunID, UpdateLifecycleRequest{RetentionClass: &protected}); err != nil {
			t.Fatalf("protect pair member: %v", err)
		}
		if err := service.DeleteControlledPairExecutionAs(owner, pair.PairID); !errors.Is(err, ErrConflict) {
			t.Fatalf("protected pair delete error=%v, want ErrConflict", err)
		}
	})

	t.Run("request identity", func(t *testing.T) {
		service, _ := newControlledPairStoreTestService(t)
		owner := testLifecycleActor(t, "pair-idempotent-owner", false)
		other := testLifecycleActor(t, "pair-idempotent-other", false)
		pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, owner)
		first, err := service.store.createControlledPairBundlesAs(owner, pair, baselineManifest, candidateManifest)
		if err != nil {
			t.Fatalf("first controlled pair create: %v", err)
		}
		second, err := service.store.createControlledPairBundlesAs(owner, pair, baselineManifest, candidateManifest)
		if err != nil || second.PairID != first.PairID {
			t.Fatalf("idempotent controlled pair create=%+v err=%v", second, err)
		}
		changed := pair
		changed.CohortDigest = digestString("different-cohort")
		if _, err := service.store.createControlledPairBundlesAs(owner, changed, baselineManifest, candidateManifest); !errors.Is(err, ErrConflict) {
			t.Fatalf("changed request identity error=%v, want ErrConflict", err)
		}
		changed.OwnerPrincipalDigest = other.principalDigest
		if _, err := service.store.createControlledPairBundlesAs(other, changed, baselineManifest, candidateManifest); !errors.Is(err, ErrForbidden) {
			t.Fatalf("cross-owner request identity error=%v, want ErrForbidden", err)
		}
	})
}

func testControlledPairAggregateRejectsInvalidActor(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	_, err := service.store.createControlledPairBundlesAs(Actor{}, pair, baselineManifest, candidateManifest)
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid actor error=%v, want ErrInvalid", err)
	}
	assertControlledPairAbsent(t, service.store, pair)
}
