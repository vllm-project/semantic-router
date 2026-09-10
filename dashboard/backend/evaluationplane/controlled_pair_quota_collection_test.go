package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestControlledPairQuotaProductRejectsOverflow(t *testing.T) {
	if _, ok := checkedPositiveInt64Product(int64(^uint64(0)>>1), 2); ok {
		t.Fatal("quota reservation multiplication accepted overflow")
	}
}

func TestControlledPairQuotaReservesMaximumAggregateEnvelopeAtExactBoundary(t *testing.T) {
	for _, allow := range []bool{false, true} {
		t.Run(fmt.Sprintf("allow_%v", allow), func(t *testing.T) {
			service, _ := newControlledPairStoreTestService(t)
			pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
			envelope, err := controlledPairIntentReservationBytes(pair)
			if err != nil {
				t.Fatalf("measure pair envelope: %v", err)
			}
			usage, err := service.store.Usage(SystemActor())
			if err != nil {
				t.Fatalf("read quota baseline: %v", err)
			}
			var owner OwnerLifecycleUsage
			for _, candidate := range usage.Owners {
				if candidate.PrincipalDigest == SystemActor().principalDigest {
					owner = candidate
				}
			}
			runGrowth, ok := checkedPositiveInt64Product(service.store.lifecyclePolicy.ReservedRunBytes, 2)
			if !ok {
				t.Fatal("run reservation overflow")
			}
			growth, err := checkedLifecycleBytes(runGrowth, envelope)
			if err != nil {
				t.Fatalf("pair reservation overflow: %v", err)
			}
			limitDelta := int64(0)
			if !allow {
				limitDelta = -1
			}
			service.store.lifecyclePolicy.Limits.MaxOwnerBytes = owner.ChargeableBytes + growth + limitDelta
			service.store.lifecyclePolicy.Limits.MaxStoreBytes = usage.ChargeableBytes + growth + limitDelta
			_, createErr := service.store.createControlledPairBundlesAs(
				SystemActor(), pair, baselineManifest, candidateManifest,
			)
			if allow && createErr != nil {
				t.Fatalf("exact pair envelope boundary rejected: %v", createErr)
			}
			if !allow && !errors.Is(createErr, ErrQuota) {
				t.Fatalf("one byte below pair envelope error=%v, want ErrQuota", createErr)
			}
		})
	}
}

func TestControlledPairAggregateStatesFitReservedEnvelope(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, _, _ := pendingControlledPairAggregate(t, service, SystemActor())
	reservation, err := controlledPairIntentReservationBytes(pair)
	if err != nil {
		t.Fatal(err)
	}
	publishing := pair
	publishing.State = controlledPairStatePublishing
	publishing.BaselineStageName = stagedRunBundlePrefix + "1234567890"
	publishing.CandidateStageName = stagedRunBundlePrefix + "0987654321"
	now := time.Unix(1_999_999_999, 999_999_999).UTC()
	running := pair
	running.State, running.StartedAt = controlledPairStateRunning, &now
	running.StartReceiptDigest = digestString("test start receipt")
	running.BaselineRun = controlledPairRunningSnapshot(running.BaselineRun, now)
	running.CandidateRun = controlledPairRunningSnapshot(running.CandidateRun, now)
	terminal := running
	terminal.State = controlledPairStateTerminal
	for _, run := range []*Run{&terminal.BaselineRun, &terminal.CandidateRun} {
		run.Status, run.CompletedAt = StatusFailed, &now
		run.Progress.Message = strings.Repeat("\x00", maxWorkerMessageBytes)
		run.Error = strings.Repeat("\x00", maxWorkerMessageBytes)
	}
	deleted := terminal
	deleted.State, deleted.DeletedAt = controlledPairStateDeleted, &now
	deleted.DeleteReceiptDigest = digestString("test delete receipt")
	for name, state := range map[string]controlledPairManifest{
		"pending": pair, "publishing": publishing, "running": running,
		"terminal": terminal, "deleted": deleted,
	} {
		encoded, err := json.MarshalIndent(state, "", "  ")
		if err != nil {
			t.Fatalf("encode %s: %v", name, err)
		}
		if size := int64(len(encoded) + 1); size > reservation {
			t.Fatalf("%s aggregate bytes=%d exceed reservation=%d", name, size, reservation)
		}
		stateReservation, err := controlledPairIntentReservationBytes(state)
		if err != nil {
			t.Fatalf("measure %s reservation: %v", name, err)
		}
		if stateReservation != reservation {
			t.Fatalf("%s reservation=%d changed from admission=%d", name, stateReservation, reservation)
		}
	}
}

func TestControlledPairEnvelopeReservationPersistsThroughTerminalState(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
		t.Fatalf("publish pair: %v", err)
	}
	envelope, operationErr := controlledPairIntentReservationBytes(pair)
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	pairBytes, operationErr := privateDirectoryBytes(filepath.Join(service.store.controlledPairRoot, pair.PairID), "")
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	usage, operationErr := service.store.Usage(SystemActor())
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	var owner OwnerLifecycleUsage
	for _, candidate := range usage.Owners {
		if candidate.PrincipalDigest == SystemActor().principalDigest {
			owner = candidate
		}
	}
	wantAggregateReservation := envelope - pairBytes
	if wantAggregateReservation < 0 {
		wantAggregateReservation = 0
	}
	if owner.ReservedBytes < wantAggregateReservation {
		t.Fatalf("owner reservation=%d omits aggregate envelope gap=%d", owner.ReservedBytes, wantAggregateReservation)
	}
	service.store.lifecyclePolicy.Limits.MaxOwnerBytes = owner.ChargeableBytes
	service.store.lifecyclePolicy.Limits.MaxStoreBytes = usage.ChargeableBytes
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest()); !errors.Is(err, ErrQuota) {
		t.Fatalf("second create consumed persistent pair reservation: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	if _, err := service.store.startControlledPairAs(SystemActor(), pair.PairID); err != nil {
		service.store.lifecycle.mu.Unlock()
		t.Fatalf("start within reserved envelope: %v", err)
	}
	if _, err := service.store.cancelControlledPairAs(SystemActor(), pair.PairID); err != nil {
		service.store.lifecycle.mu.Unlock()
		t.Fatalf("terminal transition within reserved envelope: %v", err)
	}
	service.store.lifecycle.mu.Unlock()
	terminalUsage, operationErr := service.store.Usage(SystemActor())
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	if terminalUsage.ChargeableBytes > terminalUsage.MaxStoreBytes {
		t.Fatalf("terminal pair escaped persistent envelope: usage=%+v", terminalUsage)
	}
}

func TestLifecycleCollectionReclaimsExpiredTerminalPairAtomically(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
		t.Fatalf("publish pair: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	if _, err := service.store.startControlledPairAs(SystemActor(), pair.PairID); err != nil {
		service.store.lifecycle.mu.Unlock()
		t.Fatalf("start pair: %v", err)
	}
	if _, err := service.store.cancelControlledPairAs(SystemActor(), pair.PairID); err != nil {
		service.store.lifecycle.mu.Unlock()
		t.Fatalf("cancel pair: %v", err)
	}
	service.store.lifecycle.mu.Unlock()
	service.store.lifecycleNow = func() time.Time { return time.Now().UTC().Add(31 * 24 * time.Hour) }
	beforeBytes, err := privateDirectoryBytes(service.store.root, service.store.lifecycleAuditRoot)
	if err != nil {
		t.Fatal(err)
	}
	beforeCollectionBytes, err := privateDirectoryBytes(service.store.collectionRoot, "")
	if err != nil {
		t.Fatal(err)
	}
	planResult, err := service.CollectLifecycle(SystemActor(), CollectionRequest{})
	if err != nil {
		t.Fatalf("plan pair collection: %v", err)
	}
	var pairItem *CollectionPlanItem
	for index := range planResult.Plan.Candidates {
		if planResult.Plan.Candidates[index].PairID == pair.PairID {
			pairItem = &planResult.Plan.Candidates[index]
		}
	}
	if pairItem == nil || len(pairItem.RunIDs) != 2 || pairItem.EstimatedBytes <= 0 {
		t.Fatalf("collection omitted atomic pair identity/bytes: %+v", planResult.Plan.Candidates)
	}
	applied, err := service.CollectLifecycle(SystemActor(), CollectionRequest{
		Apply: true, PlanDigest: planResult.Plan.PlanDigest,
	})
	if err != nil {
		t.Fatalf("apply pair collection: %v", err)
	}
	if len(applied.DeletedPairIDs) != 1 || applied.DeletedPairIDs[0] != pair.PairID || len(applied.DeletedRunIDs) != 2 {
		t.Fatalf("atomic pair collection result=%+v", applied)
	}
	afterBytes, err := privateDirectoryBytes(service.store.root, service.store.lifecycleAuditRoot)
	if err != nil {
		t.Fatal(err)
	}
	afterCollectionBytes, err := privateDirectoryBytes(service.store.collectionRoot, "")
	if err != nil {
		t.Fatal(err)
	}
	if reclaimed := (beforeBytes - beforeCollectionBytes) - (afterBytes - afterCollectionBytes); reclaimed != pairItem.EstimatedBytes {
		t.Fatalf("estimated reclaim=%d, physical reclaim=%d", pairItem.EstimatedBytes, reclaimed)
	}
	tombstonePath := filepath.Join(service.store.controlledPairRoot, pair.PairID, controlledPairTombstoneFile)
	tombstoneBytes, err := readEvidenceBytes(tombstonePath, maxStructuredArtifactBytes)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(tombstoneBytes), `"baseline_run":`) ||
		strings.Contains(string(tombstoneBytes), `"candidate_run":`) {
		t.Fatalf("identity tombstone retained full member snapshots: %s", tombstoneBytes)
	}
	for _, sourceID := range []string{pair.BaselineSourceRunID, pair.CandidateSourceRunID} {
		if _, err := service.store.GetRun(sourceID); err != nil {
			t.Fatalf("pair collection cascaded into source %s: %v", sourceID, err)
		}
	}
}

func TestControlledPairCancelPreservesTerminalArmAndRejectsSealingBeforeIntent(t *testing.T) {
	t.Run("half_terminal", func(t *testing.T) {
		service, _ := newControlledPairStoreTestService(t)
		pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
		if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
			t.Fatal(err)
		}
		service.store.lifecycle.mu.Lock()
		start, operationErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
		service.store.lifecycle.mu.Unlock()
		if operationErr != nil {
			t.Fatal(operationErr)
		}
		completed := start.Baseline
		completedAt := time.Now().UTC().Truncate(time.Microsecond)
		completed.Status, completed.CompletedAt = StatusCompleted, &completedAt
		completed.Progress.Percent, completed.Progress.Completed = 100, completed.Progress.Total
		completed.Progress.CurrentTrackID, completed.Progress.Message = "", "Baseline evidence sealed"
		if err := service.store.updateRunFixture(completed); err != nil {
			t.Fatal(err)
		}
		service.store.lifecycle.mu.Lock()
		cancelled, operationErr := service.store.cancelControlledPairAs(SystemActor(), pair.PairID)
		service.store.lifecycle.mu.Unlock()
		if operationErr != nil || !reflect.DeepEqual(cancelled.BaselineRun, completed) || cancelled.CandidateRun.Status != StatusCancelled {
			t.Fatalf("half-terminal cancellation=%+v err=%v", cancelled, operationErr)
		}
	})

	t.Run("sealing", func(t *testing.T) {
		service, _ := newControlledPairStoreTestService(t)
		pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
		if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
			t.Fatal(err)
		}
		service.store.lifecycle.mu.Lock()
		start, operationErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
		service.store.lifecycle.mu.Unlock()
		if operationErr != nil {
			t.Fatal(operationErr)
		}
		sealing := start.Baseline
		sealing.Status, sealing.Progress.Message = StatusSealing, "Sealing evaluation evidence"
		if err := service.store.updateRunFixture(sealing); err != nil {
			t.Fatal(err)
		}
		service.store.lifecycle.mu.Lock()
		_, operationErr = service.store.cancelControlledPairAs(SystemActor(), pair.PairID)
		service.store.lifecycle.mu.Unlock()
		if !errors.Is(operationErr, ErrConflict) {
			t.Fatalf("sealing cancellation error=%v, want ErrConflict", operationErr)
		}
		durable, operationErr := service.store.readControlledPair(pair.PairID)
		if operationErr != nil || durable.State != controlledPairStateRunning {
			t.Fatalf("sealing cancellation published intent: state=%s err=%v", durable.State, operationErr)
		}
	})
}
