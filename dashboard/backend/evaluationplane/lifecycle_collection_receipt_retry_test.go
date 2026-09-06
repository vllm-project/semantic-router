package evaluationplane

import (
	"bytes"
	"errors"
	"os"
	"reflect"
	"testing"
	"time"
)

func readLifecycleCollectionForTest(
	t *testing.T,
	service *Service,
) lifecycleCollectionTransaction {
	t.Helper()
	service.store.lifecycle.mu.Lock()
	defer service.store.lifecycle.mu.Unlock()
	transaction, exists, err := service.store.readLifecycleCollectionTransactionUnlocked()
	if err != nil || !exists {
		t.Fatalf("read lifecycle collection receipt: exists=%t err=%v", exists, err)
	}
	return transaction
}

func TestLifecycleCollectionRetryReplacesOldReceiptAfterUnpublishedHeader(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	actor := SystemActor()
	oldRuns := expiredCollectionRuns(t, service, 1)
	oldPlan, oldPlanErr := service.CollectLifecycle(actor, CollectionRequest{})
	if oldPlanErr != nil {
		t.Fatalf("build old lifecycle collection plan: %v", oldPlanErr)
	}
	oldRequest := CollectionRequest{Apply: true, PlanDigest: oldPlan.Plan.PlanDigest}
	oldReceipt, oldReceiptErr := service.CollectLifecycle(actor, oldRequest)
	if oldReceiptErr != nil || len(oldReceipt.DeletedRunIDs) != 1 {
		t.Fatalf("complete old lifecycle collection: receipt=%+v err=%v", oldReceipt, oldReceiptErr)
	}
	if oldReceipt.DeletedRunIDs[0] != oldRuns[0].ID {
		t.Fatalf("old receipt deleted runs=%v, want %s", oldReceipt.DeletedRunIDs, oldRuns[0].ID)
	}

	newRuns := expiredCollectionRuns(t, service, 1)
	newPlan, newPlanErr := service.CollectLifecycle(actor, CollectionRequest{})
	if newPlanErr != nil || len(newPlan.Plan.Candidates) != 1 {
		t.Fatalf("build new lifecycle collection plan: result=%+v err=%v", newPlan, newPlanErr)
	}
	if newPlan.Plan.PlanDigest == oldPlan.Plan.PlanDigest {
		t.Fatal("new lifecycle collection plan reused the old digest")
	}
	faults := &faultingLifecycleCollectionPersistence{
		delegate: atomicLifecycleCollectionPersistence{}, failWriteAt: 1,
	}
	service.store.collectionPersistence = faults
	newRequest := CollectionRequest{Apply: true, PlanDigest: newPlan.Plan.PlanDigest}
	if _, err := service.CollectLifecycle(actor, newRequest); !errors.Is(err, ErrConflict) {
		t.Fatalf("unpublished replacement header error=%v, want ErrConflict", err)
	}
	service.store.lifecycle.mu.Lock()
	pending := service.store.lifecycle.pendingCollection
	if pending == nil {
		service.store.lifecycle.mu.Unlock()
		t.Fatal("unpublished replacement header did not retain its exact plan")
	}
	failedPlan := pending.Transaction.Plan
	service.store.lifecycle.mu.Unlock()
	service.store.lifecycleNow = func() time.Time { return failedPlan.GeneratedAt.Add(time.Hour) }
	oldDurable := readLifecycleCollectionForTest(t, service)
	if oldDurable.State != collectionStateCompleted ||
		oldDurable.Plan.PlanDigest != oldPlan.Plan.PlanDigest {
		t.Fatalf("failed header publication changed old receipt: %+v", oldDurable)
	}

	other, actorErr := NewActor("collection-receipt-other-admin", true)
	if actorErr != nil {
		t.Fatal(actorErr)
	}
	if _, err := service.CollectLifecycle(other, newRequest); !errors.Is(err, ErrConflict) {
		t.Fatalf("cross-actor retry error=%v, want ErrConflict", err)
	}
	otherPlan := CollectionRequest{Apply: true, PlanDigest: digestString("other collection plan")}
	if _, err := service.CollectLifecycle(actor, otherPlan); !errors.Is(err, ErrConflict) {
		t.Fatalf("cross-plan retry error=%v, want ErrConflict", err)
	}
	peer, peerErr := openTestPeerStore(t, service.store, LifecycleLimits{})
	if peer != nil || !errors.Is(peerErr, ErrConflict) {
		t.Fatalf("peer advanced unpublished header: peer=%v err=%v", peer, peerErr)
	}
	if faults.writeCalls != 1 {
		t.Fatalf("non-matching retry or peer performed %d writes, want 1", faults.writeCalls)
	}
	if _, err := service.GetRunAs(SystemActor(), newRuns[0].ID); err != nil {
		t.Fatalf("peer changed new collection candidate: %v", err)
	}

	receipt, receiptErr := service.CollectLifecycle(actor, newRequest)
	if receiptErr != nil || !reflect.DeepEqual(receipt.DeletedRunIDs, []string{newRuns[0].ID}) {
		t.Fatalf("matching replacement retry: receipt=%+v err=%v", receipt, receiptErr)
	}
	if faults.resolveCalls != 1 {
		t.Fatalf("matching retry resolved the old receipt %d times, want 1", faults.resolveCalls)
	}
	durable := readLifecycleCollectionForTest(t, service)
	service.store.lifecycle.mu.Lock()
	pending = service.store.lifecycle.pendingCollection
	service.store.lifecycle.mu.Unlock()
	if durable.State != collectionStateCompleted || !reflect.DeepEqual(durable.Plan, failedPlan) ||
		pending != nil {
		t.Fatalf("replacement receipt was not completed durably: %+v", durable)
	}
}

func TestLifecycleCollectionRejectsProgressBeyondWriterFrame(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	expiredCollectionRuns(t, service, 1)
	plan, planErr := service.CollectLifecycle(SystemActor(), CollectionRequest{})
	if planErr != nil {
		t.Fatal(planErr)
	}
	request := CollectionRequest{Apply: true, PlanDigest: plan.Plan.PlanDigest}
	if _, err := service.CollectLifecycle(SystemActor(), request); err != nil {
		t.Fatalf("complete lifecycle collection fixture: %v", err)
	}
	path := service.store.lifecycleCollectionPath()
	data, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	lines := bytes.Split(bytes.TrimSuffix(data, []byte{'\n'}), []byte{'\n'})
	if len(lines) != 2 || len(lines[1]) >= maxLifecycleCollectionProgressBytes {
		t.Fatalf("unexpected lifecycle collection fixture framing: lines=%d progress=%d", len(lines), len(lines[1]))
	}
	lines[1] = append(
		lines[1],
		bytes.Repeat([]byte{' '}, maxLifecycleCollectionProgressBytes-len(lines[1]))...,
	)
	if err := os.WriteFile(path, append(bytes.Join(lines, []byte{'\n'}), '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	service.store.lifecycle.mu.Lock()
	_, _, transactionErr := service.store.readLifecycleCollectionTransactionUnlocked()
	service.store.lifecycle.mu.Unlock()
	if !errors.Is(transactionErr, ErrInvalid) {
		t.Fatalf("oversized progress frame error=%v, want ErrInvalid", transactionErr)
	}
}
