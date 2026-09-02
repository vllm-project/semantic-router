package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func createBaselineAndCandidateForReferenceTest(t *testing.T) (*Service, string, Run, Run) {
	t.Helper()
	service, root := newTestService(t, &controlledProcess{}, 1)
	baseline, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create baseline: %v", err)
	}
	baseline = completeTestRun(t, service, baseline)
	service.codeRevision = strings.Repeat("b", 40)
	request := validCreateRequest()
	request.Name = "candidate"
	request.BaselineRunID = baseline.ID
	candidate, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	return service, root, baseline, candidate
}

func TestServiceDeletePreservesRunBaselineReferences(t *testing.T) {
	service, _, baseline, candidate := createBaselineAndCandidateForReferenceTest(t)
	if err := service.DeleteRunAs(SystemActor(), baseline.ID); !errors.Is(err, ErrConflict) ||
		!strings.Contains(err.Error(), candidate.ID) {
		t.Fatalf("delete referenced baseline error=%v, want candidate-bound conflict", err)
	}
	if err := service.DeleteRunAs(SystemActor(), candidate.ID); err != nil {
		t.Fatalf("delete candidate: %v", err)
	}
	if err := service.DeleteRunAs(SystemActor(), baseline.ID); err != nil {
		t.Fatalf("delete released baseline: %v", err)
	}
}

func TestCandidateBaselineOwnershipIsAtomicAcrossStoreInstances(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "baseline-owner", false)
	other := testLifecycleActor(t, "baseline-other", false)
	baseline, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create owned baseline: %v", err)
	}
	baseline = completeTestRun(t, service, baseline)

	request := validCreateRequest()
	request.BaselineRunID = baseline.ID
	candidate, manifest := preparePendingRun(t, service, request)
	peer := newTestPeerStore(t, service.store)
	if _, err := peer.CreateBundleAs(other, candidate, manifest); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner baseline publication error=%v, want ErrForbidden", err)
	}
	if _, err := peer.GetRun(candidate.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("denied cross-owner candidate became visible: %v", err)
	}
	reopened := newTestPeerStore(t, service.store)
	if _, err := reopened.GetRun(candidate.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("denied candidate appeared after restart: %v", err)
	}
	if err := service.DeleteRunAs(owner, baseline.ID); err != nil {
		t.Fatalf("cross-owner denial left a baseline reference pin: %v", err)
	}
	assertLifecycleAuditDecision(t, lifecycleAuditRecords(service.store), "create", "denied")
}

func TestCandidateBaselineAuthorizationHidesForeignState(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "baseline-state-owner", false)
	other := testLifecycleActor(t, "baseline-state-other", false)
	completed, completedErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if completedErr != nil {
		t.Fatalf("create completed foreign baseline: %v", completedErr)
	}
	completed = completeTestRun(t, service, completed)
	pending, pendingErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if pendingErr != nil {
		t.Fatalf("create pending foreign baseline: %v", pendingErr)
	}
	incomparableRequest := validCreateRequest()
	incomparableRequest.Seed++
	incomparable, incomparableErr := service.CreateRunAs(context.Background(), owner, incomparableRequest)
	if incomparableErr != nil {
		t.Fatalf("create incomparable foreign baseline: %v", incomparableErr)
	}
	incomparable = completeTestRun(t, service, incomparable)

	cases := map[string]string{
		"missing":      newTestClientRequestID(),
		"completed":    completed.ID,
		"nonterminal":  pending.ID,
		"incomparable": incomparable.ID,
	}
	wantMessage := ""
	for name, baselineID := range cases {
		t.Run(name, func(t *testing.T) {
			request := validCreateRequest()
			request.BaselineRunID = baselineID
			_, err := service.CreateRunAs(context.Background(), other, request)
			if !errors.Is(err, ErrForbidden) {
				t.Fatalf("foreign baseline error=%v, want ErrForbidden", err)
			}
			if wantMessage == "" {
				wantMessage = err.Error()
			} else if err.Error() != wantMessage {
				t.Fatalf("foreign baseline leaked state: error=%q want=%q", err.Error(), wantMessage)
			}
			if _, lookupErr := service.store.GetRun(request.ClientRequestID); !errors.Is(lookupErr, ErrNotFound) {
				t.Fatalf("foreign baseline denial published candidate: %v", lookupErr)
			}
		})
	}

	completedPath := filepath.Join(service.store.runsRoot, completed.ID, runFileName)
	if err := os.WriteFile(completedPath, []byte("not-json\n"), 0o600); err != nil {
		t.Fatalf("corrupt foreign baseline status: %v", err)
	}
	request := validCreateRequest()
	request.BaselineRunID = completed.ID
	_, deniedErr := service.CreateRunAs(context.Background(), other, request)
	if !errors.Is(deniedErr, ErrForbidden) || deniedErr.Error() != wantMessage {
		t.Fatalf("unreadable foreign baseline leaked state: error=%q want=%q", deniedErr, wantMessage)
	}
	if err := writeJSONAtomic(completedPath, completed); err != nil {
		t.Fatalf("restore foreign baseline status: %v", err)
	}
}

func TestStoreDeleteFailsClosedWhenRunReferenceLedgerIsCorrupt(t *testing.T) {
	service, root, baseline, candidate := createBaselineAndCandidateForReferenceTest(t)
	if err := os.WriteFile(
		filepath.Join(root, "runs", candidate.ID, runFileName),
		[]byte("not-json\n"),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	if err := service.store.DeleteRunAs(SystemActor(), baseline.ID); !errors.Is(err, ErrConflict) ||
		!strings.Contains(err.Error(), "cannot be verified") {
		t.Fatalf("delete with corrupt reference ledger error=%v", err)
	}
}

func TestStoreStartupRejectsDanglingBaselineReference(t *testing.T) {
	service, root, baseline, _ := createBaselineAndCandidateForReferenceTest(t)
	if err := os.RemoveAll(filepath.Join(root, "runs", baseline.ID)); err != nil {
		t.Fatal(err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before dangling baseline restart: %v", err)
	}
	if _, err := newStandaloneStore(root); !errors.Is(err, ErrInvalid) ||
		!strings.Contains(err.Error(), "baseline reference") {
		t.Fatalf("restart with dangling baseline error=%v, want ErrInvalid", err)
	}
}
