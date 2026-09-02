package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestListRunsIsolatesCorruptBundleAndRetainsWarning(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	first, createFirstErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createFirstErr != nil {
		t.Fatalf("create first run: %v", createFirstErr)
	}
	request := validCreateRequest()
	request.Name = "second run"
	second, createSecondErr := service.CreateRunAs(context.Background(), SystemActor(), request)
	if createSecondErr != nil {
		t.Fatalf("create second run: %v", createSecondErr)
	}

	statusPath := filepath.Join(root, "runs", second.ID, runFileName)
	if writeErr := os.WriteFile(statusPath, []byte("{not-json\n"), 0o600); writeErr != nil {
		t.Fatalf("corrupt second run status: %v", writeErr)
	}
	var logged bytes.Buffer
	previousLogOutput := log.Writer()
	log.SetOutput(&logged)
	t.Cleanup(func() { log.SetOutput(previousLogOutput) })
	if err := service.store.refreshRunIndex(); err != nil {
		t.Fatalf("refresh index after external corruption: %v", err)
	}

	ledger, listErr := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: defaultRunPageLimit})
	if listErr != nil {
		t.Fatalf("ListRunLedger with one corrupt bundle: %v", listErr)
	}
	runs := ledger.Runs
	if len(runs) != 1 || runs[0].ID != first.ID {
		t.Fatalf("ListRunLedger returned runs=%+v, want only intact run %s", runs, first.ID)
	}
	if ledger.LedgerComplete || ledger.SchemaVersion != SchemaVersion || len(ledger.Warnings) != 1 {
		t.Fatalf("corrupt ledger integrity metadata=%+v", ledger)
	}
	publicWarning := ledger.Warnings[0]
	if publicWarning.Code != corruptRunBundleWarningCode || publicWarning.EvidenceID != second.ID ||
		publicWarning.EvidenceFile != runFileName || publicWarning.Message != quarantinedRunMessage ||
		strings.Contains(publicWarning.Message, root) || strings.Contains(publicWarning.Message, "decode evaluation bundle") {
		t.Fatalf("public quarantine warning is missing or leaks diagnostics: %+v", publicWarning)
	}
	if _, getErr := service.GetRunAs(SystemActor(), second.ID); getErr == nil {
		t.Fatal("GetRun silently accepted the corrupt bundle")
	}
	warnings := service.store.activeRunListWarnings()
	if len(warnings) != 1 || warnings[0].Code != corruptRunBundleWarningCode || warnings[0].EvidenceID != second.ID ||
		!strings.Contains(warnings[0].Message, "decode evaluation bundle") {
		t.Fatalf("active run-list warnings=%+v, want structured corruption warning", warnings)
	}
	if _, repeatErr := service.store.ListRuns(); repeatErr != nil {
		t.Fatalf("repeat ListRuns with unchanged corruption: %v", repeatErr)
	}
	if count := strings.Count(logged.String(), "warning_code="+corruptRunBundleWarningCode); count != 1 {
		t.Fatalf("corruption warning logged %d times, want one transition log: %q", count, logged.String())
	}

	if repairErr := writeJSONAtomic(statusPath, second); repairErr != nil {
		t.Fatalf("repair second run status: %v", repairErr)
	}
	if err := service.store.refreshRunIndex(); err != nil {
		t.Fatalf("refresh index after external repair: %v", err)
	}
	ledger, listErr = service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: defaultRunPageLimit})
	if listErr != nil || len(ledger.Runs) != 2 || !ledger.LedgerComplete || len(ledger.Warnings) != 0 {
		t.Fatalf("ListRunLedger after repair returned %+v, err=%v", ledger, listErr)
	}
	if warnings := service.store.activeRunListWarnings(); len(warnings) != 0 {
		t.Fatalf("warning did not clear after bundle repair: %+v", warnings)
	}
}

func TestEmptyRunLedgerSerializesCanonicalArrays(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	runs, err := service.store.ListRuns()
	if err != nil || runs == nil || len(runs) != 0 {
		t.Fatalf("ListRuns=%+v err=%v, want a non-nil empty list", runs, err)
	}
	ledger, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: defaultRunPageLimit})
	if err != nil || ledger.Runs == nil || ledger.Warnings == nil {
		t.Fatalf("ListRunLedger=%+v err=%v, want canonical empty arrays", ledger, err)
	}
	payload, err := json.Marshal(ledger)
	if err != nil {
		t.Fatalf("marshal empty run ledger: %v", err)
	}
	if !bytes.Contains(payload, []byte(`"runs":[]`)) || !bytes.Contains(payload, []byte(`"warnings":[]`)) {
		t.Fatalf("empty run ledger JSON=%s, want canonical arrays", payload)
	}
}

func TestListRunLedgerUsesStableBoundedPages(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	created := make([]Run, 0, 3)
	baseTime := time.Date(2026, time.August, 30, 1, 2, 3, 0, time.UTC)
	for index, id := range []string{
		"10000000-0000-4000-8000-000000000001",
		"10000000-0000-4000-8000-000000000002",
		"10000000-0000-4000-8000-000000000003",
	} {
		request := validCreateRequest()
		request.ClientRequestID = id
		request.Name = id
		run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
		if err != nil {
			t.Fatalf("CreateRun %d: %v", index, err)
		}
		lifecycle, err := service.store.readRunLifecycle(run)
		if err != nil {
			t.Fatalf("read lifecycle %d: %v", index, err)
		}
		run.CreatedAt = baseTime.Add(time.Duration(index) * time.Minute)
		if err := service.store.updateRunFixture(run); err != nil {
			t.Fatalf("UpdateRun %d: %v", index, err)
		}
		lifecycle.CreatedAt = run.CreatedAt
		lifecycle.PolicyDigest = ""
		lifecycle.PolicyDigest = lifecycleDigest(lifecycle)
		if err := writeJSONAtomic(
			filepath.Join(service.store.runsRoot, run.ID, lifecycleFileName), lifecycle,
		); err != nil {
			t.Fatalf("rewrite lifecycle %d: %v", index, err)
		}
		created = append(created, run)
	}

	first, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 2})
	if err != nil {
		t.Fatalf("first page: %v", err)
	}
	if len(first.Runs) != 2 || first.Runs[0].ID != created[2].ID || first.Runs[1].ID != created[1].ID ||
		first.NextCursor == "" || first.TotalRuns != 3 || first.WarningCount != 0 || !first.LedgerComplete {
		t.Fatalf("first page=%+v", first)
	}
	second, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 2, Cursor: first.NextCursor})
	if err != nil {
		t.Fatalf("second page: %v", err)
	}
	if len(second.Runs) != 1 || second.Runs[0].ID != created[0].ID || second.NextCursor != "" || second.TotalRuns != 3 {
		t.Fatalf("second page=%+v", second)
	}
	if _, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: maxRunPageLimit + 1}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("oversized page error=%v, want ErrInvalid", err)
	}
	if _, err := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 2, Cursor: "not-a-cursor"}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("invalid cursor error=%v, want ErrInvalid", err)
	}
}

func TestRunLedgerIndexIsMaintainedAcrossStoreInstances(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	peer := newTestPeerStore(t, service.store)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	peerPage, peerPageErr := peer.listRunLedger(SystemActor(), RunListQuery{Limit: 10})
	if peerPageErr != nil || len(peerPage.Runs) != 1 || peerPage.Runs[0].ID != run.ID {
		t.Fatalf("peer index after create=%+v err=%v", peerPage, peerPageErr)
	}

	now := time.Now().UTC()
	run.Status = StatusCancelled
	run.CompletedAt = &now
	run.Progress.Message = "Run cancelled"
	if updateErr := peer.updateRunFixture(run); updateErr != nil {
		t.Fatalf("peer UpdateRun: %v", updateErr)
	}
	servicePage, servicePageErr := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10})
	if servicePageErr != nil || len(servicePage.Runs) != 1 || servicePage.Runs[0].Status != StatusCancelled {
		t.Fatalf("service index after peer update=%+v err=%v", servicePage, servicePageErr)
	}
	if deleteErr := peer.DeleteRunAs(SystemActor(), run.ID); deleteErr != nil {
		t.Fatalf("peer DeleteRun: %v", deleteErr)
	}
	servicePage, servicePageErr = service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10})
	if servicePageErr != nil || len(servicePage.Runs) != 0 || servicePage.TotalRuns != 0 {
		t.Fatalf("service index after peer delete=%+v err=%v", servicePage, servicePageErr)
	}
}

func TestRunLedgerPageUsesIndexUntilExplicitIntegrityRefresh(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	if writeErr := os.WriteFile(filepath.Join(root, "runs", run.ID, runFileName), []byte("not-json\n"), 0o600); writeErr != nil {
		t.Fatalf("corrupt status outside Store: %v", writeErr)
	}
	page, pageErr := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10})
	if pageErr != nil || len(page.Runs) != 1 || page.Runs[0].ID != run.ID || !page.LedgerComplete {
		t.Fatalf("indexed page unexpectedly rescanned durable history: page=%+v err=%v", page, pageErr)
	}
	if integrityErr := service.requireCompleteRunLedger(); !errors.Is(integrityErr, ErrConflict) {
		t.Fatalf("integrity refresh error=%v, want ErrConflict", integrityErr)
	}
	page, pageErr = service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: 10})
	if pageErr != nil || len(page.Runs) != 0 || page.LedgerComplete || page.WarningCount != 1 {
		t.Fatalf("refreshed page=%+v err=%v", page, pageErr)
	}
}

func TestRunLedgerQuarantinesUnknownStatusFieldsAndUnexpectedEntries(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	encoded, marshalErr := json.Marshal(run)
	if marshalErr != nil {
		t.Fatalf("marshal run: %v", marshalErr)
	}
	encoded = append(encoded[:len(encoded)-1], []byte(`,"unexpected_field":true}`)...)
	if writeErr := os.WriteFile(filepath.Join(root, "runs", run.ID, runFileName), encoded, 0o600); writeErr != nil {
		t.Fatalf("write status with unknown field: %v", writeErr)
	}
	if writeErr := os.WriteFile(filepath.Join(root, "runs", "unexpected-entry"), []byte("not a bundle\n"), 0o600); writeErr != nil {
		t.Fatalf("write unexpected runs entry: %v", writeErr)
	}
	if refreshErr := service.store.refreshRunIndex(); refreshErr != nil {
		t.Fatalf("refresh index after external corruption: %v", refreshErr)
	}

	ledger, ledgerErr := service.ListRunLedgerPageAs(SystemActor(), RunListQuery{Limit: defaultRunPageLimit})
	if ledgerErr != nil {
		t.Fatalf("ListRunLedger: %v", ledgerErr)
	}
	if ledger.LedgerComplete || len(ledger.Runs) != 0 || ledger.WarningCount != 2 || len(ledger.Warnings) != 2 {
		t.Fatalf("ledger did not quarantine all non-current entries: %+v", ledger)
	}
	hashedEntryID := digestBytes([]byte("unexpected-entry"))
	foundHashedEntry := false
	for _, warning := range ledger.Warnings {
		if warning.EvidenceID == "unexpected-entry" {
			t.Fatal("quarantine warning exposed an arbitrary filesystem entry name")
		}
		if warning.EvidenceID == hashedEntryID {
			foundHashedEntry = true
		}
	}
	if !foundHashedEntry {
		t.Fatalf("quarantine warning omitted hashed evidence identity %q: %+v", hashedEntryID, ledger.Warnings)
	}
	if _, err := service.GetRunAs(SystemActor(), run.ID); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("GetRun unknown-field error=%v", err)
	}
}

func TestRecoverInterruptedRunsContinuesPastCorruptBundle(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	running, createRunningErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createRunningErr != nil {
		t.Fatalf("create running candidate: %v", createRunningErr)
	}
	running = stageRunningTestRun(t, service, running)
	request := validCreateRequest()
	request.Name = "corrupt run"
	corrupt, createCorruptErr := service.CreateRunAs(context.Background(), SystemActor(), request)
	if createCorruptErr != nil {
		t.Fatalf("create corrupt candidate: %v", createCorruptErr)
	}
	if writeErr := os.WriteFile(
		filepath.Join(root, "runs", corrupt.ID, runFileName),
		[]byte("not-json\n"),
		0o600,
	); writeErr != nil {
		t.Fatalf("corrupt status: %v", writeErr)
	}

	if recoverErr := service.RecoverInterruptedRuns(); recoverErr != nil {
		t.Fatalf("RecoverInterruptedRuns: %v", recoverErr)
	}
	recovered, getErr := service.GetRunAs(SystemActor(), running.ID)
	if getErr != nil || recovered.Status != StatusFailed || !strings.Contains(recovered.Error, "restarted") {
		t.Fatalf("valid interrupted run was not recovered: run=%+v err=%v", recovered, getErr)
	}
	warnings := service.store.activeRunListWarnings()
	if len(warnings) != 1 || warnings[0].EvidenceID != corrupt.ID {
		t.Fatalf("corrupt bundle warning was not retained during recovery: %+v", warnings)
	}
}

func TestDirectCreateWithBaselineRequiresACompleteLedger(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	baseline, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun baseline: %v", err)
	}
	baseline = completeTestRun(t, service, baseline)
	if err := os.WriteFile(filepath.Join(root, "runs", "unexpected-entry"), []byte("not a run\n"), 0o600); err != nil {
		t.Fatalf("write quarantined entry: %v", err)
	}
	service.codeRevision = strings.Repeat("b", 40)
	candidate := validCreateRequest()
	candidate.Name = "candidate"
	candidate.BaselineRunID = baseline.ID
	if _, err := service.CreateRunAs(context.Background(), SystemActor(), candidate); !errors.Is(err, ErrConflict) ||
		!strings.Contains(err.Error(), "ledger is incomplete") {
		t.Fatalf("direct baseline create error=%v, want incomplete-ledger conflict", err)
	}
}

func TestListRunsRejectsStatusIdentityAndStateCorruption(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	statusPath := filepath.Join(root, "runs", run.ID, runFileName)

	tests := []struct {
		name   string
		mutate func(*Run)
		match  string
	}{
		{name: "schema", mutate: func(candidate *Run) { candidate.SchemaVersion = "evaluation.v2" }, match: "schema_version"},
		{name: "identity", mutate: func(candidate *Run) { candidate.ID = "different-run" }, match: "identity"},
		{name: "state", mutate: func(candidate *Run) { candidate.Status = "unknown" }, match: "state"},
		{name: "pending started", mutate: func(candidate *Run) {
			started := candidate.CreatedAt.Add(time.Second)
			candidate.StartedAt = &started
		}, match: "pending status"},
		{name: "running without started", mutate: func(candidate *Run) {
			candidate.Status = StatusRunning
		}, match: "running status"},
		{name: "sealing without started", mutate: func(candidate *Run) {
			candidate.Status = StatusSealing
		}, match: "sealing status"},
		{name: "completed without terminal progress", mutate: func(candidate *Run) {
			started := candidate.CreatedAt.Add(time.Second)
			completed := started.Add(time.Second)
			candidate.Status, candidate.StartedAt, candidate.CompletedAt = StatusCompleted, &started, &completed
		}, match: "completed status"},
		{name: "failed without error", mutate: func(candidate *Run) {
			started := candidate.CreatedAt.Add(time.Second)
			completed := started.Add(time.Second)
			candidate.Status, candidate.StartedAt, candidate.CompletedAt = StatusFailed, &started, &completed
		}, match: "failed status"},
		{name: "cancelled without completion", mutate: func(candidate *Run) {
			candidate.Status = StatusCancelled
		}, match: "cancelled status"},
		{name: "started before creation", mutate: func(candidate *Run) {
			started := candidate.CreatedAt.Add(-time.Second)
			candidate.Status, candidate.StartedAt = StatusRunning, &started
		}, match: "predates"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := run
			test.mutate(&candidate)
			if err := writeJSONAtomic(statusPath, candidate); err != nil {
				t.Fatalf("write corrupt status: %v", err)
			}
			if _, err := service.GetRunAs(SystemActor(), run.ID); err == nil {
				t.Fatal("GetRun silently accepted corrupt status metadata")
			}
			runs, err := service.store.ListRuns()
			if err != nil || len(runs) != 0 {
				t.Fatalf("ListRuns returned runs=%+v err=%v, want isolated bundle", runs, err)
			}
			warnings := service.store.activeRunListWarnings()
			if len(warnings) != 1 || warnings[0].EvidenceID != run.ID || !strings.Contains(warnings[0].Message, test.match) {
				t.Fatalf("warning=%+v, want %q corruption", warnings, test.match)
			}
		})
	}
}
