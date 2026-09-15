package evaluationplane

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"
)

func createOwnedReadTestRun(t *testing.T, service *Service, actor Actor, name string) Run {
	t.Helper()
	request := validCreateRequest()
	request.Name = name
	run, err := service.CreateRunAs(context.Background(), actor, request)
	if err != nil {
		t.Fatalf("create %s: %v", name, err)
	}
	return run
}

func collectOwnedRunPages(
	t *testing.T,
	service *Service,
	actor Actor,
	expected map[string]bool,
) {
	t.Helper()
	cursor := ""
	seen := make(map[string]bool, len(expected))
	for pageIndex := 0; pageIndex < len(expected); pageIndex++ {
		page, err := service.ListRunLedgerPageAs(actor, RunListQuery{Limit: 1, Cursor: cursor})
		if err != nil {
			t.Fatalf("list owner page %d: %v", pageIndex, err)
		}
		if page.TotalRuns != len(expected) || len(page.Runs) != 1 {
			t.Fatalf("owner page %d=%+v, want one of %d runs", pageIndex, page, len(expected))
		}
		runID := page.Runs[0].ID
		if !expected[runID] || seen[runID] {
			t.Fatalf("owner page %d exposed unexpected or duplicate run %q", pageIndex, runID)
		}
		seen[runID] = true
		cursor = page.NextCursor
		if pageIndex < len(expected)-1 && cursor == "" {
			t.Fatalf("owner page %d omitted continuation cursor", pageIndex)
		}
	}
	if cursor != "" || !reflect.DeepEqual(seen, expected) {
		t.Fatalf("owner pagination seen=%v cursor=%q, want=%v and terminal cursor", seen, cursor, expected)
	}
}

func TestRunReadAPIsEnforceOwnerAndAdministratorBoundaries(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	ownerA := testLifecycleActor(t, "run-read-owner-a", false)
	ownerB := testLifecycleActor(t, "run-read-owner-b", false)
	administrator := testLifecycleActor(t, "run-read-administrator", true)
	a1 := createOwnedReadTestRun(t, service, ownerA, "owner a first")
	b1 := createOwnedReadTestRun(t, service, ownerB, "owner b first")
	a2 := createOwnedReadTestRun(t, service, ownerA, "owner a second")
	b2 := createOwnedReadTestRun(t, service, ownerB, "owner b second")
	// Status projections may be refreshed many times, but the owner projection
	// is write-once for a run identity.
	service.store.runIndex.upsertOwned(a1, ownerB.principalDigest)

	collectOwnedRunPages(t, service, ownerA, map[string]bool{a1.ID: true, a2.ID: true})
	collectOwnedRunPages(t, service, ownerB, map[string]bool{b1.ID: true, b2.ID: true})
	adminPage, adminPageErr := service.ListRunLedgerPageAs(administrator, RunListQuery{Limit: 10})
	if adminPageErr != nil || adminPage.TotalRuns != 4 || len(adminPage.Runs) != 4 {
		t.Fatalf("administrator ledger=%+v err=%v, want all four runs", adminPage, adminPageErr)
	}

	if run, err := service.GetRunAs(ownerA, a1.ID); err != nil || run.ID != a1.ID {
		t.Fatalf("owner GetRunAs run=%+v err=%v", run, err)
	}
	if _, err := service.GetRunAs(ownerB, a1.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("foreign GetRunAs error=%v, want ErrForbidden", err)
	}
	if _, err := service.GetRunAs(administrator, a1.ID); err != nil {
		t.Fatalf("administrator GetRunAs: %v", err)
	}

	if events, err := service.EventsAfterAs(ownerA, a1.ID, ""); err != nil || len(events) != 1 {
		t.Fatalf("owner EventsAfterAs events=%+v err=%v", events, err)
	}
	if _, err := service.EventsAfterAs(ownerB, a1.ID, ""); !errors.Is(err, ErrForbidden) {
		t.Fatalf("foreign EventsAfterAs error=%v, want ErrForbidden", err)
	}
	if _, err := service.EventsAfterAs(administrator, a1.ID, ""); err != nil {
		t.Fatalf("administrator EventsAfterAs: %v", err)
	}

	beforeCount, beforeRuns, beforeOwners := subscriberRegistryCounts(service)
	foreignEvents, foreignUnsubscribe, foreignSubscribeErr := service.SubscribeAs(ownerB, a1.ID)
	if !errors.Is(foreignSubscribeErr, ErrForbidden) || foreignEvents != nil || foreignUnsubscribe != nil {
		t.Fatalf("foreign SubscribeAs events=%v unsubscribeSet=%t error=%v", foreignEvents, foreignUnsubscribe != nil, foreignSubscribeErr)
	}
	afterCount, afterRuns, afterOwners := subscriberRegistryCounts(service)
	if afterCount != beforeCount || afterRuns != beforeRuns || afterOwners != beforeOwners {
		t.Fatalf(
			"foreign subscription changed registry: before=(%d,%d,%d) after=(%d,%d,%d)",
			beforeCount, beforeRuns, beforeOwners, afterCount, afterRuns, afterOwners,
		)
	}
	if _, unsubscribe, err := service.SubscribeAs(ownerA, a1.ID); err != nil {
		t.Fatalf("owner SubscribeAs: %v", err)
	} else {
		unsubscribe()
	}
	if _, unsubscribe, err := service.SubscribeAs(administrator, a1.ID); err != nil {
		t.Fatalf("administrator SubscribeAs: %v", err)
	} else {
		unsubscribe()
	}

	if _, err := service.ReportJSONAs(ownerB, a1.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("foreign ReportJSONAs error=%v, want ErrForbidden before report state", err)
	}
	if _, err := service.ReportJSONAs(ownerA, a1.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("owner pending ReportJSONAs error=%v, want ErrConflict", err)
	}
	if _, err := service.OpenArtifactAs(ownerB, a1.ID, "metrics"); !errors.Is(err, ErrForbidden) {
		t.Fatalf("foreign OpenArtifactAs error=%v, want ErrForbidden before artifact state", err)
	}
	if _, err := service.OpenArtifactAs(ownerA, a1.ID, "metrics"); !errors.Is(err, ErrConflict) {
		t.Fatalf("owner pending OpenArtifactAs error=%v, want ErrConflict", err)
	}
	if _, err := service.CompareAs(ownerA, a1.ID, b1.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("mixed-owner CompareAs error=%v, want second-arm ErrForbidden", err)
	}
	if _, err := service.CompareAs(administrator, a1.ID, b1.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("administrator pending CompareAs error=%v, want post-authorization ErrConflict", err)
	}

	firstCatalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("first shared catalog: %v", err)
	}
	secondCatalog, err := service.Catalog()
	firstCatalog.GeneratedAt = secondCatalog.GeneratedAt
	if err != nil || !reflect.DeepEqual(firstCatalog, secondCatalog) || len(firstCatalog.Suites) == 0 {
		t.Fatalf("catalog is not a shared stable capability view: equal=%v suites=%d err=%v", reflect.DeepEqual(firstCatalog, secondCatalog), len(firstCatalog.Suites), err)
	}
}

func waitForEvidenceReadReservation(t *testing.T, service *Service) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for len(service.activity.evidenceReads) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("authorized evidence read did not reach the shared lease boundary")
		}
		runtime.Gosched()
	}
}

func prepareOwnedEvidenceReadTestRun(
	t *testing.T,
	service *Service,
	owner Actor,
) (Run, string, []byte) {
	t.Helper()
	const eventMarker = "original owner evidence event"
	request := validCreateRequest()
	request.Name = "original owner report"
	run, err := service.CreateRunAs(context.Background(), owner, request)
	if err != nil {
		t.Fatalf("create authorized evidence read fixture: %v", err)
	}
	if _, err := service.store.AppendEvent(Event{
		RunID: run.ID, Type: "progress", Timestamp: time.Now().UTC(), Message: eventMarker,
	}); err != nil {
		t.Fatalf("append authorized evidence read marker: %v", err)
	}
	artifact := []byte("{\"owner\":\"original\"}\n")
	if err := os.WriteFile(
		filepath.Join(service.store.runsRoot, run.ID, "metrics.json"), artifact, 0o600,
	); err != nil {
		t.Fatalf("write authorized evidence read artifact: %v", err)
	}
	writeReportWithPublicReceipt(t, service, run, []Artifact{
		artifactForBytes("metrics", "metrics.json", "application/json", artifact),
	})
	return run, eventMarker, artifact
}

func assertEvidenceReadPinsAuthorizedIdentity(
	t *testing.T, testName string,
	read func(*Service, Actor, Run, string) (string, error),
	want func(Run, string, []byte) string,
) {
	t.Helper()
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "evidence-read-owner-"+testName, false)
	replacementOwner := testLifecycleActor(t, "evidence-read-replacement-"+testName, false)
	run, eventMarker, artifact := prepareOwnedEvidenceReadTestRun(t, service, owner)

	service.activity.evidenceMu.Lock()
	evidenceWriteLocked := true
	defer func() {
		if evidenceWriteLocked {
			service.activity.evidenceMu.Unlock()
		}
	}()
	readDone := make(chan struct {
		value string
		err   error
	}, 1)
	go func() {
		value, err := read(service, owner, run, eventMarker)
		readDone <- struct {
			value string
			err   error
		}{value: value, err: err}
	}()
	waitForEvidenceReadReservation(t, service)

	identityPinned := !service.store.lifecycle.mu.TryLock()
	if !identityPinned {
		service.store.lifecycle.mu.Unlock()
	}
	mutationDone := make(chan error, 1)
	go func() {
		if err := service.DeleteRunAs(owner, run.ID); err != nil {
			mutationDone <- err
			return
		}
		replacement := validCreateRequest()
		replacement.ClientRequestID = run.ID
		replacement.Name = "replacement owner report"
		_, err := service.CreateRunAs(context.Background(), replacementOwner, replacement)
		mutationDone <- err
	}()

	service.activity.evidenceMu.Unlock()
	evidenceWriteLocked = false
	select {
	case result := <-readDone:
		if result.err != nil || result.value != want(run, eventMarker, artifact) {
			t.Fatalf("authorized %s read value=%q err=%v", testName, result.value, result.err)
		}
	case <-time.After(3 * time.Second):
		t.Fatalf("authorized %s read did not resume", testName)
	}
	select {
	case err := <-mutationDone:
		if err != nil {
			t.Fatalf("delete and cross-owner rebind after %s read: %v", testName, err)
		}
	case <-time.After(3 * time.Second):
		t.Fatalf("delete and cross-owner rebind remained blocked after %s read", testName)
	}
	if !identityPinned {
		t.Fatalf("%s authorization released lifecycle identity before acquiring its evidence lease", testName)
	}
	if _, err := service.GetRunAs(owner, run.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("original owner read rebound %s identity: %v", testName, err)
	}
	if replacement, err := service.GetRunAs(replacementOwner, run.ID); err != nil || replacement.Name != "replacement owner report" {
		t.Fatalf("replacement owner %s run=%+v err=%v", testName, replacement, err)
	}
	if leases := len(service.activity.evidenceReads); leases != 0 {
		t.Fatalf("%s read leaked %d evidence leases", testName, leases)
	}
}

func TestRunEvidenceReadsPinAuthorizedIdentityAcrossDeleteAndRebind(t *testing.T) {
	tests := []struct {
		name string
		read func(*Service, Actor, Run, string) (string, error)
		want func(Run, string, []byte) string
	}{
		{
			name: "report",
			read: func(service *Service, actor Actor, run Run, _ string) (string, error) {
				data, err := service.ReportJSONAs(actor, run.ID)
				if err != nil {
					return "", err
				}
				report, err := decodeReportStrict(run.ID, data)
				return report.Run.Name, err
			},
			want: func(run Run, _ string, _ []byte) string { return run.Name },
		},
		{
			name: "artifact",
			read: func(service *Service, actor Actor, run Run, _ string) (string, error) {
				opened, err := service.OpenArtifactAs(actor, run.ID, "metrics")
				if err != nil {
					return "", err
				}
				data, readErr := io.ReadAll(opened.File)
				return string(data), errors.Join(readErr, opened.File.Close())
			},
			want: func(_ Run, _ string, artifact []byte) string { return string(artifact) },
		},
		{
			name: "events",
			read: func(service *Service, actor Actor, run Run, eventMarker string) (string, error) {
				events, err := service.EventsAfterAs(actor, run.ID, "")
				if err != nil {
					return "", err
				}
				for _, event := range events {
					if event.Message == eventMarker {
						return event.Message, nil
					}
				}
				return "", errors.New("original owner event is absent")
			},
			want: func(_ Run, eventMarker string, _ []byte) string { return eventMarker },
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertEvidenceReadPinsAuthorizedIdentity(t, test.name, test.read, test.want)
		})
	}
}

func TestRunLedgerScopesQuarantineWarningsToTheImmutableOwner(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	ownerA := testLifecycleActor(t, "run-ledger-owner-a", false)
	ownerB := testLifecycleActor(t, "run-ledger-owner-b", false)
	administrator := testLifecycleActor(t, "run-ledger-administrator", true)
	healthy := createOwnedReadTestRun(t, service, ownerA, "healthy owner run")
	quarantined := createOwnedReadTestRun(t, service, ownerB, "unknown owner run")
	originalLifecycle, lifecycleReadErr := service.store.readRunLifecycle(quarantined)
	if lifecycleReadErr != nil {
		t.Fatalf("read original lifecycle evidence: %v", lifecycleReadErr)
	}
	if err := os.WriteFile(
		filepath.Join(root, "runs", quarantined.ID, lifecycleFileName),
		[]byte("{not-json\n"),
		0o600,
	); err != nil {
		t.Fatalf("corrupt lifecycle evidence: %v", err)
	}
	if err := service.store.refreshRunIndex(); err != nil {
		t.Fatalf("refresh quarantined ledger: %v", err)
	}

	ownerPage, ownerPageErr := service.ListRunLedgerPageAs(ownerA, RunListQuery{Limit: 10})
	if ownerPageErr != nil || ownerPage.TotalRuns != 1 || len(ownerPage.Runs) != 1 || ownerPage.Runs[0].ID != healthy.ID {
		t.Fatalf("owner ledger=%+v err=%v", ownerPage, ownerPageErr)
	}
	if !ownerPage.LedgerComplete || ownerPage.WarningCount != 0 || len(ownerPage.Warnings) != 0 {
		t.Fatalf("unrelated owner observed another principal's quarantine state: %+v", ownerPage)
	}
	quarantinedOwnerPage, quarantinedOwnerPageErr := service.ListRunLedgerPageAs(ownerB, RunListQuery{Limit: 10})
	if quarantinedOwnerPageErr != nil || quarantinedOwnerPage.TotalRuns != 0 || len(quarantinedOwnerPage.Runs) != 0 ||
		quarantinedOwnerPage.LedgerComplete || quarantinedOwnerPage.WarningCount != 1 || len(quarantinedOwnerPage.Warnings) != 1 ||
		quarantinedOwnerPage.Warnings[0].EvidenceID != redactedRunLedgerEvidenceID {
		t.Fatalf("unknown-owner ledger=%+v err=%v", quarantinedOwnerPage, quarantinedOwnerPageErr)
	}
	adminPage, adminPageErr := service.ListRunLedgerPageAs(administrator, RunListQuery{Limit: 10})
	if adminPageErr != nil || len(adminPage.Warnings) != 1 || adminPage.Warnings[0].EvidenceID != quarantined.ID {
		t.Fatalf("administrator quarantine warning=%+v err=%v", adminPage.Warnings, adminPageErr)
	}
	changedOwner := originalLifecycle
	changedOwner.OwnerPrincipalDigest = ownerA.principalDigest
	changedOwner.PolicyDigest = ""
	changedOwner.PolicyDigest = lifecycleDigest(changedOwner)
	if err := writeJSONAtomic(
		filepath.Join(root, "runs", quarantined.ID, lifecycleFileName), changedOwner,
	); err != nil {
		t.Fatalf("write changed lifecycle owner: %v", err)
	}
	if err := service.store.refreshRunIndex(); err != nil {
		t.Fatalf("refresh changed lifecycle owner: %v", err)
	}
	if _, err := service.GetRunAs(ownerA, quarantined.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("changed lifecycle owner rebound run identity: %v", err)
	}
	adminPage, adminPageErr = service.ListRunLedgerPageAs(administrator, RunListQuery{Limit: 10})
	if adminPageErr != nil || len(adminPage.Warnings) != 1 || adminPage.Warnings[0].EvidenceID != quarantined.ID {
		t.Fatalf("changed-owner quarantine warning=%+v err=%v", adminPage.Warnings, adminPageErr)
	}
	if err := service.requireCompleteRunLedger(); !errors.Is(err, ErrConflict) {
		t.Fatalf("scientific completeness error=%v, want ErrConflict", err)
	}
}

func TestRunLedgerShowsUnattributableQuarantineOnlyToAdministrators(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-ledger-ordinary-owner", false)
	administrator := testLifecycleActor(t, "run-ledger-administrator", true)
	evidenceID := "unattributed-quarantine"
	runDirectory := filepath.Join(root, "runs", evidenceID)
	if err := os.Mkdir(runDirectory, 0o700); err != nil {
		t.Fatalf("create unattributed run directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDirectory, runFileName), []byte("{not-json\n"), 0o600); err != nil {
		t.Fatalf("write unattributed corrupt evidence: %v", err)
	}
	if err := service.store.refreshRunIndex(); err != nil {
		t.Fatalf("refresh unattributed quarantine: %v", err)
	}

	ownerPage, err := service.ListRunLedgerPageAs(owner, RunListQuery{Limit: 10})
	if err != nil || !ownerPage.LedgerComplete || ownerPage.WarningCount != 0 || len(ownerPage.Warnings) != 0 {
		t.Fatalf("ordinary owner observed unattributable quarantine: ledger=%+v err=%v", ownerPage, err)
	}
	adminPage, err := service.ListRunLedgerPageAs(administrator, RunListQuery{Limit: 10})
	if err != nil || adminPage.LedgerComplete || adminPage.WarningCount != 1 || len(adminPage.Warnings) != 1 ||
		adminPage.Warnings[0].EvidenceID != quarantinedEvidenceID(evidenceID) {
		t.Fatalf("administrator quarantine projection=%+v err=%v", adminPage, err)
	}
	if err := service.requireCompleteRunLedger(); !errors.Is(err, ErrConflict) {
		t.Fatalf("scientific completeness error=%v, want ErrConflict", err)
	}
}
