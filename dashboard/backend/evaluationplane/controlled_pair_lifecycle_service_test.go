package evaluationplane

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

type countingRunStatusPersistence struct {
	delegate runStatusPersistence
	writes   atomic.Int32
}

func (p *countingRunStatusPersistence) Write(path string, run Run) error {
	p.writes.Add(1)
	return p.delegate.Write(path, run)
}

func (p *countingRunStatusPersistence) SyncDirectory(path, description string) error {
	return p.delegate.SyncDirectory(path, description)
}

func TestControlledPairLifecycleCancelsAndDeletesAsOneAggregate(t *testing.T) {
	process := &controlledPairStoreTestProcess{controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)}}
	service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = service.Close() })
	baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	if _, err := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err != nil {
		t.Fatalf("create controlled pair execution: %v", err)
	}
	for range 2 {
		select {
		case <-process.started:
		case <-time.After(time.Second):
			t.Fatal("controlled pair worker did not start")
		}
	}
	cancelled, operationErr := service.CancelControlledPairExecutionAs(SystemActor(), request.ClientRequestID)
	if operationErr != nil {
		t.Fatalf("cancel controlled pair: %v", operationErr)
	}
	if cancelled.BaselineRun.Status != StatusCancelled || cancelled.CandidateRun.Status != StatusCancelled {
		t.Fatalf("controlled pair cancellation split: %+v", cancelled)
	}
	if cancelled.State != controlledPairStateTerminal || cancelled.Capabilities.CanCancel || cancelled.Capabilities.CanDelete {
		t.Fatalf("terminal controlled pair capabilities are invalid: %+v", cancelled.Capabilities)
	}
	deadline := time.Now().Add(time.Second)
	for {
		service.mu.Lock()
		active := len(service.active)
		service.mu.Unlock()
		if active == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("controlled pair workers did not exit: active=%d", active)
		}
		time.Sleep(time.Millisecond)
	}
	settled, operationErr := service.GetControlledPairExecutionAs(SystemActor(), request.ClientRequestID)
	if operationErr != nil || !settled.Capabilities.CanDelete {
		t.Fatalf("settled controlled pair capabilities=%+v err=%v", settled.Capabilities, operationErr)
	}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), request.ClientRequestID); err != nil {
		t.Fatalf("delete controlled pair: %v", err)
	}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), request.ClientRequestID); err != nil {
		t.Fatalf("idempotent controlled pair delete: %v", err)
	}
	for _, runID := range []string{request.BaselineRunID, request.CandidateRunID} {
		if _, err := service.store.GetRun(runID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("deleted pair member %s remains visible: %v", runID, err)
		}
	}
	tombstone, operationErr := service.store.readControlledPair(request.ClientRequestID)
	if operationErr != nil || tombstone.State != controlledPairStateDeleted ||
		tombstone.DeletedAt == nil || !digestPattern.MatchString(tombstone.DeleteReceiptDigest) {
		t.Fatalf("controlled pair tombstone=%+v err=%v", tombstone, operationErr)
	}
	if err := service.DeleteRunAs(SystemActor(), baselineSource.ID); err != nil {
		t.Fatalf("released baseline source reference: %v", err)
	}
	if err := service.DeleteRunAs(SystemActor(), candidateSource.ID); err != nil {
		t.Fatalf("released candidate source reference: %v", err)
	}
}

func TestControlledPairAuthoritativeReadUsesAggregateStateAndOwnership(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	actor := testLifecycleActor(t, "pair-read-owner", false)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, actor)
	if _, err := service.store.createControlledPairBundlesAs(actor, pair, baselineManifest, candidateManifest); err != nil {
		t.Fatalf("publish pair: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	start, operationErr := service.store.startControlledPairAs(actor, pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if operationErr != nil {
		t.Fatalf("start pair: %v", operationErr)
	}
	half := start.Baseline
	completedAt := time.Now().UTC().Truncate(time.Microsecond)
	half.Status, half.CompletedAt, half.Progress.Message = StatusCompleted, &completedAt, "Baseline completed"
	half.Progress.Percent, half.Progress.Completed, half.Progress.CurrentTrackID = 100, half.Progress.Total, ""
	if err := service.store.updateRunFixture(half); err != nil {
		t.Fatalf("persist split physical state: %v", err)
	}
	execution, operationErr := service.GetControlledPairExecutionAs(actor, pair.PairID)
	if operationErr != nil || execution.State != controlledPairStateRunning || !execution.Capabilities.CanCancel || execution.Capabilities.CanDelete {
		t.Fatalf("authoritative pair read=%+v err=%v", execution, operationErr)
	}
	sealing := start.Candidate
	sealing.Status, sealing.CompletedAt, sealing.Progress.Message = StatusSealing, nil, "Sealing evaluation evidence"
	if err := service.store.updateRunFixture(sealing); err != nil {
		t.Fatalf("persist sealing physical state: %v", err)
	}
	execution, operationErr = service.GetControlledPairExecutionAs(actor, pair.PairID)
	if operationErr != nil || execution.Capabilities.CanCancel || execution.Capabilities.CanDelete {
		t.Fatalf("sealing pair capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
	other := testLifecycleActor(t, "pair-read-other", false)
	if _, err := service.GetControlledPairExecutionAs(other, pair.PairID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner pair read error=%v, want ErrForbidden", err)
	}
	if _, err := service.GetControlledPairExecutionAs(actor, newTestClientRequestID()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing pair read error=%v, want ErrNotFound", err)
	}
}

func TestControlledPairCapabilitiesAndDeleteSeeWorkersOwnedByAnotherService(t *testing.T) {
	process := &controlledPairStoreTestProcess{controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)}}
	service, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = service.Close() })
	baselineSource := createSealedControlledPairSource(t, service, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, service, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	execution, operationErr := service.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request)
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	for range 2 {
		<-process.started
	}
	for _, run := range []Run{execution.BaselineRun, execution.CandidateRun} {
		terminal, _ := service.buildTerminalRun(run, errors.New("test terminal while worker exits"))
		if err := service.store.updateRunFixture(terminal); err != nil {
			t.Fatal(err)
		}
	}
	if err := service.store.refreshControlledPairTerminalState(execution.BaselineRun.ID); err != nil {
		t.Fatal(err)
	}

	second, operationErr := newControlledPairTestService(Options{
		DataDir: service.store.Root(), PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		DeploymentsDir: service.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: &controlledPairStoreTestProcess{},
	})
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	t.Cleanup(func() { _ = second.Close() })
	visible, operationErr := second.GetControlledPairExecutionAs(SystemActor(), execution.ID)
	if operationErr != nil || visible.State != controlledPairStateTerminal || visible.Capabilities.CanDelete {
		t.Fatalf("cross-service active capability=%+v err=%v", visible, operationErr)
	}
	if err := second.DeleteControlledPairExecutionAs(SystemActor(), execution.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("cross-service active delete error=%v, want ErrConflict", err)
	}
	second.store.lifecycleNow = func() time.Time { return time.Now().UTC().Add(31 * 24 * time.Hour) }
	activePlan, operationErr := second.CollectLifecycle(SystemActor(), CollectionRequest{})
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	for _, candidate := range activePlan.Plan.Candidates {
		if candidate.PairID == execution.ID {
			t.Fatalf("collection exposed pair with cross-service active owner: %+v", candidate)
		}
	}
	if activePlan.Plan.Skipped["active"] == 0 {
		t.Fatalf("collection did not account for cross-service active pair: %+v", activePlan.Plan.Skipped)
	}
	for _, runID := range []string{execution.BaselineRun.ID, execution.CandidateRun.ID} {
		if _, err := second.store.GetRun(runID); err != nil {
			t.Fatalf("active collection touched member %s: %v", runID, err)
		}
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close worker owner: %v", err)
	}
	settledPlan, operationErr := second.CollectLifecycle(SystemActor(), CollectionRequest{})
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	foundPair := false
	for _, candidate := range settledPlan.Plan.Candidates {
		foundPair = foundPair || candidate.PairID == execution.ID
	}
	if !foundPair {
		t.Fatalf("collection omitted pair after owner exit: %+v", settledPlan.Plan.Candidates)
	}
	if _, err := second.CollectLifecycle(SystemActor(), CollectionRequest{
		Apply: true, PlanDigest: settledPlan.Plan.PlanDigest,
	}); err != nil {
		t.Fatalf("collect pair after owner exit: %v", err)
	}
	if _, err := second.GetControlledPairExecutionAs(SystemActor(), execution.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("collected pair remains visible: %v", err)
	}
}

func TestControlledPairCancelFromSecondServiceStopsOwnerWorkers(t *testing.T) {
	process := &controlledPairStoreTestProcess{controlledProcess: controlledProcess{started: make(chan ProcessSpec, 2)}}
	owner, baselineTargetID, candidateTargetID := newControlledPairExecutionTestService(t, process, 2)
	t.Cleanup(func() { _ = owner.Close() })
	baselineSource := createSealedControlledPairSource(t, owner, baselineTargetID)
	candidateSource := createSealedControlledPairSource(t, owner, candidateTargetID)
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.ID,
		CandidateSourceRunID: candidateSource.ID, BaselineRunID: newTestClientRequestID(),
		CandidateRunID: newTestClientRequestID(),
	}
	if _, err := owner.CreateControlledPairExecutionAs(context.Background(), SystemActor(), request); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		select {
		case <-process.started:
		case <-time.After(time.Second):
			t.Fatal("owner worker did not start")
		}
	}
	ownerWrites := &countingRunStatusPersistence{delegate: owner.store.statusPersistence}
	owner.store.statusPersistence = ownerWrites

	second, err := newControlledPairTestService(Options{
		DataDir: owner.store.Root(), PythonPath: "python3", ConfigPath: owner.registrySource.configPath,
		DeploymentsDir: owner.registrySource.deploymentsDir, CodeRevision: testSourceRevision,
		MaxConcurrent: 2, Process: &controlledPairStoreTestProcess{},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = second.Close() })
	cancelled, err := second.CancelControlledPairExecutionAs(SystemActor(), request.ClientRequestID)
	if err != nil || cancelled.State != controlledPairStateTerminal {
		t.Fatalf("remote pair cancel=%+v err=%v", cancelled, err)
	}
	deadline := time.Now().Add(time.Second)
	for {
		owner.mu.Lock()
		active := len(owner.active)
		owner.mu.Unlock()
		if active == 0 && !owner.activity.contains(request.BaselineRunID) &&
			!owner.activity.contains(request.CandidateRunID) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("remote cancellation did not stop owner workers: active=%d", active)
		}
		time.Sleep(time.Millisecond)
	}
	if writes := ownerWrites.writes.Load(); writes != 0 {
		t.Fatalf("owner workers rewrote %d terminal snapshots after remote cancellation", writes)
	}
	for _, runID := range []string{request.BaselineRunID, request.CandidateRunID} {
		run, err := second.store.GetRun(runID)
		if err != nil || run.Status != StatusCancelled {
			t.Fatalf("remote cancellation member=%+v err=%v", run, err)
		}
	}
}

func TestControlledPairReadCapabilitiesPreflightActiveAndLifecyclePolicy(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	actor := testLifecycleActor(t, "pair-capability-owner", false)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, actor)
	if _, err := service.store.createControlledPairBundlesAs(actor, pair, baselineManifest, candidateManifest); err != nil {
		t.Fatalf("publish pair: %v", err)
	}

	execution, operationErr := service.GetControlledPairExecutionAs(actor, pair.PairID)
	if operationErr != nil || !execution.Capabilities.CanDelete || execution.Capabilities.CanCancel {
		t.Fatalf("pending pair capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
	service.mu.Lock()
	service.active[pair.BaselineRunID] = func() {}
	service.mu.Unlock()
	execution, operationErr = service.GetControlledPairExecutionAs(actor, pair.PairID)
	if operationErr != nil || execution.Capabilities.CanDelete {
		t.Fatalf("active pair capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
	service.mu.Lock()
	delete(service.active, pair.BaselineRunID)
	service.mu.Unlock()

	hold := true
	if _, err := service.UpdateRunLifecycle(actor, pair.CandidateRunID, UpdateLifecycleRequest{EvidenceHold: &hold}); err != nil {
		t.Fatalf("hold pair member: %v", err)
	}
	execution, operationErr = service.GetControlledPairExecutionAs(actor, pair.PairID)
	if operationErr != nil || execution.Capabilities.CanDelete {
		t.Fatalf("held pair capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
	hold = false
	protected := RetentionProtected
	if _, err := service.UpdateRunLifecycle(actor, pair.CandidateRunID, UpdateLifecycleRequest{
		EvidenceHold: &hold, RetentionClass: &protected,
	}); err != nil {
		t.Fatalf("protect pair member: %v", err)
	}
	execution, operationErr = service.GetControlledPairExecutionAs(actor, pair.PairID)
	if operationErr != nil || execution.Capabilities.CanDelete {
		t.Fatalf("protected pair capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
	standard := RetentionStandard
	if _, err := service.UpdateRunLifecycle(actor, pair.CandidateRunID, UpdateLifecycleRequest{RetentionClass: &standard}); err != nil {
		t.Fatalf("restore standard retention: %v", err)
	}

	campaign := campaignV2StoredSchemaFixture(t)
	campaign.GateBindings.G2RunID = pair.BaselineRunID
	campaign.Decision.Evidence[0].RunID = pair.BaselineRunID
	anchors, anchorErr := validateCampaignEvidenceAnchors(campaign.GateBindings, campaign.Decision.Evidence)
	if anchorErr != nil {
		t.Fatalf("validate campaign reference anchors: %v", anchorErr)
	}
	for index := range campaign.Decision.Gates {
		switch campaign.Decision.Gates[index].ID {
		case "G0", "G1":
			campaign.Decision.Gates[index].EvidenceRefs = campaignAnchorRefs(campaign.GateBindings, anchors)
		case "G2":
			campaign.Decision.Gates[index].EvidenceRefs = campaignAnchorRefsForKeys(anchors, "g2:evidence")
		}
	}
	var digestErr error
	campaign.ManifestDigest, digestErr = campaignManifestDigest(campaign)
	if digestErr != nil {
		t.Fatalf("digest campaign reference fixture: %v", digestErr)
	}
	campaign.Decision.CampaignDigest = campaign.ManifestDigest
	campaign.Decision.DecisionDigest, digestErr = campaignDecisionDigest(campaign.Decision)
	if digestErr != nil {
		t.Fatalf("digest campaign decision fixture: %v", digestErr)
	}
	if err := validateStoredCampaign(campaign.ID, campaign); err != nil {
		t.Fatalf("campaign reference fixture is invalid: %v", err)
	}
	writeCampaignLifecycleFixture(t, service.store, campaign, actor)
	execution, operationErr = service.GetControlledPairExecutionAs(actor, pair.PairID)
	if operationErr != nil || execution.Capabilities.CanDelete {
		t.Fatalf("campaign-referenced pair capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
}

func TestControlledPairDeleteClosesBothMemberSubscriptions(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
		t.Fatalf("publish pair: %v", err)
	}
	baselineEvents, _, err := service.SubscribeAs(SystemActor(), pair.BaselineRunID)
	if err != nil {
		t.Fatalf("subscribe baseline: %v", err)
	}
	candidateEvents, _, err := service.SubscribeAs(SystemActor(), pair.CandidateRunID)
	if err != nil {
		t.Fatalf("subscribe candidate: %v", err)
	}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); err != nil {
		t.Fatalf("delete pair: %v", err)
	}
	for role, channel := range map[string]<-chan Event{"baseline": baselineEvents, "candidate": candidateEvents} {
		if _, open := <-channel; open {
			t.Fatalf("%s subscription remained open after aggregate deletion", role)
		}
	}
	if count, runs, owners := subscriberRegistryCounts(service); count != 0 || runs != 0 || owners != 0 {
		t.Fatalf("pair subscribers leaked: count=%d runs=%d owners=%d", count, runs, owners)
	}
}

func TestControlledPairDeleteRejectsOrdinaryBaselineReference(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(SystemActor(), pair, baselineManifest, candidateManifest); err != nil {
		t.Fatalf("publish pair: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	start, operationErr := service.store.startControlledPairAs(SystemActor(), pair.PairID)
	service.store.lifecycle.mu.Unlock()
	if operationErr != nil {
		t.Fatalf("start pair: %v", operationErr)
	}
	completedAt := time.Now().UTC().Truncate(time.Microsecond)
	for _, run := range []Run{start.Baseline, start.Candidate} {
		run.Status, run.CompletedAt, run.Progress.Message = StatusCompleted, &completedAt, "Completed"
		run.Progress.Percent, run.Progress.Completed, run.Progress.CurrentTrackID = 100, run.Progress.Total, ""
		if err := service.store.updateRunFixture(run); err != nil {
			t.Fatalf("complete pair member: %v", err)
		}
	}
	if err := service.store.refreshControlledPairTerminalState(pair.BaselineRunID); err != nil {
		t.Fatalf("commit pair terminal state: %v", err)
	}
	baseline, operationErr := service.store.GetRun(pair.BaselineRunID)
	if operationErr != nil {
		t.Fatalf("read completed baseline: %v", operationErr)
	}
	external := baseline
	external.ID, external.ClientRequestID = newTestClientRequestID(), newTestClientRequestID()
	external.ClientRequestID = external.ID
	external.Name, external.Description = "external pair baseline reference", "reference integrity fixture"
	external.Status, external.BaselineRunID, external.ControlledPair = StatusPending, baseline.ID, nil
	external.Progress = RunProgress{Total: len(external.TrackIDs), Message: "Run created"}
	external.CreatedAt, external.StartedAt, external.CompletedAt, external.Error = completedAt.Add(time.Microsecond), nil, nil, ""
	externalManifest, _, operationErr := service.readDurableManifest(baseline.ID)
	if operationErr != nil {
		t.Fatalf("read pair baseline manifest: %v", operationErr)
	}
	externalManifest.RunID, externalManifest.Name, externalManifest.Description = external.ID, external.Name, external.Description
	externalManifest.BaselineRunID, externalManifest.CreatedAt = baseline.ID, external.CreatedAt
	refreshTestManifestDigest(t, &externalManifest)
	if _, operationErr = service.store.CreateBundleAs(SystemActor(), external, externalManifest); operationErr != nil {
		t.Fatalf("create external candidate: %v", operationErr)
	}
	execution, operationErr := service.GetControlledPairExecutionAs(SystemActor(), pair.PairID)
	if operationErr != nil || execution.Capabilities.CanDelete {
		t.Fatalf("externally referenced pair capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), pair.PairID); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete externally referenced pair error=%v, want ErrConflict", err)
	}
	if _, err := service.store.GetRun(pair.BaselineRunID); err != nil {
		t.Fatalf("reference denial removed pair member: %v", err)
	}
}

func TestControlledPairDeleteRejectsActivePairSourceReferences(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	first, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(SystemActor(), first, baselineManifest, candidateManifest); err != nil {
		t.Fatal(err)
	}
	service.store.lifecycle.mu.Lock()
	start, operationErr := service.store.startControlledPairAs(SystemActor(), first.PairID)
	service.store.lifecycle.mu.Unlock()
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	completedAt := time.Now().UTC().Truncate(time.Microsecond)
	completedRuns := make([]Run, 0, 2)
	for _, run := range []Run{start.Baseline, start.Candidate} {
		run.Status, run.CompletedAt, run.Progress.Message = StatusCompleted, &completedAt, "Completed"
		run.Progress.Percent, run.Progress.Completed, run.Progress.CurrentTrackID = 100, run.Progress.Total, ""
		if err := service.store.updateRunFixture(run); err != nil {
			t.Fatal(err)
		}
		completedRuns = append(completedRuns, run)
	}
	baselineSource := sealExistingControlledPairMember(t, service, completedRuns[0])
	candidateSource := sealExistingControlledPairMember(t, service, completedRuns[1])
	if err := service.store.refreshControlledPairTerminalState(first.BaselineRunID); err != nil {
		t.Fatal(err)
	}
	createdAt := completedAt.Add(time.Microsecond)
	secondBaseline, secondBaselineManifest, operationErr := cloneControlledPairRun(
		baselineSource, newTestClientRequestID(), "", controlledPairRoleBaseline, createdAt,
	)
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	secondCandidate, secondCandidateManifest, operationErr := cloneControlledPairRun(
		candidateSource, newTestClientRequestID(), secondBaseline.ID, controlledPairRoleCandidate, createdAt.Add(time.Microsecond),
	)
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: first.BaselineRunID,
		CandidateSourceRunID: first.CandidateRunID, BaselineRunID: secondBaseline.ID,
		CandidateRunID: secondCandidate.ID,
	}
	second, operationErr := newControlledPairManifest(
		SystemActor(), request, baselineSource, candidateSource,
		secondBaseline, secondCandidate, secondBaselineManifest, secondCandidateManifest,
	)
	if operationErr != nil {
		t.Fatal(operationErr)
	}
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), second, secondBaselineManifest, secondCandidateManifest,
	); err != nil {
		t.Fatalf("publish referencing pair: %v", err)
	}
	execution, operationErr := service.GetControlledPairExecutionAs(SystemActor(), first.PairID)
	if operationErr != nil || execution.Capabilities.CanDelete {
		t.Fatalf("pair referenced by active aggregate capabilities=%+v err=%v", execution.Capabilities, operationErr)
	}
	if err := service.DeleteControlledPairExecutionAs(SystemActor(), first.PairID); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete pair referenced by active pair error=%v, want ErrConflict", err)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); err != nil {
		t.Fatalf("reference denial left invalid graph: %v", err)
	}
}
