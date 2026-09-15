package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

type failOnceLifecycleCheckpointCleaner struct {
	targetDirectory string
	failSync        bool
	failed          bool
}

func (cleaner *failOnceLifecycleCheckpointCleaner) Remove(path string) error {
	if !cleaner.failSync && !cleaner.failed && filepath.Dir(path) == cleaner.targetDirectory {
		cleaner.failed = true
		return errors.New("injected lifecycle checkpoint cleanup failure")
	}
	return os.Remove(path)
}

func (cleaner *failOnceLifecycleCheckpointCleaner) Sync(directory, purpose string) error {
	if cleaner.failSync && !cleaner.failed && directory == cleaner.targetDirectory {
		cleaner.failed = true
		return errors.New("injected lifecycle checkpoint cleanup sync failure")
	}
	return syncEvaluationDirectory(directory, purpose)
}

func smallestLifecycleAuditLimits() LifecycleLimits {
	limits := DefaultLifecycleLimits()
	limits.MaxAuditBytes = maxLifecycleRecordSize
	limits.MaxOwnerRuns = 64
	limits.MaxOwnerCampaigns = 64
	return limits
}

func TestLifecycleAuditCheckpointRotatesAndRecoversRunBinding(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, root := newLifecycleTestService(t, limits)
	owner := testLifecycleActor(t, "checkpoint-run-owner", false)
	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create retained run: %v", err)
	}

	remnantPath, remnant := checkpointLifecycleAuditAndCaptureTail(t, service.store)
	assertLifecycleCheckpoint(t, service.store)
	if _, err := os.Lstat(filepath.Join(
		service.store.lifecycleBindingRoot,
		lifecycleBindingFileName(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID}),
	)); err != nil {
		t.Fatalf("checkpoint omitted retained run creation binding: %v", err)
	}
	if service.store.lifecycle.sequence <= service.store.lifecycle.activeCount {
		t.Fatalf(
			"audit did not rotate: sequence=%d active=%d",
			service.store.lifecycle.sequence, service.store.lifecycle.activeCount,
		)
	}
	if err := os.WriteFile(remnantPath, remnant, 0o600); err != nil {
		t.Fatalf("restore checkpoint cleanup crash remnant: %v", err)
	}

	if _, err := openTestPeerStore(t, service.store, limits); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer open error=%v, want checkpoint cleanup conflict", err)
	}
	if _, err := os.Lstat(remnantPath); err != nil {
		t.Fatalf("peer opener removed checkpoint crash remnant: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before checkpoint cleanup restart: %v", err)
	}
	reopenedService := reopenTestService(t, root)
	reopened := reopenedService.store
	if _, err := reopened.RunLifecycle(owner, run.ID); err != nil {
		t.Fatalf("restart lost checkpointed run binding: %v", err)
	}
	if _, err := os.Lstat(remnantPath); !os.IsNotExist(err) {
		t.Fatalf("restart did not collect verified checkpoint crash remnant: %v", err)
	}
	if err := reopened.DeleteRunAs(owner, run.ID); err != nil {
		t.Fatalf("delete checkpointed run: %v", err)
	}
	forceCreateDeleteAuditRotation(t, reopenedService, owner)
	if _, err := os.Lstat(filepath.Join(
		reopened.lifecycleBindingRoot,
		lifecycleBindingFileName(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID}),
	)); !os.IsNotExist(err) {
		t.Fatalf("deleted run creation binding survived a later checkpoint: %v", err)
	}
	_ = newTestPeerStoreWithLifecycleLimits(t, reopened, limits)
}

func TestLifecycleCheckpointCleanupBarrierRetriesBeforeAppend(t *testing.T) {
	service, _ := newLifecycleTestService(t, smallestLifecycleAuditLimits())
	owner := testLifecycleActor(t, "checkpoint-cleanup-retry-owner", false)
	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create retained run: %v", err)
	}
	cleaner := &failOnceLifecycleCheckpointCleaner{targetDirectory: service.store.lifecycleAuditRoot}
	service.store.lifecycleCleaner = cleaner
	service.store.lifecycle.mu.Lock()
	err = service.store.checkpointLifecycleAuditUnlocked(service.store.lifecycleNow())
	pending := service.store.lifecycle.checkpointCleanup
	service.store.lifecycle.mu.Unlock()
	if err == nil || !cleaner.failed || !pending {
		t.Fatalf("checkpoint cleanup failure was not retained: err=%v failed=%t pending=%t", err, cleaner.failed, pending)
	}
	assertLifecycleCheckpoint(t, service.store)

	hold := true
	if _, err := service.UpdateRunLifecycle(
		owner, run.ID, UpdateLifecycleRequest{EvidenceHold: &hold},
	); err != nil {
		t.Fatalf("append did not recover pending checkpoint cleanup: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	pending = service.store.lifecycle.checkpointCleanup
	active := service.store.lifecycle.activeCount
	service.store.lifecycle.mu.Unlock()
	if pending || active == 0 {
		t.Fatalf("append crossed an unfinished cleanup barrier: pending=%t active=%d", pending, active)
	}
}

func TestStartupCleanupPreservesPostCheckpointActiveSuffix(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, root := newLifecycleTestService(t, limits)
	owner := testLifecycleActor(t, "checkpoint-active-suffix-owner", false)
	anchored, anchorErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if anchorErr != nil {
		t.Fatalf("create checkpoint anchor run: %v", anchorErr)
	}
	remnantPath, remnant := checkpointLifecycleAuditAndCaptureTail(t, service.store)
	if err := os.WriteFile(remnantPath, remnant, 0o600); err != nil {
		t.Fatalf("restore checkpoint cleanup crash remnant: %v", err)
	}
	activeRun, activeRunErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if activeRunErr != nil {
		t.Fatalf("create post-checkpoint active run: %v", activeRunErr)
	}
	activeLifecycle, lifecycleErr := service.store.readRunLifecycle(activeRun)
	if lifecycleErr != nil {
		t.Fatalf("read post-checkpoint run lifecycle: %v", lifecycleErr)
	}
	service.store.lifecycle.mu.Lock()
	activeRecord, exists := service.store.lifecycle.records[activeLifecycle.CreationAuditDigest]
	checkpointSequence := service.store.lifecycle.checkpointSequence
	service.store.lifecycle.mu.Unlock()
	if !exists || activeRecord.Sequence <= checkpointSequence {
		t.Fatalf(
			"post-checkpoint create was not active: record=%+v checkpoint_sequence=%d",
			activeRecord, checkpointSequence,
		)
	}
	activePath := filepath.Join(
		service.store.lifecycleAuditRoot,
		fmt.Sprintf("%020d-%s.json", activeRecord.Sequence, trimSHA256(activeRecord.Digest)),
	)

	if err := service.Close(); err != nil {
		t.Fatalf("close before active checkpoint suffix restart: %v", err)
	}
	reopened, reopenErr := newStoreWithLifecycleLimits(root, limits)
	if reopenErr != nil {
		t.Fatalf("restart with active checkpoint suffix: %v", reopenErr)
	}
	if _, err := os.Lstat(remnantPath); !os.IsNotExist(err) {
		t.Fatalf("startup retained compacted checkpoint remnant: %v", err)
	}
	if _, err := os.Lstat(activePath); err != nil {
		t.Fatalf("startup removed post-checkpoint active record: %v", err)
	}
	reopened.lifecycle.mu.Lock()
	sequence := reopened.lifecycle.sequence
	activeCount := reopened.lifecycle.activeCount
	_, activeLoaded := reopened.lifecycle.records[activeLifecycle.CreationAuditDigest]
	_, activePlanned := reopened.lifecycle.creationBindings[activeLifecycle.CreationAuditDigest]
	reopened.lifecycle.mu.Unlock()
	if sequence != activeRecord.Sequence || activeCount != 1 || !activeLoaded || !activePlanned {
		t.Fatalf(
			"startup lost active suffix state: sequence=%d active=%d loaded=%t planned=%t",
			sequence, activeCount, activeLoaded, activePlanned,
		)
	}
	if _, err := reopened.RunLifecycle(owner, anchored.ID); err != nil {
		t.Fatalf("checkpoint anchor run was not readable: %v", err)
	}
	hold := true
	if _, err := reopened.UpdateRunLifecycle(
		owner, activeRun.ID, UpdateLifecycleRequest{EvidenceHold: &hold},
	); err != nil {
		t.Fatalf("append after recovered active suffix: %v", err)
	}
	reopened.lifecycle.mu.Lock()
	continuedSequence := reopened.lifecycle.sequence
	reopened.lifecycle.mu.Unlock()
	if continuedSequence != activeRecord.Sequence+1 {
		t.Fatalf("active chain did not continue: sequence=%d want=%d", continuedSequence, activeRecord.Sequence+1)
	}
	if _, err := newStoreWithLifecycleLimits(root, limits); err != nil {
		t.Fatalf("second restart rejected continued active suffix: %v", err)
	}
}

func TestLifecycleCheckpointBindingCleanupRecoversOnRestart(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, root := newLifecycleTestService(t, limits)
	owner := testLifecycleActor(t, "checkpoint-binding-restart-owner", false)
	run, createErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if createErr != nil {
		t.Fatalf("create retained run: %v", createErr)
	}
	_, _ = checkpointLifecycleAuditAndCaptureTail(t, service.store)
	bindingPath := filepath.Join(
		service.store.lifecycleBindingRoot,
		lifecycleBindingFileName(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID}),
	)
	if err := service.DeleteRunAs(owner, run.ID); err != nil {
		t.Fatalf("delete checkpointed run: %v", err)
	}
	cleaner := &failOnceLifecycleCheckpointCleaner{
		targetDirectory: service.store.lifecycleBindingRoot,
		failSync:        true,
	}
	service.store.lifecycleCleaner = cleaner
	service.store.lifecycle.mu.Lock()
	checkpointErr := service.store.checkpointLifecycleAuditUnlocked(service.store.lifecycleNow())
	pending := service.store.lifecycle.checkpointCleanup
	service.store.lifecycle.mu.Unlock()
	if checkpointErr == nil || !cleaner.failed || !pending {
		t.Fatalf("binding cleanup failure was not retained: err=%v failed=%t pending=%t", checkpointErr, cleaner.failed, pending)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close before binding cleanup restart: %v", err)
	}
	reopened, reopenErr := newStoreWithLifecycleLimits(root, limits)
	if reopenErr != nil {
		t.Fatalf("restart did not recover pending binding cleanup: %v", reopenErr)
	}
	reopened.lifecycle.mu.Lock()
	pending = reopened.lifecycle.checkpointCleanup
	reopened.lifecycle.mu.Unlock()
	if pending {
		t.Fatal("restart returned while checkpoint cleanup remained pending")
	}
	if _, err := os.Lstat(bindingPath); !os.IsNotExist(err) {
		t.Fatalf("restart retained stale creation binding: %v", err)
	}
}

func TestLifecycleAuditCheckpointPreservesAndCollectsCampaignBinding(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, _ := newLifecycleTestService(t, limits)
	owner := testLifecycleActor(t, "checkpoint-campaign-owner", false)
	campaign := campaignV2StoredSchemaFixture(t)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)

	for index := 0; index < 80; index++ {
		hold := index%2 == 0
		if _, err := service.UpdateCampaignLifecycle(
			owner, campaign.ID, UpdateLifecycleRequest{EvidenceHold: &hold},
		); err != nil {
			t.Fatalf("campaign lifecycle mutation %d crossed an audit boundary: %v", index, err)
		}
	}
	assertLifecycleCheckpoint(t, service.store)
	bindingPath := filepath.Join(
		service.store.lifecycleBindingRoot,
		lifecycleBindingFileName(lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: campaign.ID}),
	)
	if _, err := os.Lstat(bindingPath); err != nil {
		t.Fatalf("checkpoint omitted retained campaign creation binding: %v", err)
	}
	service.store.lifecycle.mu.Lock()
	err := service.store.validateLifecycleAuditUnlocked()
	service.store.lifecycle.mu.Unlock()
	if err != nil {
		t.Fatalf("reload checkpointed campaign audit: %v", err)
	}
	if err := service.store.validateLifecycleCampaignBindings(true); err != nil {
		t.Fatalf("checkpointed campaign binding is invalid: %v", err)
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); err != nil {
		t.Fatalf("delete checkpointed campaign: %v", err)
	}
	forceCreateDeleteAuditRotation(t, service, owner)
	if _, err := os.Lstat(bindingPath); !os.IsNotExist(err) {
		t.Fatalf("deleted campaign creation binding survived a later checkpoint: %v", err)
	}
}

func TestLifecycleCheckpointKeepsDurableCreationBindingOverIdempotentAudit(t *testing.T) {
	service, _ := newLifecycleTestService(t, smallestLifecycleAuditLimits())
	owner := testLifecycleActor(t, "checkpoint-idempotent-owner", false)
	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create retained run: %v", err)
	}
	lifecycle, err := service.store.readRunLifecycle(run)
	if err != nil {
		t.Fatalf("read retained run lifecycle: %v", err)
	}
	if _, err := service.CreateRunAs(context.Background(), owner, requestForExistingRun(run)); err != nil {
		t.Fatalf("repeat idempotent run create: %v", err)
	}
	_, _ = checkpointLifecycleAuditAndCaptureTail(t, service.store)

	path := filepath.Join(
		service.store.lifecycleBindingRoot,
		lifecycleBindingFileName(lifecycleResourceRef{Kind: lifecycleResourceRun, ID: run.ID}),
	)
	var binding lifecycleAuditRecord
	if err := readJSON(path, &binding); err != nil {
		t.Fatalf("read retained creation binding: %v", err)
	}
	if binding.Digest != lifecycle.CreationAuditDigest {
		t.Fatalf("idempotent audit replaced durable creation binding: got=%s want=%s", binding.Digest, lifecycle.CreationAuditDigest)
	}
}

func TestLifecycleAuditCheckpointRejectsUnboundCrashRemnant(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, root := newLifecycleTestService(t, limits)
	owner := testLifecycleActor(t, "checkpoint-remnant-owner", false)
	if _, err := service.CreateRunAs(context.Background(), owner, validCreateRequest()); err != nil {
		t.Fatalf("create checkpoint run: %v", err)
	}
	authorizeMissingRun(t, service.store, owner, "start")
	_, _ = checkpointLifecycleAuditAndCaptureTail(t, service.store)

	forged := lifecycleAuditRecord{
		SchemaVersion: lifecycleAuditSchemaVersion,
		Sequence:      service.store.lifecycle.sequence - 1,
		Timestamp:     service.store.lifecycleNow().UTC().Truncate(time.Microsecond),
		Action:        "start", Decision: "denied", ReasonCode: "not_found",
		ActorDigest: owner.principalDigest, ResourceKind: lifecycleResourceRun,
	}
	forged.Digest = lifecycleAuditDigest(forged)
	path := filepath.Join(
		service.store.lifecycleAuditRoot,
		fmt.Sprintf("%020d-%s.json", forged.Sequence, trimSHA256(forged.Digest)),
	)
	if err := service.store.lifecycleAuditWriter.WriteExclusive(path, forged); err != nil {
		t.Fatalf("publish forged crash remnant fixture: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close before crash remnant restart: %v", err)
	}
	if _, err := newStoreWithLifecycleLimits(root, limits); !errors.Is(err, ErrInvalid) {
		t.Fatalf("unbound checkpoint crash remnant error=%v, want ErrInvalid", err)
	}
}

func TestLifecycleAuditCheckpointRefreshPrunesExpiredDenialWindows(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, _ := newLifecycleTestService(t, limits)
	owner := testLifecycleActor(t, "checkpoint-refresh-owner", false)
	start := time.Date(2026, 9, 1, 2, 3, 4, 0, time.UTC)
	service.store.lifecycleNow = func() time.Time { return start }
	if _, err := service.CreateRunAs(context.Background(), owner, validCreateRequest()); err != nil {
		t.Fatalf("create checkpoint run: %v", err)
	}
	authorizeMissingRun(t, service.store, owner, "start")
	_, _ = checkpointLifecycleAuditAndCaptureTail(t, service.store)

	service.store.lifecycle.mu.Lock()
	before := service.store.lifecycle.checkpointDigest
	err := service.store.checkpointLifecycleAuditUnlocked(start.Add(lifecycleNotFoundDedupeWindow))
	after := service.store.lifecycle.checkpointDigest
	windows := len(service.store.lifecycle.notFoundDenials)
	service.store.lifecycle.mu.Unlock()
	if err != nil {
		t.Fatalf("refresh checkpoint after denial window expiry: %v", err)
	}
	if before == after || windows != 0 {
		t.Fatalf("checkpoint refresh did not prune denial windows: changed=%t windows=%d", before != after, windows)
	}
}

func TestLifecycleNotFoundAuditIsPersistentlyDeduplicated(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, root := newLifecycleTestService(t, limits)
	actor := testLifecycleActor(t, "not-found-writer", false)
	windowStart := time.Date(2026, 9, 1, 1, 2, 3, 0, time.UTC)
	service.store.lifecycleNow = func() time.Time { return windowStart }

	for attempt := 0; attempt < 1000; attempt++ {
		authorizeMissingRun(t, service.store, actor, "start")
	}
	assertNotFoundAuditCount(t, service.store, actor, lifecycleResourceRun, "start", 1)

	if err := service.Close(); err != nil {
		t.Fatalf("close before deduplicated audit restart: %v", err)
	}
	reopened, err := newStoreWithLifecycleLimits(root, limits)
	if err != nil {
		t.Fatalf("restart deduplicated audit: %v", err)
	}
	reopened.lifecycleNow = func() time.Time { return windowStart.Add(30 * time.Second) }
	authorizeMissingRun(t, reopened, actor, "start")
	assertNotFoundAuditCount(t, reopened, actor, lifecycleResourceRun, "start", 1)

	reopened.lifecycleNow = func() time.Time {
		return windowStart.Add(lifecycleNotFoundDedupeWindow + time.Microsecond)
	}
	authorizeMissingRun(t, reopened, actor, "start")
	assertNotFoundAuditCount(t, reopened, actor, lifecycleResourceRun, "start", 2)
}

func TestLifecycleNotFoundDedupeIsResourceScoped(t *testing.T) {
	service, _ := newLifecycleTestService(t, smallestLifecycleAuditLimits())
	actor := testLifecycleActor(t, "resource-scoped-not-found-writer", false)
	now := time.Date(2026, 9, 1, 3, 4, 5, 0, time.UTC)
	service.store.lifecycleNow = func() time.Time { return now }

	authorizeMissingRun(t, service.store, actor, "delete")
	if err := service.DeleteCampaignAs(actor, newTestClientRequestID()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing campaign deletion error=%v, want ErrNotFound", err)
	}
	assertNotFoundAuditCount(t, service.store, actor, lifecycleResourceRun, "delete", 1)
	assertNotFoundAuditCount(t, service.store, actor, lifecycleResourceCampaign, "delete", 1)
}

func TestLifecycleNotFoundWindowCapacityCannotBlockRotation(t *testing.T) {
	limits := smallestLifecycleAuditLimits()
	service, _ := newLifecycleTestService(t, limits)
	now := time.Date(2026, 9, 1, 4, 5, 6, 0, time.UTC)
	service.store.lifecycleNow = func() time.Time { return now }
	var overflow Actor
	for index := 0; index < maxLifecycleNotFoundDenialWindows+100; index++ {
		overflow = testLifecycleActor(t, fmt.Sprintf("overflow-writer-%d", index), false)
		authorizeMissingRun(t, service.store, overflow, "start")
	}

	service.store.lifecycle.mu.Lock()
	tracked := len(service.store.lifecycle.notFoundDenials)
	overflowRecorded := false
	for _, record := range service.store.lifecycle.records {
		if record.ActorDigest == overflow.principalDigest && record.ResourceKind == lifecycleResourceRun &&
			record.Action == "start" && record.Decision == "denied" && record.ReasonCode == "not_found" &&
			record.ResourceID == "" {
			overflowRecorded = true
		}
	}
	err := service.store.checkpointLifecycleAuditUnlocked(now)
	service.store.lifecycle.mu.Unlock()
	if tracked != maxLifecycleNotFoundDenialWindows {
		t.Fatalf("not-found denial windows=%d, want fixed capacity %d", tracked, maxLifecycleNotFoundDenialWindows)
	}
	if !overflowRecorded {
		t.Fatal("untracked principal denial was not written to the active audit")
	}
	if err != nil {
		t.Fatalf("checkpoint after denial-window overflow: %v", err)
	}
	checkpointInfo, err := os.Lstat(filepath.Join(
		service.store.lifecycleAuditRoot, lifecycleAuditCheckpointFileName,
	))
	if err != nil {
		t.Fatalf("stat bounded lifecycle checkpoint: %v", err)
	}
	if checkpointInfo.Size()+lifecycleAuditAppendReserveBytes > limits.MaxAuditBytes {
		t.Fatalf(
			"checkpoint consumed append reserve: checkpoint=%d reserve=%d audit_limit=%d",
			checkpointInfo.Size(), lifecycleAuditAppendReserveBytes, limits.MaxAuditBytes,
		)
	}
	if _, err := service.CreateRunAs(
		context.Background(), testLifecycleActor(t, "post-overflow-owner", false), validCreateRequest(),
	); err != nil {
		t.Fatalf("append after bounded denial checkpoint: %v", err)
	}
}

func TestLifecycleBindingFileNamesSeparateResourceKinds(t *testing.T) {
	id := newTestClientRequestID()
	runResource := lifecycleResourceRef{Kind: lifecycleResourceRun, ID: id}
	campaignResource := lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: id}
	run := lifecycleBindingFileName(runResource)
	campaign := lifecycleBindingFileName(campaignResource)
	if run == campaign {
		t.Fatalf("resource-kind binding paths collided: %q", run)
	}
	for name, expected := range map[string]lifecycleResourceRef{run: runResource, campaign: campaignResource} {
		if resource, ok := parseLifecycleBindingFileName(name); !ok || resource != expected {
			t.Fatalf(
				"binding filename %q did not round-trip: resource=%+v expected=%+v valid=%t",
				name, resource, expected, ok,
			)
		}
	}
}

func TestLifecyclePolicyAllowsOnlyDurableMonotonicExpansion(t *testing.T) {
	initial := smallestLifecycleAuditLimits()
	service, _ := newLifecycleTestService(t, initial)
	peer := newTestPeerStore(t, service.store)
	if service.store.lifecyclePolicy != peer.lifecyclePolicy ||
		!reflect.DeepEqual(peer.lifecyclePolicy.Limits, initial) {
		t.Fatalf(
			"unspecified peer did not share durable lifecycle policy: service=%+v peer=%+v",
			service.store.lifecyclePolicy.Limits, peer.lifecyclePolicy.Limits,
		)
	}
	expanded := initial
	expanded.MaxOwnerBytes *= 2
	expanded.MaxStoreBytes *= 2
	expanded.MaxOwnerRuns += 10
	expanded.MaxOwnerCampaigns += 10
	expanded.MaxAuditBytes *= 2

	if _, err := openTestPeerStore(t, service.store, expanded); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer policy expansion error=%v, want ErrConflict", err)
	}
	service.store.lifecycle.mu.Lock()
	expansionErr := service.store.initializeLifecyclePolicyUnlocked(expanded)
	service.store.lifecycle.mu.Unlock()
	if expansionErr != nil {
		t.Fatalf("owner policy expansion: %v", expansionErr)
	}
	expander := newTestPeerStoreWithLifecycleLimits(t, service.store, expanded)
	if !reflect.DeepEqual(expander.lifecyclePolicy.Limits, expanded) {
		t.Fatalf("expanded limits were not installed: %+v", expander.lifecyclePolicy.Limits)
	}
	for name, store := range map[string]*Store{
		"original": service.store,
		"peer":     peer,
		"expander": expander,
	} {
		if store.lifecyclePolicy != expander.lifecyclePolicy ||
			!reflect.DeepEqual(store.lifecyclePolicy.Limits, expanded) {
			t.Fatalf("%s store did not observe shared expansion: %+v", name, store.lifecyclePolicy.Limits)
		}
	}
	var durable lifecycleStorePolicy
	if err := readJSON(filepath.Join(service.store.lifecycleRoot, lifecyclePolicyFileName), &durable); err != nil ||
		!reflect.DeepEqual(durable.Limits, expanded) {
		t.Fatalf("expanded limits were not durable: policy=%+v err=%v", durable, err)
	}
	if _, err := openTestPeerStore(t, service.store, initial); !errors.Is(err, ErrConflict) {
		t.Fatalf("lifecycle policy contraction error=%v, want ErrConflict", err)
	}
	restarted := newTestPeerStore(t, service.store)
	if restarted.lifecyclePolicy != service.store.lifecyclePolicy ||
		!reflect.DeepEqual(restarted.lifecyclePolicy.Limits, expanded) {
		t.Fatalf("restart did not preserve expanded durable policy: %+v", restarted.lifecyclePolicy.Limits)
	}
}

func TestLifecyclePolicyExpansionIsNotSharedBeforeDirectorySync(t *testing.T) {
	initial := smallestLifecycleAuditLimits()
	service, _ := newLifecycleTestService(t, initial)
	peer := newTestPeerStore(t, service.store)
	expanded := initial
	expanded.MaxOwnerBytes *= 2
	expanded.MaxStoreBytes *= 2
	expanded.MaxOwnerRuns++
	expanded.MaxOwnerCampaigns++
	expanded.MaxAuditBytes *= 2

	faults := &faultingLifecyclePersistence{
		delegate:      atomicLifecyclePolicyPersistence{},
		writeFailures: 1,
		syncFailures:  2,
	}
	service.store.lifecyclePersistence = faults
	for attempt := 0; attempt < 3; attempt++ {
		service.store.lifecycle.mu.Lock()
		err := service.store.initializeLifecyclePolicyUnlocked(expanded)
		service.store.lifecycle.mu.Unlock()
		if err == nil {
			t.Fatalf("policy expansion attempt %d succeeded before durability retry", attempt+1)
		}
		if !reflect.DeepEqual(peer.lifecyclePolicy.Limits, initial) {
			t.Fatalf(
				"active peer observed uncommitted expansion after attempt %d: %+v",
				attempt+1, peer.lifecyclePolicy.Limits,
			)
		}
	}

	service.store.lifecyclePersistence = atomicLifecyclePolicyPersistence{}
	service.store.lifecycle.mu.Lock()
	err := service.store.initializeLifecyclePolicyUnlocked(expanded)
	service.store.lifecycle.mu.Unlock()
	if err != nil {
		t.Fatalf("commit expanded lifecycle policy after durability retry: %v", err)
	}
	if !reflect.DeepEqual(peer.lifecyclePolicy.Limits, expanded) {
		t.Fatalf("active peer did not observe committed expansion: %+v", peer.lifecyclePolicy.Limits)
	}
}

func forceCreateDeleteAuditRotation(t *testing.T, service *Service, owner Actor) {
	t.Helper()
	for index := 0; index < 40; index++ {
		run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
		if err != nil {
			t.Fatalf("create rotation run %d: %v", index, err)
		}
		if err := service.DeleteRunAs(owner, run.ID); err != nil {
			t.Fatalf("delete rotation run %d: %v", index, err)
		}
	}
}

func checkpointLifecycleAuditAndCaptureTail(t *testing.T, store *Store) (string, []byte) {
	t.Helper()
	store.lifecycle.mu.Lock()
	defer store.lifecycle.mu.Unlock()
	entries, err := os.ReadDir(store.lifecycleAuditRoot)
	if err != nil {
		t.Fatalf("list lifecycle audit before checkpoint: %v", err)
	}
	var tailPath string
	for _, entry := range entries {
		if lifecycleAuditFilePattern.MatchString(entry.Name()) {
			tailPath = filepath.Join(store.lifecycleAuditRoot, entry.Name())
		}
	}
	if tailPath == "" {
		t.Fatal("lifecycle audit has no checkpointable record")
	}
	tail, err := os.ReadFile(tailPath)
	if err != nil {
		t.Fatalf("read lifecycle audit tail: %v", err)
	}
	if err := store.checkpointLifecycleAuditUnlocked(store.lifecycleNow()); err != nil {
		t.Fatalf("checkpoint lifecycle audit: %v", err)
	}
	return tailPath, tail
}

func authorizeMissingRun(t *testing.T, store *Store, actor Actor, action string) {
	t.Helper()
	store.lifecycle.mu.Lock()
	defer store.lifecycle.mu.Unlock()
	err := store.authorizeRunActionUnlocked(actor, newTestClientRequestID(), action)
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing run authorization error=%v, want ErrNotFound", err)
	}
}

func assertLifecycleCheckpoint(t *testing.T, store *Store) {
	t.Helper()
	var checkpoint lifecycleAuditCheckpoint
	if err := readJSON(
		filepath.Join(store.lifecycleAuditRoot, lifecycleAuditCheckpointFileName), &checkpoint,
	); err != nil {
		t.Fatalf("read lifecycle audit checkpoint: %v", err)
	}
	if err := validateLifecycleAuditCheckpoint(checkpoint); err != nil {
		t.Fatalf("validate lifecycle audit checkpoint: %v", err)
	}
}

func assertNotFoundAuditCount(
	t *testing.T,
	store *Store,
	actor Actor,
	resourceKind string,
	action string,
	want int,
) {
	t.Helper()
	store.lifecycle.mu.Lock()
	defer store.lifecycle.mu.Unlock()
	count := 0
	for _, record := range store.lifecycle.records {
		if record.ActorDigest == actor.principalDigest && record.ResourceKind == resourceKind &&
			record.Action == action &&
			record.Decision == "denied" && record.ReasonCode == "not_found" {
			if record.ResourceID != "" {
				t.Fatalf("not-found denial persisted attacker-controlled resource id: %+v", record)
			}
			count++
		}
	}
	if count != want {
		t.Fatalf("not-found audit count=%d, want %d", count, want)
	}
}
