package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

type faultingLifecyclePersistence struct {
	delegate      lifecyclePolicyPersistence
	writeFailures int
	syncFailures  int
}

type unreadableFailedLifecyclePersistence struct {
	delegate lifecyclePolicyPersistence
	path     string
	hidden   string
	failed   bool
}

func (p *unreadableFailedLifecyclePersistence) Write(path string, value any) error {
	if p.failed {
		return p.delegate.Write(path, value)
	}
	p.failed, p.path, p.hidden = true, path, path+".hidden-by-fault"
	if err := os.Rename(path, p.hidden); err != nil {
		return err
	}
	return errors.New("injected lifecycle write and canonical read failure")
}

func (p *unreadableFailedLifecyclePersistence) SyncDirectory(path, description string) error {
	return p.delegate.SyncDirectory(path, description)
}

func (p *unreadableFailedLifecyclePersistence) restore() error {
	return os.Rename(p.hidden, p.path)
}

func TestMatchingLifecycleRetryRecoversWhenFailedWriteCouldNotReadCanonicalValue(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "lifecycle-unreadable-write-owner", false)
	run, lifecycleErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if lifecycleErr != nil {
		t.Fatalf("create lifecycle retry run: %v", lifecycleErr)
	}
	faults := &unreadableFailedLifecyclePersistence{delegate: atomicLifecyclePolicyPersistence{}}
	service.store.lifecyclePersistence = faults
	hold := true
	request := UpdateLifecycleRequest{EvidenceHold: &hold}
	if _, err := service.UpdateRunLifecycle(owner, run.ID, request); err == nil {
		t.Fatal("lifecycle update unexpectedly survived the injected unreadable canonical value")
	}
	if err := faults.restore(); err != nil {
		t.Fatalf("restore canonical lifecycle after transient read fault: %v", err)
	}
	if got := len(service.store.lifecycle.pendingLifecycle); got != 1 {
		t.Fatalf("pending lifecycle mutations=%d, want 1", got)
	}
	view, lifecycleErr := service.UpdateRunLifecycle(owner, run.ID, request)
	if lifecycleErr != nil || !view.EvidenceHold {
		t.Fatalf("matching retry did not reconcile unpublished lifecycle write: view=%+v err=%v", view, lifecycleErr)
	}
	if got := len(service.store.lifecycle.pendingLifecycle); got != 0 {
		t.Fatalf("pending lifecycle mutations=%d after retry, want 0", got)
	}
}

func (p *faultingLifecyclePersistence) Write(path string, value any) error {
	if p.writeFailures == 0 {
		return p.delegate.Write(path, value)
	}
	p.writeFailures--
	if err := publishJSONWithoutParentSync(path, value); err != nil {
		return err
	}
	return errors.New("injected lifecycle directory sync failure after visible rename")
}

func (p *faultingLifecyclePersistence) SyncDirectory(path, description string) error {
	if p.syncFailures > 0 {
		p.syncFailures--
		return errors.New("injected lifecycle retry sync failure")
	}
	return p.delegate.SyncDirectory(path, description)
}

func publishJSONWithoutParentSync(path string, value any) error {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	temporary, err := os.CreateTemp(filepath.Dir(path), ".tmp-lifecycle-fault-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o600); err != nil {
		_ = temporary.Close()
		return err
	}
	if _, err := temporary.Write(encoded); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryPath, path)
}

func TestRunLifecycleRetryClosesVisiblePolicySyncFailure(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-lifecycle-durability-owner", false)
	run, lifecycleErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if lifecycleErr != nil {
		t.Fatalf("create lifecycle durability run: %v", lifecycleErr)
	}
	completedAt := time.Now().UTC().Truncate(time.Microsecond)
	run.Status, run.CompletedAt = StatusCancelled, &completedAt
	run.Progress.Message = "Run cancelled"
	if err := service.store.updateRunFixture(run); err != nil {
		t.Fatalf("make lifecycle durability run collectable: %v", err)
	}

	faults := &faultingLifecyclePersistence{
		delegate:      atomicLifecyclePolicyPersistence{},
		writeFailures: 1,
		syncFailures:  1,
	}
	service.store.lifecyclePersistence = faults
	hold, protected := true, RetentionProtected
	request := UpdateLifecycleRequest{RetentionClass: &protected, EvidenceHold: &hold}
	if _, err := service.UpdateRunLifecycle(owner, run.ID, request); err == nil {
		t.Fatal("run lifecycle update succeeded after an uncertain visible write")
	}
	if _, err := service.store.GetRun(run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("generic run read committed pending lifecycle mutation: %v", err)
	}
	if err := service.store.updateRunFixture(run); !errors.Is(err, ErrConflict) {
		t.Fatalf("status writer committed pending lifecycle mutation: %v", err)
	}
	foreign := testLifecycleActor(t, "run-lifecycle-durability-foreign", false)
	if _, err := service.UpdateRunLifecycle(foreign, run.ID, request); !errors.Is(err, ErrForbidden) {
		t.Fatalf("foreign lifecycle retry error=%v, want ErrForbidden", err)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer opener recovered pending lifecycle mutation: %v", err)
	}
	if _, err := service.UpdateRunLifecycle(owner, run.ID, request); err == nil {
		t.Fatal("run lifecycle idempotent retry bypassed persistent directory sync failure")
	}
	view, lifecycleErr := service.UpdateRunLifecycle(owner, run.ID, request)
	if lifecycleErr != nil || !view.EvidenceHold || view.RetentionClass != RetentionProtected || view.DeleteAfter != nil {
		t.Fatalf("run lifecycle retry did not commit protected hold: view=%+v err=%v", view, lifecycleErr)
	}

	service.store.lifecycleNow = func() time.Time { return completedAt.Add(365 * 24 * time.Hour) }
	plan, lifecycleErr := service.CollectLifecycle(SystemActor(), CollectionRequest{})
	if lifecycleErr != nil || len(plan.Plan.Candidates) != 0 || plan.Plan.Skipped["held"] != 1 {
		t.Fatalf("protected held run entered collection before restart: plan=%+v err=%v", plan.Plan, lifecycleErr)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close lifecycle durability service: %v", err)
	}
	restarted := reopenTestService(t, root)
	restarted.store.lifecycleNow = func() time.Time { return completedAt.Add(365 * 24 * time.Hour) }
	view, lifecycleErr = restarted.RunLifecycle(owner, run.ID)
	if lifecycleErr != nil || !view.EvidenceHold || view.RetentionClass != RetentionProtected || view.DeleteAfter != nil {
		t.Fatalf("protected hold changed after restart: view=%+v err=%v", view, lifecycleErr)
	}
	plan, lifecycleErr = restarted.CollectLifecycle(SystemActor(), CollectionRequest{})
	if lifecycleErr != nil || len(plan.Plan.Candidates) != 0 || plan.Plan.Skipped["held"] != 1 {
		t.Fatalf("protected held run entered collection after restart: plan=%+v err=%v", plan.Plan, lifecycleErr)
	}
	if err := restarted.DeleteRunAs(owner, run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("protected held run deletion error=%v, want ErrConflict", err)
	}
}

func TestCampaignLifecycleRetryClosesVisiblePolicySyncFailure(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-lifecycle-durability-owner", false)
	campaign := campaignV2StoredSchemaFixture(t)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)
	faults := &faultingLifecyclePersistence{
		delegate:      atomicLifecyclePolicyPersistence{},
		writeFailures: 1,
		syncFailures:  1,
	}
	service.store.lifecyclePersistence = faults
	hold, protected := true, RetentionProtected
	request := UpdateLifecycleRequest{RetentionClass: &protected, EvidenceHold: &hold}
	if _, err := service.UpdateCampaignLifecycle(owner, campaign.ID, request); err == nil {
		t.Fatal("campaign lifecycle update succeeded after an uncertain visible write")
	}
	if _, err := service.GetCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("generic campaign read committed pending lifecycle mutation: %v", err)
	}
	foreign := testLifecycleActor(t, "campaign-lifecycle-durability-foreign", false)
	if _, err := service.UpdateCampaignLifecycle(foreign, campaign.ID, request); !errors.Is(err, ErrForbidden) {
		t.Fatalf("foreign campaign lifecycle retry error=%v, want ErrForbidden", err)
	}
	if _, err := service.UpdateCampaignLifecycle(owner, campaign.ID, request); err == nil {
		t.Fatal("campaign lifecycle idempotent retry bypassed persistent directory sync failure")
	}
	view, err := service.UpdateCampaignLifecycle(owner, campaign.ID, request)
	if err != nil || !view.EvidenceHold || view.RetentionClass != RetentionProtected || view.DeleteAfter != nil {
		t.Fatalf("campaign lifecycle retry did not commit protected hold: view=%+v err=%v", view, err)
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("protected held campaign deletion error=%v, want ErrConflict", err)
	}
}

func TestRunLifecycleReleaseCannotBypassPendingCommitDuringCollection(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "run-lifecycle-release-owner", false)
	run, lifecycleErr := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if lifecycleErr != nil {
		t.Fatalf("create lifecycle release run: %v", lifecycleErr)
	}
	completedAt := run.CreatedAt.Add(time.Second)
	run.Status, run.CompletedAt = StatusCancelled, &completedAt
	run.Progress.Message = "Run cancelled"
	if err := service.store.updateRunFixture(run); err != nil {
		t.Fatalf("make lifecycle release run terminal: %v", err)
	}
	hold, protected := true, RetentionProtected
	if _, err := service.UpdateRunLifecycle(owner, run.ID, UpdateLifecycleRequest{
		RetentionClass: &protected, EvidenceHold: &hold,
	}); err != nil {
		t.Fatalf("protect lifecycle release run: %v", err)
	}

	faults := &faultingLifecyclePersistence{
		delegate: atomicLifecyclePolicyPersistence{}, writeFailures: 1, syncFailures: 1,
	}
	service.store.lifecyclePersistence = faults
	hold, ephemeral := false, RetentionEphemeral
	release := UpdateLifecycleRequest{RetentionClass: &ephemeral, EvidenceHold: &hold}
	if _, err := service.UpdateRunLifecycle(owner, run.ID, release); err == nil {
		t.Fatal("run lifecycle release unexpectedly committed through injected sync failure")
	}
	if _, err := service.UpdateRunLifecycle(
		owner, run.ID, UpdateLifecycleRequest{EvidenceHold: &hold},
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("different run lifecycle request recovered pending mutation: %v", err)
	}
	service.store.lifecycleNow = func() time.Time { return completedAt.Add(365 * 24 * time.Hour) }
	if _, err := service.CollectLifecycle(SystemActor(), CollectionRequest{}); !errors.Is(err, ErrConflict) {
		t.Fatalf("collection consumed an uncertain run release: %v", err)
	}
	if _, err := service.UpdateRunLifecycle(owner, run.ID, release); !errors.Is(err, ErrConflict) {
		t.Fatalf("run release retry bypassed persistent commit failure: %v", err)
	}
	view, lifecycleErr := service.UpdateRunLifecycle(owner, run.ID, release)
	if lifecycleErr != nil || view.EvidenceHold || view.RetentionClass != RetentionEphemeral {
		t.Fatalf("run release retry did not close commit: view=%+v err=%v", view, lifecycleErr)
	}
	plan, lifecycleErr := service.CollectLifecycle(SystemActor(), CollectionRequest{})
	if lifecycleErr != nil || len(plan.Plan.Candidates) != 1 || plan.Plan.Candidates[0].RunID != run.ID {
		t.Fatalf("durable run release did not become collectable: plan=%+v err=%v", plan.Plan, lifecycleErr)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close lifecycle release service: %v", err)
	}
	restarted := reopenTestService(t, root)
	restarted.store.lifecycleNow = func() time.Time { return completedAt.Add(365 * 24 * time.Hour) }
	plan, lifecycleErr = restarted.CollectLifecycle(SystemActor(), CollectionRequest{})
	if lifecycleErr != nil || len(plan.Plan.Candidates) != 1 || plan.Plan.Candidates[0].RunID != run.ID {
		t.Fatalf("run release changed after restart: plan=%+v err=%v", plan.Plan, lifecycleErr)
	}
}

func TestCampaignLifecycleReleaseCannotBypassPendingCommitDuringDelete(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-lifecycle-release-owner", false)
	campaign := campaignV2StoredSchemaFixture(t)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)
	hold, protected := true, RetentionProtected
	if _, err := service.UpdateCampaignLifecycle(owner, campaign.ID, UpdateLifecycleRequest{
		RetentionClass: &protected, EvidenceHold: &hold,
	}); err != nil {
		t.Fatalf("protect lifecycle release campaign: %v", err)
	}

	faults := &faultingLifecyclePersistence{
		delegate: atomicLifecyclePolicyPersistence{}, writeFailures: 1, syncFailures: 1,
	}
	service.store.lifecyclePersistence = faults
	hold, standard := false, RetentionStandard
	release := UpdateLifecycleRequest{RetentionClass: &standard, EvidenceHold: &hold}
	if _, err := service.UpdateCampaignLifecycle(owner, campaign.ID, release); err == nil {
		t.Fatal("campaign lifecycle release unexpectedly committed through injected sync failure")
	}
	if _, err := service.UpdateCampaignLifecycle(
		owner, campaign.ID, UpdateLifecycleRequest{EvidenceHold: &hold},
	); !errors.Is(err, ErrConflict) {
		t.Fatalf("different campaign lifecycle request recovered pending mutation: %v", err)
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete consumed an uncertain campaign release: %v", err)
	}
	if _, err := service.UpdateCampaignLifecycle(owner, campaign.ID, release); !errors.Is(err, ErrConflict) {
		t.Fatalf("campaign release retry bypassed persistent commit failure: %v", err)
	}
	view, err := service.UpdateCampaignLifecycle(owner, campaign.ID, release)
	if err != nil || view.EvidenceHold || view.RetentionClass != RetentionStandard {
		t.Fatalf("campaign release retry did not close commit: view=%+v err=%v", view, err)
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); err != nil {
		t.Fatalf("delete durable released campaign: %v", err)
	}

	if err := service.Close(); err != nil {
		t.Fatalf("close campaign lifecycle release service: %v", err)
	}
	restarted := reopenTestService(t, root)
	if _, err := restarted.GetCampaignAs(owner, campaign.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("deleted released campaign reappeared after restart: %v", err)
	}
}
