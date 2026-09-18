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

func writeCampaignLifecycleFixture(
	t *testing.T,
	store *Store,
	campaign Campaign,
	owner Actor,
) CampaignLifecycle {
	t.Helper()
	store.lifecycle.mu.Lock()
	defer store.lifecycle.mu.Unlock()
	if err := validateActor(owner); err != nil {
		t.Fatal(err)
	}
	audit, err := store.appendLifecycleAuditUnlocked(
		owner, lifecycleResourceCampaign, "create", "allowed",
		lifecycleOwnerAuthorizationReason(owner, owner.principalDigest),
		campaign.ID, owner.principalDigest,
	)
	if err != nil {
		t.Fatalf("audit campaign fixture creation: %v", err)
	}
	lifecycle := newCampaignLifecycle(campaign, owner)
	lifecycle.CreationAuditDigest = audit.Digest
	lifecycle.PolicyDigest = lifecycleDigest(lifecycle)
	directory := filepath.Join(store.campaignRoot, campaign.ID)
	if err := os.Mkdir(directory, 0o700); err != nil {
		t.Fatalf("create campaign fixture directory: %v", err)
	}
	if err := writeJSONAtomic(filepath.Join(directory, campaignFileName), campaign); err != nil {
		t.Fatalf("write campaign fixture: %v", err)
	}
	if err := writeJSONAtomic(filepath.Join(directory, lifecycleFileName), lifecycle); err != nil {
		t.Fatalf("write campaign lifecycle fixture: %v", err)
	}
	return lifecycle
}

func TestCampaignLifecycleOwnerRetentionAuditAndDelete(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-owner", false)
	other := testLifecycleActor(t, "campaign-other", false)
	campaign := campaignV2StoredSchemaFixture(t)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)

	if _, err := service.CampaignLifecycle(other, campaign.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner campaign lifecycle error=%v, want ErrForbidden", err)
	}
	usage, usageErr := service.LifecycleUsage(owner)
	if usageErr != nil || len(usage.Owners) != 1 || usage.Owners[0].CampaignCount != 1 ||
		usage.ManagedPhysicalBytes != 0 || usage.ReservedBytes != 0 || usage.ChargeableBytes != 0 ||
		usage.MaxStoreBytes != 0 || usage.AuditBytes != 0 || usage.MaxAuditBytes != 0 ||
		usage.RunCount != 0 || usage.CampaignCount != 0 {
		t.Fatalf("campaign usage=%+v err=%v", usage, usageErr)
	}
	if _, err := service.CreateRunAs(context.Background(), other, validCreateRequest()); err != nil {
		t.Fatalf("create other owner run: %v", err)
	}
	usageAfterOtherOwnerMutation, updatedUsageErr := service.LifecycleUsage(owner)
	if updatedUsageErr != nil {
		t.Fatalf("campaign usage after other owner mutation: %v", updatedUsageErr)
	}
	if !reflect.DeepEqual(usage, usageAfterOtherOwnerMutation) {
		t.Fatalf(
			"other owner mutation changed campaign owner usage: before=%+v after=%+v",
			usage, usageAfterOtherOwnerMutation,
		)
	}
	hold := true
	if _, err := service.UpdateCampaignLifecycle(owner, campaign.ID, UpdateLifecycleRequest{
		EvidenceHold: &hold,
	}); err != nil {
		t.Fatalf("hold campaign: %v", err)
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("delete held campaign error=%v, want ErrConflict", err)
	}
	hold = false
	ephemeral := RetentionEphemeral
	view, err := service.UpdateCampaignLifecycle(owner, campaign.ID, UpdateLifecycleRequest{
		RetentionClass: &ephemeral, EvidenceHold: &hold,
	})
	if err != nil || view.RetentionClass != RetentionEphemeral || view.EvidenceHold {
		t.Fatalf("release campaign lifecycle=%+v err=%v", view, err)
	}
	if err := service.DeleteCampaignAs(other, campaign.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner campaign delete error=%v, want ErrForbidden", err)
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); err != nil {
		t.Fatalf("owner campaign delete: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(service.store.campaignRoot, campaign.ID)); !os.IsNotExist(err) {
		t.Fatalf("deleted campaign remains: %v", err)
	}
	records := lifecycleAuditRecords(service.store)
	assertLifecycleAuditDecision(t, records, "create", "allowed")
	assertLifecycleAuditDecision(t, records, "hold", "allowed")
	assertLifecycleAuditDecision(t, records, "release", "allowed")
	assertLifecycleAuditDecision(t, records, "delete", "denied")
	assertLifecycleAuditDecision(t, records, "delete", "allowed")
}

func TestCampaignCollectionExpiresReferenceOwnerAndQuota(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-owner", false)
	administrator := testLifecycleActor(t, "campaign-admin", true)
	campaign := campaignV2StoredSchemaFixture(t)
	service.store.lifecycleNow = func() time.Time { return campaign.CreatedAt.Add(31 * 24 * time.Hour) }
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)
	service.store.lifecyclePolicy.Limits.MaxOwnerCampaigns = 1
	if reason, err := service.store.requireCampaignCreateQuotaUnlocked(owner, 1); !errors.Is(err, ErrQuota) || reason != "quota_owner_campaigns" {
		t.Fatalf("campaign count quota reason=%q err=%v", reason, err)
	}
	dryRun, err := service.CollectLifecycle(administrator, CollectionRequest{})
	if err != nil || len(dryRun.Plan.Candidates) != 1 ||
		dryRun.Plan.Candidates[0].CampaignID != campaign.ID {
		t.Fatalf("campaign collection plan=%+v err=%v", dryRun.Plan, err)
	}
	applied, err := service.CollectLifecycle(administrator, CollectionRequest{
		Apply: true, PlanDigest: dryRun.Plan.PlanDigest,
	})
	if err != nil || len(applied.DeletedCampaignIDs) != 1 ||
		applied.DeletedCampaignIDs[0] != campaign.ID {
		t.Fatalf("campaign collection result=%+v err=%v", applied, err)
	}
}

func TestCampaignCollectionReleasesControlledPairInSamePlan(t *testing.T) {
	service, _ := newControlledPairStoreTestService(t)
	pair, baselineManifest, candidateManifest := pendingControlledPairAggregate(t, service, SystemActor())
	if _, err := service.store.createControlledPairBundlesAs(
		SystemActor(), pair, baselineManifest, candidateManifest,
	); err != nil {
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

	campaign := campaignV2StoredSchemaFixture(t)
	campaign.GateBindings.G2RunID = pair.BaselineRunID
	campaign.GateBindings.G4RunID = pair.CandidateRunID
	for index := range campaign.Decision.Evidence {
		switch campaign.Decision.Evidence[index].GateID {
		case "G2":
			campaign.Decision.Evidence[index].RunID = pair.BaselineRunID
		case "G4":
			campaign.Decision.Evidence[index].RunID = pair.CandidateRunID
		}
	}
	anchors, anchorErr := validateCampaignEvidenceAnchors(campaign.GateBindings, campaign.Decision.Evidence)
	if anchorErr != nil {
		t.Fatalf("validate campaign collection anchors: %v", anchorErr)
	}
	for index := range campaign.Decision.Gates {
		switch campaign.Decision.Gates[index].ID {
		case "G0", "G1":
			campaign.Decision.Gates[index].EvidenceRefs = campaignAnchorRefs(campaign.GateBindings, anchors)
		case "G2":
			campaign.Decision.Gates[index].EvidenceRefs = campaignAnchorRefsForKeys(anchors, "g2:evidence")
		case "G4":
			campaign.Decision.Gates[index].EvidenceRefs = campaignAnchorRefsForKeys(anchors, "g4:evidence")
		}
	}
	var digestErr error
	campaign.ManifestDigest, digestErr = campaignManifestDigest(campaign)
	if digestErr != nil {
		t.Fatalf("digest campaign: %v", digestErr)
	}
	campaign.Decision.CampaignDigest = campaign.ManifestDigest
	campaign.Decision.DecisionDigest, digestErr = campaignDecisionDigest(campaign.Decision)
	if digestErr != nil {
		t.Fatalf("digest campaign decision: %v", digestErr)
	}
	if err := validateStoredCampaign(campaign.ID, campaign); err != nil {
		t.Fatalf("validate campaign fixture: %v", err)
	}
	writeCampaignLifecycleFixture(t, service.store, campaign, SystemActor())
	service.store.lifecycleNow = func() time.Time { return time.Now().UTC().Add(31 * 24 * time.Hour) }

	planResult, err := service.CollectLifecycle(SystemActor(), CollectionRequest{})
	if err != nil {
		t.Fatalf("plan campaign and pair collection: %v", err)
	}
	campaignIndex, pairIndex := -1, -1
	for index, candidate := range planResult.Plan.Candidates {
		if candidate.CampaignID == campaign.ID {
			campaignIndex = index
		}
		if candidate.PairID == pair.PairID {
			pairIndex = index
		}
	}
	if campaignIndex < 0 || pairIndex < 0 || campaignIndex >= pairIndex {
		t.Fatalf("collection did not order campaign before newly unpinned pair: %+v", planResult.Plan.Candidates)
	}
	applied, err := service.CollectLifecycle(SystemActor(), CollectionRequest{
		Apply: true, PlanDigest: planResult.Plan.PlanDigest,
	})
	if err != nil {
		t.Fatalf("apply campaign and pair collection: %v", err)
	}
	if len(applied.DeletedCampaignIDs) != 1 || applied.DeletedCampaignIDs[0] != campaign.ID ||
		len(applied.DeletedPairIDs) != 1 || applied.DeletedPairIDs[0] != pair.PairID {
		t.Fatalf("campaign and pair collection result=%+v", applied)
	}
}

func TestCampaignEvidenceOwnershipIsActorScoped(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "evidence-owner", false)
	other := testLifecycleActor(t, "campaign-owner", false)
	administrator := testLifecycleActor(t, "campaign-admin", true)
	run, err := service.CreateRunAs(context.Background(), owner, validCreateRequest())
	if err != nil {
		t.Fatalf("create owned evidence: %v", err)
	}
	campaign := Campaign{Decision: CampaignDecision{Evidence: []CampaignEvidenceAnchor{{RunID: run.ID}}}}
	request := CreateCampaignRequest{
		ClientRequestID: "f451dc66-09ac-4a48-86b5-63f84e272599",
		Name:            "cross-owner campaign", Description: "must fail before private evidence read",
		ChangeProfile: "schema_adapter", GateBindings: CampaignGateBindings{G4RunID: run.ID},
	}
	if _, err := service.CreateCampaignAs(other, request); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner service admission error=%v, want ErrForbidden", err)
	}
	if err := service.store.validateCampaignRunOwnersUnlocked(other, campaign); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner evidence admission error=%v, want ErrForbidden", err)
	}
	if err := service.store.validateCampaignRunOwnersUnlocked(administrator, campaign); err != nil {
		t.Fatalf("administrator evidence admission: %v", err)
	}
}
