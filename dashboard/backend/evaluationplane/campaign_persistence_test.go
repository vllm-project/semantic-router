package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestCampaignCreationRetryClosesParentSyncUncertainty(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-create-durability-owner", false)
	campaign := campaignV2StoredSchemaFixture(t)
	stableCampaign := campaignWithIdentity(t, campaign, newTestClientRequestID())
	writeCampaignLifecycleFixture(t, service.store, stableCampaign, owner)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)

	destination := filepath.Join(service.store.campaignRoot, campaign.ID)
	staged := filepath.Join(service.store.campaignRoot, stagedCampaignPrefix+"durability-test")
	if err := os.Rename(destination, staged); err != nil {
		t.Fatalf("stage visible campaign publication: %v", err)
	}
	faults := &faultingCampaignNamespacePersistence{
		delegate:     atomicCampaignNamespacePersistence{},
		syncFailures: 2,
	}
	service.store.campaignPersistence = faults
	if err := service.store.publishCampaignCreationUnlocked(owner, campaign, staged, destination); !errors.Is(err, ErrConflict) {
		t.Fatalf("visible publication sync error=%v, want ErrConflict", err)
	}
	if err := requirePrivateDirectory(destination); err != nil {
		t.Fatalf("failed publication did not leave a retryable visible bundle: %v", err)
	}

	request := CreateCampaignRequest{
		ClientRequestID: campaign.ID,
		Name:            campaign.Name,
		Description:     campaign.Description,
		ChangeProfile:   campaign.ChangeProfile,
		GateBindings:    campaign.GateBindings,
	}
	if _, err := service.GetCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("generic campaign read exposed pending publication: %v", err)
	}
	foreign := testLifecycleActor(t, "campaign-create-durability-foreign", false)
	if _, err := service.CreateCampaignAs(foreign, request); !errors.Is(err, ErrForbidden) {
		t.Fatalf("foreign campaign create retry error=%v, want ErrForbidden", err)
	}
	if _, err := openTestPeerStore(t, service.store, LifecycleLimits{}); !errors.Is(err, ErrConflict) {
		t.Fatalf("peer opener recovered pending campaign publication: %v", err)
	}
	callsBeforeUnrelatedRetry := faults.syncCalls
	stableRequest := campaignCreateRequest(stableCampaign)
	if _, err := service.CreateCampaignAs(owner, stableRequest); !errors.Is(err, ErrConflict) {
		t.Fatalf("unrelated idempotent campaign retry error=%v, want pending publication conflict", err)
	}
	if faults.syncCalls != callsBeforeUnrelatedRetry {
		t.Fatalf("unrelated campaign retry synced another publication: calls=%d want=%d", faults.syncCalls, callsBeforeUnrelatedRetry)
	}
	administrator := testLifecycleActor(t, "campaign-create-durability-admin", true)
	if _, err := service.CreateCampaignAs(administrator, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("different administrator adopted pending campaign publication: %v", err)
	}
	if faults.syncCalls != callsBeforeUnrelatedRetry {
		t.Fatalf("different administrator synced pending campaign publication: calls=%d want=%d", faults.syncCalls, callsBeforeUnrelatedRetry)
	}
	if _, err := service.CreateCampaignAs(owner, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("idempotent retry bypassed persistent parent sync failure: %v", err)
	}
	if existing, err := service.CreateCampaignAs(owner, request); err != nil || existing.ID != campaign.ID {
		t.Fatalf("idempotent retry did not close parent durability: campaign=%s err=%v", existing.ID, err)
	}
}

func campaignCreateRequest(campaign Campaign) CreateCampaignRequest {
	return CreateCampaignRequest{
		ClientRequestID: campaign.ID,
		Name:            campaign.Name,
		Description:     campaign.Description,
		ChangeProfile:   campaign.ChangeProfile,
		GateBindings:    campaign.GateBindings,
	}
}

func campaignWithIdentity(t *testing.T, campaign Campaign, id string) Campaign {
	t.Helper()
	campaign.ID = id
	manifestDigest, err := campaignManifestDigest(campaign)
	if err != nil {
		t.Fatalf("digest campaign fixture identity: %v", err)
	}
	campaign.ManifestDigest = manifestDigest
	campaign.Decision.CampaignID = id
	campaign.Decision.CampaignDigest = manifestDigest
	campaign.Decision.DecisionDigest = ""
	campaign.Decision.DecisionDigest, err = campaignDecisionDigest(campaign.Decision)
	if err != nil {
		t.Fatalf("digest campaign fixture decision: %v", err)
	}
	if err := validateStoredCampaign(id, campaign); err != nil {
		t.Fatalf("validate re-identified campaign fixture: %v", err)
	}
	return campaign
}
