package evaluationplane

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type faultingCampaignNamespacePersistence struct {
	delegate     campaignNamespacePersistence
	syncFailures int
	syncCalls    int
	removeAll    func(string) error
}

func (p *faultingCampaignNamespacePersistence) Rename(source, destination string) error {
	return p.delegate.Rename(source, destination)
}

func (p *faultingCampaignNamespacePersistence) RemoveAll(path string) error {
	if p.removeAll != nil {
		return p.removeAll(path)
	}
	return p.delegate.RemoveAll(path)
}

func (p *faultingCampaignNamespacePersistence) SyncDirectory(path, description string) error {
	p.syncCalls++
	if p.syncFailures > 0 {
		p.syncFailures--
		return errors.New("injected campaign parent sync failure")
	}
	return p.delegate.SyncDirectory(path, description)
}

func TestCampaignDeletionParentSyncFailureKeepsReferencesFailClosed(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-delete-owner", false)
	other := testLifecycleActor(t, "campaign-delete-other", false)
	campaign := campaignV2StoredSchemaFixture(t)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)

	faults := &faultingCampaignNamespacePersistence{
		delegate:     atomicCampaignNamespacePersistence{},
		syncFailures: 1,
	}
	service.store.campaignPersistence = faults
	if err := service.DeleteCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("campaign deletion sync error=%v, want ErrConflict", err)
	}
	tombstone := campaignDeletionPath(service.store.campaignRoot, campaign.ID)
	if _, err := os.Lstat(filepath.Join(service.store.campaignRoot, campaign.ID)); !os.IsNotExist(err) {
		t.Fatalf("live campaign remained after atomic hide: %v", err)
	}
	if err := requirePrivateDirectory(tombstone); err != nil {
		t.Fatalf("deletion intent is unavailable after uncertain sync: %v", err)
	}

	// A reference scan must first make the namespace transition durable. If
	// that sync is still unavailable it cannot treat the hidden Campaign as
	// deleted and therefore cannot authorize evidence reclamation.
	faults.syncFailures = 1
	runID := campaign.Decision.Evidence[0].RunID
	if err := service.store.ensureRunNotCampaignReferencedUnlocked(runID); !errors.Is(err, ErrConflict) {
		t.Fatalf("uncertain campaign reference scan error=%v, want ErrConflict", err)
	}
	if err := service.DeleteCampaignAs(other, campaign.ID); !errors.Is(err, ErrForbidden) {
		t.Fatalf("cross-owner deletion retry error=%v, want ErrForbidden", err)
	}
	if err := requirePrivateDirectory(tombstone); err != nil {
		t.Fatalf("cross-owner retry consumed the deletion intent: %v", err)
	}

	// Startup observes the hidden namespace, syncs it, and only then reclaims
	// the tombstone. The Campaign's referenced evidence can be unpinned only
	// after this recovery boundary succeeds.
	if err := service.Close(); err != nil {
		t.Fatalf("close before campaign deletion recovery: %v", err)
	}
	reopened, err := newStandaloneStore(root)
	if err != nil {
		t.Fatalf("recover campaign deletion on restart: %v", err)
	}
	if _, err := os.Lstat(tombstone); !os.IsNotExist(err) {
		t.Fatalf("restart retained committed campaign deletion intent: %v", err)
	}
	if _, err := reopened.getCampaignUnlocked(campaign.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("recovered campaign lookup error=%v, want ErrNotFound", err)
	}
}

func TestPeerOpenCannotRecoverLiveCampaignDeletionIntent(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-peer-delete-owner", false)
	campaign := campaignV2StoredSchemaFixture(t)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)
	faults := &faultingCampaignNamespacePersistence{
		delegate: atomicCampaignNamespacePersistence{}, syncFailures: 1,
	}
	service.store.campaignPersistence = faults
	if err := service.DeleteCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("campaign deletion sync error=%v, want ErrConflict", err)
	}
	tombstone := campaignDeletionPath(service.store.campaignRoot, campaign.ID)
	peer, err := NewService(Options{
		DataDir: service.store.Root(), PythonPath: "python3", ConfigPath: service.registrySource.configPath,
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid",
		CodeRevision: testSourceRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if peer != nil {
		_ = peer.Close()
	}
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("peer opener error=%v, want deletion recovery ErrConflict", err)
	}
	if err := requirePrivateDirectory(tombstone); err != nil {
		t.Fatalf("peer opener consumed deletion intent: %v", err)
	}
	if faults.syncFailures != 0 {
		t.Fatalf("initial delete did not consume its injected sync failure: %d", faults.syncFailures)
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); err != nil {
		t.Fatalf("owner deletion retry: %v", err)
	}
	if _, err := os.Lstat(tombstone); !os.IsNotExist(err) {
		t.Fatalf("owner retry retained deletion intent: %v", err)
	}
}

func TestCampaignCreateCannotReuseIdentityBehindDeletionIntent(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	owner := testLifecycleActor(t, "campaign-delete-identity-owner", false)
	other := testLifecycleActor(t, "campaign-delete-identity-other", false)
	campaign := campaignV2StoredSchemaFixture(t)
	writeCampaignLifecycleFixture(t, service.store, campaign, owner)
	service.store.campaignPersistence = &faultingCampaignNamespacePersistence{
		delegate: atomicCampaignNamespacePersistence{}, syncFailures: 1,
	}
	if err := service.DeleteCampaignAs(owner, campaign.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("campaign deletion sync error=%v, want ErrConflict", err)
	}
	request := CreateCampaignRequest{
		ClientRequestID: campaign.ID, Name: campaign.Name, Description: campaign.Description,
		ChangeProfile: campaign.ChangeProfile, GateBindings: campaign.GateBindings,
	}
	if _, err := service.CreateCampaignAs(other, request); !errors.Is(err, ErrConflict) {
		t.Fatalf("cross-owner identity reuse error=%v, want ErrConflict", err)
	}
	if err := requirePrivateDirectory(campaignDeletionPath(service.store.campaignRoot, campaign.ID)); err != nil {
		t.Fatalf("identity reuse consumed deletion intent: %v", err)
	}
}

func TestCampaignDeletionRecoversPartialTombstoneCleanup(t *testing.T) {
	for _, test := range []struct {
		name    string
		restart bool
	}{
		{name: "same process"},
		{name: "restart", restart: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			owner := testLifecycleActor(t, "campaign-partial-delete-owner", false)
			campaign := campaignV2StoredSchemaFixture(t)
			writeCampaignLifecycleFixture(t, service.store, campaign, owner)

			tombstone := campaignDeletionPath(service.store.campaignRoot, campaign.ID)
			faults := &faultingCampaignNamespacePersistence{
				delegate: atomicCampaignNamespacePersistence{},
				removeAll: func(path string) error {
					if path != tombstone {
						t.Fatalf("cleanup path=%q, want %q", path, tombstone)
					}
					if err := os.Remove(filepath.Join(path, campaignFileName)); err != nil {
						return err
					}
					return errors.New("injected partial campaign tombstone cleanup")
				},
			}
			service.store.campaignPersistence = faults
			if err := service.DeleteCampaignAs(owner, campaign.ID); err != nil {
				t.Fatalf("durably committed campaign deletion returned cleanup failure: %v", err)
			}
			if err := requirePrivateDirectory(tombstone); err != nil {
				t.Fatalf("partial deletion tombstone is unavailable: %v", err)
			}
			if _, err := os.Lstat(filepath.Join(tombstone, campaignFileName)); !os.IsNotExist(err) {
				t.Fatalf("partial cleanup did not remove the injected member: %v", err)
			}

			if test.restart {
				if err := service.Close(); err != nil {
					t.Fatalf("close before partial campaign deletion restart: %v", err)
				}
				if _, err := newStandaloneStore(root); err != nil {
					t.Fatalf("restart did not reclaim partial campaign tombstone: %v", err)
				}
			} else {
				service.store.campaignPersistence = atomicCampaignNamespacePersistence{}
				campaigns, err := service.store.loadStoredCampaignsUnlocked()
				if err != nil || len(campaigns) != 0 {
					t.Fatalf("same-process deletion recovery campaigns=%d err=%v", len(campaigns), err)
				}
			}
			if _, err := os.Lstat(tombstone); !os.IsNotExist(err) {
				t.Fatalf("recovery retained partial campaign tombstone: %v", err)
			}
		})
	}
}
