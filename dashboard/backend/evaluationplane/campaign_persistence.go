package evaluationplane

import (
	"fmt"
	"os"
)

// campaignNamespacePersistence is the narrow durability seam for Campaign
// publication and deletion. Namespace visibility is never treated as durable
// until the campaigns parent directory has synced successfully.
type campaignNamespacePersistence interface {
	Rename(source, destination string) error
	RemoveAll(path string) error
	SyncDirectory(path, description string) error
}

type atomicCampaignNamespacePersistence struct{}

func (atomicCampaignNamespacePersistence) Rename(source, destination string) error {
	return os.Rename(source, destination)
}

func (atomicCampaignNamespacePersistence) RemoveAll(path string) error {
	return os.RemoveAll(path)
}

func (atomicCampaignNamespacePersistence) SyncDirectory(path, description string) error {
	return syncEvaluationDirectory(path, description)
}

func (s *Store) beginCampaignPublicationDurability(actor Actor, campaign Campaign) {
	s.lifecycle.campaignNamespaceMu.Lock()
	defer s.lifecycle.campaignNamespaceMu.Unlock()
	s.lifecycle.pendingCampaignPublications[campaign.ID] = pendingNamespacePublication{
		actorDigest:    actor.principalDigest,
		identityDigest: lifecycleDigest(campaign),
	}
}

func (s *Store) abandonCampaignPublicationDurability(id string) {
	s.lifecycle.campaignNamespaceMu.Lock()
	defer s.lifecycle.campaignNamespaceMu.Unlock()
	delete(s.lifecycle.pendingCampaignPublications, id)
}

func (s *Store) requireNoPendingCampaignPublications() error {
	s.lifecycle.campaignNamespaceMu.Lock()
	defer s.lifecycle.campaignNamespaceMu.Unlock()
	if len(s.lifecycle.pendingCampaignPublications) != 0 {
		return fmt.Errorf("%w: evaluation campaign publication requires the startup owner or explicit create retry", ErrConflict)
	}
	return nil
}

func (s *Store) publishCampaignCreationUnlocked(
	actor Actor,
	campaign Campaign,
	staged string,
	destination string,
) error {
	s.beginCampaignPublicationDurability(actor, campaign)
	if err := s.campaignPersistence.Rename(staged, destination); err != nil {
		if _, statErr := os.Lstat(destination); statErr == nil {
			return fmt.Errorf("%w: campaign %s already exists", ErrConflict, campaign.ID)
		} else if os.IsNotExist(statErr) {
			s.abandonCampaignPublicationDurability(campaign.ID)
		}
		return fmt.Errorf("publish evaluation campaign: %w", err)
	}
	return s.resolveCampaignPublicationDurability(actor, campaign)
}

func (s *Store) resolveCampaignPublicationDurability(actor Actor, campaign Campaign) error {
	s.lifecycle.campaignNamespaceMu.Lock()
	defer s.lifecycle.campaignNamespaceMu.Unlock()
	if len(s.lifecycle.pendingCampaignPublications) == 0 {
		return nil
	}
	pending, exists := s.lifecycle.pendingCampaignPublications[campaign.ID]
	if !exists || len(s.lifecycle.pendingCampaignPublications) != 1 ||
		pending.actorDigest != actor.principalDigest || pending.identityDigest != lifecycleDigest(campaign) {
		return fmt.Errorf("%w: evaluation campaign create retry does not match the pending publication", ErrConflict)
	}
	if err := s.campaignPersistence.SyncDirectory(s.campaignRoot, "evaluation campaign publication"); err != nil {
		return fmt.Errorf("%w: evaluation campaign publication durability is uncertain: %w", ErrConflict, err)
	}
	delete(s.lifecycle.pendingCampaignPublications, campaign.ID)
	return nil
}

func (s *Store) recoverCampaignPublicationDurability() error {
	s.lifecycle.campaignNamespaceMu.Lock()
	defer s.lifecycle.campaignNamespaceMu.Unlock()
	if err := s.campaignPersistence.SyncDirectory(s.campaignRoot, "evaluation campaign publication recovery"); err != nil {
		return fmt.Errorf("%w: evaluation campaign publication durability is uncertain: %w", ErrConflict, err)
	}
	clear(s.lifecycle.pendingCampaignPublications)
	return nil
}
