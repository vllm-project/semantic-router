package evaluationplane

import (
	"fmt"
	"reflect"
)

func (s *Store) requireFrozenCollectionIdentityUnlocked(
	candidate CollectionPlanItem,
	expected collectionItemIdentity,
) error {
	current, err := s.currentCollectionItemIdentityUnlocked(candidate)
	if err != nil {
		return fmt.Errorf("%w: collection candidate identity cannot be revalidated: %w", ErrConflict, err)
	}
	current.EstimatedBytes = 0
	expected.EstimatedBytes = 0
	if !reflect.DeepEqual(current, expected) {
		return fmt.Errorf("%w: collection candidate identity changed after planning", ErrConflict)
	}
	return nil
}

func (s *Store) currentCollectionItemIdentityUnlocked(candidate CollectionPlanItem) (collectionItemIdentity, error) {
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()

	if candidate.CampaignID != "" {
		campaign, err := s.getCampaignUnlocked(candidate.CampaignID)
		if err != nil {
			return collectionItemIdentity{}, err
		}
		lifecycle, err := s.readCampaignLifecycle(campaign)
		if err != nil {
			return collectionItemIdentity{}, err
		}
		if lifecycle.DeleteAfter == nil {
			return collectionItemIdentity{}, fmt.Errorf("campaign is no longer an expired collection candidate")
		}
		return collectionItemIdentity{
			CampaignID: candidate.CampaignID, CompletedAt: campaign.CreatedAt,
			RetentionClass: lifecycle.RetentionClass, DeleteAfter: *lifecycle.DeleteAfter,
			LifecycleDigest: lifecycle.PolicyDigest, EvidenceDigest: campaign.ManifestDigest,
		}, nil
	}
	if candidate.PairID != "" {
		pair, err := s.readControlledPair(candidate.PairID)
		if err != nil {
			return collectionItemIdentity{}, err
		}
		ids := []string{pair.BaselineRunID, pair.CandidateRunID}
		runs := make([]Run, 0, 2)
		lifecycles := make([]RunLifecycle, 0, 2)
		references := make(map[string]bool)
		for _, id := range ids {
			run, err := s.getRunUnlocked(id)
			if err != nil {
				return collectionItemIdentity{}, err
			}
			lifecycle, err := s.readRunLifecycle(run)
			if err != nil {
				return collectionItemIdentity{}, err
			}
			if err := s.markRunCASReferences(id, references); err != nil {
				return collectionItemIdentity{}, err
			}
			runs, lifecycles = append(runs, run), append(lifecycles, lifecycle)
		}
		if runs[0].CompletedAt == nil || runs[1].CompletedAt == nil ||
			lifecycles[0].DeleteAfter == nil || lifecycles[1].DeleteAfter == nil {
			return collectionItemIdentity{}, fmt.Errorf("controlled pair is no longer an expired terminal candidate")
		}
		completedAt := *runs[0].CompletedAt
		if runs[1].CompletedAt.After(completedAt) {
			completedAt = *runs[1].CompletedAt
		}
		deleteAfter := *lifecycles[0].DeleteAfter
		if lifecycles[1].DeleteAfter.After(deleteAfter) {
			deleteAfter = *lifecycles[1].DeleteAfter
		}
		return collectionItemIdentity{
			PairID: pair.PairID, RunIDs: ids, Status: StatusCompleted, CompletedAt: completedAt,
			RetentionClass: lifecycles[0].RetentionClass, DeleteAfter: deleteAfter,
			LifecycleDigest: digestString(lifecycles[0].PolicyDigest + ":" + lifecycles[1].PolicyDigest),
			EvidenceDigest:  collectionReferenceDigest(references),
		}, nil
	}

	run, err := s.getRunUnlocked(candidate.RunID)
	if err != nil {
		return collectionItemIdentity{}, err
	}
	lifecycle, err := s.readRunLifecycle(run)
	if err != nil {
		return collectionItemIdentity{}, err
	}
	references := make(map[string]bool)
	if err := s.markRunCASReferences(run.ID, references); err != nil {
		return collectionItemIdentity{}, err
	}
	if run.CompletedAt == nil || lifecycle.DeleteAfter == nil {
		return collectionItemIdentity{}, fmt.Errorf("run is no longer an expired terminal collection candidate")
	}
	completedAt := *run.CompletedAt
	return collectionItemIdentity{
		RunID: run.ID, Status: run.Status, CompletedAt: completedAt,
		RetentionClass: lifecycle.RetentionClass, DeleteAfter: *lifecycle.DeleteAfter,
		LifecycleDigest: lifecycle.PolicyDigest, EvidenceDigest: collectionReferenceDigest(references),
	}, nil
}
