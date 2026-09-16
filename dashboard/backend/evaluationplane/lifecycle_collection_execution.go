package evaluationplane

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

func (s *Store) executeLifecycleCollectionItemUnlocked(
	actor Actor,
	candidate CollectionPlanItem,
	expected collectionItemIdentity,
	hooks collectionExecutionHooks,
) error {
	runIDs := collectionCandidateRunIDs(candidate)
	if hooks.active != nil && hooks.active(runIDs) {
		return fmt.Errorf("%w: collection candidate is still exiting", ErrConflict)
	}

	switch {
	case candidate.CampaignID != "":
		return s.executeCampaignCollectionItemUnlocked(actor, candidate, expected)
	case candidate.PairID != "":
		if err := s.executePairCollectionItemUnlocked(actor, candidate, expected); err != nil {
			return err
		}
	default:
		if err := s.executeRunCollectionItemUnlocked(actor, candidate, expected); err != nil {
			return err
		}
	}
	if hooks.closeSubscribers != nil {
		hooks.closeSubscribers(runIDs)
	}
	return nil
}

func (s *Store) executeCampaignCollectionItemUnlocked(
	actor Actor,
	candidate CollectionPlanItem,
	expected collectionItemIdentity,
) error {
	intent := campaignDeletionPath(s.campaignRoot, candidate.CampaignID)
	live := filepath.Join(s.campaignRoot, candidate.CampaignID)
	intentExists, err := collectionPathExists(intent)
	if err != nil {
		return err
	}
	if intentExists {
		return s.deleteCampaignAsUnlocked(actor, candidate.CampaignID, true)
	}
	liveExists, err := collectionPathExists(live)
	if err != nil || !liveExists {
		return err
	}
	if err := s.requireFrozenCollectionIdentityUnlocked(candidate, expected); err != nil {
		return err
	}
	return s.deleteCampaignAsUnlocked(actor, candidate.CampaignID, true)
}

func (s *Store) executePairCollectionItemUnlocked(
	actor Actor,
	candidate CollectionPlanItem,
	expected collectionItemIdentity,
) error {
	pair, err := s.readControlledPair(candidate.PairID)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return fmt.Errorf("%w: controlled pair collection identity disappeared", ErrConflict)
		}
		return err
	}
	if pair.State != controlledPairStateDeleting && pair.State != controlledPairStateDeleted {
		if err := s.requireFrozenCollectionIdentityUnlocked(candidate, expected); err != nil {
			return err
		}
	}
	return s.deleteControlledPairAs(actor, candidate.PairID)
}

func (s *Store) executeRunCollectionItemUnlocked(
	actor Actor,
	candidate CollectionPlanItem,
	expected collectionItemIdentity,
) error {
	resumed, err := s.resumeRunDeletionAsUnlocked(actor, candidate.RunID)
	if resumed {
		return err
	}
	liveExists, err := collectionPathExists(filepath.Join(s.runsRoot, candidate.RunID))
	if err != nil || !liveExists {
		return err
	}
	if err := s.requireFrozenCollectionIdentityUnlocked(candidate, expected); err != nil {
		return err
	}
	if err := s.authorizeRunActionUnlocked(actor, candidate.RunID, "delete"); err != nil {
		return err
	}
	return s.deleteRunAuthorizedUnlocked(actor, candidate.RunID)
}

func collectionPathExists(path string) (bool, error) {
	_, err := os.Lstat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, fmt.Errorf("inspect lifecycle collection resource: %w", err)
}

func collectionCandidateRunIDs(candidate CollectionPlanItem) []string {
	if candidate.RunID != "" {
		return []string{candidate.RunID}
	}
	return append([]string(nil), candidate.RunIDs...)
}

func collectionResultThrough(plan CollectionPlan, next int) CollectionResult {
	result := CollectionResult{
		SchemaVersion: lifecyclePolicySchemaVersion,
		Applied:       true,
		Plan:          plan,
		DeletedRunIDs: []string{}, DeletedPairIDs: []string{}, DeletedCampaignIDs: []string{},
	}
	for _, candidate := range plan.Candidates[:next] {
		switch {
		case candidate.CampaignID != "":
			result.DeletedCampaignIDs = append(result.DeletedCampaignIDs, candidate.CampaignID)
		case candidate.PairID != "":
			result.DeletedPairIDs = append(result.DeletedPairIDs, candidate.PairID)
			result.DeletedRunIDs = append(result.DeletedRunIDs, candidate.RunIDs...)
		default:
			result.DeletedRunIDs = append(result.DeletedRunIDs, candidate.RunID)
		}
	}
	return result
}
