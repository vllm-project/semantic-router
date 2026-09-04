package evaluationplane

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const deletingCampaignPrefix = ".deleting-evaluation-campaign-"

func campaignDeletionPath(root, id string) string {
	return filepath.Join(root, deletingCampaignPrefix+id)
}

func campaignDeletionID(name string) (string, bool) {
	if !strings.HasPrefix(name, deletingCampaignPrefix) {
		return "", false
	}
	id := strings.TrimPrefix(name, deletingCampaignPrefix)
	return id, validClientRequestID(id)
}

type campaignDeletionIntent struct {
	id       string
	path     string
	complete bool
}

// recoverCampaignDeletionsUnlocked commits every visible deletion intent
// before reclaiming it. The first parent sync is the safety boundary: no
// caller may omit the hidden Campaign from a reference scan before it succeeds.
func (s *Store) recoverCampaignDeletionsUnlocked() error {
	intents, err := s.listCampaignDeletionIntentsUnlocked()
	if err != nil {
		return err
	}
	if len(intents) == 0 {
		return nil
	}
	for _, intent := range intents {
		if intent.complete {
			if _, _, err := readCampaignDeletionBundle(intent.path, intent.id); err != nil {
				return err
			}
		}
	}
	if err := s.campaignPersistence.SyncDirectory(s.campaignRoot, "evaluation campaign deletion intent"); err != nil {
		return fmt.Errorf("%w: evaluation campaign deletion durability is uncertain: %w", ErrConflict, err)
	}
	for _, intent := range intents {
		s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: intent.id})
	}
	for _, intent := range intents {
		if err := s.campaignPersistence.RemoveAll(intent.path); err != nil {
			return fmt.Errorf("%w: reclaim evaluation campaign deletion intent: %w", ErrConflict, err)
		}
	}
	if err := s.campaignPersistence.SyncDirectory(s.campaignRoot, "evaluation campaign deletion recovery"); err != nil {
		return fmt.Errorf("%w: evaluation campaign deletion cleanup is uncertain: %w", ErrConflict, err)
	}
	return nil
}

func (s *Store) requireNoCampaignDeletionIntentsUnlocked() error {
	intents, err := s.listCampaignDeletionIntentsUnlocked()
	if err != nil {
		return err
	}
	if len(intents) != 0 {
		return fmt.Errorf("%w: evaluation campaign deletion recovery is required", ErrConflict)
	}
	return nil
}

// Generic reads may reclaim only a partial tombstone, which can be produced
// exclusively after the campaignRoot commit cut by best-effort cleanup. An
// intact bundle still carries the owner's deletion authorization and remains
// fail-closed until that owner retries or the first startup owner recovers it.
func (s *Store) requireStableCampaignDeletionLedgerUnlocked() error {
	intents, err := s.listCampaignDeletionIntentsUnlocked()
	if err != nil {
		return err
	}
	partials := make([]campaignDeletionIntent, 0, len(intents))
	for _, intent := range intents {
		if intent.complete {
			return fmt.Errorf("%w: evaluation campaign deletion recovery is required", ErrConflict)
		}
		partials = append(partials, intent)
	}
	if len(partials) == 0 {
		return nil
	}
	for _, intent := range partials {
		if err := s.campaignPersistence.RemoveAll(intent.path); err != nil {
			return fmt.Errorf("%w: reclaim committed evaluation campaign deletion: %w", ErrConflict, err)
		}
		s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: intent.id})
	}
	if err := s.campaignPersistence.SyncDirectory(
		s.campaignRoot, "evaluation campaign committed deletion cleanup",
	); err != nil {
		return fmt.Errorf("%w: evaluation campaign deletion cleanup is uncertain: %w", ErrConflict, err)
	}
	return nil
}

func (s *Store) listCampaignDeletionIntentsUnlocked() ([]campaignDeletionIntent, error) {
	if err := requirePrivateDirectory(s.campaignRoot); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(s.campaignRoot)
	if err != nil {
		return nil, fmt.Errorf("list evaluation campaign deletions: %w", err)
	}
	intents := make([]campaignDeletionIntent, 0)
	for _, entry := range entries {
		id, recognized := campaignDeletionID(entry.Name())
		if !recognized {
			if strings.HasPrefix(entry.Name(), deletingCampaignPrefix) {
				return nil, fmt.Errorf("%w: evaluation campaign deletion intent is invalid", ErrInvalid)
			}
			continue
		}
		path := filepath.Join(s.campaignRoot, entry.Name())
		if entry.Type()&os.ModeSymlink != 0 || !entry.IsDir() || requirePrivateDirectory(path) != nil {
			return nil, fmt.Errorf("%w: evaluation campaign deletion intent is invalid", ErrInvalid)
		}
		if _, liveErr := os.Lstat(filepath.Join(s.campaignRoot, id)); liveErr == nil {
			return nil, fmt.Errorf("%w: evaluation campaign has both live and deleting identities", ErrInvalid)
		} else if !os.IsNotExist(liveErr) {
			return nil, fmt.Errorf("inspect live evaluation campaign during deletion recovery: %w", liveErr)
		}
		complete, inspectErr := inspectCampaignDeletionFiles(path)
		if inspectErr != nil {
			return nil, inspectErr
		}
		intents = append(intents, campaignDeletionIntent{id: id, path: path, complete: complete})
	}
	return intents, nil
}

func inspectCampaignDeletionFiles(path string) (bool, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return false, err
	}
	seen := make(map[string]bool, 2)
	for _, entry := range entries {
		if entry.Name() != campaignFileName && entry.Name() != lifecycleFileName {
			return false, fmt.Errorf("%w: evaluation campaign deletion bundle is invalid", ErrInvalid)
		}
		if !entry.Type().IsRegular() || entry.Type()&os.ModeSymlink != 0 || seen[entry.Name()] {
			return false, fmt.Errorf("%w: evaluation campaign deletion bundle is invalid", ErrInvalid)
		}
		seen[entry.Name()] = true
	}
	return seen[campaignFileName] && seen[lifecycleFileName], nil
}

// resumeCampaignDeletionAsUnlocked handles a retry after rename succeeded but
// the parent sync result was uncertain. The intact hidden bundle retains the
// owner boundary until the authorized caller commits the namespace transition.
func (s *Store) resumeCampaignDeletionAsUnlocked(actor Actor, id string) (bool, error) {
	if !validClientRequestID(id) {
		return false, fmt.Errorf("%w: campaign id must be a canonical UUID", ErrInvalid)
	}
	tombstone := campaignDeletionPath(s.campaignRoot, id)
	if _, err := os.Lstat(tombstone); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return true, fmt.Errorf("inspect evaluation campaign deletion intent: %w", err)
	}
	_, lifecycle, err := readCampaignDeletionBundle(tombstone, id)
	if err != nil {
		complete, inspectErr := inspectCampaignDeletionFiles(tombstone)
		if inspectErr != nil {
			return true, inspectErr
		}
		if complete {
			return true, err
		}
		if cleanupErr := s.campaignPersistence.RemoveAll(tombstone); cleanupErr != nil {
			return true, fmt.Errorf("%w: reclaim committed evaluation campaign deletion: %w", ErrConflict, cleanupErr)
		}
		if syncErr := s.campaignPersistence.SyncDirectory(
			s.campaignRoot, "evaluation campaign committed deletion retry",
		); syncErr != nil {
			return true, fmt.Errorf("%w: evaluation campaign deletion cleanup is uncertain: %w", ErrConflict, syncErr)
		}
		s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: id})
		return true, nil
	}
	if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, "delete", "denied", "not_owner",
			id, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return true, auditErr
		}
		return true, fmt.Errorf("%w: campaign belongs to another evaluation principal", ErrForbidden)
	}
	if err := campaignDeletionBlocked(lifecycle); err != nil {
		return true, err
	}
	if err := s.campaignPersistence.SyncDirectory(s.campaignRoot, "evaluation campaign deletion retry"); err != nil {
		return true, fmt.Errorf("%w: evaluation campaign deletion durability is uncertain: %w", ErrConflict, err)
	}
	s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: id})
	s.cleanupCommittedCampaignDeletion(tombstone)
	return true, nil
}

func readCampaignDeletionBundle(directory, id string) (Campaign, CampaignLifecycle, error) {
	if err := requirePrivateDirectory(directory); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	files, err := os.ReadDir(directory)
	if err != nil || len(files) != 2 || files[0].Name() != campaignFileName ||
		files[1].Name() != lifecycleFileName || !files[0].Type().IsRegular() ||
		!files[1].Type().IsRegular() || files[0].Type()&os.ModeSymlink != 0 ||
		files[1].Type()&os.ModeSymlink != 0 {
		return Campaign{}, CampaignLifecycle{}, fmt.Errorf(
			"%w: evaluation campaign deletion bundle is invalid", ErrInvalid,
		)
	}
	var campaign Campaign
	if err := readJSON(filepath.Join(directory, campaignFileName), &campaign); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if err := validateStoredCampaign(id, campaign); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	var lifecycle CampaignLifecycle
	if err := readJSON(filepath.Join(directory, lifecycleFileName), &lifecycle); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if err := validateCampaignLifecycle(campaign, lifecycle); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	return campaign, lifecycle, nil
}

func (s *Store) publishCampaignDeletionUnlocked(directory, id string) error {
	tombstone := campaignDeletionPath(s.campaignRoot, id)
	if _, err := os.Lstat(tombstone); err == nil {
		return fmt.Errorf("%w: campaign deletion is already in progress", ErrConflict)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("inspect evaluation campaign deletion destination: %w", err)
	}
	if err := s.campaignPersistence.Rename(directory, tombstone); err != nil {
		return fmt.Errorf("begin evaluation campaign deletion: %w", err)
	}
	if err := s.campaignPersistence.SyncDirectory(s.campaignRoot, "evaluation campaign deletion"); err != nil {
		// The tombstone is intentionally retained. Every campaign reference
		// scan must commit or reject this intent before it can omit the Campaign.
		return fmt.Errorf("%w: evaluation campaign deletion durability is uncertain: %w", ErrConflict, err)
	}
	s.forgetLifecycleResourceDurability(lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: id})
	s.cleanupCommittedCampaignDeletion(tombstone)
	return nil
}

func (s *Store) cleanupCommittedCampaignDeletion(tombstone string) {
	if err := s.campaignPersistence.RemoveAll(tombstone); err != nil {
		log.Printf("evaluationplane: committed campaign deletion cleanup deferred: %v", err)
		return
	}
	if err := s.campaignPersistence.SyncDirectory(s.campaignRoot, "evaluation campaign deletion cleanup"); err != nil {
		log.Printf("evaluationplane: committed campaign deletion cleanup sync deferred: %v", err)
	}
}
