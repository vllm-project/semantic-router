package evaluationplane

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

func newCampaignLifecycle(campaign Campaign, actor Actor) CampaignLifecycle {
	createdAt := campaign.CreatedAt.UTC().Truncate(time.Microsecond)
	deleteAfter, err := retentionDeleteAfter(RetentionStandard, createdAt)
	if err != nil {
		panic(err)
	}
	return CampaignLifecycle{
		SchemaVersion: campaignLifecycleSchemaVersion, CampaignID: campaign.ID,
		OwnerPrincipalDigest: actor.principalDigest,
		RetentionClass:       RetentionStandard, DeleteAfter: deleteAfter,
		CreatedAt: createdAt, UpdatedAt: createdAt, PolicyRevision: lifecyclePolicyRevision,
	}
}

func validateCampaignLifecycle(campaign Campaign, lifecycle CampaignLifecycle) error {
	if lifecycle.SchemaVersion != campaignLifecycleSchemaVersion || lifecycle.CampaignID != campaign.ID ||
		!digestPattern.MatchString(lifecycle.OwnerPrincipalDigest) ||
		!digestPattern.MatchString(lifecycle.CreationAuditDigest) ||
		lifecycle.PolicyRevision != lifecyclePolicyRevision || lifecycle.CreatedAt.IsZero() ||
		lifecycle.UpdatedAt.Before(lifecycle.CreatedAt) || !lifecycle.CreatedAt.Equal(campaign.CreatedAt) ||
		lifecycle.PolicyDigest != lifecycleDigest(lifecycle) {
		return fmt.Errorf("%w: campaign lifecycle identity is invalid", ErrInvalid)
	}
	switch lifecycle.RetentionClass {
	case RetentionEphemeral, RetentionStandard:
		if lifecycle.DeleteAfter == nil || lifecycle.DeleteAfter.Before(lifecycle.CreatedAt) {
			return fmt.Errorf("%w: campaign lifecycle expiry is invalid", ErrInvalid)
		}
	case RetentionProtected:
		if lifecycle.DeleteAfter != nil {
			return fmt.Errorf("%w: protected campaign lifecycle cannot expire", ErrInvalid)
		}
	default:
		return fmt.Errorf("%w: campaign retention class is invalid", ErrInvalid)
	}
	return nil
}

func publicCampaignLifecycle(lifecycle CampaignLifecycle) CampaignLifecycleView {
	return CampaignLifecycleView{
		SchemaVersion: lifecycle.SchemaVersion, CampaignID: lifecycle.CampaignID,
		RetentionClass: lifecycle.RetentionClass, EvidenceHold: lifecycle.EvidenceHold,
		DeleteAfter: lifecycle.DeleteAfter, CreatedAt: lifecycle.CreatedAt, UpdatedAt: lifecycle.UpdatedAt,
	}
}

func (s *Store) readCampaignLifecycle(campaign Campaign) (CampaignLifecycle, error) {
	lifecycle, err := s.readCampaignLifecycleEvidence(campaign)
	if err != nil {
		return CampaignLifecycle{}, err
	}
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: campaign.ID},
	); err != nil {
		return CampaignLifecycle{}, err
	}
	return lifecycle, nil
}

func (s *Store) readCampaignLifecycleEvidence(campaign Campaign) (CampaignLifecycle, error) {
	directory, err := s.checkedCampaignDir(campaign.ID)
	if err != nil {
		return CampaignLifecycle{}, err
	}
	var lifecycle CampaignLifecycle
	if err := readJSON(filepath.Join(directory, lifecycleFileName), &lifecycle); err != nil {
		return CampaignLifecycle{}, fmt.Errorf("validate campaign lifecycle: %w", err)
	}
	if err := validateCampaignLifecycle(campaign, lifecycle); err != nil {
		return CampaignLifecycle{}, err
	}
	return lifecycle, nil
}

func (s *Store) checkedCampaignDir(id string) (string, error) {
	if !validClientRequestID(id) {
		return "", fmt.Errorf("%w: campaign id must be a canonical UUID", ErrInvalid)
	}
	directory := filepath.Join(s.campaignRoot, id)
	if err := requirePrivateDirectory(directory); err != nil {
		if os.IsNotExist(err) {
			return "", fmt.Errorf("%w: campaign %s", ErrNotFound, id)
		}
		return "", err
	}
	return directory, nil
}

func (s *Store) campaignForActorUnlocked(actor Actor, id string) (Campaign, CampaignLifecycle, error) {
	campaign, lifecycle, err := s.campaignForCreateActorUnlocked(actor, id)
	if err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if err := s.requireNoPendingCampaignPublications(); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: campaign.ID},
	); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	return campaign, lifecycle, nil
}

// campaignForCreateActorUnlocked authenticates an existing identity without
// committing its namespace. It is used only so the same actor and exact create
// request can close a prior visible parent-sync failure.
func (s *Store) campaignForCreateActorUnlocked(actor Actor, id string) (Campaign, CampaignLifecycle, error) {
	if err := validateActor(actor); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if err := s.requireStableCampaignDeletionLedgerUnlocked(); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	campaign, err := s.getCampaignUnlocked(id)
	if err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	lifecycle, err := s.readCampaignLifecycleEvidence(campaign)
	if err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
		return campaign, lifecycle, fmt.Errorf("%w: campaign belongs to another evaluation principal", ErrForbidden)
	}
	return campaign, lifecycle, nil
}

func (s *Store) campaignForMutationActorUnlocked(actor Actor, id string) (Campaign, CampaignLifecycle, error) {
	if err := validateActor(actor); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if err := s.requireStableCampaignDeletionLedgerUnlocked(); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	campaign, err := s.getCampaignUnlocked(id)
	if err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	lifecycle, err := s.readCampaignLifecycleEvidence(campaign)
	if err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
		return campaign, lifecycle, fmt.Errorf("%w: campaign belongs to another evaluation principal", ErrForbidden)
	}
	if err := s.requireNoPendingCampaignPublications(); err != nil {
		return Campaign{}, CampaignLifecycle{}, err
	}
	return campaign, lifecycle, nil
}

func (s *Store) CampaignLifecycle(actor Actor, id string) (CampaignLifecycleView, error) {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	_, lifecycle, err := s.campaignForActorUnlocked(actor, id)
	if err != nil {
		return CampaignLifecycleView{}, err
	}
	return publicCampaignLifecycle(lifecycle), nil
}

func (s *Store) UpdateCampaignLifecycle(
	actor Actor,
	id string,
	request UpdateLifecycleRequest,
) (CampaignLifecycleView, error) {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	campaign, lifecycle, err := s.campaignForMutationActorUnlocked(actor, id)
	if err != nil {
		if auditErr := s.auditRejectedCampaignLifecycleMutation(actor, id, request, err); auditErr != nil {
			return CampaignLifecycleView{}, auditErr
		}
		return CampaignLifecycleView{}, err
	}
	if request.RetentionClass == nil && request.EvidenceHold == nil {
		return CampaignLifecycleView{}, fmt.Errorf("%w: campaign lifecycle update contains no mutation", ErrInvalid)
	}
	now := s.lifecycleNow().UTC().Truncate(time.Microsecond)
	updated, actions, err := planCampaignLifecycleMutation(lifecycle, request, now)
	if err != nil {
		if auditErr := s.auditInvalidCampaignRetention(actor, id, lifecycle.OwnerPrincipalDigest); auditErr != nil {
			return CampaignLifecycleView{}, auditErr
		}
		return CampaignLifecycleView{}, err
	}
	if err := s.reconcileUnpublishedLifecycleRetry(
		lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: id},
		actor.principalDigest,
		lifecycleDigest(request),
		lifecycle.PolicyDigest,
	); err != nil {
		return CampaignLifecycleView{}, err
	}
	if len(actions) == 0 {
		return s.resolveUnchangedCampaignLifecycle(actor, id, request, lifecycle)
	}
	return s.persistCampaignLifecycleMutation(actor, campaign, request, lifecycle, updated, actions, now)
}

func planCampaignLifecycleMutation(
	lifecycle CampaignLifecycle,
	request UpdateLifecycleRequest,
	now time.Time,
) (CampaignLifecycle, []string, error) {
	updated := lifecycle
	actions := make([]string, 0, 2)
	if request.RetentionClass != nil && *request.RetentionClass != lifecycle.RetentionClass {
		deleteAfter, err := retentionDeleteAfter(*request.RetentionClass, now)
		if err != nil {
			return CampaignLifecycle{}, nil, err
		}
		updated.RetentionClass, updated.DeleteAfter = *request.RetentionClass, deleteAfter
		actions = append(actions, "retention")
	}
	if request.EvidenceHold != nil && *request.EvidenceHold != lifecycle.EvidenceHold {
		updated.EvidenceHold = *request.EvidenceHold
		if updated.EvidenceHold {
			actions = append(actions, "hold")
		} else {
			actions = append(actions, "release")
		}
	}
	return updated, actions, nil
}

func (s *Store) resolveUnchangedCampaignLifecycle(
	actor Actor,
	id string,
	request UpdateLifecycleRequest,
	lifecycle CampaignLifecycle,
) (CampaignLifecycleView, error) {
	directory, err := s.checkedCampaignDir(id)
	if err != nil {
		return CampaignLifecycleView{}, err
	}
	if err := s.resolveLifecycleResourceDurability(
		lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: id},
		directory,
		actor.principalDigest,
		lifecycleDigest(request),
		lifecycle.PolicyDigest,
	); err != nil {
		return CampaignLifecycleView{}, err
	}
	return publicCampaignLifecycle(lifecycle), nil
}

func (s *Store) persistCampaignLifecycleMutation(
	actor Actor,
	campaign Campaign,
	request UpdateLifecycleRequest,
	previous, updated CampaignLifecycle,
	actions []string,
	now time.Time,
) (CampaignLifecycleView, error) {
	id := campaign.ID
	if err := s.requireLifecycleResourceDurable(
		lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: id},
	); err != nil {
		return CampaignLifecycleView{}, err
	}
	reason := lifecycleOwnerAuthorizationReason(actor, previous.OwnerPrincipalDigest)
	for _, action := range actions {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, action, "allowed", reason, id, previous.OwnerPrincipalDigest,
		); err != nil {
			return CampaignLifecycleView{}, err
		}
	}
	updated.UpdatedAt, updated.PolicyDigest = now, ""
	updated.PolicyDigest = lifecycleDigest(updated)
	if err := validateCampaignLifecycle(campaign, updated); err != nil {
		return CampaignLifecycleView{}, err
	}
	directory, err := s.checkedCampaignDir(id)
	if err != nil {
		return CampaignLifecycleView{}, err
	}
	if err := s.writeCampaignLifecycleResource(actor, request, filepath.Join(directory, lifecycleFileName), updated); err != nil {
		return CampaignLifecycleView{}, err
	}
	return publicCampaignLifecycle(updated), nil
}

func (s *Store) auditRejectedCampaignLifecycleMutation(
	actor Actor,
	id string,
	request UpdateLifecycleRequest,
	rejection error,
) error {
	if !validClientRequestID(id) || validateActor(actor) != nil {
		return nil
	}
	owner := ""
	if current, err := s.getCampaignUnlocked(id); err == nil {
		if lifecycle, lifecycleErr := s.readCampaignLifecycleEvidence(current); lifecycleErr == nil {
			owner = lifecycle.OwnerPrincipalDigest
		}
	}
	return s.auditCampaignLifecycleDenialUnlocked(
		actor, id, owner, request, lifecycleDenialReason(rejection),
	)
}

func (s *Store) auditInvalidCampaignRetention(actor Actor, id, ownerDigest string) error {
	_, err := s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceCampaign, "retention", "denied", "invalid_request", id, ownerDigest,
	)
	return err
}

func (s *Store) auditCampaignLifecycleDenialUnlocked(
	actor Actor,
	id string,
	ownerDigest string,
	request UpdateLifecycleRequest,
	reason string,
) error {
	actions := make([]string, 0, 2)
	if request.RetentionClass != nil {
		actions = append(actions, "retention")
	}
	if request.EvidenceHold != nil {
		if *request.EvidenceHold {
			actions = append(actions, "hold")
		} else {
			actions = append(actions, "release")
		}
	}
	for _, action := range actions {
		if _, err := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, action, "denied", reason, id, ownerDigest,
		); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) validateLifecycleCampaignBindings(startupAuthority bool) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	campaigns, err := s.loadStoredCampaignsForOpenUnlocked(startupAuthority)
	if err != nil {
		return err
	}
	for _, campaign := range campaigns {
		lifecycle, err := s.readCampaignLifecycle(campaign)
		if err != nil {
			return err
		}
		directory, err := s.checkedCampaignDir(campaign.ID)
		if err != nil {
			return err
		}
		if startupAuthority {
			if err := s.syncLifecycleResourceDirectory(
				directory, "evaluation campaign lifecycle startup recovery",
			); err != nil {
				return err
			}
		}
		record, exists := s.lifecycle.records[lifecycle.CreationAuditDigest]
		if !exists {
			record, exists = s.lifecycle.creationBindings[lifecycle.CreationAuditDigest]
		}
		if !exists || record.Action != "create" || record.Decision != "allowed" ||
			record.ResourceKind != lifecycleResourceCampaign || record.ResourceID != campaign.ID ||
			record.OwnerDigest != lifecycle.OwnerPrincipalDigest {
			return fmt.Errorf("%w: campaign lifecycle is not bound to its creation audit", ErrInvalid)
		}
	}
	return nil
}

func campaignDeletionBlocked(lifecycle CampaignLifecycle) error {
	if lifecycle.EvidenceHold {
		return fmt.Errorf("%w: held campaign evidence cannot be deleted", ErrConflict)
	}
	if lifecycle.RetentionClass == RetentionProtected {
		return fmt.Errorf("%w: protected campaign evidence cannot be deleted", ErrConflict)
	}
	return nil
}
