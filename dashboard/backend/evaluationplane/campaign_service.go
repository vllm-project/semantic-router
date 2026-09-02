package evaluationplane

import (
	"errors"
	"fmt"
	"reflect"
	"time"
)

func (s *Service) CreateCampaignAs(actor Actor, request CreateCampaignRequest) (Campaign, error) {
	releaseOperation, operationErr := s.beginOperation()
	if operationErr != nil {
		return Campaign{}, operationErr
	}
	defer releaseOperation()
	if err := validateActor(actor); err != nil {
		return Campaign{}, err
	}
	if err := validateCampaignRequest(request); err != nil {
		return Campaign{}, err
	}
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	if existing, lifecycle, lookupErr := s.store.campaignForCreateActorUnlocked(actor, request.ClientRequestID); lookupErr == nil {
		if campaignMatchesRequest(existing, request) {
			if err := s.store.requireLifecycleResourceDurable(
				lifecycleResourceRef{Kind: lifecycleResourceCampaign, ID: existing.ID},
			); err != nil {
				return Campaign{}, err
			}
			if err := s.store.resolveCampaignPublicationDurability(actor, existing); err != nil {
				return Campaign{}, err
			}
			return existing, nil
		}
		return Campaign{}, fmt.Errorf("%w: campaign id belongs to another request", ErrConflict)
	} else if errors.Is(lookupErr, ErrForbidden) {
		if _, auditErr := s.store.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, "create", "denied", "not_owner",
			request.ClientRequestID, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return Campaign{}, auditErr
		}
		return Campaign{}, lookupErr
	} else if !errors.Is(lookupErr, ErrNotFound) {
		return Campaign{}, lookupErr
	}
	if err := s.store.requireNoPendingCampaignPublications(); err != nil {
		return Campaign{}, err
	}
	// Refresh the complete projection before entering the non-reentrant evidence
	// publication critical section. createCampaignAsUnlocked revalidates the
	// physical bundles, owners, anchors, and quota while publication is locked,
	// so this early refresh cannot introduce a TOCTOU publication window.
	if err := s.requireCompleteRunLedgerWithinLifecycle(); err != nil {
		return Campaign{}, err
	}
	s.store.lifecycle.evidenceMu.Lock()
	defer s.store.lifecycle.evidenceMu.Unlock()
	if err := s.store.validateCampaignBindingOwnersUnlocked(actor, request.GateBindings); err != nil {
		if _, auditErr := s.store.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, "create", "denied", lifecycleDenialReason(err),
			request.ClientRequestID, "",
		); auditErr != nil {
			return Campaign{}, auditErr
		}
		return Campaign{}, err
	}
	// The lifecycle evidence write lock already pins every bound run. Reserve
	// only bounded decoder capacity here; attempting to acquire its read side
	// would be a non-reentrant RWMutex upgrade in reverse.
	release, acquireErr := s.reserveEvidenceReadCapacity()
	if acquireErr != nil {
		return Campaign{}, acquireErr
	}
	defer release()
	evidence, err := s.loadCampaignEvidence(request.ChangeProfile, request.GateBindings, nil)
	if err != nil {
		return Campaign{}, err
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	campaign := Campaign{
		SchemaVersion: SchemaVersion, ContractVersion: CampaignContractVersion,
		ID: request.ClientRequestID, Name: request.Name, Description: request.Description,
		ChangeProfile: request.ChangeProfile, Status: CampaignStatusDecided,
		GateBindings: request.GateBindings, CreatedAt: now,
	}
	campaign.ManifestDigest, err = campaignManifestDigest(campaign)
	if err != nil {
		return Campaign{}, err
	}
	campaign.Decision, err = buildCampaignDecision(campaign, evidence, now)
	if err != nil {
		return Campaign{}, err
	}
	if err := s.store.createCampaignAsUnlocked(actor, campaign); err != nil {
		return Campaign{}, err
	}
	return campaign, nil
}

func (s *Service) GetCampaignAs(actor Actor, id string) (Campaign, error) {
	releaseOperation, operationErr := s.beginOperation()
	if operationErr != nil {
		return Campaign{}, operationErr
	}
	defer releaseOperation()
	s.store.lifecycle.mu.Lock()
	defer s.store.lifecycle.mu.Unlock()
	campaign, _, err := s.store.campaignForActorUnlocked(actor, id)
	if err != nil {
		return Campaign{}, err
	}
	s.store.lifecycle.evidenceMu.Lock()
	defer s.store.lifecycle.evidenceMu.Unlock()
	release, err := s.reserveEvidenceReadCapacity()
	if err != nil {
		return Campaign{}, err
	}
	defer release()
	expected := make(map[string]CampaignEvidenceAnchor, len(campaign.Decision.Evidence))
	for _, anchor := range campaign.Decision.Evidence {
		expected[campaignEvidenceKey(anchor.SlotID, anchor.BindingRole)] = anchor
	}
	evidence, err := s.loadCampaignEvidence(campaign.ChangeProfile, campaign.GateBindings, expected)
	if err != nil {
		return Campaign{}, err
	}
	rebuilt, err := buildCampaignDecision(campaign, evidence, campaign.Decision.CreatedAt)
	if err != nil || !reflect.DeepEqual(rebuilt, campaign.Decision) {
		return Campaign{}, fmt.Errorf("%w: campaign decision differs from its sealed private evidence", ErrInvalid)
	}
	return campaign, nil
}

func (s *Service) DeleteCampaignAs(actor Actor, id string) error {
	release, err := s.beginOperation()
	if err != nil {
		return err
	}
	defer release()
	return s.store.DeleteCampaignAs(actor, id)
}

func (s *Service) CampaignLifecycle(actor Actor, id string) (CampaignLifecycleView, error) {
	release, err := s.beginOperation()
	if err != nil {
		return CampaignLifecycleView{}, err
	}
	defer release()
	return s.store.CampaignLifecycle(actor, id)
}

func (s *Service) UpdateCampaignLifecycle(
	actor Actor,
	id string,
	request UpdateLifecycleRequest,
) (CampaignLifecycleView, error) {
	release, err := s.beginOperation()
	if err != nil {
		return CampaignLifecycleView{}, err
	}
	defer release()
	return s.store.UpdateCampaignLifecycle(actor, id, request)
}

func campaignMatchesRequest(campaign Campaign, request CreateCampaignRequest) bool {
	return campaign.ID == request.ClientRequestID && campaign.Name == request.Name &&
		campaign.Description == request.Description && campaign.ChangeProfile == request.ChangeProfile &&
		reflect.DeepEqual(campaign.GateBindings, request.GateBindings)
}
