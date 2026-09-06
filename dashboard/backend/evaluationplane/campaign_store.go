package evaluationplane

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func recoverStagedCampaigns(root string) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("list staged evaluation campaigns: %w", err)
	}
	removed := false
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), stagedCampaignPrefix) {
			continue
		}
		path := filepath.Join(root, entry.Name())
		if entry.Type()&os.ModeSymlink != 0 || !entry.IsDir() || requirePrivateDirectory(path) != nil {
			return fmt.Errorf("%w: staged evaluation campaign is invalid", ErrInvalid)
		}
		if err := os.RemoveAll(path); err != nil {
			return fmt.Errorf("remove staged evaluation campaign: %w", err)
		}
		removed = true
	}
	if removed {
		return syncEvaluationDirectory(root, "evaluation campaign recovery")
	}
	return nil
}

func requireNoStagedCampaigns(root string) error {
	entries, err := os.ReadDir(root)
	if err != nil {
		return fmt.Errorf("list staged evaluation campaigns: %w", err)
	}
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), stagedCampaignPrefix) {
			continue
		}
		path := filepath.Join(root, entry.Name())
		if entry.Type()&os.ModeSymlink != 0 || !entry.IsDir() || requirePrivateDirectory(path) != nil {
			return fmt.Errorf("%w: staged evaluation campaign is invalid", ErrInvalid)
		}
		return fmt.Errorf("%w: staged evaluation campaign recovery requires the startup owner", ErrConflict)
	}
	return nil
}

const (
	campaignFileName     = "campaign.json"
	stagedCampaignPrefix = ".staged-evaluation-campaign-"
)

type campaignManifestSubject struct {
	SchemaVersion   string               `json:"schema_version"`
	ContractVersion string               `json:"contract_version"`
	ID              string               `json:"id"`
	Name            string               `json:"name"`
	Description     string               `json:"description"`
	ChangeProfile   ChangeProfile        `json:"change_profile"`
	GateBindings    CampaignGateBindings `json:"gate_bindings"`
	CreatedAt       time.Time            `json:"created_at"`
}

func campaignManifestDigest(campaign Campaign) (string, error) {
	subject := campaignManifestSubject{
		SchemaVersion: campaign.SchemaVersion, ContractVersion: campaign.ContractVersion,
		ID: campaign.ID, Name: campaign.Name, Description: campaign.Description,
		ChangeProfile: campaign.ChangeProfile, GateBindings: campaign.GateBindings, CreatedAt: campaign.CreatedAt,
	}
	encoded, err := json.Marshal(subject)
	if err != nil {
		return "", fmt.Errorf("encode evaluation campaign identity: %w", err)
	}
	return fmt.Sprintf("sha256:%x", sha256.Sum256(encoded)), nil
}

func campaignDecisionDigest(decision CampaignDecision) (string, error) {
	decision.DecisionDigest = ""
	encoded, err := json.Marshal(decision)
	if err != nil {
		return "", fmt.Errorf("encode evaluation campaign decision: %w", err)
	}
	return fmt.Sprintf("sha256:%x", sha256.Sum256(encoded)), nil
}

func (s *Store) createCampaignAsUnlocked(actor Actor, campaign Campaign) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	if err := validateStoredCampaign(campaign.ID, campaign); err != nil {
		return err
	}
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.validateCampaignRunReferencesUnlocked(campaign); err != nil {
		return fmt.Errorf("%w: campaign run evidence is unavailable: %w", ErrInvalid, err)
	}
	if err := s.validateCampaignRunOwnersUnlocked(actor, campaign); err != nil {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, "create", "denied", lifecycleDenialReason(err), campaign.ID, "",
		); auditErr != nil {
			return auditErr
		}
		return err
	}
	if err := requirePrivateDirectory(s.campaignRoot); err != nil {
		return err
	}
	destination := filepath.Join(s.campaignRoot, campaign.ID)
	if _, err := os.Lstat(destination); err == nil {
		return fmt.Errorf("%w: campaign %s already exists", ErrConflict, campaign.ID)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("inspect evaluation campaign destination: %w", err)
	}
	lifecycle := newCampaignLifecycle(campaign, actor)
	campaignBytes, campaignSizeErr := campaignJSONSize(campaign)
	if campaignSizeErr != nil {
		return campaignSizeErr
	}
	lifecycleBytes, lifecycleSizeErr := campaignLifecycleAdmissionBytes(lifecycle)
	if lifecycleSizeErr != nil {
		return lifecycleSizeErr
	}
	bundleBytes, bundleSizeErr := checkedLifecycleBytes(campaignBytes, lifecycleBytes)
	if bundleSizeErr != nil {
		return bundleSizeErr
	}
	if reason, quotaErr := s.requireCampaignCreateQuotaUnlocked(actor, bundleBytes); quotaErr != nil {
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, "create", "denied", reason, campaign.ID, actor.principalDigest,
		); auditErr != nil {
			return auditErr
		}
		return quotaErr
	}
	createAudit, auditErr := s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceCampaign, "create", "allowed", lifecycleOwnerAuthorizationReason(actor, actor.principalDigest),
		campaign.ID, actor.principalDigest,
	)
	if auditErr != nil {
		return auditErr
	}
	lifecycle.CreationAuditDigest, lifecycle.PolicyDigest = createAudit.Digest, ""
	lifecycle.PolicyDigest = lifecycleDigest(lifecycle)
	if err := validateCampaignLifecycle(campaign, lifecycle); err != nil {
		return err
	}
	staged, stageErr := os.MkdirTemp(s.campaignRoot, stagedCampaignPrefix)
	if stageErr != nil {
		return fmt.Errorf("stage evaluation campaign: %w", stageErr)
	}
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(staged)
		}
	}()
	if err := writeJSONAtomic(filepath.Join(staged, campaignFileName), campaign); err != nil {
		return err
	}
	if err := writeJSONAtomic(filepath.Join(staged, lifecycleFileName), lifecycle); err != nil {
		return err
	}
	if err := syncEvaluationDirectory(staged, "staged evaluation campaign"); err != nil {
		return err
	}
	if err := s.publishCampaignCreationUnlocked(actor, campaign, staged, destination); err != nil {
		return err
	}
	published = true
	return nil
}

func campaignJSONSize(value any) (int64, error) {
	encoded, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return 0, fmt.Errorf("encode evaluation campaign bundle: %w", err)
	}
	return int64(len(encoded) + 1), nil
}

func campaignLifecycleAdmissionBytes(lifecycle CampaignLifecycle) (int64, error) {
	if lifecycle.CreationAuditDigest != "" || lifecycle.PolicyDigest != "" {
		return 0, fmt.Errorf("%w: campaign lifecycle admission requires unresolved digests", ErrInvalid)
	}
	baseBytes, err := campaignJSONSize(lifecycle)
	if err != nil {
		return 0, err
	}
	// Both digest fields are required JSON strings. Replacing their empty
	// values with canonical SHA-256 text adds an exact, content-independent
	// number of bytes, so quota admission does not need fabricated evidence.
	const canonicalDigestTextBytes = int64(len("sha256:") + sha256.Size*2)
	resolvedDigestBytes, err := checkedLifecycleBytes(canonicalDigestTextBytes, canonicalDigestTextBytes)
	if err != nil {
		return 0, err
	}
	return checkedLifecycleBytes(baseBytes, resolvedDigestBytes)
}

func (s *Store) getCampaignUnlocked(id string) (Campaign, error) {
	directory, err := s.checkedCampaignDir(id)
	if err != nil {
		return Campaign{}, err
	}
	var campaign Campaign
	if err := readJSON(filepath.Join(directory, campaignFileName), &campaign); err != nil {
		return Campaign{}, err
	}
	if err := validateStoredCampaign(id, campaign); err != nil {
		return Campaign{}, err
	}
	return campaign, nil
}

func (s *Store) DeleteCampaignAs(actor Actor, id string) error {
	s.lifecycle.mu.Lock()
	defer s.lifecycle.mu.Unlock()
	return s.deleteCampaignAsUnlocked(actor, id, false)
}

func (s *Store) deleteCampaignAsUnlocked(actor Actor, id string, collection bool) error {
	if err := validateActor(actor); err != nil {
		return err
	}
	if resumed, err := s.resumeCampaignDeletionAsUnlocked(actor, id); resumed || err != nil {
		return err
	}
	campaign, lifecycle, lookupErr := s.campaignForActorUnlocked(actor, id)
	if lookupErr != nil {
		if validClientRequestID(id) {
			owner := ""
			if lifecycle.OwnerPrincipalDigest != "" {
				owner = lifecycle.OwnerPrincipalDigest
			}
			if _, auditErr := s.appendLifecycleAuditUnlocked(
				actor, lifecycleResourceCampaign, "delete", "denied", lifecycleDenialReason(lookupErr), id, owner,
			); auditErr != nil {
				return auditErr
			}
		}
		return lookupErr
	}
	if blockErr := campaignDeletionBlocked(lifecycle); blockErr != nil {
		reason := "protected_retention"
		if lifecycle.EvidenceHold {
			reason = "evidence_hold"
		}
		if _, auditErr := s.appendLifecycleAuditUnlocked(
			actor, lifecycleResourceCampaign, "delete", "denied", reason, id, lifecycle.OwnerPrincipalDigest,
		); auditErr != nil {
			return auditErr
		}
		return blockErr
	}
	if collection && (lifecycle.DeleteAfter == nil || lifecycle.DeleteAfter.After(s.lifecycleNow().UTC())) {
		return fmt.Errorf("%w: campaign collection candidate is no longer expired", ErrConflict)
	}
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	directory, err := s.checkedCampaignDir(campaign.ID)
	if err != nil {
		return err
	}
	if _, auditErr := s.appendLifecycleAuditUnlocked(
		actor, lifecycleResourceCampaign, "delete", "allowed", lifecycleOwnerAuthorizationReason(actor, lifecycle.OwnerPrincipalDigest),
		id, lifecycle.OwnerPrincipalDigest,
	); auditErr != nil {
		return auditErr
	}
	return s.publishCampaignDeletionUnlocked(directory, id)
}

func validateStoredCampaign(id string, campaign Campaign) error {
	if campaign.SchemaVersion != SchemaVersion || campaign.ContractVersion != CampaignContractVersion ||
		campaign.ID != id || !validClientRequestID(id) || campaign.Status != CampaignStatusDecided ||
		campaign.CreatedAt.IsZero() || !validChangeProfile(campaign.ChangeProfile) {
		return fmt.Errorf("%w: evaluation campaign identity is invalid", ErrInvalid)
	}
	digest, err := campaignManifestDigest(campaign)
	if err != nil || campaign.ManifestDigest != digest || campaign.Decision.CampaignDigest != digest {
		return fmt.Errorf("%w: evaluation campaign manifest digest is invalid", ErrInvalid)
	}
	decision := campaign.Decision
	decisionDigest, digestErr := campaignDecisionDigest(decision)
	if decision.SchemaVersion != SchemaVersion || decision.ContractVersion != CampaignContractVersion ||
		decision.AttestationRevision != ServerAttestationRevision || decision.CampaignID != id ||
		digestErr != nil || decision.DecisionDigest != decisionDigest ||
		decision.CreatedAt.IsZero() || len(decision.Gates) != len(requiredGateIDs) || decision.Evidence == nil ||
		decision.Recommendations == nil {
		return fmt.Errorf("%w: evaluation campaign decision is invalid", ErrInvalid)
	}
	if err := validateCampaignDecisionContract(campaign); err != nil {
		return err
	}
	return nil
}
