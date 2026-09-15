package evaluationplane

import (
	"fmt"
	"os"
	"path/filepath"
)

func (s *Store) validateCampaignReferenceIntegrity(startupAuthority bool) error {
	s.lifecycle.evidenceMu.Lock()
	defer s.lifecycle.evidenceMu.Unlock()
	s.runIndex.coordinator.Lock()
	defer s.runIndex.coordinator.Unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	campaigns, err := s.loadStoredCampaignsForOpenUnlocked(startupAuthority)
	if err != nil {
		return err
	}
	for _, campaign := range campaigns {
		if err := s.validateCampaignRunReferencesUnlocked(campaign); err != nil {
			return fmt.Errorf("%w: campaign %s has unavailable sealed run evidence: %w", ErrInvalid, campaign.ID, err)
		}
	}
	return nil
}

func (s *Store) validateCampaignRunReferencesUnlocked(campaign Campaign) error {
	for _, expected := range campaign.Decision.Evidence {
		runDir, err := s.checkedRunDir(expected.RunID)
		if err != nil {
			return err
		}
		anchor, err := s.readReportAnchor(expected.RunID)
		if err != nil {
			return err
		}
		if anchor.ManifestSemanticDigest != expected.ManifestSemanticDigest ||
			anchor.ManifestArtifactDigest != expected.ManifestArtifactDigest ||
			anchor.ReportDigest != expected.ReportDigest ||
			anchor.PrivateReceiptDigest != expected.PrivateReceiptDigest ||
			anchor.ExecutionAttestationDigest != expected.ExecutionAttestationDigest ||
			anchor.CreatedAt.After(campaign.CreatedAt) {
			return fmt.Errorf("campaign evidence anchor does not match its sealed run")
		}
		manifestPath := filepath.Join(runDir, manifestFileName)
		manifestBytes, err := readEvidenceBytes(manifestPath, maxStructuredArtifactBytes)
		if err != nil || digestBytes(manifestBytes) != expected.ManifestArtifactDigest {
			return fmt.Errorf("campaign manifest evidence is unavailable or changed")
		}
		manifest, _, err := readRunManifestStrict(manifestPath)
		if err != nil || manifest.RunID != expected.RunID || manifest.ManifestDigest != expected.ManifestSemanticDigest {
			return fmt.Errorf("campaign manifest contract is unavailable or changed")
		}
		report, err := s.ReadReport(expected.RunID)
		if err != nil {
			return err
		}
		reportDigest, reportSize := digestAndSize(report)
		if reportDigest != expected.ReportDigest || reportSize != anchor.ReportSize {
			return fmt.Errorf("campaign report evidence is unavailable or changed")
		}
		privateReceipt, err := readEvidenceBytes(
			filepath.Join(runDir, privateChecksumArtifactName), maxStructuredArtifactBytes,
		)
		if err != nil || digestBytes(privateReceipt) != expected.PrivateReceiptDigest {
			return fmt.Errorf("campaign private receipt is unavailable or changed")
		}
		if expected.ExecutionAttestationDigest != "" {
			attestation, attestationErr := s.readExecutionAttestationForManifest(expected.RunID, manifest)
			if attestationErr != nil || attestation.Digest != expected.ExecutionAttestationDigest {
				return fmt.Errorf("campaign execution attestation is unavailable or changed")
			}
		}
	}
	return nil
}

func (s *Store) validateCampaignRunOwnersUnlocked(actor Actor, campaign Campaign) error {
	runIDs := make([]string, 0, len(campaign.Decision.Evidence))
	for _, anchor := range campaign.Decision.Evidence {
		runIDs = append(runIDs, anchor.RunID)
	}
	return s.validateCampaignRunIDOwnersUnlocked(actor, runIDs)
}

func (s *Store) validateCampaignBindingOwnersUnlocked(actor Actor, bindings CampaignGateBindings) error {
	evidenceBindings, err := campaignEvidenceBindings(bindings)
	if err != nil {
		return err
	}
	runIDs := make([]string, 0, len(evidenceBindings))
	for _, binding := range evidenceBindings {
		runIDs = append(runIDs, binding.runID)
	}
	return s.validateCampaignRunIDOwnersUnlocked(actor, runIDs)
}

func (s *Store) validateCampaignRunIDOwnersUnlocked(actor Actor, runIDs []string) error {
	seen := make(map[string]bool, len(runIDs))
	for _, runID := range runIDs {
		if seen[runID] {
			continue
		}
		seen[runID] = true
		run, err := s.getRunUnlocked(runID)
		if err != nil {
			return err
		}
		lifecycle, err := s.readRunLifecycle(run)
		if err != nil {
			return err
		}
		if !actor.administrator && lifecycle.OwnerPrincipalDigest != actor.principalDigest {
			return fmt.Errorf("%w: campaign evidence belongs to another evaluation principal", ErrForbidden)
		}
	}
	return nil
}

func (s *Store) ensureRunNotCampaignReferencedUnlocked(runID string) error {
	return s.ensureRunNotCampaignReferencedExceptUnlocked(runID, nil)
}

func (s *Store) ensureRunNotCampaignReferencedExceptUnlocked(
	runID string,
	excludedCampaigns map[string]bool,
) error {
	campaigns, err := s.loadStoredCampaignsUnlocked()
	if err != nil {
		return fmt.Errorf("%w: campaign reference ledger cannot be verified: %w", ErrConflict, err)
	}
	for _, campaign := range campaigns {
		if excludedCampaigns[campaign.ID] {
			continue
		}
		bindings, bindingErr := campaignEvidenceBindings(campaign.GateBindings)
		if bindingErr != nil {
			return fmt.Errorf("%w: campaign binding ledger is invalid: %w", ErrConflict, bindingErr)
		}
		for _, binding := range bindings {
			if binding.runID == runID {
				return fmt.Errorf("%w: run is referenced by immutable campaign %s", ErrConflict, campaign.ID)
			}
		}
	}
	return nil
}

func (s *Store) loadStoredCampaignsUnlocked() ([]Campaign, error) {
	return s.loadStoredCampaignsForOpenUnlocked(true)
}

func (s *Store) loadStoredCampaignsForOpenUnlocked(startupAuthority bool) ([]Campaign, error) {
	if err := s.requireNoPendingCampaignPublications(); err != nil {
		return nil, err
	}
	if startupAuthority {
		if err := s.requireStableCampaignDeletionLedgerUnlocked(); err != nil {
			return nil, err
		}
	} else if err := s.requireNoCampaignDeletionIntentsUnlocked(); err != nil {
		return nil, err
	}
	if err := requirePrivateDirectory(s.campaignRoot); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(s.campaignRoot)
	if err != nil {
		return nil, fmt.Errorf("list evaluation campaigns: %w", err)
	}
	campaigns := make([]Campaign, 0, len(entries))
	for _, entry := range entries {
		if !validClientRequestID(entry.Name()) || !entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("evaluation campaign store contains an invalid entry")
		}
		directory := filepath.Join(s.campaignRoot, entry.Name())
		if err := requirePrivateDirectory(directory); err != nil {
			return nil, err
		}
		files, err := os.ReadDir(directory)
		if err != nil || len(files) != 2 || files[0].Name() != campaignFileName ||
			files[1].Name() != lifecycleFileName || !files[0].Type().IsRegular() ||
			!files[1].Type().IsRegular() || files[0].Type()&os.ModeSymlink != 0 ||
			files[1].Type()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("evaluation campaign bundle is invalid")
		}
		var campaign Campaign
		if err := readJSON(filepath.Join(directory, campaignFileName), &campaign); err != nil {
			return nil, err
		}
		if err := validateStoredCampaign(entry.Name(), campaign); err != nil {
			return nil, err
		}
		campaigns = append(campaigns, campaign)
	}
	return campaigns, nil
}
