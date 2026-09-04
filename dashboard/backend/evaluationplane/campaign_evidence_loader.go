package evaluationplane

import (
	"fmt"
	"reflect"
)

type campaignRunEvidence struct {
	report      Report
	records     []executionRecordEvidence
	attestation *executionAttestation
	manifest    RunManifest
	anchor      CampaignEvidenceAnchor
}

func (s *Service) loadCampaignEvidence(
	profile ChangeProfile,
	gateBindings CampaignGateBindings,
	expected map[string]CampaignEvidenceAnchor,
) (map[string]campaignRunEvidence, error) {
	bindings, err := campaignEvidenceBindings(gateBindings)
	if err != nil {
		return nil, err
	}
	if expected != nil && len(expected) != len(bindings) {
		return nil, fmt.Errorf("%w: campaign evidence anchor set is incomplete", ErrInvalid)
	}
	loaded := make(map[string]campaignRunEvidence, len(bindings))
	for _, binding := range bindings {
		key := campaignEvidenceKey(binding.slotID, binding.bindingRole)
		var sealed *CampaignEvidenceAnchor
		if expected != nil {
			value, ok := expected[key]
			if !ok {
				return nil, fmt.Errorf("%w: campaign evidence anchor %s is missing", ErrInvalid, key)
			}
			sealed = &value
		}
		item, loadErr := s.loadCampaignRunEvidence(binding, sealed)
		if loadErr != nil {
			return nil, loadErr
		}
		loaded[key] = item
	}
	if err := validateCampaignEvidenceSet(profile, gateBindings, loaded); err != nil {
		return nil, err
	}
	return loaded, nil
}

func (s *Service) loadCampaignRunEvidence(
	binding campaignEvidenceBinding,
	expected *CampaignEvidenceAnchor,
) (campaignRunEvidence, error) {
	item, err := s.loadSealedCampaignRunEvidence(binding.runID)
	if err != nil {
		return campaignRunEvidence{}, err
	}
	if binding.gateID == "G3" || binding.gateID == "G5" {
		item, err = s.loadCampaignRunRecords(item)
		if err != nil {
			return campaignRunEvidence{}, err
		}
	}
	return bindCampaignRunEvidence(binding, item, expected)
}

// loadSealedCampaignRunEvidence verifies the durable run bundle without
// assigning it to a campaign gate or role. Binding is a separate projection so
// report comparison and readiness never impersonate a different gate.
func (s *Service) loadSealedCampaignRunEvidence(runID string) (campaignRunEvidence, error) {
	label := "run " + runID
	report, err := s.decodedReport(runID)
	if err != nil {
		return campaignRunEvidence{}, fmt.Errorf("%s report: %w", label, err)
	}
	storedAnchor, err := s.store.readReportAnchor(runID)
	if err != nil {
		return campaignRunEvidence{}, fmt.Errorf("%s anchor: %w", label, err)
	}
	manifest, manifestBytes, err := s.readDurableManifest(runID)
	if err != nil {
		return campaignRunEvidence{}, fmt.Errorf("%s manifest: %w", label, err)
	}
	manifestArtifactDigest, _ := digestAndSize(manifestBytes)
	if storedAnchor.ManifestSemanticDigest != manifest.ManifestDigest ||
		storedAnchor.ManifestArtifactDigest != manifestArtifactDigest {
		return campaignRunEvidence{}, fmt.Errorf("%w: %s manifest no longer matches its sealed anchor", ErrInvalid, label)
	}
	item := campaignRunEvidence{report: report, manifest: manifest}
	if report.Run.Mode == ModeLive {
		attestation, attestationErr := s.store.readExecutionAttestationForManifest(runID, manifest)
		if attestationErr != nil || storedAnchor.ExecutionAttestationDigest == "" ||
			storedAnchor.ExecutionAttestationDigest != attestation.Digest {
			return campaignRunEvidence{}, fmt.Errorf("%w: %s lacks an exact execution attestation", ErrInvalid, label)
		}
		item.attestation = &attestation
	} else if storedAnchor.ExecutionAttestationDigest != "" {
		return campaignRunEvidence{}, fmt.Errorf("%w: %s replay evidence claims a live attestation", ErrInvalid, label)
	}
	item.anchor = CampaignEvidenceAnchor{
		RunID:                  runID,
		ManifestSemanticDigest: storedAnchor.ManifestSemanticDigest,
		ManifestArtifactDigest: storedAnchor.ManifestArtifactDigest,
		ReportDigest:           storedAnchor.ReportDigest, PrivateReceiptDigest: storedAnchor.PrivateReceiptDigest,
		ExecutionAttestationDigest: storedAnchor.ExecutionAttestationDigest,
	}
	return item, nil
}

func (s *Service) loadCampaignRunRecords(item campaignRunEvidence) (campaignRunEvidence, error) {
	records, err := s.loadPrivateComparisonRecords(item.report.Run.ID)
	if err != nil {
		return campaignRunEvidence{}, fmt.Errorf("run %s records: %w", item.report.Run.ID, err)
	}
	item.records = records
	return item, nil
}

func bindCampaignRunEvidence(
	binding campaignEvidenceBinding,
	item campaignRunEvidence,
	expected *CampaignEvidenceAnchor,
) (campaignRunEvidence, error) {
	label := binding.slotID + " " + binding.bindingRole
	item.anchor.SlotID = binding.slotID
	item.anchor.GateID = binding.gateID
	item.anchor.BindingRole = binding.bindingRole
	if binding.candidate {
		digest, digestErr := candidateSubjectDigest(item.manifest, item.report)
		if digestErr != nil {
			return campaignRunEvidence{}, fmt.Errorf("%s candidate subject: %w", label, digestErr)
		}
		item.anchor.CandidateSubjectDigest = digest
	}
	if expected != nil && !reflect.DeepEqual(item.anchor, *expected) {
		return campaignRunEvidence{}, fmt.Errorf("%w: campaign evidence anchor %s changed", ErrInvalid, label)
	}
	return item, nil
}
