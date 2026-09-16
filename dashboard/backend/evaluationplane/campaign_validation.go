package evaluationplane

import (
	"fmt"
	"strings"
)

const campaignSingleBindingRole = "evidence"

type campaignEvidenceBinding struct {
	slotID      string
	gateID      string
	bindingRole string
	runID       string
	candidate   bool
}

func validateCampaignRequest(request CreateCampaignRequest) error {
	if !validClientRequestID(request.ClientRequestID) {
		return fmt.Errorf("%w: campaign client_request_id must be a canonical UUID", ErrInvalid)
	}
	if request.Name == "" || request.Name != strings.TrimSpace(request.Name) || len(request.Name) > maxRunNameLength {
		return fmt.Errorf("%w: campaign name must be 1-%d trimmed characters", ErrInvalid, maxRunNameLength)
	}
	if request.Description != strings.TrimSpace(request.Description) || len(request.Description) > maxRunDescriptionLength {
		return fmt.Errorf("%w: campaign description exceeds its bound", ErrInvalid)
	}
	profile, ok := campaignProfileContract(request.ChangeProfile)
	if !ok {
		return fmt.Errorf("%w: campaign change_profile is invalid", ErrInvalid)
	}
	for _, slot := range profile.CampaignSlots {
		present := campaignSlotBound(request.GateBindings, slot.GateID)
		if slot.Disposition == GateDispositionRequired && !present {
			return fmt.Errorf("%w: change_profile %q requires campaign slot %s", ErrInvalid, request.ChangeProfile, strings.ToLower(slot.GateID))
		}
		if slot.Disposition == GateDispositionNotApplicable && present {
			return fmt.Errorf("%w: change_profile %q does not accept campaign slot %s", ErrInvalid, request.ChangeProfile, strings.ToLower(slot.GateID))
		}
	}
	bindings, err := campaignEvidenceBindings(request.GateBindings)
	if err != nil {
		return err
	}
	seen := make(map[string]string, len(bindings)+1)
	seen[request.ClientRequestID] = "campaign"
	for _, binding := range bindings {
		if !validClientRequestID(binding.runID) {
			return fmt.Errorf("%w: campaign %s %s run ID is invalid", ErrInvalid, binding.slotID, binding.bindingRole)
		}
		key := binding.slotID + ":" + binding.bindingRole
		if prior, duplicate := seen[binding.runID]; duplicate {
			return fmt.Errorf("%w: campaign evidence identities %s and %s must be distinct", ErrInvalid, prior, key)
		}
		seen[binding.runID] = key
	}
	return nil
}

func campaignProfileContract(id ChangeProfile) (CatalogChangeProfile, bool) {
	for _, profile := range builtinChangeProfiles() {
		if profile.ID == id {
			return copyCatalogChangeProfile(profile), true
		}
	}
	return CatalogChangeProfile{}, false
}

func campaignSlotContract(profile ChangeProfile, gateID string) (CatalogCampaignSlot, bool) {
	definition, ok := campaignProfileContract(profile)
	if !ok {
		return CatalogCampaignSlot{}, false
	}
	for _, slot := range definition.CampaignSlots {
		if slot.GateID == gateID {
			return slot, true
		}
	}
	return CatalogCampaignSlot{}, false
}

func campaignSlotBound(bindings CampaignGateBindings, gateID string) bool {
	switch gateID {
	case "G2":
		return bindings.G2RunID != ""
	case "G3":
		return bindings.G3ControlledPair != nil
	case "G4":
		return bindings.G4RunID != ""
	case "G5":
		return bindings.G5Fidelity != nil
	case "G6":
		return bindings.G6RunID != ""
	case "G7":
		return bindings.G7RunID != ""
	case "G8":
		return bindings.G8RunID != ""
	case "G9":
		return bindings.G9RunID != ""
	default:
		return false
	}
}

func campaignEvidenceBindings(bindings CampaignGateBindings) ([]campaignEvidenceBinding, error) {
	result := make([]campaignEvidenceBinding, 0, 10)
	appendRun := func(gateID, runID string) {
		if runID != "" {
			result = append(result, campaignEvidenceBinding{
				slotID: strings.ToLower(gateID), gateID: gateID,
				bindingRole: campaignSingleBindingRole, runID: runID, candidate: true,
			})
		}
	}
	appendRun("G2", bindings.G2RunID)
	if pair := bindings.G3ControlledPair; pair != nil {
		if pair.BaselineRunID == "" || pair.CandidateRunID == "" {
			return nil, fmt.Errorf("%w: g3_controlled_pair requires baseline_run_id and candidate_run_id", ErrInvalid)
		}
		result = append(result,
			campaignEvidenceBinding{slotID: "g3", gateID: "G3", bindingRole: "baseline", runID: pair.BaselineRunID},
			campaignEvidenceBinding{slotID: "g3", gateID: "G3", bindingRole: "candidate", runID: pair.CandidateRunID, candidate: true},
		)
	}
	appendRun("G4", bindings.G4RunID)
	if fidelity := bindings.G5Fidelity; fidelity != nil {
		if fidelity.ReferenceRunID == "" || fidelity.LiveRunID == "" {
			return nil, fmt.Errorf("%w: g5_fidelity requires reference_run_id and live_run_id", ErrInvalid)
		}
		result = append(result,
			campaignEvidenceBinding{slotID: "g5", gateID: "G5", bindingRole: "reference", runID: fidelity.ReferenceRunID, candidate: true},
			campaignEvidenceBinding{slotID: "g5", gateID: "G5", bindingRole: "live", runID: fidelity.LiveRunID, candidate: true},
		)
	}
	appendRun("G6", bindings.G6RunID)
	appendRun("G7", bindings.G7RunID)
	appendRun("G8", bindings.G8RunID)
	appendRun("G9", bindings.G9RunID)
	return result, nil
}

func campaignEvidenceKey(slotID, bindingRole string) string {
	return slotID + ":" + bindingRole
}

func validateCampaignEvidenceSet(
	profile ChangeProfile,
	bindings CampaignGateBindings,
	evidence map[string]campaignRunEvidence,
) error {
	expected, err := campaignEvidenceBindings(bindings)
	if err != nil {
		return err
	}
	if len(evidence) != len(expected) {
		return fmt.Errorf("%w: campaign evidence bindings are incomplete", ErrInvalid)
	}
	candidateDigest := ""
	for _, binding := range expected {
		item, present := evidence[campaignEvidenceKey(binding.slotID, binding.bindingRole)]
		if !present {
			return fmt.Errorf("%w: campaign evidence for %s %s is missing", ErrInvalid, binding.slotID, binding.bindingRole)
		}
		slot, ok := campaignSlotContract(profile, binding.gateID)
		if !ok || slot.Disposition == GateDispositionNotApplicable {
			return fmt.Errorf("%w: campaign slot %s is not registered", ErrInvalid, binding.slotID)
		}
		if err := validateCampaignBoundRun(profile, binding, slot, item); err != nil {
			return err
		}
		if binding.candidate {
			if item.anchor.CandidateSubjectDigest == "" {
				return fmt.Errorf("%w: campaign %s candidate subject is missing", ErrInvalid, binding.slotID)
			}
			if candidateDigest == "" {
				candidateDigest = item.anchor.CandidateSubjectDigest
			} else if item.anchor.CandidateSubjectDigest != candidateDigest {
				return fmt.Errorf("%w: campaign candidate evidence does not identify one exact subject", ErrInvalid)
			}
		}
	}
	if candidateDigest == "" {
		return fmt.Errorf("%w: campaign has no candidate evidence", ErrInvalid)
	}
	if pair := bindings.G3ControlledPair; pair != nil {
		baseline := evidence[campaignEvidenceKey("g3", "baseline")]
		candidate := evidence[campaignEvidenceKey("g3", "candidate")]
		if candidate.report.Run.BaselineRunID != pair.BaselineRunID ||
			candidate.manifest.BaselineRunID != pair.BaselineRunID {
			return fmt.Errorf("%w: G3 candidate does not bind its controlled baseline", ErrInvalid)
		}
		if err := validateCampaignPairedLiveSources(baseline, candidate); err != nil {
			return fmt.Errorf("%w: G3 controlled pair: %w", ErrInvalid, err)
		}
	}
	if bindings.G5Fidelity != nil {
		reference := evidence[campaignEvidenceKey("g5", "reference")]
		live := evidence[campaignEvidenceKey("g5", "live")]
		if err := validateCampaignFidelitySources(reference, live); err != nil {
			return fmt.Errorf("%w: G5 fidelity pair: %w", ErrInvalid, err)
		}
	}
	return nil
}

func validateCampaignBoundRun(
	profile ChangeProfile,
	binding campaignEvidenceBinding,
	slot CatalogCampaignSlot,
	evidence campaignRunEvidence,
) error {
	report, manifest := evidence.report, evidence.manifest
	if report.SchemaVersion != SchemaVersion || report.AttestationRevision != ServerAttestationRevision ||
		report.Run.Status != StatusCompleted || report.Run.Error != "" ||
		report.Run.ID != binding.runID || report.Run.ClientRequestID != binding.runID ||
		manifest.RunID != binding.runID || manifest.ManifestDigest != evidence.anchor.ManifestSemanticDigest ||
		report.Run.ChangeProfile != manifest.ChangeProfile || report.Run.ChangeProfile != profile {
		return fmt.Errorf("%w: %s %s is not one completed server-sealed report", ErrInvalid, binding.slotID, binding.bindingRole)
	}
	executorID, ok := manifestExecutorIdentity(manifest)
	if !ok || !campaignStringMember(slot.AcceptedExecutorIDs, executorID) {
		return fmt.Errorf("%w: %s %s uses executor %q outside the slot contract", ErrInvalid, binding.slotID, binding.bindingRole, executorID)
	}
	if slot.Mode != "" && report.Run.Mode != slot.Mode {
		return fmt.Errorf("%w: %s %s must use mode %s", ErrInvalid, binding.slotID, binding.bindingRole, slot.Mode)
	}
	if report.Run.Mode != manifest.Mode {
		return fmt.Errorf("%w: %s %s manifest/report mode mismatch", ErrInvalid, binding.slotID, binding.bindingRole)
	}
	if report.Run.Mode == ModeLive {
		if evidence.attestation == nil || evidence.anchor.ExecutionAttestationDigest == "" ||
			evidence.attestation.Digest != evidence.anchor.ExecutionAttestationDigest {
			return fmt.Errorf("%w: %s %s lacks an exact execution attestation", ErrInvalid, binding.slotID, binding.bindingRole)
		}
	} else if report.Run.Mode != ModeReplay || evidence.anchor.ExecutionAttestationDigest != "" {
		return fmt.Errorf("%w: %s %s has invalid replay provenance", ErrInvalid, binding.slotID, binding.bindingRole)
	}
	if binding.gateID == "G3" {
		return validateCampaignG3SourceTracks(binding, evidence)
	}
	if binding.gateID == "G5" {
		return validateCampaignG5SourceLevel(binding, evidence)
	}
	track, found := campaignTrackReport(report, slot.TrackID)
	if !found || track.Status != "completed" ||
		evidenceLevelRank(track.EvidenceLevel) < evidenceLevelRank(slot.MinimumEvidenceLevel) {
		return fmt.Errorf("%w: %s lacks %s track evidence at %s", ErrInvalid, binding.slotID, slot.TrackID, slot.MinimumEvidenceLevel)
	}
	gate, found := reportGate(report, binding.gateID)
	if !found || gate.TrackID != slot.TrackID || gate.ChangeProfile != profile ||
		gate.ContractVersion != GateContractVersion || gate.Disposition != slot.Disposition ||
		evidenceLevelRank(gate.EvidenceLevel) < evidenceLevelRank(slot.MinimumEvidenceLevel) ||
		(gate.Verdict != "pass" && gate.Verdict != "fail") {
		return fmt.Errorf("%w: %s lacks a conclusive qualified %s method receipt", ErrInvalid, binding.slotID, binding.gateID)
	}
	return nil
}

func validateCampaignG3SourceTracks(binding campaignEvidenceBinding, evidence campaignRunEvidence) error {
	if evidence.report.Run.Mode != ModeLive || evidence.attestation == nil ||
		evidenceLevelRank(evidence.report.Run.EvidenceLevel) < evidenceLevelRank("E3") {
		return fmt.Errorf("%w: %s must be server-attested paired-live evidence", ErrInvalid, binding.slotID)
	}
	required := map[TrackID]EvidenceLevel{"routing": "E3", "model_pool": "E4", "joint": "E5"}
	for trackID, level := range required {
		track, found := campaignTrackReport(evidence.report, trackID)
		if !found || track.Status != "completed" || evidenceLevelRank(track.EvidenceLevel) < evidenceLevelRank(level) {
			return fmt.Errorf("%w: %s %s lacks dense %s evidence at %s", ErrInvalid, binding.slotID, binding.bindingRole, trackID, level)
		}
	}
	return nil
}

func validateCampaignG5SourceLevel(binding campaignEvidenceBinding, evidence campaignRunEvidence) error {
	slot, registered := campaignSlotContract(evidence.report.Run.ChangeProfile, "G5")
	if !registered || slot.Disposition == GateDispositionNotApplicable || slot.TrackID == "" {
		return fmt.Errorf("%w: G5 is not registered for the evidence change profile", ErrInvalid)
	}
	minimum := slot.MinimumEvidenceLevel
	if binding.bindingRole == "reference" && evidenceLevelRank(minimum) > evidenceLevelRank("E4") {
		minimum = "E4"
	}
	if evidence.report.Run.Mode != ModeLive || evidence.attestation == nil {
		return fmt.Errorf("%w: G5 %s evidence must be a qualified attested live run", ErrInvalid, binding.bindingRole)
	}
	track, found := campaignTrackReport(evidence.report, slot.TrackID)
	if !found || track.Status != "completed" ||
		evidenceLevelRank(track.EvidenceLevel) < evidenceLevelRank(minimum) {
		return fmt.Errorf(
			"%w: G5 %s evidence must include completed %s track evidence at %s",
			ErrInvalid, binding.bindingRole, slot.TrackID, minimum,
		)
	}
	return nil
}

func campaignTrackReport(report Report, trackID TrackID) (TrackReport, bool) {
	for _, track := range report.Tracks {
		if track.TrackID == trackID {
			return track, true
		}
	}
	return TrackReport{}, false
}

func campaignStringMember(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func validateControlledPairedReportCohort(
	baseline campaignRunEvidence,
	candidate campaignRunEvidence,
) error {
	if baseline.manifest.ManifestDigest == "" || candidate.manifest.ManifestDigest == "" ||
		baseline.manifest.ManifestDigest != baseline.anchor.ManifestSemanticDigest ||
		candidate.manifest.ManifestDigest != candidate.anchor.ManifestSemanticDigest ||
		baseline.manifest.RunID != baseline.report.Run.ID ||
		candidate.manifest.RunID != candidate.report.Run.ID ||
		candidate.manifest.BaselineRunID != baseline.manifest.RunID {
		return fmt.Errorf(
			"%w: distinct deployment targets require a server-owned controlled pair with exact manifest bindings",
			ErrInvalid,
		)
	}
	if err := validateControlledPairAddressability(baseline.manifest, candidate.manifest); err != nil {
		return err
	}
	normalizedCandidate := normalizeControlledPairCandidate(baseline.report, candidate.report)
	if err := validatePairedReportCohort(baseline.report, normalizedCandidate); err != nil {
		return err
	}
	if !comparisonTreatment(baseline.report.Run.ChangeProfile).environment &&
		baseline.manifest.Target.BackendTopologyDigest != candidate.manifest.Target.BackendTopologyDigest {
		return fmt.Errorf(
			"%w: controlled pair backend topology changed outside the declared treatment",
			ErrInvalid,
		)
	}
	return nil
}

// A deployment target ID is an addressable treatment locator, not the logical
// Mixture cohort. The caller must validate both arms against their own sealed
// manifests and attestations before using this normalization.
func normalizeControlledPairCandidate(baseline, candidate Report) Report {
	normalized := candidate
	normalized.Run.TargetID = baseline.Run.TargetID
	if baseline.Run.ChangeProfile != "runtime_capacity" &&
		baseline.Run.ChangeProfile != "model_pool" {
		normalized.Provenance.EnvironmentSnapshotDigest = baseline.Provenance.EnvironmentSnapshotDigest
	}
	return normalized
}
