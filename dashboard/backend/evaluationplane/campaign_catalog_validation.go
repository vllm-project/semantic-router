package evaluationplane

import "fmt"

// validateCampaignCatalogContracts keeps the public campaign composition
// catalog honest: every advertised slot must name an executable track/mode/
// evidence combination, and a required controlled pair must have an
// independently identifiable treatment factor.
func validateCampaignCatalogContracts(registry *Registry, profiles []CatalogChangeProfile) error {
	campaignGates := canonicalCampaignGateDefinitions()
	for _, profile := range profiles {
		if len(profile.CampaignSlots) != len(campaignGates) {
			return fmt.Errorf("campaign profile %q must define every campaign gate exactly once", profile.ID)
		}
		seen := make(map[string]bool, len(profile.CampaignSlots))
		for index, slot := range profile.CampaignSlots {
			expectedGateID := campaignGates[index].ID
			if slot.GateID != expectedGateID || seen[slot.GateID] {
				return fmt.Errorf("campaign profile %q has invalid slot order or duplicate %q", profile.ID, slot.GateID)
			}
			seen[slot.GateID] = true
			if !validGateDisposition(slot.Disposition) {
				return fmt.Errorf("campaign profile %q slot %s has invalid disposition", profile.ID, slot.GateID)
			}
			if slot.BindingKind != campaignBindingKindForGate(slot.GateID) {
				return fmt.Errorf("campaign profile %q slot %s has invalid binding kind", profile.ID, slot.GateID)
			}
			track, ok := registry.tracks[slot.TrackID]
			if !ok || !containsMode(track.Modes, slot.Mode) ||
				!trackSupportsCampaignLevel(track, slot.MinimumEvidenceLevel) {
				return fmt.Errorf("campaign profile %q slot %s is not reachable on track %q", profile.ID, slot.GateID, slot.TrackID)
			}
			if !campaignSlotHasCapableExecutor(registry, slot) {
				return fmt.Errorf("campaign profile %q slot %s has no capable executor", profile.ID, slot.GateID)
			}
			if slot.GateID == "G3" && slot.Disposition == GateDispositionRequired &&
				!comparisonTreatment(profile.ID).supported {
				return fmt.Errorf("campaign profile %q requires G3 without a paired treatment protocol", profile.ID)
			}
		}
	}
	return nil
}

func campaignBindingKindForGate(gateID string) CampaignBindingKind {
	for _, gate := range releaseGateDefinitions() {
		if gate.ID == gateID && gate.Campaign != nil {
			return gate.Campaign.BindingKind
		}
	}
	return ""
}

func trackSupportsCampaignLevel(track CatalogTrack, minimum EvidenceLevel) bool {
	for _, level := range track.EvidenceLevels {
		if evidenceLevelRank(level) >= evidenceLevelRank(minimum) {
			return true
		}
	}
	return false
}

func campaignSlotHasCapableExecutor(registry *Registry, slot CatalogCampaignSlot) bool {
	for _, executorID := range slot.AcceptedExecutorIDs {
		executor, ok := registry.executors[executorID]
		if !ok || executor.Mode != slot.Mode || !containsTrack(executor.TrackIDs, slot.TrackID) {
			continue
		}
		if executor.EvidenceLevelCeiling == "" ||
			evidenceLevelRank(executor.EvidenceLevelCeiling) >= evidenceLevelRank(slot.MinimumEvidenceLevel) {
			return true
		}
	}
	return false
}
