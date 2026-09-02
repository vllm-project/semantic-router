package evaluationplane

import (
	"fmt"
	"reflect"
	"testing"
)

func TestReleaseGateContractIsCompleteAndUnique(t *testing.T) {
	gates := releaseGateDefinitions()
	if len(gates) != 10 {
		t.Fatalf("release gate contract must define G0-G9 exactly once, got %d gates", len(gates))
	}
	seen := make(map[string]bool, len(gates))
	for index, gate := range gates {
		expectedID := fmt.Sprintf("G%d", index)
		if gate.ID != expectedID || seen[gate.ID] {
			t.Fatalf("release gate %d is missing, out of order, or duplicated: %+v", index, gate)
		}
		seen[gate.ID] = true
		if gate.Name == "" || gate.Description == "" || gate.Owner == "" ||
			evidenceLevelRank(gate.EvidenceLevel) < 0 || len(gate.EvidenceRefs) == 0 {
			t.Fatalf("release gate %s is incomplete: %+v", gate.ID, gate)
		}
		if gate.Campaign == nil {
			if gate.CampaignName != "" {
				t.Fatalf("non-campaign gate %s cannot advertise a campaign name", gate.ID)
			}
			continue
		}
		if gate.CampaignName == "" || gate.Campaign == nil || gate.Campaign.BindingKind == "" ||
			gate.Campaign.TrackID == "" || gate.Campaign.Mode == "" ||
			gate.Campaign.MinimumEvidenceLevel == "" || len(gate.Campaign.AcceptedExecutorIDs) == 0 {
			t.Fatalf("campaign gate %s is incomplete: %+v", gate.ID, gate)
		}
	}
	if !reflect.DeepEqual(requiredGateIDs, canonicalReleaseGateIDs()) {
		t.Fatal("report validation gate inventory drifted from the canonical release contract")
	}
	for _, gate := range gates {
		projected, ok := releaseGateDefinitionByID(gate.ID)
		if !ok || !reflect.DeepEqual(projected, gate) {
			t.Fatalf("gate %s lookup drifted from canonical definition", gate.ID)
		}
	}
}

func TestReleaseProfileCatalogAndDispositionProjectionsStayCanonical(t *testing.T) {
	definitions := releaseProfileDefinitions()
	catalogProfiles := builtinChangeProfiles()
	gates := releaseGateDefinitions()
	campaignGates := canonicalCampaignGateDefinitions()
	if len(definitions) == 0 || len(catalogProfiles) != len(definitions) {
		t.Fatalf("release profile projections are incomplete: definitions=%d catalog=%d",
			len(definitions), len(catalogProfiles))
	}
	seen := make(map[ChangeProfile]bool, len(definitions))
	for profileIndex, definition := range definitions {
		if definition.ID == "" || seen[definition.ID] {
			t.Fatalf("release profile is missing an identity or duplicated: %+v", definition)
		}
		seen[definition.ID] = true
		catalogProfile := catalogProfiles[profileIndex]
		if catalogProfile.ID != definition.ID || catalogProfile.Name != definition.Name ||
			catalogProfile.Description != definition.Description {
			t.Fatalf("release profile %s drifted from its canonical definition", definition.ID)
		}
		dispositionByGateID := make(map[string]GateDisposition, len(gates))
		for _, gate := range gates {
			disposition := definition.Dispositions[gate.ID]
			projected, ok := releaseProfileDisposition(definition.ID, gate.ID)
			if !validGateDisposition(disposition) || !ok || projected != disposition {
				t.Fatalf("release profile %s gate %s has invalid disposition %q",
					definition.ID, gate.ID, disposition)
			}
			dispositionByGateID[gate.ID] = disposition
		}
		if len(catalogProfile.CampaignSlots) != len(campaignGates) {
			t.Fatalf("release profile %s must project one slot for every campaign gate", definition.ID)
		}
		for gateIndex, gate := range campaignGates {
			slot := catalogProfile.CampaignSlots[gateIndex]
			campaign := *gate.Campaign
			description := gate.Description
			if override, ok := definition.CampaignOverrides[gate.ID]; ok {
				campaign = override
				if override.Description != "" {
					description = override.Description
				}
			}
			if slot.GateID != gate.ID || slot.Name != gate.CampaignName ||
				slot.Description != description || slot.Disposition != dispositionByGateID[gate.ID] ||
				slot.BindingKind != campaign.BindingKind || slot.TrackID != campaign.TrackID ||
				slot.Mode != campaign.Mode || slot.MinimumEvidenceLevel != campaign.MinimumEvidenceLevel ||
				!reflect.DeepEqual(slot.AcceptedExecutorIDs, campaign.AcceptedExecutorIDs) {
				t.Fatalf("release profile %s slot %s drifted from the canonical definition: %+v",
					definition.ID, gate.ID, slot)
			}
		}
	}
}

func TestReleaseProfileDefinitionRejectsMissingExtraAndUnknownGateKeys(t *testing.T) {
	gates := releaseGateDefinitions()
	base := releaseProfileDefinitions()[0]
	tests := []struct {
		name   string
		mutate func(map[string]GateDisposition)
	}{
		{
			name: "missing gate",
			mutate: func(dispositions map[string]GateDisposition) {
				delete(dispositions, "G9")
			},
		},
		{
			name: "extra gate",
			mutate: func(dispositions map[string]GateDisposition) {
				dispositions["G10"] = "required"
			},
		},
		{
			name: "unknown gate substitution",
			mutate: func(dispositions map[string]GateDisposition) {
				delete(dispositions, "G9")
				dispositions["G10"] = "required"
			},
		},
		{
			name: "unknown disposition",
			mutate: func(dispositions map[string]GateDisposition) {
				dispositions["G9"] = "waived"
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dispositions := make(map[string]GateDisposition, len(base.Dispositions)+1)
			for gateID, disposition := range base.Dispositions {
				dispositions[gateID] = disposition
			}
			test.mutate(dispositions)
			candidate := base
			candidate.Dispositions = dispositions
			if err := validateReleaseProfileDefinitions([]releaseProfileDefinition{candidate}, gates); err == nil {
				t.Fatal("invalid release profile definition was accepted")
			}
		})
	}
}
