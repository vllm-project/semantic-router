package evaluationplane

import "fmt"

// releaseGateDefinition is the server-owned definition of one release gate.
// Report validation and the campaign catalog are projections of this contract;
// neither layer keeps a second handwritten gate inventory.
type releaseGateDefinition struct {
	ID            string
	Name          string
	CampaignName  string
	Description   string
	TrackID       TrackID
	EvidenceLevel EvidenceLevel
	Owner         string
	EvidenceRefs  []string
	Campaign      *releaseGateCampaignDefinition
}

type releaseGateCampaignDefinition struct {
	Description          string
	BindingKind          CampaignBindingKind
	TrackID              TrackID
	Mode                 Mode
	MinimumEvidenceLevel EvidenceLevel
	AcceptedExecutorIDs  []string
}

type releaseProfileDefinition struct {
	ID                ChangeProfile
	Name              string
	Description       string
	Dispositions      map[string]GateDisposition
	CampaignOverrides map[string]releaseGateCampaignDefinition
}

func foundationalReleaseGateDefinitions() []releaseGateDefinition {
	return []releaseGateDefinition{
		{
			ID: "G0", Name: "Reproducibility",
			EvidenceLevel: "E0", Owner: "evaluation-platform",
			Description:  "Verifies that the evaluation can be traced to pinned inputs, settings, and outputs.",
			EvidenceRefs: []string{manifestFileName, "lineage.json", "provenance.json", publicChecksumArtifactName},
		},
		{
			ID: "G1", Name: "Static correctness",
			EvidenceLevel: "E0", Owner: "evaluation-platform",
			Description:  "Verifies that the saved evaluation bundle is complete, valid, and internally consistent.",
			EvidenceRefs: []string{manifestFileName, "records.jsonl"},
		},
	}
}

func releaseGateDefinitions() []releaseGateDefinition {
	return append(foundationalReleaseGateDefinitions(), []releaseGateDefinition{
		{
			ID: "G2", Name: "Hard policy", CampaignName: "Policy enforcement",
			EvidenceLevel: "E3", Owner: "router-policy",
			Description:  "Checks that required safety and routing policies are enforced on the proposed system.",
			TrackID:      "safety",
			EvidenceRefs: []string{"records.jsonl", "metric:safety.violation_rate", "metric:safety.block_accuracy", "method:evaluation-hard-policy-proof.v1"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingRun, TrackID: "safety", Mode: ModeLive,
				MinimumEvidenceLevel: "E3", AcceptedExecutorIDs: []string{liveRuntimeExecutorID},
			},
		},
		{
			ID: "G3", Name: "Offline value", CampaignName: "Controlled value comparison",
			EvidenceLevel: "E4", Owner: "recipe-and-model-pool",
			Description:  "Compares baseline and candidate outcomes on the same live cases with balanced execution order.",
			TrackID:      "joint",
			EvidenceRefs: []string{"metrics.json", "metric:joint.normalized_regret"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingControlledPair, TrackID: "joint", Mode: ModeLive,
				MinimumEvidenceLevel: "E4", AcceptedExecutorIDs: []string{liveRuntimeExecutorID},
			},
		},
		{
			ID: "G4", Name: "Declared-shift robustness", CampaignName: "Workload-shift robustness",
			EvidenceLevel: "E4", Owner: "evaluation-workload",
			Description:  "Measures quality and reliability under the workload changes declared for this release.",
			TrackID:      "routing",
			EvidenceRefs: []string{"records.jsonl", "metric:routing.robustness_pass_rate", "metric:routing.robustness_worst_slice_pass_rate"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingRun, TrackID: "routing", Mode: ModeLive,
				MinimumEvidenceLevel: "E4", AcceptedExecutorIDs: []string{normalizedSuiteLiveExecutorID},
			},
		},
		{
			ID: "G5", Name: "Live fidelity", CampaignName: "Live consistency",
			EvidenceLevel: "E5", Owner: "router-and-serving-runtime",
			Description:  "Checks that a fresh live run agrees with the saved candidate on the same evaluation cases.",
			TrackID:      "joint",
			EvidenceRefs: []string{"records.jsonl", "provenance.json"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingFidelityPair, TrackID: "joint", Mode: ModeLive,
				MinimumEvidenceLevel: "E5", AcceptedExecutorIDs: []string{liveRuntimeExecutorID},
			},
		},
		{
			ID: "G6", Name: "Live fault-recovery continuity", CampaignName: "Fault recovery",
			EvidenceLevel: "E5", Owner: "agent-runtime",
			Description:  "Measures fallback, retry, state continuity, and side effects during injected failures.",
			TrackID:      "agentic",
			EvidenceRefs: []string{"records.jsonl", "metric:agentic.recovery_cluster_pass_rate_lower_95"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingRun, TrackID: "agentic", Mode: ModeLive,
				MinimumEvidenceLevel: "E5", AcceptedExecutorIDs: []string{liveRuntimeExecutorID},
			},
		},
		{
			ID: "G7", Name: "Cost / latency / capacity", CampaignName: "Cost, latency, and capacity",
			EvidenceLevel: "E5", Owner: "serving-capacity",
			Description:  "Measures whether the proposed system meets its service objectives under repeated load.",
			TrackID:      "capacity",
			EvidenceRefs: []string{manifestFileName, capacityProfileArtifactName, "metric:capacity.error_rate_upper_bound", "metric:capacity.error_rate_cluster_range_max", "metric:capacity.measurement_cluster_count_min", "metric:capacity.slo_headroom"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingRun, TrackID: "capacity", Mode: ModeLive,
				MinimumEvidenceLevel: "E5", AcceptedExecutorIDs: []string{liveRuntimeExecutorID},
			},
		},
		{
			ID: "G8", Name: "Shadow / canary", CampaignName: "Canary safety",
			EvidenceLevel: "E5", Owner: "release-operations",
			Description:  "Checks production assignment, exposure limits, stop conditions, and rollback controls.",
			TrackID:      "preference",
			EvidenceRefs: []string{manifestFileName, "records.jsonl", "metric:experiment.assignment_balance_p_value", "metric:experiment.risk_event_upper_confidence_bound", "metric:experiment.risk_budget_max_rate", "metric:experiment.candidate_safe"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingRun, TrackID: "preference", Mode: ModeLive,
				MinimumEvidenceLevel: "E5", AcceptedExecutorIDs: []string{liveRuntimeExecutorID},
			},
		},
		{
			ID: "G9", Name: "Online preference", CampaignName: "Online preference",
			EvidenceLevel: "E5", Owner: "online-learning",
			Description:  "Measures assigned user-preference outcomes for the baseline and proposed system.",
			TrackID:      "preference",
			EvidenceRefs: []string{"records.jsonl", "metric:preference.online_reward_lift", "metric:preference.online_effective_sample_size", "metric:preference.online_segment_coverage"},
			Campaign: &releaseGateCampaignDefinition{
				BindingKind: CampaignBindingRun, TrackID: "preference", Mode: ModeLive,
				MinimumEvidenceLevel: "E5", AcceptedExecutorIDs: []string{liveRuntimeExecutorID},
			},
		},
	}...)
}

func releaseProfileDefinitions() []releaseProfileDefinition {
	return []releaseProfileDefinition{
		{
			ID: "schema_adapter", Name: "API and integration",
			Description: "Request or response formats, provider integrations, and adapter behavior changes.",
			Dispositions: map[string]GateDisposition{
				"G0": "required", "G1": "required", "G2": "advisory", "G3": "advisory", "G4": "required",
				"G5": "advisory", "G6": "not_applicable", "G7": "advisory", "G8": "not_applicable", "G9": "not_applicable",
			},
		},
		{
			ID: "recipe", Name: "Routing recipe",
			Description: "Routing rules, signals, decision logic, or selection policy changes.",
			Dispositions: map[string]GateDisposition{
				"G0": "required", "G1": "required", "G2": "required", "G3": "required", "G4": "required",
				"G5": "required", "G6": "not_applicable", "G7": "required", "G8": "advisory", "G9": "not_applicable",
			},
		},
		{
			ID: "selector", Name: "Model selection",
			Description: "Model scoring, prediction, classification, or model-binding changes.",
			Dispositions: map[string]GateDisposition{
				"G0": "required", "G1": "required", "G2": "required", "G3": "required", "G4": "required",
				"G5": "required", "G6": "advisory", "G7": "required", "G8": "required", "G9": "not_applicable",
			},
		},
		{
			ID: "model_pool", Name: "Model pool",
			Description: "Available models, their capabilities, quality, reliability, or pricing changes.",
			Dispositions: map[string]GateDisposition{
				"G0": "required", "G1": "required", "G2": "required", "G3": "required", "G4": "required",
				"G5": "required", "G6": "advisory", "G7": "required", "G8": "required", "G9": "not_applicable",
			},
		},
		{
			ID: "runtime_capacity", Name: "Runtime and capacity",
			Description: "Serving software, deployment placement, throughput, latency, or transport changes.",
			Dispositions: map[string]GateDisposition{
				"G0": "required", "G1": "required", "G2": "required", "G3": "advisory", "G4": "advisory",
				"G5": "required", "G6": "advisory", "G7": "required", "G8": "required", "G9": "not_applicable",
			},
		},
		{
			ID: "agent_multimodal", Name: "Agents and multimodal",
			Description: "Diagnostic-only checks for tool use, multi-step agent behavior, state handling, and multimodal inputs; this profile does not provide release-promotion evidence.",
			Dispositions: map[string]GateDisposition{
				"G0": "required", "G1": "required", "G2": "required", "G3": "not_applicable", "G4": "required",
				"G5": "required", "G6": "required", "G7": "required", "G8": "required", "G9": "advisory",
			},
			CampaignOverrides: map[string]releaseGateCampaignDefinition{
				"G5": {
					Description: "Checks that a fresh multimodal run agrees with the saved candidate on the same evaluation cases.",
					BindingKind: CampaignBindingFidelityPair, TrackID: "multimodal", Mode: ModeLive,
					MinimumEvidenceLevel: "E4", AcceptedExecutorIDs: []string{normalizedSuiteLiveExecutorID},
				},
			},
		},
		{
			ID: "online_adaptation", Name: "Online learning and feedback",
			Description: "Traffic assignment, user preferences, feedback, or adaptive policy changes.",
			Dispositions: map[string]GateDisposition{
				"G0": "required", "G1": "required", "G2": "required", "G3": "required", "G4": "required",
				"G5": "required", "G6": "required", "G7": "required", "G8": "required", "G9": "required",
			},
		},
	}
}

func canonicalReleaseGateIDs() []string {
	definitions := releaseGateDefinitions()
	ids := make([]string, 0, len(definitions))
	for _, gate := range definitions {
		ids = append(ids, gate.ID)
	}
	return ids
}

func releaseGateDefinitionByID(id string) (releaseGateDefinition, bool) {
	for _, gate := range releaseGateDefinitions() {
		if gate.ID == id {
			return gate, true
		}
	}
	return releaseGateDefinition{}, false
}

func releaseProfileDefinitionByID(id ChangeProfile) (releaseProfileDefinition, bool) {
	for _, profile := range releaseProfileDefinitions() {
		if profile.ID == id {
			return profile, true
		}
	}
	return releaseProfileDefinition{}, false
}

func releaseProfileDisposition(profileID ChangeProfile, gateID string) (GateDisposition, bool) {
	profile, ok := releaseProfileDefinitionByID(profileID)
	if !ok {
		return "", false
	}
	disposition, ok := profile.Dispositions[gateID]
	return disposition, ok
}

func validateReleaseProfileDefinitions(
	profiles []releaseProfileDefinition,
	gates []releaseGateDefinition,
) error {
	if len(profiles) == 0 || len(gates) == 0 {
		return fmt.Errorf("release profile or gate definitions are empty")
	}
	gateIDs := make(map[string]releaseGateDefinition, len(gates))
	for index, gate := range gates {
		if gate.ID == "" || gate.Name == "" || gate.Description == "" || gate.Owner == "" ||
			evidenceLevelRank(gate.EvidenceLevel) < 0 || len(gate.EvidenceRefs) == 0 {
			return fmt.Errorf("release gate %q is incomplete", gate.ID)
		}
		if gate.ID != fmt.Sprintf("G%d", index) {
			return fmt.Errorf("release gate %q is out of canonical order", gate.ID)
		}
		if _, duplicate := gateIDs[gate.ID]; duplicate {
			return fmt.Errorf("release gate %q is duplicated", gate.ID)
		}
		gateIDs[gate.ID] = gate
	}
	profileIDs := make(map[ChangeProfile]struct{}, len(profiles))
	for _, profile := range profiles {
		if profile.ID == "" || profile.Name == "" || profile.Description == "" ||
			len(profile.Dispositions) != len(gateIDs) {
			return fmt.Errorf("release profile %q is incomplete", profile.ID)
		}
		if _, duplicate := profileIDs[profile.ID]; duplicate {
			return fmt.Errorf("release profile %q is duplicated", profile.ID)
		}
		profileIDs[profile.ID] = struct{}{}
		for gateID, disposition := range profile.Dispositions {
			if _, exists := gateIDs[gateID]; !exists {
				return fmt.Errorf("release profile %q contains unknown gate %q", profile.ID, gateID)
			}
			if !validGateDisposition(disposition) {
				return fmt.Errorf("release profile %q gate %q has invalid disposition", profile.ID, gateID)
			}
		}
		for gateID, override := range profile.CampaignOverrides {
			gate, exists := gateIDs[gateID]
			if !exists || gate.Campaign == nil || override.BindingKind == "" {
				return fmt.Errorf("release profile %q contains invalid campaign override %q", profile.ID, gateID)
			}
		}
	}
	return nil
}

func canonicalCampaignGateDefinitions() []releaseGateDefinition {
	definitions := releaseGateDefinitions()
	campaignGates := make([]releaseGateDefinition, 0, len(definitions))
	for _, gate := range definitions {
		if gate.Campaign != nil {
			campaignGates = append(campaignGates, gate)
		}
	}
	return campaignGates
}

func builtinChangeProfiles() []CatalogChangeProfile {
	campaignGates := canonicalCampaignGateDefinitions()
	profiles := releaseProfileDefinitions()
	result := make([]CatalogChangeProfile, 0, len(profiles))
	for _, profile := range profiles {
		slots := make([]CatalogCampaignSlot, 0, len(campaignGates))
		for _, gate := range campaignGates {
			campaign := *gate.Campaign
			description := gate.Description
			if override, ok := profile.CampaignOverrides[gate.ID]; ok {
				campaign = override
				if override.Description != "" {
					description = override.Description
				}
			}
			slots = append(slots, CatalogCampaignSlot{
				GateID: gate.ID, Name: gate.CampaignName, Description: description,
				Disposition: profile.Dispositions[gate.ID], BindingKind: campaign.BindingKind,
				TrackID: campaign.TrackID, Mode: campaign.Mode,
				MinimumEvidenceLevel: campaign.MinimumEvidenceLevel,
				AcceptedExecutorIDs:  append([]string(nil), campaign.AcceptedExecutorIDs...),
			})
		}
		result = append(result, CatalogChangeProfile{
			ID: profile.ID, Name: profile.Name, Description: profile.Description, CampaignSlots: slots,
		})
	}
	return result
}
