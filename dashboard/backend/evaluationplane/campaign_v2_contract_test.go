package evaluationplane

import (
	"strings"
	"testing"
)

func campaignV2Request(profile ChangeProfile) CreateCampaignRequest {
	ids := []string{
		"11111111-1111-4111-8111-111111111111",
		"22222222-2222-4222-8222-222222222222",
		"33333333-3333-4333-8333-333333333333",
		"44444444-4444-4444-8444-444444444444",
		"55555555-5555-4555-8555-555555555555",
		"66666666-6666-4666-8666-666666666666",
		"77777777-7777-4777-8777-777777777777",
		"88888888-8888-4888-8888-888888888888",
		"99999999-9999-4999-8999-999999999999",
		"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
	}
	bindings := CampaignGateBindings{
		G2RunID:          ids[0],
		G3ControlledPair: &CampaignControlledPairBinding{BaselineRunID: ids[1], CandidateRunID: ids[2]},
		G4RunID:          ids[3],
		G5Fidelity:       &CampaignFidelityBinding{ReferenceRunID: ids[4], LiveRunID: ids[5]},
		G6RunID:          ids[6], G7RunID: ids[7], G8RunID: ids[8], G9RunID: ids[9],
	}
	definition, _ := campaignProfileContract(profile)
	for _, slot := range definition.CampaignSlots {
		if slot.Disposition == "not_applicable" {
			switch slot.GateID {
			case "G2":
				bindings.G2RunID = ""
			case "G3":
				bindings.G3ControlledPair = nil
			case "G4":
				bindings.G4RunID = ""
			case "G5":
				bindings.G5Fidelity = nil
			case "G6":
				bindings.G6RunID = ""
			case "G7":
				bindings.G7RunID = ""
			case "G8":
				bindings.G8RunID = ""
			case "G9":
				bindings.G9RunID = ""
			}
		}
	}
	return CreateCampaignRequest{
		ClientRequestID: "63f7b8f0-a839-40af-a2cf-e84800823948",
		Name:            "typed campaign", Description: "independent gate evidence",
		ChangeProfile: profile, GateBindings: bindings,
	}
}

func TestCampaignV2AcceptsTypedIndependentSlotsForCoreProfiles(t *testing.T) {
	for _, profile := range []ChangeProfile{"schema_adapter", "recipe", "model_pool"} {
		request := campaignV2Request(profile)
		if err := validateCampaignRequest(request); err != nil {
			t.Fatalf("profile %s rejected: %v bindings=%+v", profile, err, request.GateBindings)
		}
	}
}

func TestCampaignV2RejectsEveryMissingRequiredSlot(t *testing.T) {
	request := campaignV2Request("recipe")
	mutations := []struct {
		gate   string
		mutate func(*CampaignGateBindings)
	}{
		{"g2", func(value *CampaignGateBindings) { value.G2RunID = "" }},
		{"g3", func(value *CampaignGateBindings) { value.G3ControlledPair = nil }},
		{"g4", func(value *CampaignGateBindings) { value.G4RunID = "" }},
		{"g5", func(value *CampaignGateBindings) { value.G5Fidelity = nil }},
		{"g7", func(value *CampaignGateBindings) { value.G7RunID = "" }},
	}
	for _, test := range mutations {
		t.Run(test.gate, func(t *testing.T) {
			copy := request
			copy.GateBindings = request.GateBindings
			test.mutate(&copy.GateBindings)
			if err := validateCampaignRequest(copy); err == nil || !strings.Contains(err.Error(), test.gate) {
				t.Fatalf("missing %s error=%v", test.gate, err)
			}
		})
	}
}

func TestCampaignV2RejectsRunReuseAndSlotSubstitution(t *testing.T) {
	request := campaignV2Request("recipe")
	request.GateBindings.G4RunID = request.GateBindings.G2RunID
	if err := validateCampaignRequest(request); err == nil || !strings.Contains(err.Error(), "must be distinct") {
		t.Fatalf("G2-to-G4 substitution error=%v", err)
	}

	request = campaignV2Request("schema_adapter")
	request.GateBindings = CampaignGateBindings{G4RunID: request.ClientRequestID}
	if err := validateCampaignRequest(request); err == nil || !strings.Contains(err.Error(), "campaign and g4:evidence") {
		t.Fatalf("campaign/run identity reuse error=%v", err)
	}

	request = campaignV2Request("schema_adapter")
	request.GateBindings.G6RunID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
	if err := validateCampaignRequest(request); err == nil || !strings.Contains(err.Error(), "does not accept campaign slot g6") {
		t.Fatalf("not-applicable slot error=%v", err)
	}
}

func TestCampaignV2CatalogIsTheSlotSourceOfTruth(t *testing.T) {
	for _, profile := range builtinChangeProfiles() {
		if len(profile.CampaignSlots) != 8 {
			t.Fatalf("profile %s slots=%d", profile.ID, len(profile.CampaignSlots))
		}
		for index, slot := range profile.CampaignSlots {
			gateIndex := index + 2
			disposition, ok := releaseProfileDisposition(profile.ID, slot.GateID)
			if !ok || slot.GateID != requiredGateIDs[gateIndex] || slot.Disposition != disposition {
				t.Fatalf("profile %s slot=%+v canonical disposition=%q", profile.ID, slot, disposition)
			}
		}
	}
	g4, _ := campaignSlotContract("recipe", "G4")
	if g4.Mode != ModeLive || g4.MinimumEvidenceLevel != "E4" ||
		len(g4.AcceptedExecutorIDs) != 1 || g4.AcceptedExecutorIDs[0] != normalizedSuiteLiveExecutorID {
		t.Fatalf("G4 campaign contract=%+v", g4)
	}
	g5, _ := campaignSlotContract("recipe", "G5")
	if g5.Mode != ModeLive || g5.MinimumEvidenceLevel != "E5" ||
		len(g5.AcceptedExecutorIDs) != 1 || g5.AcceptedExecutorIDs[0] != liveRuntimeExecutorID {
		t.Fatalf("G5 campaign contract=%+v", g5)
	}
	agentG3, _ := campaignSlotContract("agent_multimodal", "G3")
	if agentG3.Disposition != "not_applicable" {
		t.Fatalf("agent_multimodal G3 campaign contract=%+v", agentG3)
	}
	agentG5, _ := campaignSlotContract("agent_multimodal", "G5")
	if agentG5.TrackID != "multimodal" || agentG5.Mode != ModeLive ||
		agentG5.MinimumEvidenceLevel != "E4" || len(agentG5.AcceptedExecutorIDs) != 1 ||
		agentG5.AcceptedExecutorIDs[0] != normalizedSuiteLiveExecutorID {
		t.Fatalf("agent_multimodal G5 campaign contract=%+v", agentG5)
	}
}

func campaignV2SingleRunEvidence(
	profile ChangeProfile,
	gateID string,
	runID string,
	subject string,
) campaignRunEvidence {
	slot, _ := campaignSlotContract(profile, gateID)
	executorID := slot.AcceptedExecutorIDs[0]
	mode := slot.Mode
	if mode == "" {
		mode = ModeLive
	}
	manifestDigest := digestBytes([]byte("manifest-" + runID))
	manifestArtifactDigest := digestBytes([]byte("manifest-artifact-" + runID))
	attestationDigest := ""
	var attestation *executionAttestation
	if mode == ModeLive {
		attestationDigest = digestBytes([]byte("attestation-" + runID))
		attestation = &executionAttestation{Digest: attestationDigest}
	}
	report := Report{
		SchemaVersion: SchemaVersion, AttestationRevision: ServerAttestationRevision,
		Run: Run{
			SchemaVersion: SchemaVersion, ID: runID, ClientRequestID: runID,
			Status: StatusCompleted, Mode: mode, EvidenceLevel: slot.MinimumEvidenceLevel,
			TrackEvidenceLevels: map[TrackID]EvidenceLevel{slot.TrackID: slot.MinimumEvidenceLevel},
			ChangeProfile:       profile, TrackIDs: []TrackID{slot.TrackID},
		},
		Tracks: []TrackReport{{TrackID: slot.TrackID, Status: "completed", EvidenceLevel: slot.MinimumEvidenceLevel}},
		Gates: []Gate{{
			ID: gateID, TrackID: slot.TrackID, Disposition: slot.Disposition,
			Verdict: "pass", ChangeProfile: profile, ContractVersion: GateContractVersion,
			EvidenceLevel: slot.MinimumEvidenceLevel, Observed: float64Reference(1),
			Threshold:   &GateThreshold{Operator: ">=", Value: 1, Unit: "boolean"},
			SampleCount: intReference(1), Rationale: "The slot-specific method receipt passed.",
		}},
	}
	manifest := RunManifest{
		SchemaVersion: SchemaVersion, ManifestDigest: manifestDigest, RunID: runID,
		Mode: mode, ChangeProfile: profile, SuiteIDs: []string{"suite"},
		SuiteExecutors: map[string]string{"suite": executorID}, TrackIDs: []TrackID{slot.TrackID},
	}
	return campaignRunEvidence{
		report: report, manifest: manifest, attestation: attestation,
		anchor: CampaignEvidenceAnchor{
			SlotID: strings.ToLower(gateID), GateID: gateID, BindingRole: campaignSingleBindingRole,
			RunID: runID, CandidateSubjectDigest: subject,
			ManifestSemanticDigest: manifestDigest, ManifestArtifactDigest: manifestArtifactDigest,
			ReportDigest:               digestBytes([]byte("report-" + runID)),
			PrivateReceiptDigest:       digestBytes([]byte("private-" + runID)),
			ExecutionAttestationDigest: attestationDigest,
		},
	}
}

func TestCampaignV2G3RejectsSyntheticOrForgedEvidence(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	binding := campaignEvidenceBinding{
		slotID: "g3", gateID: "G3", bindingRole: "candidate",
		runID: fixture.candidate.report.Run.ID, candidate: true,
	}

	fixture.candidate.report.Run.Mode = ModeReplay
	fixture.candidate.manifest.Mode = ModeReplay
	fixture.candidate.report.Run.EvidenceLevel = "E0"
	fixture.candidate.attestation = nil
	fixture.candidate.anchor.ExecutionAttestationDigest = ""
	if err := validateCampaignG3SourceTracks(binding, fixture.candidate); err == nil ||
		!strings.Contains(err.Error(), "paired-live") {
		t.Fatalf("synthetic G3 evidence error=%v", err)
	}

	fixture = withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	fixture.candidate.report.Run.EvidenceLevel = "E5"
	fixture.candidate.report.Tracks = nil
	if err := validateCampaignG3SourceTracks(binding, fixture.candidate); err == nil ||
		!strings.Contains(err.Error(), "lacks dense") {
		t.Fatalf("forged G3 level error=%v", err)
	}
}

func TestCampaignV2SingleRunSlotValidationFailsClosed(t *testing.T) {
	runID := "44444444-4444-4444-8444-444444444444"
	subject := digestBytes([]byte("candidate"))
	binding := campaignEvidenceBinding{
		slotID: "g4", gateID: "G4", bindingRole: campaignSingleBindingRole,
		runID: runID, candidate: true,
	}
	slot, _ := campaignSlotContract("schema_adapter", "G4")
	valid := campaignV2SingleRunEvidence("schema_adapter", "G4", runID, subject)
	if err := validateCampaignBoundRun("schema_adapter", binding, slot, valid); err != nil {
		t.Fatalf("valid G4 evidence rejected: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*campaignRunEvidence)
	}{
		{"wrong executor", func(value *campaignRunEvidence) { value.manifest.SuiteExecutors["suite"] = liveRuntimeExecutorID }},
		{"wrong track", func(value *campaignRunEvidence) { value.report.Tracks[0].TrackID = "joint" }},
		{"missing gate", func(value *campaignRunEvidence) { value.report.Gates = nil }},
		{"forged level", func(value *campaignRunEvidence) { value.report.Gates[0].EvidenceLevel = "E0" }},
		{"wrong profile", func(value *campaignRunEvidence) { value.report.Run.ChangeProfile = "recipe" }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := campaignV2SingleRunEvidence("schema_adapter", "G4", runID, subject)
			test.mutate(&candidate)
			if err := validateCampaignBoundRun("schema_adapter", binding, slot, candidate); err == nil {
				t.Fatalf("%s evidence was accepted", test.name)
			}
		})
	}
}

func TestCampaignV2RejectsCandidateSubjectDriftAcrossSlots(t *testing.T) {
	request := campaignV2Request("schema_adapter")
	request.GateBindings = CampaignGateBindings{
		G2RunID: "11111111-1111-4111-8111-111111111111",
		G4RunID: "44444444-4444-4444-8444-444444444444",
	}
	evidence := map[string]campaignRunEvidence{
		"g2:evidence": campaignV2SingleRunEvidence(
			"schema_adapter", "G2", request.GateBindings.G2RunID, digestBytes([]byte("candidate-a")),
		),
		"g4:evidence": campaignV2SingleRunEvidence(
			"schema_adapter", "G4", request.GateBindings.G4RunID, digestBytes([]byte("candidate-b")),
		),
	}
	if err := validateCampaignEvidenceSet("schema_adapter", request.GateBindings, evidence); err == nil ||
		!strings.Contains(err.Error(), "one exact subject") {
		t.Fatalf("candidate subject drift error=%v", err)
	}
}
