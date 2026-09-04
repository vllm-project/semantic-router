package evaluationplane

import (
	"testing"
	"time"
)

func TestCampaignV2BuildsStoredRecipeAndModelPoolDecisionsFromIndependentSlots(t *testing.T) {
	for _, profile := range []ChangeProfile{"recipe", "model_pool"} {
		t.Run(string(profile), func(t *testing.T) {
			campaign, evidence := campaignV2CompleteDecisionFixture(t, profile)
			if err := validateCampaignEvidenceSet(profile, campaign.GateBindings, evidence); err != nil {
				t.Fatalf("qualified typed evidence rejected: %v", err)
			}
			var err error
			campaign.Decision, err = buildCampaignDecision(campaign, evidence, campaign.CreatedAt)
			if err != nil {
				t.Fatalf("build campaign decision: %v", err)
			}
			if campaign.Decision.Verdict != "pass" || campaign.Decision.PairedLiveEvidence == nil ||
				campaign.Decision.PairedLiveEvidence.ContractVersion != CampaignPairedLiveContractVersion ||
				campaign.Decision.FidelityEvidence == nil || campaign.Decision.FidelityEvidence.Verdict != "pass" {
				t.Fatalf("decision=%+v", campaign.Decision)
			}
			if err := validateStoredCampaign(campaign.ID, campaign); err != nil {
				t.Fatalf("stored campaign rejected: %v", err)
			}
		})
	}
}

func campaignV2CompleteDecisionFixture(
	t *testing.T,
	profile ChangeProfile,
) (Campaign, map[string]campaignRunEvidence) {
	t.Helper()
	const (
		g2ID          = "11111111-1111-4111-8111-111111111111"
		g4ID          = "44444444-4444-4444-8444-444444444444"
		g5ReferenceID = "55555555-5555-4555-8555-555555555555"
		g5LiveID      = "66666666-6666-4666-8666-666666666666"
		g7ID          = "88888888-8888-4888-8888-888888888888"
		g8ID          = "99999999-9999-4999-8999-999999999999"
	)
	subject := digestBytes([]byte("complete-" + string(profile) + "-candidate"))
	pair := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	if profile == "model_pool" {
		pair = withChangedCandidatePool(pair)
	}
	prepareCampaignV2G3Run(&pair.baseline, profile, "")
	prepareCampaignV2G3Run(&pair.candidate, profile, subject)

	reference, live := campaignFidelityV2Fixture(t)
	reidentifyCampaignV2Run(&reference, g5ReferenceID)
	reidentifyCampaignV2Run(&live, g5LiveID)
	prepareCampaignV2G5Run(&reference, profile, subject)
	prepareCampaignV2G5Run(&live, profile, subject)

	bindings := CampaignGateBindings{
		G2RunID: g2ID,
		G3ControlledPair: &CampaignControlledPairBinding{
			BaselineRunID: pair.baseline.report.Run.ID, CandidateRunID: pair.candidate.report.Run.ID,
		},
		G4RunID: g4ID,
		G5Fidelity: &CampaignFidelityBinding{
			ReferenceRunID: g5ReferenceID, LiveRunID: g5LiveID,
		},
		G7RunID: g7ID,
	}
	evidence := map[string]campaignRunEvidence{
		"g2:evidence":  campaignV2SingleRunEvidence(profile, "G2", g2ID, subject),
		"g3:baseline":  pair.baseline,
		"g3:candidate": pair.candidate,
		"g4:evidence":  campaignV2SingleRunEvidence(profile, "G4", g4ID, subject),
		"g5:reference": reference,
		"g5:live":      live,
		"g7:evidence":  campaignV2SingleRunEvidence(profile, "G7", g7ID, subject),
	}
	if profile == "model_pool" {
		bindings.G8RunID = g8ID
		evidence["g8:evidence"] = campaignV2SingleRunEvidence(profile, "G8", g8ID, subject)
	}
	request := CreateCampaignRequest{
		ClientRequestID: "63f7b8f0-a839-40af-a2cf-e84800823948",
		Name:            "complete typed campaign", Description: "independent purpose-qualified slots",
		ChangeProfile: profile, GateBindings: bindings,
	}
	if err := validateCampaignRequest(request); err != nil {
		t.Fatalf("validate campaign request: %v", err)
	}
	now := time.Date(2026, time.August, 31, 9, 0, 0, 0, time.UTC)
	campaign := Campaign{
		SchemaVersion: SchemaVersion, ContractVersion: CampaignContractVersion,
		ID: request.ClientRequestID, Name: request.Name, Description: request.Description,
		ChangeProfile: profile, Status: CampaignStatusDecided,
		GateBindings: bindings, CreatedAt: now,
	}
	var err error
	campaign.ManifestDigest, err = campaignManifestDigest(campaign)
	if err != nil {
		t.Fatal(err)
	}
	return campaign, evidence
}

func prepareCampaignV2G3Run(
	evidence *campaignRunEvidence,
	profile ChangeProfile,
	subject string,
) {
	evidence.report.Run.ClientRequestID = evidence.report.Run.ID
	evidence.report.Run.ChangeProfile = profile
	evidence.report.Tracks = []TrackReport{
		{TrackID: "routing", Status: "completed", EvidenceLevel: "E3"},
		{TrackID: "model_pool", Status: "completed", EvidenceLevel: "E4"},
		{TrackID: "joint", Status: "completed", EvidenceLevel: "E5"},
	}
	evidence.report.Run.TrackIDs = []TrackID{"routing", "model_pool", "joint"}
	evidence.report.Run.TrackEvidenceLevels = map[TrackID]EvidenceLevel{
		"routing": "E3", "model_pool": "E4", "joint": "E5",
	}
	evidence.manifest.ChangeProfile = profile
	evidence.manifest.SuiteIDs = append([]string(nil), evidence.report.Run.SuiteIDs...)
	evidence.manifest.TrackIDs = append([]TrackID(nil), evidence.report.Run.TrackIDs...)
	evidence.manifest.SuiteExecutors = map[string]string{
		evidence.report.Run.SuiteIDs[0]: liveRuntimeExecutorID,
	}
	evidence.anchor.CandidateSubjectDigest = subject
}

func prepareCampaignV2G5Run(
	evidence *campaignRunEvidence,
	profile ChangeProfile,
	subject string,
) {
	evidence.report.Run.ClientRequestID = evidence.report.Run.ID
	evidence.report.Run.ChangeProfile = profile
	evidence.manifest.ChangeProfile = profile
	evidence.manifest.SuiteIDs = append([]string(nil), evidence.report.Run.SuiteIDs...)
	evidence.manifest.SuiteExecutors = map[string]string{
		evidence.report.Run.SuiteIDs[0]: liveRuntimeExecutorID,
	}
	evidence.anchor.CandidateSubjectDigest = subject
}

func reidentifyCampaignV2Run(evidence *campaignRunEvidence, runID string) {
	evidence.report.Run.ID = runID
	evidence.report.Run.ClientRequestID = runID
	evidence.manifest.RunID = runID
	evidence.anchor.RunID = runID
	if evidence.attestation != nil {
		evidence.attestation.RunID = runID
	}
}
