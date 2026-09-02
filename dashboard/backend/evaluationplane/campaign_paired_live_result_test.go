package evaluationplane

import "testing"

func TestCampaignPairedLiveEvidenceRejectsSemanticTamperEvenWithNewDigest(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	campaign := Campaign{ChangeProfile: "recipe", GateBindings: CampaignGateBindings{
		G3ControlledPair: &CampaignControlledPairBinding{
			BaselineRunID: fixture.baseline.report.Run.ID, CandidateRunID: fixture.candidate.report.Run.ID,
		},
	}}
	anchors := map[string]CampaignEvidenceAnchor{
		"g3:baseline": fixture.baseline.anchor, "g3:candidate": fixture.candidate.anchor,
	}
	if validationErr := validateCampaignPairedLiveEvidence(campaign, anchors, *evidence); validationErr != nil {
		t.Fatalf("valid evidence rejected: %v statistics=%+v", validationErr, evidence.Statistics)
	}
	*evidence.Statistics[0].Delta += 0.01
	evidence.Digest, err = campaignPairedLiveEvidenceDigest(*evidence)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateCampaignPairedLiveEvidence(campaign, anchors, *evidence); err == nil {
		t.Fatal("semantically tampered paired evidence with a recomputed digest was accepted")
	}
}

func TestCampaignPairedLiveEvidenceRejectsFrozenArmTamperEvenWithNewDigest(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	campaign := Campaign{
		ChangeProfile: "recipe",
		GateBindings: CampaignGateBindings{
			G3ControlledPair: &CampaignControlledPairBinding{
				BaselineRunID: fixture.baseline.report.Run.ID, CandidateRunID: fixture.candidate.report.Run.ID,
			},
		},
	}
	anchors := map[string]CampaignEvidenceAnchor{
		"g3:baseline": fixture.baseline.anchor, "g3:candidate": fixture.candidate.anchor,
	}
	*evidence.ModelPoolArmReliability[0].CandidateFailureRate = 0.5
	evidence.Digest, err = campaignPairedLiveEvidenceDigest(*evidence)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateCampaignPairedLiveEvidence(campaign, anchors, *evidence); err == nil {
		t.Fatal("semantically tampered frozen-arm reliability survived a recomputed evidence digest")
	}
}
