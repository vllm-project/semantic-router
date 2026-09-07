package evaluationplane

import "testing"

func TestCampaignPairedLiveReducerFailsRegressedQualityAndRisk(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, true))
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	g3 := campaignTestPairedGate("G3", *evidence, fixture)
	if g3.Verdict != "fail" || g3.Observed == nil || g3.Threshold == nil {
		t.Fatalf("G3=%+v", g3)
	}
	g8 := campaignTestPairedGate("G8", *evidence, fixture)
	if g8.Verdict != "unavailable" || g8.Observed != nil || g8.Threshold != nil {
		t.Fatalf("G8=%+v", g8)
	}
	failedDiagnostics := 0
	for _, statistic := range evidence.Statistics {
		if statistic.GateID == "G8" && statistic.Verdict == "fail" {
			failedDiagnostics++
		}
	}
	if failedDiagnostics == 0 {
		t.Fatal("regressed paired risk diagnostics were not retained")
	}
}

func TestCampaignPairedLiveReducerKeepsSmallCohortsUnavailable(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases-1, false))
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	for _, gateID := range []string{"G3", "G8"} {
		gate := campaignTestPairedGate(gateID, *evidence, fixture)
		if gate.Verdict != "unavailable" || gate.Observed != nil || gate.Threshold != nil {
			t.Fatalf("gate %s=%+v", gateID, gate)
		}
	}
}
