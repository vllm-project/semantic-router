package evaluationplane

import "testing"

func TestCampaignPairedLiveReducerRejectsMissingTamperedAndMisalignedEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*campaignPairedLiveFixture)
	}{
		{name: "missing attestation", mutate: func(fixture *campaignPairedLiveFixture) {
			fixture.candidate.attestation = nil
		}},
		{name: "tampered hidden grade", mutate: func(fixture *campaignPairedLiveFixture) {
			fixture.candidate.records[0].Quality = float64Reference(0.5)
		}},
		{name: "misaligned analysis identity", mutate: func(fixture *campaignPairedLiveFixture) {
			fixture.candidate.records[0].AttemptID = "attempt-other-case"
		}},
		{name: "missing attested operation", mutate: func(fixture *campaignPairedLiveFixture) {
			fixture.candidate.attestation.Entries = fixture.candidate.attestation.Entries[1:]
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fixture := newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)
			test.mutate(&fixture)
			if _, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate); err == nil {
				t.Fatal("invalid paired live evidence was accepted")
			}
		})
	}
}

func TestCampaignPairedLiveRejectsMoMRecordThatDiffersFromServerArmAttestation(t *testing.T) {
	for _, trackID := range []TrackID{"model_pool", "joint"} {
		t.Run(string(trackID), func(t *testing.T) {
			fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
			for index := range fixture.candidate.records {
				record := &fixture.candidate.records[index]
				if record.TrackID != trackID {
					continue
				}
				wrong := "arm-outside-attestation"
				if trackID == "model_pool" {
					record.ArmID = &wrong
				} else {
					record.SelectedArmID = &wrong
				}
				break
			}
			if _, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate); err == nil {
				t.Fatal("record-side arm tamper was accepted over the server attestation")
			}
		})
	}
}
