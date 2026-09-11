package evaluationplane

import (
	"strings"
	"testing"
	"time"
)

func campaignFidelityV2Fixture(t *testing.T) (campaignRunEvidence, campaignRunEvidence) {
	t.Helper()
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(minimumFidelityCases, false))
	reference, live := fixture.baseline, fixture.candidate
	subject := digestBytes([]byte("fidelity-candidate-subject"))
	reference.report.Run.EvidenceLevel = "E3"
	reference.report.Run.TrackIDs = []TrackID{"routing", "joint"}
	reference.report.Run.TrackEvidenceLevels = map[TrackID]EvidenceLevel{"routing": "E3", "joint": "E4"}
	reference.manifest.TrackIDs = append([]TrackID(nil), reference.report.Run.TrackIDs...)
	reference.report.Tracks = []TrackReport{
		{TrackID: "routing", Status: "completed", EvidenceLevel: "E3"},
		{TrackID: "joint", Status: "completed", EvidenceLevel: "E4"},
	}
	reference.anchor = CampaignEvidenceAnchor{
		SlotID: "g5", GateID: "G5", BindingRole: "reference", RunID: reference.report.Run.ID,
		CandidateSubjectDigest: subject,
		ManifestSemanticDigest: reference.anchor.ManifestSemanticDigest,
		ManifestArtifactDigest: reference.anchor.ManifestArtifactDigest,
		ReportDigest:           reference.anchor.ReportDigest, PrivateReceiptDigest: reference.anchor.PrivateReceiptDigest,
		ExecutionAttestationDigest: reference.attestation.Digest,
	}
	completedAt := time.Date(2026, time.August, 30, 2, 0, 0, 0, time.UTC)
	reference.report.Run.CompletedAt = &completedAt
	live.report.Run.EvidenceLevel = "E3"
	live.report.Run.TrackIDs = []TrackID{"routing", "joint"}
	live.report.Run.TrackEvidenceLevels = map[TrackID]EvidenceLevel{"routing": "E3", "joint": "E5"}
	live.manifest.TrackIDs = append([]TrackID(nil), live.report.Run.TrackIDs...)
	live.report.Tracks = []TrackReport{
		{TrackID: "routing", Status: "completed", EvidenceLevel: "E3"},
		{TrackID: "joint", Status: "completed", EvidenceLevel: "E5"},
	}
	live.anchor.SlotID, live.anchor.GateID, live.anchor.BindingRole = "g5", "G5", "live"
	live.anchor.CandidateSubjectDigest = subject
	live.attestation.StartedAt = completedAt.Add(time.Minute)
	liveCompletedAt := completedAt.Add(2 * time.Minute)
	live.report.Run.CompletedAt = &liveCompletedAt
	live.attestation.CompletedAt = liveCompletedAt
	for index := range live.records {
		left, right := reference.records[index], &live.records[index]
		right.Status, right.Success = left.Status, left.Success
		right.SelectedArmID, right.ArmID, right.Quality = left.SelectedArmID, left.ArmID, left.Quality
		for entryIndex := range live.attestation.Entries {
			entry := &live.attestation.Entries[entryIndex]
			if right.BrokerReceipt != nil && entry.BrokerReceipt == *right.BrokerReceipt {
				entry.Success, entry.Quality = *right.Success, right.Quality
				entry.ArmID = right.SelectedArmID
				if right.TrackID == "model_pool" {
					entry.ArmID = right.ArmID
				}
				break
			}
		}
	}
	return reference, live
}

func TestCampaignV2FidelityAcceptsQualifiedReferenceToFreshLive(t *testing.T) {
	reference, live := campaignFidelityV2Fixture(t)
	for _, item := range []struct {
		role     string
		evidence campaignRunEvidence
	}{
		{role: "reference", evidence: reference},
		{role: "live", evidence: live},
	} {
		binding := campaignEvidenceBinding{slotID: "g5", gateID: "G5", bindingRole: item.role}
		if err := validateCampaignG5SourceLevel(binding, item.evidence); err != nil {
			t.Fatalf("%s track-qualified evidence rejected from weaker run headline: %v", item.role, err)
		}
	}
	evidence, err := buildCampaignFidelityEvidence(reference, live)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Verdict != "pass" || evidence.MatchedCases != minimumFidelityCases ||
		evidence.TrackID != "joint" ||
		evidence.DecisionMismatches != 0 || evidence.OutcomeMismatches != 0 ||
		evidence.UnavailableCases != 0 || !digestPattern.MatchString(evidence.Digest) {
		t.Fatalf("fidelity evidence=%+v", evidence)
	}
}

func TestCampaignV2FidelityRejectsWeakJointTrackDespiteStrongRunHeadline(t *testing.T) {
	_, live := campaignFidelityV2Fixture(t)
	live.report.Run.EvidenceLevel = "E5"
	live.report.Tracks[1].EvidenceLevel = "E4"
	binding := campaignEvidenceBinding{slotID: "g5", gateID: "G5", bindingRole: "live"}
	if err := validateCampaignG5SourceLevel(binding, live); err == nil ||
		!strings.Contains(err.Error(), "joint track evidence at E5") {
		t.Fatalf("weak joint track error=%v", err)
	}
}

func TestCampaignV2FidelityRejectsStaleLiveAndCaseDrift(t *testing.T) {
	reference, live := campaignFidelityV2Fixture(t)
	live.attestation.StartedAt = reference.report.Run.CompletedAt.Add(-time.Second)
	if err := validateCampaignFidelitySources(reference, live); err == nil || !strings.Contains(err.Error(), "exact candidate suite/workload/case cohort") {
		t.Fatalf("stale live error=%v", err)
	}

	reference, live = campaignFidelityV2Fixture(t)
	for index, record := range live.records {
		if record.TrackID != "joint" || record.BrokerReceipt == nil {
			continue
		}
		receipt := *record.BrokerReceipt
		live.records = append(live.records[:index], live.records[index+1:]...)
		for entryIndex, entry := range live.attestation.Entries {
			if entry.BrokerReceipt == receipt {
				live.attestation.Entries = append(
					live.attestation.Entries[:entryIndex],
					live.attestation.Entries[entryIndex+1:]...,
				)
				break
			}
		}
		break
	}
	if err := validateCampaignFidelitySources(reference, live); err == nil || !strings.Contains(err.Error(), "same case cohort") {
		t.Fatalf("case drift error=%v", err)
	}
}

func TestCampaignV2FidelityRejectsReplayReferenceAndSubjectDrift(t *testing.T) {
	reference, live := campaignFidelityV2Fixture(t)
	reference.report.Run.Mode = ModeReplay
	reference.manifest.Mode = ModeReplay
	reference.attestation = nil
	reference.anchor.ExecutionAttestationDigest = ""
	binding := campaignEvidenceBinding{slotID: "g5", gateID: "G5", bindingRole: "reference", runID: reference.report.Run.ID, candidate: true}
	if err := validateCampaignG5SourceLevel(binding, reference); err == nil || !strings.Contains(err.Error(), "attested live") {
		t.Fatalf("replay reference error=%v", err)
	}

	reference, live = campaignFidelityV2Fixture(t)
	live.anchor.CandidateSubjectDigest = digestBytes([]byte("different-subject"))
	if err := validateCampaignFidelitySources(reference, live); err == nil {
		t.Fatal("cross-subject fidelity pair was accepted")
	}
}

func TestCampaignV2FidelityRequiresFreshLiveAfterQualifiedLiveReference(t *testing.T) {
	reference, live := campaignFidelityV2Fixture(t)
	if err := validateCampaignFidelitySources(reference, live); err != nil {
		t.Fatalf("fresh live/live fidelity rejected: %v", err)
	}
	live.attestation.StartedAt = reference.attestation.CompletedAt
	if err := validateCampaignFidelitySources(reference, live); err == nil {
		t.Fatal("non-fresh live/live fidelity was accepted")
	}
}

func TestCampaignV2AgentMultimodalFidelityUsesExactMMRTrackAtE4(t *testing.T) {
	reference, live := campaignAgentMultimodalFidelityFixture(t)
	for _, item := range []struct {
		role     string
		evidence campaignRunEvidence
	}{
		{role: "reference", evidence: reference},
		{role: "live", evidence: live},
	} {
		binding := campaignEvidenceBinding{slotID: "g5", gateID: "G5", bindingRole: item.role}
		if err := validateCampaignG5SourceLevel(binding, item.evidence); err != nil {
			t.Fatalf("%s multimodal E4 source rejected: %v", item.role, err)
		}
	}
	evidence, err := buildCampaignFidelityEvidence(reference, live)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.TrackID != "multimodal" || evidence.Verdict != "pass" ||
		evidence.MatchedCases != minimumFidelityCases {
		t.Fatalf("multimodal fidelity evidence=%+v", evidence)
	}
}

func TestCampaignV2FidelityDoesNotCountMatchedFailuresAsAgreement(t *testing.T) {
	reference, live := campaignFidelityV2Fixture(t)
	for index := range reference.records {
		if reference.records[index].TrackID != "joint" {
			continue
		}
		reference.records[index].Status = "failed"
		live.records[index].Status = "failed"
		*reference.records[index].Success = false
		*live.records[index].Success = false
		for _, evidence := range []*campaignRunEvidence{&reference, &live} {
			receipt := *evidence.records[index].BrokerReceipt
			for entryIndex := range evidence.attestation.Entries {
				if evidence.attestation.Entries[entryIndex].BrokerReceipt == receipt {
					evidence.attestation.Entries[entryIndex].Success = false
					break
				}
			}
		}
		break
	}
	evidence, err := buildCampaignFidelityEvidence(reference, live)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Verdict != "fail" || evidence.UnavailableCases != 1 ||
		evidence.MatchedCases != minimumFidelityCases-1 {
		t.Fatalf("failed-pair fidelity evidence=%+v", evidence)
	}
}

func campaignAgentMultimodalFidelityFixture(t *testing.T) (campaignRunEvidence, campaignRunEvidence) {
	t.Helper()
	reference, live := campaignFidelityV2Fixture(t)
	for _, evidence := range []*campaignRunEvidence{&reference, &live} {
		evidence.report.Run.ClientRequestID = evidence.report.Run.ID
		evidence.report.Run.ChangeProfile = "agent_multimodal"
		evidence.report.Run.EvidenceLevel = "E4"
		evidence.report.Run.TrackIDs = []TrackID{"multimodal"}
		evidence.report.Run.TrackEvidenceLevels = map[TrackID]EvidenceLevel{"multimodal": "E4"}
		evidence.report.Tracks = []TrackReport{{TrackID: "multimodal", Status: "completed", EvidenceLevel: "E4"}}
		evidence.report.Provenance.BenchmarkRevisions = map[string]string{
			"installed-multimodal": digestBytes([]byte("installed-multimodal-revision")),
		}
		evidence.manifest.ChangeProfile = "agent_multimodal"
		evidence.manifest.SuiteIDs = []string{"installed-multimodal"}
		evidence.manifest.TrackIDs = []TrackID{"multimodal"}
		evidence.manifest.SuiteExecutors = map[string]string{
			"installed-multimodal": normalizedSuiteLiveExecutorID,
		}
		evidence.report.Run.SuiteIDs = append([]string(nil), evidence.manifest.SuiteIDs...)

		records := make([]executionRecordEvidence, 0, minimumFidelityCases)
		entries := make([]executionAttestationEntry, 0, minimumFidelityCases)
		for _, record := range evidence.records {
			if record.TrackID != "joint" || record.BrokerReceipt == nil {
				continue
			}
			record.TrackID = "multimodal"
			modality := "image"
			record.Modality = &modality
			records = append(records, record)
			for _, entry := range evidence.attestation.Entries {
				if entry.BrokerReceipt == *record.BrokerReceipt {
					entry.TrackID = "multimodal"
					entries = append(entries, entry)
					break
				}
			}
		}
		evidence.records = records
		evidence.attestation.Entries = entries
	}
	return reference, live
}
