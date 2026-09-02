package evaluationplane

import (
	"strings"
	"testing"
)

func TestCampaignPairedLiveReducerPassesServerAttestedNonInferiorityAndRisk(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	if len(evidence.Statistics) != 10 || len(evidence.PromotionStatistics) != 5 || !digestPattern.MatchString(evidence.Digest) {
		t.Fatalf("paired evidence=%+v", evidence)
	}
	for _, statistic := range evidence.Statistics {
		if statistic.Verdict != "pass" || statistic.SampleCount != campaignPairedMinimumCases || len(statistic.ConfidenceInterval) != 2 {
			t.Fatalf("statistic=%+v", statistic)
		}
	}
	g3 := campaignTestPairedGate("G3", *evidence, fixture)
	if g3.Verdict != "pass" || g3.Observed == nil || g3.Threshold == nil || g3.SampleCount != campaignPairedMinimumCases ||
		g3.Source != "server_attested_paired_live" {
		t.Fatalf("G3=%+v", g3)
	}
	g8 := campaignTestPairedGate("G8", *evidence, fixture)
	if g8.Verdict != "unavailable" || g8.Observed != nil || g8.Threshold != nil || g8.SampleCount != campaignPairedMinimumCases ||
		g8.Source != "server_attested_paired_live_diagnostic" {
		t.Fatalf("G8=%+v", g8)
	}
}

func TestCampaignPairedLiveReducerCoversMoMRecipePoolAndJointLayers(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	if len(evidence.Statistics) != 10 || len(evidence.ModelPoolArmReliability) != 2 {
		t.Fatalf("statistics=%+v", evidence.Statistics)
	}
	expectedUnits := map[string]string{
		"campaign.g3.routing.quality_non_inferiority":                  campaignQualityUnit,
		"campaign.g3.model_pool.quality_non_inferiority":               campaignPoolQualityUnit,
		"campaign.g3.model_pool.worst_arm_reliability_non_inferiority": campaignPoolWorstArmReliabilityUnit,
		"campaign.g3.joint.quality_non_inferiority":                    campaignQualityUnit,
		"campaign.g8.model_pool.failure_risk":                          campaignPoolFailureUnit,
	}
	for _, statistic := range evidence.Statistics {
		if statistic.SampleCount != campaignPairedMinimumCases || statistic.Verdict != "pass" {
			t.Fatalf("statistic=%+v", statistic)
		}
		if unit, present := expectedUnits[statistic.ID]; present && statistic.AnalysisUnit != unit {
			t.Fatalf("statistic %s unit=%q, want %q", statistic.ID, statistic.AnalysisUnit, unit)
		}
	}
	for _, statistic := range evidence.ModelPoolArmReliability {
		if statistic.Cohort != campaignArmCohortPaired || statistic.BaselineFailureRate == nil ||
			statistic.CandidateFailureRate == nil || *statistic.BaselineFailureRate != 0 ||
			*statistic.CandidateFailureRate != 0 || statistic.Verdict != "pass" ||
			statistic.BaselineSampleCount != campaignPairedMinimumCases ||
			statistic.CandidateSampleCount != campaignPairedMinimumCases {
			t.Fatalf("arm reliability=%+v", statistic)
		}
	}
	g3 := campaignTestPairedGate("G3", *evidence, fixture)
	if g3.Verdict != "pass" || g3.SampleCount != campaignPairedMinimumCases {
		t.Fatalf("G3=%+v", g3)
	}
}

func TestCampaignPairedLiveModelPoolReliabilityBlocksSingleArmRegression(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	failed := false
	for index := range fixture.candidate.records {
		record := &fixture.candidate.records[index]
		if record.TrackID == "model_pool" && stringValue(record.ArmID) == "arm-wrong" {
			record.Status = "failed"
			record.Success = &failed
			record.Quality = nil
		}
	}
	for index := range fixture.candidate.attestation.Entries {
		entry := &fixture.candidate.attestation.Entries[index]
		if entry.TrackID == "model_pool" && stringValue(entry.ArmID) == "arm-wrong" {
			entry.Success = false
			entry.Quality = nil
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	var availability, worst *CampaignPairedStatistic
	for index := range evidence.Statistics {
		statistic := &evidence.Statistics[index]
		switch statistic.AnalysisUnit {
		case campaignPoolFailureUnit:
			availability = statistic
		case campaignPoolWorstArmReliabilityUnit:
			worst = statistic
		}
	}
	if availability == nil || availability.CandidateValue == nil || *availability.CandidateValue != 0 ||
		availability.Verdict != "pass" {
		t.Fatalf("pool availability=%+v", availability)
	}
	if worst == nil || worst.BaselineValue == nil || *worst.BaselineValue != 1 ||
		worst.CandidateValue == nil || *worst.CandidateValue != 0 || worst.Verdict != "fail" {
		t.Fatalf("worst-arm reliability=%+v", worst)
	}
	armFailed := false
	for _, statistic := range evidence.ModelPoolArmReliability {
		if statistic.ArmID == "arm-wrong" && statistic.CandidateFailureRate != nil &&
			*statistic.CandidateFailureRate == 1 && statistic.Verdict == "fail" {
			armFailed = true
		}
	}
	if !armFailed || campaignTestPairedGate("G3", *evidence, fixture).Verdict != "fail" {
		t.Fatalf("single-arm regression did not block promotion: %+v", evidence.ModelPoolArmReliability)
	}
}

func TestCampaignPairedLiveG3BothWorstArmsAtZeroCannotPass(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	failed := false
	for _, evidence := range []*campaignRunEvidence{&fixture.baseline, &fixture.candidate} {
		for index := range evidence.records {
			record := &evidence.records[index]
			if record.TrackID == "model_pool" && stringValue(record.ArmID) == "arm-wrong" {
				record.Status, record.Success, record.Quality = "failed", &failed, nil
			}
		}
		for index := range evidence.attestation.Entries {
			entry := &evidence.attestation.Entries[index]
			if entry.TrackID == "model_pool" && stringValue(entry.ArmID) == "arm-wrong" {
				entry.Success, entry.Quality = false, nil
			}
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	var worst *CampaignPairedStatistic
	for index := range evidence.Statistics {
		if evidence.Statistics[index].AnalysisUnit == campaignPoolWorstArmReliabilityUnit {
			worst = &evidence.Statistics[index]
			break
		}
	}
	if worst == nil || worst.BaselineValue == nil || *worst.BaselineValue != 0 ||
		worst.CandidateValue == nil || *worst.CandidateValue != 0 || worst.Delta == nil || *worst.Delta != 0 ||
		worst.Verdict != "fail" || campaignTestPairedGate("G3", *evidence, fixture).Verdict != "fail" {
		t.Fatalf("both-worst-zero statistic=%+v gate=%+v", worst, campaignTestPairedGate("G3", *evidence, fixture))
	}
}

func TestCampaignPairedLivePerArmGateCatchesRegressionHiddenByExistingWorstArm(t *testing.T) {
	fixture := withCampaignPoolArm(
		withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false)),
		ModelArm{ID: "arm-stable", Model: "model-stable"},
		0.9,
	)
	failed := false
	for _, side := range []struct {
		evidence     *campaignRunEvidence
		degradeRight bool
	}{
		{evidence: &fixture.baseline},
		{evidence: &fixture.candidate, degradeRight: true},
	} {
		for index := range side.evidence.records {
			record := &side.evidence.records[index]
			if record.TrackID == "model_pool" &&
				(stringValue(record.ArmID) == "arm-wrong" ||
					(side.degradeRight && stringValue(record.ArmID) == "arm-right")) {
				record.Status, record.Success, record.Quality = "failed", &failed, nil
			}
		}
		for index := range side.evidence.attestation.Entries {
			entry := &side.evidence.attestation.Entries[index]
			if entry.TrackID == "model_pool" &&
				(stringValue(entry.ArmID) == "arm-wrong" ||
					(side.degradeRight && stringValue(entry.ArmID) == "arm-right")) {
				entry.Success, entry.Quality = false, nil
			}
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	var poolQuality, availability, worst *CampaignPairedStatistic
	for index := range evidence.Statistics {
		statistic := &evidence.Statistics[index]
		switch statistic.AnalysisUnit {
		case campaignPoolQualityUnit:
			poolQuality = statistic
		case campaignPoolFailureUnit:
			availability = statistic
		case campaignPoolWorstArmReliabilityUnit:
			worst = statistic
		}
	}
	if poolQuality == nil || poolQuality.Verdict != "pass" ||
		availability == nil || availability.Verdict != "pass" ||
		worst == nil || worst.BaselineValue == nil || *worst.BaselineValue != 0 ||
		worst.CandidateValue == nil || *worst.CandidateValue != 0 || worst.Verdict != "fail" {
		t.Fatalf("aggregate pool boundaries quality=%+v availability=%+v worst=%+v", poolQuality, availability, worst)
	}
	for _, statistic := range evidence.ModelPoolArmReliability {
		if statistic.ArmID == "arm-right" {
			if statistic.BaselineFailureRate == nil || *statistic.BaselineFailureRate != 0 ||
				statistic.CandidateFailureRate == nil || *statistic.CandidateFailureRate != 1 ||
				statistic.Verdict != "fail" || campaignTestPairedGate("G3", *evidence, fixture).Verdict != "fail" {
				t.Fatalf("shared arm regression=%+v gate=%+v", statistic, campaignTestPairedGate("G3", *evidence, fixture))
			}
			return
		}
	}
	t.Fatal("arm-right reliability statistic is missing")
}

func TestCampaignPairedLiveG3FailsWhenRoutedJointQualityRegresses(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	for index := range fixture.candidate.records {
		if fixture.candidate.records[index].TrackID == "joint" {
			fixture.candidate.records[index].Quality = float64Reference(0)
		}
	}
	for index := range fixture.candidate.attestation.Entries {
		if fixture.candidate.attestation.Entries[index].TrackID == "joint" {
			fixture.candidate.attestation.Entries[index].Quality = float64Reference(0)
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	g3 := campaignTestPairedGate("G3", *evidence, fixture)
	if g3.Verdict != "fail" {
		t.Fatalf("G3=%+v", g3)
	}
	for _, statistic := range evidence.Statistics {
		if statistic.ID == "campaign.g3.joint.quality_non_inferiority" && statistic.Verdict != "fail" {
			t.Fatalf("joint quality=%+v", statistic)
		}
	}
}

func TestCampaignPairedLiveModelPoolTreatmentUsesCaseOracleAcrossChangedArms(t *testing.T) {
	fixture := withChangedCandidatePool(withCampaignMoMCoreTracks(
		newCampaignPairedLiveFixture(campaignPairedMinimumCases, false),
	))
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	var poolQuality *CampaignPairedStatistic
	for index := range evidence.Statistics {
		if evidence.Statistics[index].ID == "campaign.g3.model_pool.quality_non_inferiority" {
			poolQuality = &evidence.Statistics[index]
			break
		}
	}
	if poolQuality == nil || poolQuality.AnalysisUnit != campaignPoolQualityUnit ||
		poolQuality.SampleCount != campaignPairedMinimumCases || poolQuality.MissingPairs != 0 ||
		poolQuality.BaselineValue == nil || *poolQuality.BaselineValue != 0.9 ||
		poolQuality.CandidateValue == nil || *poolQuality.CandidateValue != 0.9 ||
		poolQuality.Verdict != "pass" {
		t.Fatalf("model-pool quality=%+v", poolQuality)
	}
	if len(evidence.ModelPoolArmReliability) != 3 ||
		evidence.ModelPoolArmReliability[0].ArmID != "arm-extra" ||
		evidence.ModelPoolArmReliability[0].Cohort != campaignArmCohortCandidateOnly ||
		evidence.ModelPoolArmReliability[0].BaselineFailureRate != nil ||
		evidence.ModelPoolArmReliability[0].CandidateFailureRate == nil ||
		*evidence.ModelPoolArmReliability[0].CandidateFailureRate != 0 ||
		evidence.ModelPoolArmReliability[0].Verdict != "pass" {
		t.Fatalf("changed-arm reliability=%+v", evidence.ModelPoolArmReliability)
	}

	baseline, err := campaignAttestedObservations("g3_baseline", fixture.baseline)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := campaignAttestedObservations("g3_candidate", fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := alignCampaignPairedLiveObservations(
		baseline, candidate, "recipe", fixture.baseline.report.Run.Mixture,
		fixture.candidate.report.Run.Mixture,
		fixture.baseline.report.Provenance.PoolSnapshotDigest,
		fixture.candidate.report.Provenance.PoolSnapshotDigest,
	); err == nil {
		t.Fatal("changed model-pool arm coordinates were accepted outside the model_pool treatment")
	}

	candidate = candidate[:len(candidate)-1]
	if _, err := alignCampaignPairedLiveObservations(
		baseline, candidate, "model_pool", fixture.baseline.report.Run.Mixture,
		fixture.candidate.report.Run.Mixture,
		fixture.baseline.report.Provenance.PoolSnapshotDigest,
		fixture.candidate.report.Provenance.PoolSnapshotDigest,
	); err == nil {
		t.Fatal("non-dense changed model-pool evidence was accepted")
	}
}

func TestCampaignPairedLiveChangedPoolRequiresSharedArmCaseAttemptAlignment(t *testing.T) {
	fixture := withChangedCandidatePool(withCampaignMoMCoreTracks(
		newCampaignPairedLiveFixture(campaignPairedMinimumCases, false),
	))
	for index := range fixture.candidate.records {
		record := &fixture.candidate.records[index]
		if record.TrackID == "model_pool" && stringValue(record.ArmID) == "arm-right" {
			record.AttemptID += "-different"
			for entryIndex := range fixture.candidate.attestation.Entries {
				entry := &fixture.candidate.attestation.Entries[entryIndex]
				if entry.BrokerReceipt == *record.BrokerReceipt {
					entry.AttemptID = record.AttemptID
					break
				}
			}
			break
		}
	}
	if _, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate); err == nil ||
		!strings.Contains(err.Error(), "same case and attempt") {
		t.Fatalf("shared-arm attempt mismatch error=%v", err)
	}
}

func TestCampaignPairedLiveChangedPoolWorstArmCoversUnpairableAddedArm(t *testing.T) {
	fixture := withChangedCandidatePool(withCampaignMoMCoreTracks(
		newCampaignPairedLiveFixture(campaignPairedMinimumCases, false),
	))
	failed := false
	for index := range fixture.candidate.records {
		record := &fixture.candidate.records[index]
		if record.TrackID == "model_pool" && stringValue(record.ArmID) == "arm-extra" {
			record.Status, record.Success, record.Quality = "failed", &failed, nil
		}
	}
	for index := range fixture.candidate.attestation.Entries {
		entry := &fixture.candidate.attestation.Entries[index]
		if entry.TrackID == "model_pool" && stringValue(entry.ArmID) == "arm-extra" {
			entry.Success, entry.Quality = false, nil
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	var worst *CampaignPairedStatistic
	for index := range evidence.Statistics {
		if evidence.Statistics[index].AnalysisUnit == campaignPoolWorstArmReliabilityUnit {
			worst = &evidence.Statistics[index]
			break
		}
	}
	if worst == nil || worst.CandidateValue == nil || *worst.CandidateValue != 0 ||
		worst.Verdict != "fail" || campaignTestPairedGate("G3", *evidence, fixture).Verdict != "fail" {
		t.Fatalf("added-arm reliability did not block promotion: worst=%+v gate=%+v", worst, campaignTestPairedGate("G3", *evidence, fixture))
	}
	extra := evidence.ModelPoolArmReliability[0]
	if extra.ArmID != "arm-extra" || extra.Cohort != campaignArmCohortCandidateOnly ||
		extra.CandidateFailureRate == nil || *extra.CandidateFailureRate != 1 || extra.Verdict != "fail" {
		t.Fatalf("added-arm diagnostic=%+v", extra)
	}
}

func TestCampaignPairedLiveOperationOwnershipIncludesMoMCore(t *testing.T) {
	tests := []struct {
		operation string
		trackID   TrackID
		want      bool
	}{
		{operation: workerBrokerRouterEvaluate, trackID: "routing", want: true},
		{operation: workerBrokerArmChatCompletion, trackID: "model_pool", want: true},
		{operation: workerBrokerRoutedChatCompletion, trackID: "joint", want: true},
		{operation: workerBrokerRoutedChatCompletion, trackID: "model_pool", want: false},
		{operation: workerBrokerArmChatCompletion, trackID: "joint", want: false},
	}
	for _, test := range tests {
		if got := campaignOperationOwnsTrack(test.operation, test.trackID); got != test.want {
			t.Fatalf("operation=%q track=%q got=%t want=%t", test.operation, test.trackID, got, test.want)
		}
	}
	for _, trackID := range []TrackID{"routing", "model_pool", "joint"} {
		if !campaignTrackHasExecutionContract(trackID) {
			t.Fatalf("track %q lacks a campaign execution contract", trackID)
		}
	}
}

func TestCampaignPairedLiveG3EnforcesAbsoluteRegretNotOnlyRelativeLift(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	for index := range fixture.candidate.records {
		if fixture.candidate.records[index].TrackID == "joint" {
			fixture.candidate.records[index].Quality = float64Reference(0.5)
		}
	}
	for index := range fixture.candidate.attestation.Entries {
		if fixture.candidate.attestation.Entries[index].TrackID == "joint" {
			fixture.candidate.attestation.Entries[index].Quality = float64Reference(0.5)
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	absolute := campaignG3PromotionStatisticForTest(t, evidence.PromotionStatistics, campaignG3CandidateNormalizedRegretID)
	paired := campaignG3PromotionStatisticForTest(t, evidence.PromotionStatistics, campaignG3PairedNormalizedRegretID)
	if absolute.Verdict != "fail" || paired.Verdict != "pass" ||
		campaignTestPairedGate("G3", *evidence, fixture).Verdict != "fail" {
		t.Fatalf("absolute=%+v paired=%+v gate=%+v", absolute, paired, campaignTestPairedGate("G3", *evidence, fixture))
	}
}

func TestCampaignPairedLiveG3RequiresLiftOverBestFixedArm(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	for index := range fixture.candidate.records {
		if fixture.candidate.records[index].TrackID == "joint" {
			fixture.candidate.records[index].Quality = float64Reference(0.8)
		}
	}
	for index := range fixture.candidate.attestation.Entries {
		if fixture.candidate.attestation.Entries[index].TrackID == "joint" {
			fixture.candidate.attestation.Entries[index].Quality = float64Reference(0.8)
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	frontier := campaignG3PromotionStatisticForTest(t, evidence.PromotionStatistics, campaignG3NoInformationFrontierID)
	absolute := campaignG3PromotionStatisticForTest(t, evidence.PromotionStatistics, campaignG3CandidateNormalizedRegretID)
	if frontier.Estimate >= 0 || frontier.Verdict != "fail" || absolute.Verdict != "pass" ||
		campaignTestPairedGate("G3", *evidence, fixture).Verdict != "fail" {
		t.Fatalf("frontier=%+v absolute=%+v gate=%+v", frontier, absolute, campaignTestPairedGate("G3", *evidence, fixture))
	}
}

func TestCampaignPairedLiveG3CountsZeroOracleAsAllArmFailure(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	for index := range fixture.candidate.records {
		if fixture.candidate.records[index].TrackID == "model_pool" {
			fixture.candidate.records[index].Quality = float64Reference(0)
		}
	}
	for index := range fixture.candidate.attestation.Entries {
		if fixture.candidate.attestation.Entries[index].TrackID == "model_pool" {
			fixture.candidate.attestation.Entries[index].Quality = float64Reference(0)
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	failure := campaignG3PromotionStatisticForTest(t, evidence.PromotionStatistics, campaignG3AllArmFailureID)
	if failure.Estimate != 1 || failure.Verdict != "fail" ||
		campaignTestPairedGate("G3", *evidence, fixture).Verdict != "fail" {
		t.Fatalf("all-arm failure=%+v gate=%+v", failure, campaignTestPairedGate("G3", *evidence, fixture))
	}
}

func TestCampaignPairedLiveG3DoesNotDropMissingQualityCase(t *testing.T) {
	fixture := withCampaignMoMCoreTracks(newCampaignPairedLiveFixture(campaignPairedMinimumCases, false))
	for index := range fixture.candidate.records {
		record := &fixture.candidate.records[index]
		if record.TrackID == "model_pool" {
			record.Quality = nil
			for entryIndex := range fixture.candidate.attestation.Entries {
				entry := &fixture.candidate.attestation.Entries[entryIndex]
				if entry.BrokerReceipt == *record.BrokerReceipt {
					entry.Quality = nil
					break
				}
			}
			break
		}
	}
	evidence, err := buildCampaignPairedLiveEvidence(fixture.baseline, fixture.candidate)
	if err != nil {
		t.Fatal(err)
	}
	absolute := campaignG3PromotionStatisticForTest(t, evidence.PromotionStatistics, campaignG3CandidateNormalizedRegretID)
	if absolute.MissingCases != 1 || absolute.SampleCount != campaignPairedMinimumCases ||
		absolute.Verdict != "unavailable" || campaignTestPairedGate("G3", *evidence, fixture).Verdict != "unavailable" {
		t.Fatalf("missing-quality statistic=%+v gate=%+v", absolute, campaignTestPairedGate("G3", *evidence, fixture))
	}
}

func campaignG3PromotionStatisticForTest(
	t *testing.T,
	statistics []CampaignG3PromotionStatistic,
	id string,
) CampaignG3PromotionStatistic {
	t.Helper()
	for _, statistic := range statistics {
		if statistic.ID == id {
			return statistic
		}
	}
	t.Fatalf("promotion statistic %s is missing: %+v", id, statistics)
	return CampaignG3PromotionStatistic{}
}
