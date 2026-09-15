package evaluationplane

import (
	"fmt"
	"math"
)

func campaignPairedLiveGate(
	base CampaignGate,
	evidence CampaignPairedLiveEvidence,
	baseline campaignRunEvidence,
	candidate campaignRunEvidence,
) CampaignGate {
	statistics, armStatistics := campaignPairedGateStatistics(base.ID, evidence)
	base.Source = "server_attested_paired_live"
	base.EvidenceRefs = append(
		append(anchorEvidenceRefs(baseline.anchor), anchorEvidenceRefs(candidate.anchor)...),
		"campaign-paired-live:"+evidence.Digest,
	)
	base.EvidenceLevel = "E5"
	if base.ID == "G8" {
		return campaignPairedG8Diagnostic(base, statistics)
	}
	if base.ID != "G3" || len(evidence.PromotionStatistics) == 0 || len(statistics) == 0 || len(armStatistics) == 0 {
		base.Rationale = "The paired target execution has no typed statistic for this gate."
		return base
	}

	verdict, passed, failed, unavailable, sampleCount := reduceCampaignG3Verdicts(
		statistics, evidence.PromotionStatistics, armStatistics,
	)
	base.SampleCount = sampleCount
	base.Verdict = verdict
	if verdict == "unavailable" {
		base.Rationale = fmt.Sprintf(
			"Paired target evidence is inconclusive: %d of %d typed case-clustered statistics lack a decisive 95%% interval.",
			unavailable, len(statistics)+len(evidence.PromotionStatistics)+len(armStatistics),
		)
		return base
	}

	if base.ID == "G3" {
		base.Observed = float64Reference(campaignPairedG3Headroom(statistics, evidence.PromotionStatistics, armStatistics))
		base.Threshold = &GateThreshold{Operator: ">=", Value: 0, Unit: "non-inferiority-headroom"}
		base.Rationale = fmt.Sprintf(
			"Server-attested absolute regret, paired frontier, reliability, all-arm failure, quality, and frozen-arm boundaries: %d passed, %d failed, and %d were inconclusive.",
			passed, failed, unavailable,
		)
	}
	return base
}

func campaignPairedGateStatistics(
	gateID string,
	evidence CampaignPairedLiveEvidence,
) ([]CampaignPairedStatistic, []CampaignArmReliabilityStatistic) {
	statistics := make([]CampaignPairedStatistic, 0, len(evidence.Statistics))
	for _, statistic := range evidence.Statistics {
		if statistic.GateID == gateID {
			statistics = append(statistics, statistic)
		}
	}
	armStatistics := []CampaignArmReliabilityStatistic{}
	if gateID == "G3" {
		for _, statistic := range evidence.ModelPoolArmReliability {
			if statistic.Cohort != campaignArmCohortBaselineOnly {
				armStatistics = append(armStatistics, statistic)
			}
		}
	}
	return statistics, armStatistics
}

func reduceCampaignG3Verdicts(
	statistics []CampaignPairedStatistic,
	promotion []CampaignG3PromotionStatistic,
	armStatistics []CampaignArmReliabilityStatistic,
) (GateVerdict, int, int, int, int) {
	verdict, passed, failed, unavailable, sampleCount := reduceCampaignPairedVerdicts(statistics, armStatistics)
	update := func(value GateVerdict) {
		switch value {
		case "pass":
			passed++
		case "fail":
			failed++
			verdict = "fail"
		default:
			unavailable++
			if verdict != "fail" {
				verdict = "unavailable"
			}
		}
	}
	for _, statistic := range promotion {
		if sampleCount == 0 {
			sampleCount = statistic.SampleCount
		} else {
			sampleCount = min(sampleCount, statistic.SampleCount)
		}
		update(statistic.Verdict)
	}
	return verdict, passed, failed, unavailable, sampleCount
}

func campaignPairedG8Diagnostic(
	base CampaignGate,
	statistics []CampaignPairedStatistic,
) CampaignGate {
	base.Source = "server_attested_paired_live_diagnostic"
	failed := 0
	if len(statistics) > 0 {
		base.SampleCount = statistics[0].SampleCount
	}
	for _, statistic := range statistics {
		base.SampleCount = min(base.SampleCount, statistic.SampleCount)
		if statistic.Verdict == "fail" {
			failed++
		}
	}
	base.Rationale = fmt.Sprintf(
		"Controlled paired probes produced %d typed risk diagnostic(s), including %d regression(s), but G8 remains unavailable without server-owned production assignment/exposure, sample-ratio, risk-budget, stop, and rollback evidence.",
		len(statistics), failed,
	)
	return base
}

func reduceCampaignPairedVerdicts(
	statistics []CampaignPairedStatistic,
	armStatistics []CampaignArmReliabilityStatistic,
) (GateVerdict, int, int, int, int) {
	verdict := GateVerdict("pass")
	passed, failed, unavailable := 0, 0, 0
	sampleCount := 0
	if len(statistics) > 0 {
		sampleCount = statistics[0].SampleCount
	} else {
		sampleCount = min(
			armStatistics[0].BaselineSampleCount,
			armStatistics[0].CandidateSampleCount,
		)
	}
	update := func(value GateVerdict) {
		switch value {
		case "pass":
			passed++
		case "fail":
			failed++
			verdict = "fail"
		case "unavailable":
			unavailable++
			if verdict != "fail" {
				verdict = "unavailable"
			}
		}
	}
	for _, statistic := range statistics {
		sampleCount = min(sampleCount, statistic.SampleCount)
		update(statistic.Verdict)
	}
	for _, statistic := range armStatistics {
		armSampleCount := statistic.CandidateSampleCount
		if statistic.Cohort == campaignArmCohortPaired {
			armSampleCount = min(statistic.BaselineSampleCount, statistic.CandidateSampleCount)
		}
		sampleCount = min(sampleCount, armSampleCount)
		update(statistic.Verdict)
	}
	return verdict, passed, failed, unavailable, sampleCount
}

func campaignPairedG3Headroom(
	statistics []CampaignPairedStatistic,
	promotion []CampaignG3PromotionStatistic,
	armStatistics []CampaignArmReliabilityStatistic,
) float64 {
	headroom := math.Inf(1)
	for _, statistic := range statistics {
		if len(statistic.ConfidenceInterval) == 2 {
			headroom = math.Min(headroom, statistic.ConfidenceInterval[0]+statistic.Margin)
		}
		if statistic.AnalysisUnit == campaignPoolWorstArmReliabilityUnit &&
			len(statistic.CandidateConfidenceInterval) == 2 {
			headroom = math.Min(
				headroom,
				statistic.CandidateConfidenceInterval[0]-frozenCampaignG3PromotionPolicy.MinimumCandidateArmReliability,
			)
		}
	}
	for _, statistic := range promotion {
		if len(statistic.ConfidenceInterval) != 2 {
			return math.Inf(-1)
		}
		if statistic.Direction == "higher_is_better" {
			headroom = math.Min(headroom, statistic.ConfidenceInterval[0]-statistic.Threshold.Value)
		} else {
			headroom = math.Min(headroom, statistic.Threshold.Value-statistic.ConfidenceInterval[1])
		}
	}
	for _, statistic := range armStatistics {
		if statistic.Cohort == campaignArmCohortPaired && len(statistic.ConfidenceInterval) == 2 {
			headroom = math.Min(headroom, -statistic.ConfidenceInterval[1]+statistic.Margin)
		}
		if len(statistic.CandidateConfidenceInterval) == 2 {
			absoluteMaximum := 1 - frozenCampaignG3PromotionPolicy.MinimumCandidateArmReliability
			headroom = math.Min(headroom, absoluteMaximum-statistic.CandidateConfidenceInterval[1])
		}
	}
	return headroom
}
