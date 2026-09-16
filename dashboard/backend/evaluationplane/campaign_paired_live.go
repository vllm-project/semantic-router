package evaluationplane

const (
	campaignPairedBootstrapSamples = 1000
	campaignPairedConfidenceLevel  = 0.95
	campaignPairedMinimumCases     = 20
	campaignQualityMargin          = 0.05
	campaignFailureRiskMargin      = 0.02
	campaignLatencyRiskMargin      = 0.05
)

const (
	campaignQualityUnit                 = "case_mean_quality"
	campaignPoolQualityUnit             = "case_pool_oracle_quality"
	campaignPoolWorstArmReliabilityUnit = "pool_worst_arm_reliability"
	campaignFailureUnit                 = "case_failure_fraction"
	campaignPoolFailureUnit             = "case_all_arm_failure"
	campaignLatencyUnit                 = "case_max_latency_relative_delta"
)

const (
	campaignArmCohortPaired        = "paired"
	campaignArmCohortBaselineOnly  = "baseline_only"
	campaignArmCohortCandidateOnly = "candidate_only"
)

type campaignAttestedObservation struct {
	trackID        TrackID
	caseID         string
	attemptID      string
	operation      string
	armID          string
	concurrency    *int64
	modality       *string
	loadPhase      *string
	loadRepeat     *int64
	loadIndex      *int64
	success        bool
	quality        *float64
	latencyMS      float64
	controlledPair *controlledPairObservation
}

type campaignObservationPair struct {
	baseline  campaignAttestedObservation
	candidate campaignAttestedObservation
}

type campaignPoolCasePair struct {
	caseID    string
	baseline  []campaignAttestedObservation
	candidate []campaignAttestedObservation
}

type campaignPairedCohort struct {
	exactPairs []campaignObservationPair
	poolCases  []campaignPoolCasePair
}

type campaignCasePair struct {
	trackID             TrackID
	caseID              string
	baselineQuality     *float64
	candidateQuality    *float64
	baselineFailure     float64
	candidateFailure    float64
	baselineMaxLatency  float64
	candidateMaxLatency float64
}

func buildCampaignPairedLiveEvidence(
	baseline campaignRunEvidence,
	candidate campaignRunEvidence,
) (*CampaignPairedLiveEvidence, error) {
	if err := validateCampaignPairedLiveSources(baseline, candidate); err != nil {
		return nil, err
	}
	baselineRecords, err := campaignAttestedObservations("g3_baseline", baseline)
	if err != nil {
		return nil, err
	}
	candidateRecords, err := campaignAttestedObservations("g3_candidate", candidate)
	if err != nil {
		return nil, err
	}
	cohort, err := alignCampaignPairedLiveObservations(
		baselineRecords,
		candidateRecords,
		baseline.report.Run.ChangeProfile,
		baseline.report.Run.Mixture,
		candidate.report.Run.Mixture,
		baseline.report.Provenance.PoolSnapshotDigest,
		candidate.report.Provenance.PoolSnapshotDigest,
	)
	if err != nil {
		return nil, err
	}
	sessionID, err := validateCampaignControlledPairCohort(cohort)
	if err != nil {
		return nil, err
	}
	return reduceCampaignPairedLiveEvidence(baseline, candidate, cohort, sessionID)
}
