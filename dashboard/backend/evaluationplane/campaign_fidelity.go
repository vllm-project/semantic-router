package evaluationplane

import (
	"fmt"
	"math"
	"reflect"
	"sort"
)

const (
	minimumFidelityCases    = 59
	fidelityMinimum         = 0.95
	fidelityConfidenceLevel = 0.95
	qualityGapMaximum       = 0.10
)

type campaignFidelityResult struct {
	PointEstimate      float64
	LowerBound         float64
	Matched            int
	DecisionMismatches int
	OutcomeMismatches  int
	UnavailableCases   int
	SampleCount        int
	Verdict            GateVerdict
	Rationale          string
}

type campaignFidelityKey struct {
	TrackID   TrackID
	CaseID    string
	AttemptID string
	ArmID     string
}

func computeCampaignFidelity(
	reference, live []executionRecordEvidence,
	trackID TrackID,
) (campaignFidelityResult, error) {
	referenceByKey, err := campaignFidelityRecords(reference, trackID)
	if err != nil {
		return campaignFidelityResult{}, fmt.Errorf("%w: reference fidelity records: %w", ErrInvalid, err)
	}
	liveByKey, err := campaignFidelityRecords(live, trackID)
	if err != nil {
		return campaignFidelityResult{}, fmt.Errorf("%w: fresh-live fidelity records: %w", ErrInvalid, err)
	}
	if len(referenceByKey) == 0 || len(liveByKey) == 0 || len(referenceByKey) != len(liveByKey) {
		return campaignFidelityResult{}, fmt.Errorf("%w: reference and fresh-live fidelity cohorts are not case-aligned", ErrInvalid)
	}
	type caseAccounting struct {
		matched, decisionMismatch, outcomeMismatch, unavailable bool
	}
	cases := make(map[string]caseAccounting)
	for key, referenceRecord := range referenceByKey {
		liveRecord, ok := liveByKey[key]
		if !ok {
			return campaignFidelityResult{}, fmt.Errorf("%w: reference and fresh-live fidelity cohorts are not case-aligned", ErrInvalid)
		}
		accounting, seen := cases[key.CaseID]
		if !seen {
			accounting.matched = true
		}
		classification := campaignFidelityPairClassification(referenceRecord, liveRecord)
		switch classification {
		case "unavailable":
			accounting.unavailable, accounting.matched = true, false
		case "decision":
			accounting.decisionMismatch, accounting.matched = true, false
		case "outcome":
			accounting.outcomeMismatch, accounting.matched = true, false
		}
		cases[key.CaseID] = accounting
	}
	result := campaignFidelityResult{}
	for _, accounting := range cases {
		switch {
		case accounting.unavailable:
			result.UnavailableCases++
		case accounting.decisionMismatch:
			result.DecisionMismatches++
		case accounting.outcomeMismatch:
			result.OutcomeMismatches++
		case accounting.matched:
			result.Matched++
		}
	}
	result.SampleCount = len(cases)
	result.PointEstimate = float64(result.Matched) / float64(result.SampleCount)
	result.LowerBound = oneSidedExactBinomialLowerBound(result.Matched, result.SampleCount, fidelityConfidenceLevel)
	result.Verdict = "fail"
	result.Rationale = fmt.Sprintf(
		"Reference-to-fresh-live fidelity failed: point agreement %.6g, one-sided 95%% exact lower bound %.6g, required lower bound %.6g; %d decision, %d outcome, and %d unavailable case(s).",
		result.PointEstimate, result.LowerBound, fidelityMinimum,
		result.DecisionMismatches, result.OutcomeMismatches, result.UnavailableCases,
	)
	if result.SampleCount < minimumFidelityCases {
		result.Verdict = "unavailable"
		result.Rationale = fmt.Sprintf(
			"Reference-to-fresh-live fidelity needs at least %d exact case pairs before even an all-match cohort can prove a one-sided 95%% lower bound of %.2f; observed %d.",
			minimumFidelityCases, fidelityMinimum, result.SampleCount,
		)
	} else if result.LowerBound >= fidelityMinimum && result.UnavailableCases == 0 {
		result.Verdict = "pass"
		result.Rationale = fmt.Sprintf(
			"Reference-to-fresh-live fidelity passed: %d/%d exact decisions and outcomes matched; one-sided 95%% exact lower bound %.6g met %.6g.",
			result.Matched, result.SampleCount, result.LowerBound, fidelityMinimum,
		)
	}
	return result, nil
}

// oneSidedExactBinomialLowerBound returns the Clopper-Pearson lower bound.
// Inverting the exact binomial upper tail makes the frozen 59/59 minimum
// explicit: 58/58 cannot prove a 0.95 lower bound, while 59/59 can.
func oneSidedExactBinomialLowerBound(successes, trials int, confidence float64) float64 {
	if trials <= 0 || successes <= 0 {
		return 0
	}
	if successes > trials || confidence <= 0 || confidence >= 1 {
		return math.NaN()
	}
	alphaLog := math.Log1p(-confidence)
	low, high := 0.0, float64(successes)/float64(trials)
	for range 80 {
		mid := (low + high) / 2
		if logBinomialUpperTail(successes, trials, mid) > alphaLog {
			high = mid
		} else {
			low = mid
		}
	}
	return (low + high) / 2
}

func logBinomialUpperTail(successes, trials int, probability float64) float64 {
	if probability <= 0 {
		return math.Inf(-1)
	}
	if probability >= 1 {
		return 0
	}
	trialLog, _ := math.Lgamma(float64(trials + 1))
	successLog, _ := math.Lgamma(float64(successes + 1))
	failureLog, _ := math.Lgamma(float64(trials - successes + 1))
	logProbability, logFailure := math.Log(probability), math.Log1p(-probability)
	term := trialLog - successLog - failureLog +
		float64(successes)*logProbability + float64(trials-successes)*logFailure
	total := term
	for count := successes; count < trials; count++ {
		term += math.Log(float64(trials-count)) - math.Log(float64(count+1)) + logProbability - logFailure
		total = logAddExp(total, term)
	}
	return min(total, 0)
}

func logAddExp(left, right float64) float64 {
	if left < right {
		left, right = right, left
	}
	if math.IsInf(left, -1) {
		return left
	}
	return left + math.Log1p(math.Exp(right-left))
}

func campaignFidelityRecords(
	records []executionRecordEvidence,
	trackID TrackID,
) (map[campaignFidelityKey]executionRecordEvidence, error) {
	if !campaignTrackHasFidelityContract(trackID) {
		return nil, fmt.Errorf("track %s has no campaign fidelity contract", trackID)
	}
	selected := make(map[campaignFidelityKey]executionRecordEvidence)
	for _, record := range records {
		if record.TrackID != trackID {
			continue
		}
		armID := ""
		if record.TrackID == "model_pool" {
			armID = stringValue(record.ArmID)
			if armID == "" {
				return nil, fmt.Errorf("model_pool fidelity record omits its frozen arm")
			}
		}
		key := campaignFidelityKey{
			TrackID: record.TrackID, CaseID: record.CaseID,
			AttemptID: record.AttemptID, ArmID: armID,
		}
		if _, duplicate := selected[key]; duplicate {
			return nil, fmt.Errorf("duplicate semantic record for track %s case %s attempt %s arm %s", key.TrackID, key.CaseID, key.AttemptID, key.ArmID)
		}
		selected[key] = record
	}
	return selected, nil
}

func campaignTrackHasFidelityContract(trackID TrackID) bool {
	return trackID == "routing" || trackID == "model_pool" || trackID == "joint" || trackID == "multimodal"
}

func campaignFidelityPairClassification(reference, live executionRecordEvidence) string {
	if reference.Status == "unavailable" || live.Status == "unavailable" ||
		reference.Success == nil || live.Success == nil {
		return "unavailable"
	}
	if reference.SelectedArmID != nil || live.SelectedArmID != nil {
		if reference.SelectedArmID == nil || live.SelectedArmID == nil ||
			*reference.SelectedArmID != *live.SelectedArmID {
			return "decision"
		}
	}
	if reference.ArmID != nil || live.ArmID != nil {
		if reference.ArmID == nil || live.ArmID == nil || *reference.ArmID != *live.ArmID {
			return "decision"
		}
	}
	if *reference.Success != *live.Success {
		return "outcome"
	}
	// A pair of identically failed requests is not evidence that the candidate
	// reproduced a useful reference decision. Keep transport/runtime failures
	// visible in the unavailable accounting instead of inflating fidelity.
	if !*reference.Success {
		return "unavailable"
	}
	if reference.Quality != nil && live.Quality != nil &&
		math.Abs(*reference.Quality-*live.Quality) > qualityGapMaximum {
		return "outcome"
	}
	if (reference.Quality == nil) != (live.Quality == nil) {
		return "outcome"
	}
	return "matched"
}

func validateCampaignFidelitySources(reference, live campaignRunEvidence) error {
	left, right := reference.report, live.report
	trackID, profileErr := campaignFidelityTrack(left.Run.ChangeProfile)
	if profileErr != nil || left.Run.ChangeProfile != right.Run.ChangeProfile {
		return fmt.Errorf("reference and fresh-live runs do not bind one registered profile fidelity contract")
	}
	if reference.anchor.CandidateSubjectDigest == "" ||
		reference.anchor.CandidateSubjectDigest != live.anchor.CandidateSubjectDigest ||
		left.Run.ID == right.Run.ID || left.Run.Mode != ModeLive || right.Run.Mode != ModeLive ||
		reference.attestation == nil || live.attestation == nil ||
		left.Run.CompletedAt == nil || !live.attestation.StartedAt.After(left.Run.CompletedAt.UTC()) ||
		!reflect.DeepEqual(left.Run.SuiteIDs, right.Run.SuiteIDs) ||
		!reflect.DeepEqual(left.Run.TrackIDs, right.Run.TrackIDs) ||
		left.Run.Seed != right.Run.Seed || left.Run.SampleLimit != right.Run.SampleLimit ||
		left.Provenance.WorkloadSnapshotDigest == "" ||
		left.Provenance.WorkloadSnapshotDigest != right.Provenance.WorkloadSnapshotDigest ||
		!reflect.DeepEqual(left.Provenance.BenchmarkRevisions, right.Provenance.BenchmarkRevisions) {
		return fmt.Errorf("reference and fresh-live runs do not bind one exact candidate suite/workload/case cohort")
	}
	if _, err := campaignAttestedObservations("g5_live", live); err != nil {
		return err
	}
	if _, err := campaignAttestedObservations("g5_reference", reference); err != nil {
		return err
	}
	leftRecords, err := campaignFidelityRecords(reference.records, trackID)
	if err != nil {
		return err
	}
	rightRecords, err := campaignFidelityRecords(live.records, trackID)
	if err != nil {
		return err
	}
	if len(leftRecords) == 0 || !reflect.DeepEqual(campaignFidelityKeys(leftRecords), campaignFidelityKeys(rightRecords)) {
		return fmt.Errorf("reference and fresh-live records do not exactly cover the same case cohort")
	}
	return nil
}

func campaignFidelityKeys(records map[campaignFidelityKey]executionRecordEvidence) []campaignFidelityKey {
	keys := make([]campaignFidelityKey, 0, len(records))
	for key := range records {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(left, right int) bool {
		if keys[left].TrackID != keys[right].TrackID {
			return keys[left].TrackID < keys[right].TrackID
		}
		if keys[left].CaseID != keys[right].CaseID {
			return keys[left].CaseID < keys[right].CaseID
		}
		if keys[left].AttemptID != keys[right].AttemptID {
			return keys[left].AttemptID < keys[right].AttemptID
		}
		return keys[left].ArmID < keys[right].ArmID
	})
	return keys
}

func buildCampaignFidelityEvidence(reference, live campaignRunEvidence) (*CampaignFidelityEvidence, error) {
	if err := validateCampaignFidelitySources(reference, live); err != nil {
		return nil, err
	}
	trackID, err := campaignFidelityTrack(reference.report.Run.ChangeProfile)
	if err != nil {
		return nil, err
	}
	result, err := computeCampaignFidelity(reference.records, live.records, trackID)
	if err != nil {
		return nil, err
	}
	evidence := &CampaignFidelityEvidence{
		SchemaVersion: SchemaVersion, ContractVersion: CampaignFidelityContractVersion,
		ReferenceRunID: reference.report.Run.ID, LiveRunID: live.report.Run.ID,
		CandidateSubjectDigest:         reference.anchor.CandidateSubjectDigest,
		ReferenceManifestDigest:        reference.anchor.ManifestSemanticDigest,
		LiveManifestDigest:             live.anchor.ManifestSemanticDigest,
		LiveExecutionAttestationDigest: live.anchor.ExecutionAttestationDigest,
		TrackID:                        trackID,
		SuiteIDs:                       append([]string(nil), reference.report.Run.SuiteIDs...),
		WorkloadSnapshotDigest:         reference.report.Provenance.WorkloadSnapshotDigest,
		BenchmarkRevisions:             copyCampaignRevisionMap(reference.report.Provenance.BenchmarkRevisions),
		MatchedCases:                   result.Matched, DecisionMismatches: result.DecisionMismatches,
		OutcomeMismatches: result.OutcomeMismatches, UnavailableCases: result.UnavailableCases,
		SampleCount: result.SampleCount, PointEstimate: result.PointEstimate,
		LowerBound: result.LowerBound, ConfidenceLevel: fidelityConfidenceLevel, Verdict: result.Verdict,
	}
	evidence.Digest, err = campaignFidelityEvidenceDigest(*evidence)
	if err != nil {
		return nil, err
	}
	return evidence, nil
}

func campaignFidelityTrack(profile ChangeProfile) (TrackID, error) {
	slot, ok := campaignSlotContract(profile, "G5")
	if !ok || slot.Disposition == GateDispositionNotApplicable || !campaignTrackHasFidelityContract(slot.TrackID) {
		return "", fmt.Errorf("%w: change profile %q has no registered G5 fidelity track", ErrInvalid, profile)
	}
	return slot.TrackID, nil
}

func campaignFidelityEvidenceDigest(evidence CampaignFidelityEvidence) (string, error) {
	evidence.Digest = ""
	digest, err := canonicalValueDigest(evidence)
	if err != nil {
		return "", fmt.Errorf("digest campaign fidelity evidence: %w", err)
	}
	return digest, nil
}
