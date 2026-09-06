package evaluationplane

import "fmt"

type robustnessMethodAttestation struct {
	PairCount          int
	NativePairCount    int
	PassRate           *float64
	WorstSlicePassRate *float64
	Passed             *bool
	SourceQualified    bool
	records            []executionRecordEvidence
	brokerReceipts     map[string]string
}

func reduceRobustnessMethod(records []executionRecordEvidence) (robustnessMethodAttestation, error) {
	attestation := robustnessMethodAttestation{records: append([]executionRecordEvidence(nil), records...)}
	actions := make(map[string]string)
	for _, record := range records {
		if record.TrackID == "routing" && record.SelectedArmID != nil {
			actions[record.CaseID] = *record.SelectedArmID
		}
	}
	seenPairs := make(map[string]struct{})
	slicePasses := make(map[string][]bool)
	passes := make([]bool, 0)
	nativeCount := 0
	for _, record := range records {
		method := record.Robustness
		if method == nil {
			continue
		}
		if nativeCount == 0 {
			nativeCount = method.NativePairCount
		} else if nativeCount != method.NativePairCount {
			return robustnessMethodAttestation{}, fmt.Errorf("robustness rows mix native export sizes")
		}
		if _, duplicate := seenPairs[method.PairID]; duplicate {
			return robustnessMethodAttestation{}, fmt.Errorf("robustness pair identities must be unique")
		}
		seenPairs[method.PairID] = struct{}{}
		sourceAction, hasSource := actions[method.SourceCaseID]
		targetAction, hasTarget := actions[method.TargetCaseID]
		if !hasSource || !hasTarget || sourceAction != method.SourceActionID || record.SelectedArmID == nil || targetAction != *record.SelectedArmID {
			return robustnessMethodAttestation{}, fmt.Errorf("robustness pair does not bind its source and target decisions")
		}
		expected := method.SourceActionID
		if method.ExpectedActionID != nil {
			expected = *method.ExpectedActionID
		}
		passed := targetAction == expected
		passes = append(passes, passed)
		for _, sliceID := range method.SliceIDs {
			slicePasses[sliceID] = append(slicePasses[sliceID], passed)
		}
	}
	if len(passes) == 0 {
		return attestation, nil
	}
	passedCount := 0
	for _, passed := range passes {
		if passed {
			passedCount++
		}
	}
	passRate := float64(passedCount) / float64(len(passes))
	worstSlice := 1.0
	for _, values := range slicePasses {
		count := 0
		for _, passed := range values {
			if passed {
				count++
			}
		}
		rate := float64(count) / float64(len(values))
		if rate < worstSlice {
			worstSlice = rate
		}
	}
	attestation.PairCount = len(passes)
	attestation.NativePairCount = nativeCount
	attestation.PassRate = &passRate
	attestation.WorstSlicePassRate = &worstSlice
	if len(passes) == nativeCount {
		passed := passRate == 1 && worstSlice == 1
		attestation.Passed = &passed
	}
	return attestation, nil
}
