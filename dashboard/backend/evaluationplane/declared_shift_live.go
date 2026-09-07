package evaluationplane

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

const (
	declaredShiftLiveMethodID         = "declared-shift.server-live.v1"
	declaredShiftLiveEvidenceSourceID = "declared-shift.server-live.v1"
)

type installedDeclaredShiftPair struct {
	SchemaVersion      string   `json:"schema_version"`
	PairID             string   `json:"pair_id"`
	SourceCaseID       string   `json:"source_case_id"`
	PerturbedCaseID    string   `json:"perturbed_case_id"`
	Relation           string   `json:"relation"`
	ExpectedActionID   *string  `json:"expected_action_id,omitempty"`
	SliceIDs           []string `json:"slice_ids"`
	NativePairCount    int      `json:"native_pair_count"`
	SourceRecordDigest string   `json:"source_record_digest"`
}

type declaredShiftSuiteSource struct {
	document                   installedSuiteDocument
	qualificationReceiptDigest string
	perturbationArtifactDigest string
	pairs                      []installedDeclaredShiftPair
}

type declaredShiftPairObservation struct {
	sourceCaseID, targetCaseID   string
	sourceReceipt, targetReceipt string
	passed                       bool
}

func installedDeclaredShiftSourceEligible(root string, document installedSuiteDocument) (bool, error) {
	_, eligible, err := loadInstalledDeclaredShiftSource(root, document)
	return eligible, err
}

func loadInstalledDeclaredShiftSource(
	root string,
	document installedSuiteDocument,
) (declaredShiftSuiteSource, bool, error) {
	var provenance unqualifiedSuiteEvidence
	if err := decodeExactJSON(document.Manifest.QualificationReceipt.Qualification, &provenance); err != nil {
		return declaredShiftSuiteSource{}, false, fmt.Errorf("%w: normalized suite parser qualification is invalid", ErrInvalid)
	}
	var artifacts map[string]json.RawMessage
	if err := json.Unmarshal(document.Manifest.Artifacts, &artifacts); err != nil {
		return declaredShiftSuiteSource{}, false, fmt.Errorf("%w: normalized suite artifact set is invalid", ErrInvalid)
	}
	rawPerturbations, hasPerturbations := artifacts["perturbations"]
	if !provenance.ParserVerified || !hasPerturbations || !containsTrack(document.Manifest.TrackIDs, "routing") {
		return declaredShiftSuiteSource{}, false, nil
	}
	var ref suiteArtifactReference
	if err := decodeExactJSON(rawPerturbations, &ref); err != nil || !digestPattern.MatchString(ref.Digest) {
		return declaredShiftSuiteSource{}, false, fmt.Errorf("%w: declared-shift artifact reference is invalid", ErrInvalid)
	}
	pairs := make([]installedDeclaredShiftPair, 0)
	seenPairIDs := make(map[string]struct{})
	seenCoordinates := make(map[string]struct{})
	seenCaseIDs := make(map[string]struct{})
	err := scanInstalledSuiteRole(root, document.Manifest, "perturbations", true, func(line []byte, lineNumber int) error {
		var pair installedDeclaredShiftPair
		if err := decodeStrictJSONLine(line, &pair); err != nil || pair.SchemaVersion != normalizedSuiteSchemaVersion ||
			!validMethodID(pair.PairID) || pair.SourceCaseID == "" || pair.PerturbedCaseID == "" ||
			pair.SourceCaseID == pair.PerturbedCaseID || !validMethodDigest(pair.SourceRecordDigest) ||
			len(pair.SliceIDs) == 0 || (pair.Relation != "invariant" && pair.Relation != "expected_change") ||
			(pair.Relation == "invariant") != (pair.ExpectedActionID == nil) {
			return fmt.Errorf("%w: installed declared-shift line %d is invalid", ErrInvalid, lineNumber)
		}
		coordinate := pair.SourceCaseID + "\x00" + pair.PerturbedCaseID
		seenSlices := make(map[string]struct{}, len(pair.SliceIDs))
		for _, sliceID := range pair.SliceIDs {
			if strings.TrimSpace(sliceID) == "" || strings.TrimSpace(sliceID) != sliceID {
				return fmt.Errorf("%w: installed declared-shift slice identity is invalid", ErrInvalid)
			}
			if _, duplicate := seenSlices[sliceID]; duplicate {
				return fmt.Errorf("%w: installed declared-shift slice identity is duplicated", ErrInvalid)
			}
			seenSlices[sliceID] = struct{}{}
		}
		if _, duplicate := seenPairIDs[pair.PairID]; duplicate {
			return fmt.Errorf("%w: installed declared-shift pair identity is duplicated", ErrInvalid)
		}
		if _, duplicate := seenCoordinates[coordinate]; duplicate {
			return fmt.Errorf("%w: installed declared-shift coordinate is duplicated", ErrInvalid)
		}
		for _, caseID := range []string{pair.SourceCaseID, pair.PerturbedCaseID} {
			if _, duplicate := seenCaseIDs[caseID]; duplicate {
				return fmt.Errorf("%w: installed declared-shift pairs reuse a case identity", ErrInvalid)
			}
			seenCaseIDs[caseID] = struct{}{}
		}
		seenPairIDs[pair.PairID] = struct{}{}
		seenCoordinates[coordinate] = struct{}{}
		pairs = append(pairs, pair)
		return nil
	})
	if err != nil {
		return declaredShiftSuiteSource{}, false, err
	}
	if len(pairs) == 0 {
		return declaredShiftSuiteSource{}, false, nil
	}
	casePlans, err := installedVisibleCasePlans(root, document)
	if err != nil {
		return declaredShiftSuiteSource{}, false, err
	}
	for _, pair := range pairs {
		if pair.NativePairCount != len(pairs) {
			return declaredShiftSuiteSource{}, false, fmt.Errorf("%w: installed declared-shift native pair count drifted", ErrInvalid)
		}
		for _, caseID := range []string{pair.SourceCaseID, pair.PerturbedCaseID} {
			plan, exists := casePlans[caseID]
			if !exists || !containsTrack(plan.TrackIDs, "routing") {
				return declaredShiftSuiteSource{}, false, fmt.Errorf("%w: installed declared-shift pair references a non-routing case", ErrInvalid)
			}
		}
	}
	receiptBytes, err := json.Marshal(document.Manifest.QualificationReceipt)
	if err != nil {
		return declaredShiftSuiteSource{}, false, err
	}
	receiptDigest, err := canonicalJSONDigest(receiptBytes)
	if err != nil {
		return declaredShiftSuiteSource{}, false, err
	}
	return declaredShiftSuiteSource{
		document: document, qualificationReceiptDigest: receiptDigest,
		perturbationArtifactDigest: ref.Digest, pairs: pairs,
	}, true, nil
}

func validateLiveDeclaredShiftRecords(
	runDir string, manifest RunManifest, attestation robustnessMethodAttestation,
) (robustnessMethodAttestation, error) {
	caseIdentities, err := validatedNormalizedCaseIdentities(
		runDir, manifest, builtinNormalizedLiveExecutorContract(),
	)
	if err != nil {
		return robustnessMethodAttestation{}, err
	}
	source, err := loadLiveDeclaredShiftSource(runDir, manifest)
	if err != nil {
		return robustnessMethodAttestation{}, err
	}
	opaqueBySource := make(map[string]string, len(caseIdentities))
	for opaqueID, identity := range caseIdentities {
		opaqueBySource[normalizedSourceKey(identity.SuiteID, identity.SourceID)] = opaqueID
	}
	routingRows, methodRows, err := indexLiveDeclaredShiftRecords(attestation.records)
	if err != nil {
		return robustnessMethodAttestation{}, err
	}
	result := robustnessMethodAttestation{
		NativePairCount: len(source.pairs), records: attestation.records,
		brokerReceipts: make(map[string]string, len(source.pairs)*2),
	}
	if len(routingRows) != len(source.pairs)*2 {
		return result, nil
	}
	passes, slicePasses, complete, err := collectLiveDeclaredShiftOutcomes(
		source, manifest, opaqueBySource, routingRows, methodRows, &result,
	)
	if err != nil {
		return robustnessMethodAttestation{}, err
	}
	if !complete || len(methodRows) != len(source.pairs) || len(result.brokerReceipts) != len(source.pairs)*2 {
		return result, nil
	}
	sealLiveDeclaredShiftResult(&result, passes, slicePasses)
	return result, nil
}

func loadLiveDeclaredShiftSource(runDir string, manifest RunManifest) (declaredShiftSuiteSource, error) {
	suiteRoot := filepath.Join(filepath.Dir(filepath.Dir(runDir)), "suites")
	documents, err := loadInstalledLineageSuites(suiteRoot, manifest)
	if err != nil {
		return declaredShiftSuiteSource{}, err
	}
	var source declaredShiftSuiteSource
	eligibleCount := 0
	for _, suiteID := range manifest.SuiteIDs {
		document := documents[suiteID]
		if !containsTrack(document.Manifest.TrackIDs, "routing") {
			continue
		}
		candidate, eligible, loadErr := loadInstalledDeclaredShiftSource(suiteRoot, document)
		if loadErr != nil {
			return declaredShiftSuiteSource{}, loadErr
		}
		if eligible {
			source = candidate
			eligibleCount++
		}
	}
	if eligibleCount != 1 {
		return declaredShiftSuiteSource{}, fmt.Errorf("%w: server-live declared-shift evidence requires one exact eligible routing suite", ErrInvalid)
	}
	return source, nil
}

func indexLiveDeclaredShiftRecords(records []executionRecordEvidence) (map[string][]executionRecordEvidence, map[string]executionRecordEvidence, error) {
	routingRows := make(map[string][]executionRecordEvidence)
	methodRows := make(map[string]executionRecordEvidence)
	for _, record := range records {
		if record.TrackID == "routing" {
			routingRows[record.CaseID] = append(routingRows[record.CaseID], record)
		}
		if record.Robustness == nil {
			continue
		}
		if _, duplicate := methodRows[record.Robustness.PairID]; duplicate {
			return nil, nil, fmt.Errorf("%w: declared-shift pair is duplicated", ErrInvalid)
		}
		methodRows[record.Robustness.PairID] = record
	}
	return routingRows, methodRows, nil
}

func collectLiveDeclaredShiftOutcomes(
	source declaredShiftSuiteSource, manifest RunManifest,
	opaqueBySource map[string]string,
	routingRows map[string][]executionRecordEvidence,
	methodRows map[string]executionRecordEvidence,
	result *robustnessMethodAttestation,
) ([]bool, map[string][]bool, bool, error) {
	passes := make([]bool, 0, len(source.pairs))
	slicePasses := make(map[string][]bool)
	for _, pair := range source.pairs {
		observation, complete, err := validateLiveDeclaredShiftPair(
			source, pair, manifest, opaqueBySource, routingRows, methodRows,
		)
		if err != nil {
			return nil, nil, false, err
		}
		if !complete {
			return nil, nil, false, nil
		}
		for _, binding := range []struct{ receipt, caseID string }{
			{observation.sourceReceipt, observation.sourceCaseID},
			{observation.targetReceipt, observation.targetCaseID},
		} {
			receipt, caseID := binding.receipt, binding.caseID
			if _, duplicate := result.brokerReceipts[receipt]; duplicate {
				return nil, nil, false, fmt.Errorf("%w: declared-shift broker receipt is duplicated", ErrInvalid)
			}
			result.brokerReceipts[receipt] = caseID
		}
		passes = append(passes, observation.passed)
		for _, sliceID := range pair.SliceIDs {
			slicePasses[sliceID] = append(slicePasses[sliceID], observation.passed)
		}
	}
	return passes, slicePasses, true, nil
}

func validateLiveDeclaredShiftPair(
	source declaredShiftSuiteSource, pair installedDeclaredShiftPair, manifest RunManifest,
	opaqueBySource map[string]string,
	routingRows map[string][]executionRecordEvidence,
	methodRows map[string]executionRecordEvidence,
) (declaredShiftPairObservation, bool, error) {
	sourceCaseID := opaqueBySource[normalizedSourceKey(source.document.Manifest.ID, pair.SourceCaseID)]
	targetCaseID := opaqueBySource[normalizedSourceKey(source.document.Manifest.ID, pair.PerturbedCaseID)]
	if sourceCaseID == "" || targetCaseID == "" {
		return declaredShiftPairObservation{}, false, nil
	}
	sourceRecords, targetRecords := routingRows[sourceCaseID], routingRows[targetCaseID]
	if len(sourceRecords) != 1 || len(targetRecords) != 1 {
		return declaredShiftPairObservation{}, false, nil
	}
	sourceRecord, targetRecord := sourceRecords[0], targetRecords[0]
	if !completeDeclaredShiftRoutingRecord(sourceRecord) || !completeDeclaredShiftRoutingRecord(targetRecord) {
		return declaredShiftPairObservation{}, false, nil
	}
	methodRecord, present := methodRows[pair.PairID]
	if !present || methodRecord.CaseID != targetCaseID || methodRecord.Robustness == nil {
		return declaredShiftPairObservation{}, false, nil
	}
	method := methodRecord.Robustness
	expectedAction := resolveDeclaredShiftExpectedAction(pair.ExpectedActionID, manifest.Target.Mixture)
	if pair.Relation == "expected_change" && expectedAction == nil {
		return declaredShiftPairObservation{}, false, nil
	}
	if method.SuiteID == nil || *method.SuiteID != source.document.Manifest.ID ||
		method.SuiteRevision == nil || *method.SuiteRevision != source.document.Manifest.Revision ||
		method.QualificationReceiptDigest == nil || *method.QualificationReceiptDigest != source.qualificationReceiptDigest ||
		method.PerturbationArtifactDigest == nil || *method.PerturbationArtifactDigest != source.perturbationArtifactDigest ||
		method.PairID != pair.PairID || method.SourceCaseID != sourceCaseID || method.TargetCaseID != targetCaseID ||
		method.Relation != pair.Relation || method.NativePairCount != len(source.pairs) ||
		method.SourceRecordDigest != pair.SourceRecordDigest || method.SourceActionID != *sourceRecord.SelectedArmID ||
		!sameOptionalString(method.ExpectedActionID, expectedAction) ||
		!sameStringSet(method.SliceIDs, pair.SliceIDs) {
		return declaredShiftPairObservation{}, false, fmt.Errorf("%w: declared-shift record differs from its exact installed relation", ErrInvalid)
	}
	if *sourceRecord.BrokerReceipt == *targetRecord.BrokerReceipt {
		return declaredShiftPairObservation{}, false, fmt.Errorf("%w: declared-shift broker receipt is duplicated", ErrInvalid)
	}
	expected := *sourceRecord.SelectedArmID
	if expectedAction != nil {
		expected = *expectedAction
	}
	return declaredShiftPairObservation{
		sourceCaseID: sourceCaseID, targetCaseID: targetCaseID,
		sourceReceipt: *sourceRecord.BrokerReceipt, targetReceipt: *targetRecord.BrokerReceipt,
		passed: *targetRecord.SelectedArmID == expected,
	}, true, nil
}

func sealLiveDeclaredShiftResult(result *robustnessMethodAttestation, passes []bool, slicePasses map[string][]bool) {
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
	passed := passRate == 1 && worstSlice == 1
	result.PairCount = len(passes)
	result.PassRate = &passRate
	result.WorstSlicePassRate = &worstSlice
	result.Passed = &passed
	result.SourceQualified = true
}

func completeDeclaredShiftRoutingRecord(record executionRecordEvidence) bool {
	return record.Status == "succeeded" && record.Success != nil && *record.Success &&
		record.SelectedArmID != nil && record.BrokerReceipt != nil && record.EvidenceKind != nil &&
		*record.EvidenceKind == declaredShiftLiveEvidenceSourceID
}

func resolveDeclaredShiftExpectedAction(selector *string, mixture *ManifestMixture) *string {
	if selector == nil || mixture == nil {
		return nil
	}
	matched := ""
	for _, arm := range mixture.ModelArms {
		if *selector == arm.ID || *selector == arm.Model {
			if matched != "" {
				return nil
			}
			matched = arm.ID
		}
	}
	if matched == "" {
		return nil
	}
	return &matched
}

func sameStringSet(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	values := make(map[string]int, len(left))
	for _, value := range left {
		values[value]++
	}
	for _, value := range right {
		values[value]--
	}
	for _, count := range values {
		if count != 0 {
			return false
		}
	}
	return true
}

func builtinNormalizedLiveExecutorContract() executorContract {
	for _, contract := range builtinExecutorContracts() {
		if contract.ID == normalizedSuiteLiveExecutorID {
			return contract
		}
	}
	return executorContract{}
}
