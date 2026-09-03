package evaluationplane

import (
	"fmt"
	"path/filepath"
	"reflect"
)

type sealedEvidenceLevels struct {
	Run     EvidenceLevel
	ByTrack map[TrackID]EvidenceLevel
}

const (
	liveMoMRoutingEvidenceKind   = "live-routing-diagnostic.v1"
	liveMoMModelPoolEvidenceKind = "live-mom-arm-outcome.v1"
	liveMoMJointEvidenceKind     = "live-mom-routed-outcome.v1"
)

func deriveSealedEvidenceLevels(
	runDir string,
	manifest RunManifest,
	records recordAttestation,
	qualification suiteGateQualification,
	executor executorContract,
	capacitySLO *capacitySLOAttestation,
	executionAttestation *executionAttestation,
) (sealedEvidenceLevels, error) {
	levels := sealedEvidenceLevels{Run: "E0", ByTrack: make(map[TrackID]EvidenceLevel, len(manifest.TrackIDs))}
	for _, trackID := range manifest.TrackIDs {
		levels.ByTrack[trackID] = "E0"
	}
	if manifest.Mode == ModeLive {
		deriveLiveMoMEvidenceLevels(&levels, manifest, records, executor, executionAttestation)
		deriveLiveDeclaredShiftEvidenceLevels(&levels, manifest, records, executor, executionAttestation)
		deriveLiveMultimodalEvidenceLevels(&levels, manifest, records, executor, executionAttestation)
		deriveLiveMethodEvidenceLevels(&levels, manifest, records)
		if containsTrack(manifest.TrackIDs, "capacity") && capacitySLO != nil {
			levels.ByTrack["capacity"] = "E5"
		}
		levels.Run = weakestSelectedTrackLevel(manifest.TrackIDs, levels.ByTrack)
		return levels, nil
	}
	// The built-in MoM replay is a frozen synthetic counterfactual. It is useful
	// for diagnostics, but can never establish source-bound execution evidence.
	if executorIsMoMCohortReplay(executor) {
		return levels, nil
	}
	if len(qualification.suiteTrackIDs) == 0 {
		return levels, nil
	}

	caseIdentities, err := validatedNormalizedCaseIdentities(runDir, manifest, executor)
	if err != nil {
		return sealedEvidenceLevels{}, err
	}
	if err := validateNormalizedCaseTrackPlans(runDir, manifest, records, qualification, caseIdentities); err != nil {
		return sealedEvidenceLevels{}, err
	}
	if !qualification.normalizedSuiteRun {
		return levels, nil
	}

	for _, trackID := range manifest.TrackIDs {
		trackLevel := EvidenceLevel("E5")
		qualified := true
		for caseID := range records.PlannedCaseIDsByTrack[trackID] {
			identity, bound := caseIdentities[caseID]
			suiteLevel, hasLevel := qualification.suiteTrackLevels[identity.SuiteID][trackID]
			cell := records.CellEvidence[trackID][caseID]
			expectedKind := executor.ID + ";ceiling=" + string(suiteLevel)
			if !bound || !hasLevel || cell == nil || cell.Rows == 0 || cell.Unavailable ||
				len(cell.EvidenceKinds) != 1 {
				qualified = false
				break
			}
			if _, exact := cell.EvidenceKinds[expectedKind]; !exact {
				qualified = false
				break
			}
			if evidenceLevelRank(suiteLevel) < evidenceLevelRank(trackLevel) {
				trackLevel = suiteLevel
			}
		}
		if qualified {
			levels.ByTrack[trackID] = trackLevel
		}
	}
	levels.Run = weakestSelectedTrackLevel(manifest.TrackIDs, levels.ByTrack)
	return levels, nil
}

func deriveLiveMultimodalEvidenceLevels(
	levels *sealedEvidenceLevels,
	manifest RunManifest,
	records recordAttestation,
	executor executorContract,
	attestation *executionAttestation,
) {
	if executor.ID != normalizedSuiteLiveExecutorID || !records.validated || attestation == nil ||
		!containsTrack(manifest.TrackIDs, "multimodal") || manifest.Target.Mixture == nil ||
		attestation.RunID != manifest.RunID || attestation.ManifestDigest != manifest.ManifestDigest ||
		attestation.TargetID != manifest.Target.ID || attestation.Mode != ModeLive ||
		attestation.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		attestation.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
		!digestPattern.MatchString(attestation.Digest) {
		return
	}

	cases := records.PlannedCaseIDsByTrack["multimodal"]
	if len(cases) == 0 || !sameEvidenceCaseSet(cases, records.EvaluatedCaseIDsByTrack["multimodal"]) {
		return
	}
	receiptsByCase := make(map[string]int, len(cases))
	seenReceipts := make(map[string]struct{}, len(attestation.Entries))
	for _, entry := range attestation.Entries {
		if !digestPattern.MatchString(entry.BrokerReceipt) {
			return
		}
		if _, duplicate := seenReceipts[entry.BrokerReceipt]; duplicate {
			return
		}
		seenReceipts[entry.BrokerReceipt] = struct{}{}
		if entry.TrackID != "multimodal" {
			continue
		}
		if entry.Operation != workerBrokerRoutedChatCompletion {
			return
		}
		if _, planned := cases[entry.CaseID]; !planned {
			return
		}
		receiptsByCase[entry.CaseID]++
	}
	for caseID := range cases {
		cell := records.CellEvidence["multimodal"][caseID]
		if cell == nil || cell.Unavailable || cell.Rows != 1 || receiptsByCase[caseID] != 1 ||
			len(cell.EvidenceKinds) != 1 {
			return
		}
		if _, exact := cell.EvidenceKinds[normalizedSuiteLiveExecutorID]; !exact {
			return
		}
	}

	// The normalized lineage validator binds every visible media case to the
	// pinned private suite and media manifest. Execution-attestation validation
	// independently joins every row to one routed broker receipt and recomputes
	// hidden-answer quality. A complete matrix therefore qualifies as E4 live
	// multimodal evidence; a later exact-cohort fidelity pair owns the E5 claim.
	levels.ByTrack["multimodal"] = "E4"
}

func deriveLiveDeclaredShiftEvidenceLevels(
	levels *sealedEvidenceLevels,
	manifest RunManifest,
	records recordAttestation,
	executor executorContract,
	attestation *executionAttestation,
) {
	method := records.Methods.Robustness
	if executor.ID != normalizedSuiteLiveExecutorID || !records.validated ||
		!containsTrack(manifest.TrackIDs, "routing") || !method.SourceQualified ||
		method.Passed == nil || len(method.brokerReceipts) == 0 || attestation == nil ||
		attestation.RunID != manifest.RunID || attestation.ManifestDigest != manifest.ManifestDigest ||
		attestation.TargetID != manifest.Target.ID || attestation.Mode != ModeLive ||
		attestation.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		attestation.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
		!digestPattern.MatchString(attestation.Digest) {
		return
	}
	entries := make(map[string]executionAttestationEntry, len(attestation.Entries))
	for _, entry := range attestation.Entries {
		if _, duplicate := entries[entry.BrokerReceipt]; duplicate || !digestPattern.MatchString(entry.BrokerReceipt) {
			return
		}
		entries[entry.BrokerReceipt] = entry
	}
	for receipt, caseID := range method.brokerReceipts {
		entry, present := entries[receipt]
		if !present || entry.Operation != workerBrokerRouterEvaluate ||
			entry.TrackID != "routing" || entry.CaseID != caseID {
			return
		}
	}
	levels.ByTrack["routing"] = "E4"
}

func deriveLiveMoMEvidenceLevels(
	levels *sealedEvidenceLevels,
	manifest RunManifest,
	records recordAttestation,
	executor executorContract,
	attestation *executionAttestation,
) {
	if !validLiveMoMEvidenceSubject(manifest, records, executor, attestation) {
		return
	}
	routingCases := records.PlannedCaseIDsByTrack["routing"]
	poolCases := records.PlannedCaseIDsByTrack["model_pool"]
	jointCases := records.PlannedCaseIDsByTrack["joint"]
	if !liveMoMSelectedCohortsMatch(manifest.TrackIDs, routingCases, poolCases, jointCases) {
		return
	}
	armIDs, validArms := liveMoMArmIDs(manifest.Target.Mixture)
	if !validArms {
		return
	}
	entries, validEntries := countLiveMoMAttestationEntries(attestation, routingCases, poolCases, jointCases, armIDs)
	if !validEntries {
		return
	}

	// Each selected aspect earns its own level from its complete broker-bound
	// case matrix. This keeps routing-only and pool-only diagnostics useful
	// without allowing an incomplete aspect to inherit a sibling's evidence.
	if containsTrack(manifest.TrackIDs, "routing") && completeLiveMoMRouting(records, routingCases, entries) {
		levels.ByTrack["routing"] = "E3"
	}
	if containsTrack(manifest.TrackIDs, "model_pool") && completeLiveMoMModelPool(records, poolCases, armIDs, entries) {
		levels.ByTrack["model_pool"] = "E4"
	}
	if containsTrack(manifest.TrackIDs, "joint") && completeLiveMoMJoint(records, jointCases, entries) {
		levels.ByTrack["joint"] = "E5"
	}
}

func validLiveMoMEvidenceSubject(
	manifest RunManifest,
	records recordAttestation,
	executor executorContract,
	attestation *executionAttestation,
) bool {
	executorID, singleExecutor := manifestExecutorIdentity(manifest)
	return executor.Mode == ModeLive && executor.SuiteClass == executorSuiteRuntime &&
		executor.TargetProfile == targetProfileRuntime && singleExecutor && executorID == executor.ID &&
		records.validated && attestation != nil &&
		(containsTrack(manifest.TrackIDs, "routing") ||
			containsTrack(manifest.TrackIDs, "model_pool") ||
			containsTrack(manifest.TrackIDs, "joint")) &&
		manifest.Target.Mixture != nil && len(manifest.Target.Mixture.ModelArms) >= 2 &&
		attestation.RunID == manifest.RunID && attestation.ManifestDigest == manifest.ManifestDigest &&
		attestation.TargetID == manifest.Target.ID && attestation.Mode == ModeLive &&
		attestation.PolicySnapshotDigest == manifest.PolicySnapshotDigest &&
		attestation.BackendTopologyDigest == manifest.Target.BackendTopologyDigest &&
		digestPattern.MatchString(attestation.Digest)
}

func liveMoMSelectedCohortsMatch(
	trackIDs []TrackID,
	routingCases map[string]struct{},
	poolCases map[string]struct{},
	jointCases map[string]struct{},
) bool {
	byTrack := map[TrackID]map[string]struct{}{
		"routing": routingCases, "model_pool": poolCases, "joint": jointCases,
	}
	var reference map[string]struct{}
	for _, trackID := range trackIDs {
		cases, isMoMTrack := byTrack[trackID]
		if !isMoMTrack {
			continue
		}
		if len(cases) == 0 {
			return false
		}
		if reference != nil && !sameEvidenceCaseSet(reference, cases) {
			return false
		}
		reference = cases
	}
	return reference != nil
}

func liveMoMArmIDs(mixture *ManifestMixture) (map[string]struct{}, bool) {
	armIDs := make(map[string]struct{}, len(mixture.ModelArms))
	for _, arm := range mixture.ModelArms {
		if _, duplicate := armIDs[arm.ID]; arm.ID == "" || duplicate {
			return nil, false
		}
		armIDs[arm.ID] = struct{}{}
	}
	return armIDs, true
}

type liveMoMEntryCounts struct {
	routing map[string]int
	pool    map[string]map[string]int
	joint   map[string]int
}

func countLiveMoMAttestationEntries(
	attestation *executionAttestation,
	routingCases map[string]struct{},
	poolCases map[string]struct{},
	jointCases map[string]struct{},
	armIDs map[string]struct{},
) (liveMoMEntryCounts, bool) {
	counts := liveMoMEntryCounts{
		routing: make(map[string]int, len(routingCases)),
		pool:    make(map[string]map[string]int, len(poolCases)),
		joint:   make(map[string]int, len(jointCases)),
	}
	seenReceipts := make(map[string]struct{}, len(attestation.Entries))
	for _, entry := range attestation.Entries {
		if entry.Operation == workerBrokerListModels {
			continue
		}
		if !digestPattern.MatchString(entry.BrokerReceipt) {
			return liveMoMEntryCounts{}, false
		}
		if _, duplicate := seenReceipts[entry.BrokerReceipt]; duplicate {
			return liveMoMEntryCounts{}, false
		}
		seenReceipts[entry.BrokerReceipt] = struct{}{}
		switch entry.Operation {
		case workerBrokerRouterEvaluate:
			if _, planned := routingCases[entry.CaseID]; !planned || entry.TrackID != "routing" {
				return liveMoMEntryCounts{}, false
			}
			counts.routing[entry.CaseID]++
		case workerBrokerArmChatCompletion:
			if _, planned := poolCases[entry.CaseID]; !planned || entry.TrackID != "model_pool" || entry.ArmID == nil {
				return liveMoMEntryCounts{}, false
			}
			if _, frozen := armIDs[*entry.ArmID]; !frozen {
				return liveMoMEntryCounts{}, false
			}
			if counts.pool[entry.CaseID] == nil {
				counts.pool[entry.CaseID] = make(map[string]int, len(armIDs))
			}
			counts.pool[entry.CaseID][*entry.ArmID]++
		case workerBrokerRoutedChatCompletion:
			if entry.TrackID != "joint" {
				continue
			}
			if _, planned := jointCases[entry.CaseID]; !planned {
				return liveMoMEntryCounts{}, false
			}
			counts.joint[entry.CaseID]++
		}
	}
	return counts, true
}

func completeLiveMoMRouting(records recordAttestation, cases map[string]struct{}, entries liveMoMEntryCounts) bool {
	if !sameEvidenceCaseSet(cases, records.EvaluatedCaseIDsByTrack["routing"]) {
		return false
	}
	for caseID := range cases {
		cell := records.CellEvidence["routing"][caseID]
		if cell == nil || cell.Unavailable || cell.Rows != 1 || entries.routing[caseID] != 1 ||
			!hasExactLiveMoMEvidenceKind(cell, liveMoMRoutingEvidenceKind) {
			return false
		}
	}
	return true
}

func completeLiveMoMModelPool(
	records recordAttestation,
	cases map[string]struct{},
	armIDs map[string]struct{},
	entries liveMoMEntryCounts,
) bool {
	if !sameEvidenceCaseSet(cases, records.EvaluatedCaseIDsByTrack["model_pool"]) {
		return false
	}
	for caseID := range cases {
		cell := records.CellEvidence["model_pool"][caseID]
		if cell == nil || cell.Unavailable || cell.Rows != len(armIDs) || len(entries.pool[caseID]) != len(armIDs) ||
			!hasExactLiveMoMEvidenceKind(cell, liveMoMModelPoolEvidenceKind) {
			return false
		}
		for armID := range armIDs {
			if entries.pool[caseID][armID] != 1 {
				return false
			}
		}
	}
	return true
}

func completeLiveMoMJoint(records recordAttestation, cases map[string]struct{}, entries liveMoMEntryCounts) bool {
	if !sameEvidenceCaseSet(cases, records.EvaluatedCaseIDsByTrack["joint"]) {
		return false
	}
	for caseID := range cases {
		cell := records.CellEvidence["joint"][caseID]
		if cell == nil || cell.Unavailable || cell.Rows != 1 || entries.joint[caseID] != 1 ||
			!hasExactLiveMoMEvidenceKind(cell, liveMoMJointEvidenceKind) {
			return false
		}
	}
	return true
}

func hasExactLiveMoMEvidenceKind(cell *recordCellAttestation, expected string) bool {
	if len(cell.EvidenceKinds) != 1 {
		return false
	}
	_, exact := cell.EvidenceKinds[expected]
	return exact
}

func sameEvidenceCaseSet(left, right map[string]struct{}) bool {
	if len(left) != len(right) {
		return false
	}
	for id := range left {
		if _, present := right[id]; !present {
			return false
		}
	}
	return true
}

func deriveLiveMethodEvidenceLevels(levels *sealedEvidenceLevels, manifest RunManifest, records recordAttestation) {
	if containsTrack(manifest.TrackIDs, "safety") {
		method := records.Methods.HardPolicy
		if method.StaticPassed != nil && method.DynamicPassed != nil && method.ObservationCount > 0 &&
			method.ObservationCount == method.TotalObservationCount {
			levels.ByTrack["safety"] = "E4"
		}
	}
	if containsTrack(manifest.TrackIDs, "agentic") {
		tasks := records.Methods.AgentTask
		recovery := records.Methods.Recovery
		// Task/trajectory outcomes and G6 injected-fault continuity are
		// independent E5 methods. Either can establish an agentic evidence
		// level; G6 still requires the conclusive recovery reducer separately.
		if tasks.Complete || (recovery.Passed != nil && recovery.PairCount == recovery.LedgerTotalPairCount &&
			recovery.PairCount >= minimumRecoveryPairCount && recovery.DistinctSeedCount >= minimumRecoveryDistinctSeedCount) {
			levels.ByTrack["agentic"] = "E5"
		}
	}
	if containsTrack(manifest.TrackIDs, "preference") {
		method := records.Methods.Production
		if method.CandidateSafe != nil && method.AssignmentCount == method.LedgerTotalAssignmentCount &&
			method.AssignmentCount >= minimumProductionAssignmentCount {
			levels.ByTrack["preference"] = "E5"
		}
	}
}

func validatedNormalizedCaseIdentities(
	runDir string,
	manifest RunManifest,
	executor executorContract,
) (map[string]normalizedLineageIdentity, error) {
	lineageBytes, err := readEvidenceBytes(filepath.Join(runDir, "lineage.json"), maxStructuredArtifactBytes)
	if err != nil {
		return nil, fmt.Errorf("read lineage evidence: %w", err)
	}
	document, err := decodeLineageDocument(lineageBytes)
	if err != nil {
		return nil, err
	}
	identities, err := validateNormalizedSuiteLineage(runDir, manifest, document.NormalizedSuiteIdentities, executor)
	if err != nil {
		return nil, err
	}
	if identities == nil {
		return nil, fmt.Errorf("%w: normalized evidence omits suite-bound case identities", ErrInvalid)
	}
	caseIdentities := make(map[string]normalizedLineageIdentity, len(identities.CaseIdentities))
	for _, identity := range identities.CaseIdentities {
		caseIdentities[identity.OpaqueID] = identity
	}
	return caseIdentities, nil
}

func validateNormalizedCaseTrackPlans(
	runDir string,
	manifest RunManifest,
	records recordAttestation,
	qualification suiteGateQualification,
	caseIdentities map[string]normalizedLineageIdentity,
) error {
	if len(caseIdentities) != len(records.CaseIDs) {
		return fmt.Errorf("%w: normalized case plan is not fully suite-bound", ErrInvalid)
	}
	suiteRoot := filepath.Join(filepath.Dir(filepath.Dir(runDir)), "suites")
	documents, err := loadInstalledLineageSuites(suiteRoot, manifest)
	if err != nil {
		return err
	}
	plansBySuite := make(map[string]map[string]installedVisibleCasePlan, len(documents))
	for caseID := range records.CaseIDs {
		identity, bound := caseIdentities[caseID]
		document, installedDocument := documents[identity.SuiteID]
		suiteTrackIDs, installed := qualification.suiteTrackIDs[identity.SuiteID]
		if !bound || !installed {
			return fmt.Errorf("%w: normalized case plan references an uninstalled suite", ErrInvalid)
		}
		if !installedDocument || !reflect.DeepEqual(suiteTrackIDs, document.Manifest.TrackIDs) {
			return fmt.Errorf("%w: normalized case plan qualification does not match its installed suite", ErrInvalid)
		}
		sourcePlans, loaded := plansBySuite[identity.SuiteID]
		if !loaded {
			sourcePlans, err = installedVisibleCasePlans(suiteRoot, document)
			if err != nil {
				return err
			}
			plansBySuite[identity.SuiteID] = sourcePlans
		}
		sourcePlan, knownSource := sourcePlans[identity.SourceID]
		if !knownSource || records.CaseModalities[caseID] != sourcePlan.Modality {
			return fmt.Errorf("%w: normalized case %q does not match its installed source modality", ErrInvalid, caseID)
		}
		expected := make([]TrackID, 0, len(manifest.TrackIDs))
		for _, trackID := range manifest.TrackIDs {
			if containsTrack(sourcePlan.TrackIDs, trackID) {
				expected = append(expected, trackID)
			}
		}
		if len(expected) == 0 || !reflect.DeepEqual(records.CaseTrackIDs[caseID], expected) {
			return fmt.Errorf("%w: normalized case %q track plan does not match its executable suite", ErrInvalid, caseID)
		}
	}
	return nil
}

func weakestSelectedTrackLevel(trackIDs []TrackID, levels map[TrackID]EvidenceLevel) EvidenceLevel {
	if len(trackIDs) == 0 {
		return "E0"
	}
	weakest := levels[trackIDs[0]]
	for _, trackID := range trackIDs[1:] {
		if evidenceLevelRank(levels[trackID]) < evidenceLevelRank(weakest) {
			weakest = levels[trackID]
		}
	}
	return weakest
}
