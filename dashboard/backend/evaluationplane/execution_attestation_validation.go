package evaluationplane

import (
	"fmt"
	"path/filepath"
	"reflect"
	"time"
)

// persistExecutionAttestationDuringPublication validates and publishes the
// server transcript while the caller owns the lifecycle/evidence transaction.
func (s *Service) persistExecutionAttestationDuringPublication(
	runID string,
	transcript *brokerExecutionTranscript,
) (string, error) {
	manifest, _, err := s.readDurableManifest(runID)
	if err != nil {
		return "", err
	}
	if manifest.Mode != ModeLive {
		if transcript != nil && len(transcript.Entries) != 0 {
			return "", fmt.Errorf("%w: replay execution cannot contain broker observations", ErrInvalid)
		}
		return "", nil
	}
	registry, err := s.registrySnapshot()
	if err != nil {
		return "", err
	}
	executionContract, err := registry.executionContracts().resolve(manifest)
	if err != nil {
		return "", err
	}
	if transcript == nil {
		return "", fmt.Errorf("%w: live execution omitted its server broker transcript", ErrInvalid)
	}
	run, err := s.store.GetRun(runID)
	if err != nil {
		return "", err
	}
	if validationErr := validateBrokerTranscriptIdentity(run, manifest, *transcript); validationErr != nil {
		return "", validationErr
	}
	runDir, err := s.store.checkedRunDir(runID)
	if err != nil {
		return "", err
	}
	if executionContract.Executor.NormalizedSuite {
		if workloadErr := validateNormalizedWorkloadFromLineage(runDir, manifest, executionContract.Executor); workloadErr != nil {
			return "", workloadErr
		}
	}
	caseLimit, err := manifestVisibleCaseLimit(manifest, executionContract.Executor)
	if err != nil {
		return "", err
	}
	cases, err := validateVisibleCaseSet(filepath.Join(runDir, "cases.jsonl"), caseLimit, manifest.TrackIDs)
	if err != nil {
		return "", err
	}
	grading, err := loadGradingCases(filepath.Join(runDir, "grading-cases.jsonl"), cases.IDs)
	if err != nil {
		return "", err
	}
	records, err := s.loadPrivateComparisonRecords(runID)
	if err != nil {
		return "", err
	}
	entries, err := validateBrokerRecordBindings(
		manifest, transcript.Entries, records, cases, grading,
		transcript.StartedAt, transcript.CompletedAt,
	)
	if err != nil {
		return "", err
	}
	attestation := executionAttestation{
		SchemaVersion: SchemaVersion, ContractVersion: executionAttestationContractVersion,
		RunID: runID, ManifestDigest: manifest.ManifestDigest, TargetID: manifest.Target.ID,
		Mode: manifest.Mode, PolicySnapshotDigest: manifest.PolicySnapshotDigest,
		BackendTopologyDigest: manifest.Target.BackendTopologyDigest,
		StartedAt:             transcript.StartedAt.UTC(), CompletedAt: transcript.CompletedAt.UTC(), Entries: entries,
	}
	attestation.Digest, err = executionAttestationDigest(attestation)
	if err != nil {
		return "", err
	}
	if err := s.store.writeLifecycleBoundExecutionAttestationDuringPublication(attestation, manifest); err != nil {
		return "", err
	}
	return attestation.Digest, nil
}

func validateBrokerTranscriptIdentity(run Run, manifest RunManifest, transcript brokerExecutionTranscript) error {
	if transcript.SchemaVersion != SchemaVersion || transcript.ContractVersion != executionAttestationContractVersion ||
		transcript.RunID != run.ID || transcript.ManifestDigest != manifest.ManifestDigest ||
		transcript.TargetID != manifest.Target.ID || transcript.Mode != ModeLive ||
		transcript.PolicySnapshotDigest != manifest.PolicySnapshotDigest ||
		transcript.BackendTopologyDigest != manifest.Target.BackendTopologyDigest ||
		!digestPattern.MatchString(transcript.BackendTopologyDigest) || run.StartedAt == nil ||
		transcript.StartedAt.Before(run.StartedAt.UTC()) || transcript.CompletedAt.Before(transcript.StartedAt) ||
		transcript.CompletedAt.After(time.Now().UTC().Add(time.Second)) ||
		len(transcript.Entries) == 0 || len(transcript.Entries) > maxWorkerBrokerRequests {
		return fmt.Errorf("%w: live broker transcript does not match the immutable run", ErrInvalid)
	}
	return nil
}

func validateControlledPairObservation(
	manifest RunManifest,
	entry executionAttestationEntry,
	transcriptStartedAt time.Time,
	transcriptCompletedAt time.Time,
) error {
	pair := entry.ControlledPair
	if pair == nil || pair.ContractVersion != controlledPairProtocolVersion ||
		pair.Protocol != controlledPairInterleaveABBA || !validClientRequestID(pair.SessionID) ||
		(pair.Role != controlledPairRoleBaseline && pair.Role != controlledPairRoleCandidate) ||
		pair.VariantManifestDigest != manifest.ManifestDigest ||
		pair.AttemptID != entry.AttemptID || !digestPattern.MatchString(pair.CoordinateDigest) ||
		!digestPattern.MatchString(pair.BlockID) || pair.ObservedAt.IsZero() || pair.CompletedAt.IsZero() ||
		pair.ObservedAt.Before(transcriptStartedAt) || pair.CompletedAt.Before(pair.ObservedAt) ||
		pair.CompletedAt.After(transcriptCompletedAt) {
		return fmt.Errorf("observation envelope is not bound to the frozen run")
	}
	armID := ""
	if entry.Operation == workerBrokerArmChatCompletion {
		armID = stringValue(entry.ArmID)
		if armID == "" {
			return fmt.Errorf("model-pool observation omits its frozen arm")
		}
	}
	coordinate := controlledPairCoordinate{
		trackID: entry.TrackID, caseID: entry.CaseID, attemptID: entry.AttemptID,
		operation: entry.Operation, armID: armID,
	}
	if pair.CoordinateDigest != digestString("controlled-pair-coordinate:"+coordinate.canonical()) {
		return fmt.Errorf("coordinate digest differs from the attested attempt")
	}
	expectedLoad, err := controlledPairRequestLoad(manifest, workerBrokerRequest{
		Operation: entry.Operation, TrackID: entry.TrackID, CaseID: entry.CaseID, AttemptID: entry.AttemptID,
	})
	if err != nil || !reflect.DeepEqual(pair.Load, expectedLoad) {
		return fmt.Errorf("load context differs from the frozen attempt")
	}
	switch pair.Cohort {
	case campaignArmCohortPaired:
		if (pair.Order != "AB" && pair.Order != "BA") || (pair.Position != 1 && pair.Position != 2) {
			return fmt.Errorf("paired block order or position is invalid")
		}
		if (pair.Order == "AB" && ((pair.Role == controlledPairRoleBaseline) != (pair.Position == 1))) ||
			(pair.Order == "BA" && ((pair.Role == controlledPairRoleCandidate) != (pair.Position == 1))) {
			return fmt.Errorf("paired block role contradicts its AB/BA order")
		}
	case campaignArmCohortBaselineOnly:
		if entry.Operation != workerBrokerArmChatCompletion || pair.Role != controlledPairRoleBaseline ||
			pair.Order != "A" || pair.Position != 1 {
			return fmt.Errorf("baseline-only block is invalid")
		}
	case campaignArmCohortCandidateOnly:
		if entry.Operation != workerBrokerArmChatCompletion || pair.Role != controlledPairRoleCandidate ||
			pair.Order != "B" || pair.Position != 1 {
			return fmt.Errorf("candidate-only block is invalid")
		}
	default:
		return fmt.Errorf("observation cohort is invalid")
	}
	return nil
}

func validateBrokerMixtureBinding(mixture *ManifestMixture, entry executionAttestationEntry) error {
	switch entry.Operation {
	case workerBrokerAgentTaskLedger, workerBrokerFaultRecoveryLedger, workerBrokerHardPolicyLedger, workerBrokerProductionExperimentLedger:
		return nil
	case workerBrokerRouterEvaluate, workerBrokerRoutedChatCompletion, workerBrokerArmChatCompletion:
	default:
		return fmt.Errorf("broker operation has no mixture binding contract")
	}
	if mixture == nil || entry.RequestedModel == nil {
		return fmt.Errorf("broker operation omits its frozen mixture request model")
	}
	if entry.Operation == workerBrokerArmChatCompletion {
		arm, present := frozenArmByRequestModel(mixture.ModelArms, *entry.RequestedModel)
		if !present || *entry.RequestedModel == mixture.EntrypointModel ||
			containsString(mixture.Aliases, *entry.RequestedModel) || entry.ArmID == nil || *entry.ArmID != arm.ID {
			return fmt.Errorf("arm chat request is outside the frozen mixture pool")
		}
		if entry.SelectedModel != nil {
			selectedArmID, resolved := frozenArmID(mixture.ModelArms, *entry.SelectedModel)
			if !resolved || selectedArmID != arm.ID {
				return fmt.Errorf("arm chat response crossed its requested frozen arm")
			}
		}
		if header := entry.Headers["x-vsr-selected-model"]; header != "" {
			headerArmID, resolved := frozenArmID(mixture.ModelArms, header)
			if !resolved || headerArmID != arm.ID {
				return fmt.Errorf("arm chat response header crossed its requested frozen arm")
			}
		}
		if entry.Success && !completeChatObservation(entry) {
			return fmt.Errorf("successful arm chat omitted response or usage attestation")
		}
		return nil
	}
	if *entry.RequestedModel != mixture.EntrypointModel {
		return fmt.Errorf("routed request does not use the frozen mixture entrypoint")
	}
	if entry.Recipe == nil || *entry.Recipe != mixture.RecipeName {
		return fmt.Errorf("routed response does not bind the frozen mixture recipe")
	}
	if header := entry.Headers["x-vsr-selected-recipe"]; header != "" && header != mixture.RecipeName {
		return fmt.Errorf("routed response header disagrees with the frozen mixture recipe")
	}
	if entry.SelectedModel != nil {
		selectedArmID, resolved := frozenArmID(mixture.ModelArms, *entry.SelectedModel)
		if !resolved || entry.ArmID == nil || selectedArmID != *entry.ArmID {
			return fmt.Errorf("routed response selected outside the frozen mixture pool")
		}
	}
	if header := entry.Headers["x-vsr-selected-model"]; header != "" {
		headerArmID, resolved := frozenArmID(mixture.ModelArms, header)
		if !resolved || entry.ArmID == nil || headerArmID != *entry.ArmID {
			return fmt.Errorf("routed response header disagrees with its resolved frozen arm")
		}
	}
	if !entry.Success {
		if entry.Operation == workerBrokerRoutedChatCompletion &&
			(entry.SelectionStatus != nil || entry.SelectionMethod != nil || entry.Algorithm != nil) {
			return fmt.Errorf("failed routed chat contains a successful selection projection")
		}
		if entry.ArmID != nil {
			if armID, present := frozenArmID(mixture.ModelArms, *entry.ArmID); !present || armID != *entry.ArmID {
				return fmt.Errorf("failed routed request resolved outside the frozen mixture pool")
			}
		}
		return nil
	}
	if entry.Operation == workerBrokerRouterEvaluate && entry.RoutingRecipeDecision != nil {
		switch entry.RoutingRecipeDecision.SelectionStatus {
		case "abstained", "error", "unavailable":
			if entry.SelectedModel != nil || entry.ArmID != nil {
				return fmt.Errorf("non-final routing decision claims a selected frozen arm")
			}
			return nil
		}
	}
	if entry.SelectedModel == nil || entry.ArmID == nil {
		return fmt.Errorf("successful routed request omitted its resolved frozen arm")
	}
	if entry.Algorithm == nil || !mixtureAuthorizesSelection(mixture, entry) {
		return fmt.Errorf("routed response selection is outside the frozen mixture decision boundary")
	}
	if header := entry.Headers["x-vsr-selected-algorithm"]; header != "" && header != *entry.Algorithm {
		return fmt.Errorf("routed response algorithm header disagrees with its attestation")
	}
	if header := entry.Headers["x-vsr-selected-decision"]; header != "" &&
		(entry.DecisionName == nil || header != *entry.DecisionName) {
		return fmt.Errorf("routed response decision header disagrees with its attestation")
	}
	if entry.Operation == workerBrokerRoutedChatCompletion &&
		(entry.SelectionStatus == nil || *entry.SelectionStatus != "selected" ||
			entry.SelectionMethod == nil || *entry.SelectionMethod != *entry.Algorithm) {
		return fmt.Errorf("successful routed chat does not bind the server-owned selection projection")
	}
	if entry.Operation == workerBrokerRoutedChatCompletion && !completeChatObservation(entry) {
		return fmt.Errorf("successful routed chat omitted response or usage attestation")
	}
	return nil
}

func mixtureAuthorizesSelection(mixture *ManifestMixture, entry executionAttestationEntry) bool {
	if entry.Algorithm == nil || entry.ArmID == nil {
		return false
	}
	// A provider-default selection has no matched decision, so its frozen
	// fallback arm is the complete authorization boundary. A selector fallback
	// is not a wildcard: it must still match one exact frozen decision below.
	if mixture.FallbackArmID != "" && *entry.ArmID == mixture.FallbackArmID &&
		*entry.Algorithm == "default" && entry.DecisionName == nil {
		return true
	}
	if entry.DecisionName == nil {
		return false
	}
	for _, decision := range mixture.Decisions {
		if decision.Algorithm != *entry.Algorithm || decision.Name != *entry.DecisionName {
			continue
		}
		if containsString(decision.ArmIDs, *entry.ArmID) {
			return true
		}
	}
	return false
}

func completeChatObservation(entry executionAttestationEntry) bool {
	return entry.ResponseContentDigest != nil && entry.InputTokens != nil && entry.OutputTokens != nil
}

func frozenArmByRequestModel(arms []ModelArm, model string) (ModelArm, bool) {
	for _, arm := range arms {
		if arm.Model == model {
			return arm, true
		}
	}
	return ModelArm{}, false
}

func frozenArmID(arms []ModelArm, identity string) (string, bool) {
	for _, arm := range arms {
		if arm.ID == identity || arm.Model == identity {
			return arm.ID, true
		}
	}
	return "", false
}

func validateMixtureRecordDensity(manifest RunManifest, records []executionRecordEvidence, cases visibleCaseSet) error {
	poolSelected := containsTrack(manifest.TrackIDs, "model_pool")
	jointSelected := containsTrack(manifest.TrackIDs, "joint")
	if !poolSelected && !jointSelected {
		return nil
	}
	if manifest.Target.Mixture == nil || len(manifest.Target.Mixture.ModelArms) == 0 {
		return fmt.Errorf("%w: mixture evaluation has no frozen model pool", ErrInvalid)
	}
	poolCounts := make(map[string]map[string]int, len(cases.CaseIDsByTrack["model_pool"]))
	jointCounts := make(map[string]int, len(cases.CaseIDsByTrack["joint"]))
	for _, record := range records {
		switch record.TrackID {
		case "model_pool":
			if _, planned := cases.CaseIDsByTrack["model_pool"][record.CaseID]; !planned {
				return fmt.Errorf("%w: model_pool record %q is outside the visible case matrix", ErrInvalid, record.ID)
			}
			if record.ArmID == nil {
				return fmt.Errorf("%w: model_pool record %q omits its frozen arm", ErrInvalid, record.ID)
			}
			armID, present := frozenArmID(manifest.Target.Mixture.ModelArms, *record.ArmID)
			if !present || armID != *record.ArmID {
				return fmt.Errorf("%w: model_pool record %q names an arm outside the frozen mixture", ErrInvalid, record.ID)
			}
			if poolCounts[record.CaseID] == nil {
				poolCounts[record.CaseID] = make(map[string]int, len(manifest.Target.Mixture.ModelArms))
			}
			poolCounts[record.CaseID][armID]++
		case "joint":
			if _, planned := cases.CaseIDsByTrack["joint"][record.CaseID]; !planned {
				return fmt.Errorf("%w: joint record %q is outside the visible case plan", ErrInvalid, record.ID)
			}
			jointCounts[record.CaseID]++
		}
	}
	if poolSelected {
		for caseID := range cases.CaseIDsByTrack["model_pool"] {
			for _, arm := range manifest.Target.Mixture.ModelArms {
				if poolCounts[caseID][arm.ID] != 1 {
					return fmt.Errorf("%w: model_pool requires exactly one broker record for case %q and frozen arm %q", ErrInvalid, caseID, arm.ID)
				}
			}
			if len(poolCounts[caseID]) != len(manifest.Target.Mixture.ModelArms) {
				return fmt.Errorf("%w: model_pool record matrix contains an extra frozen-arm coordinate", ErrInvalid)
			}
		}
	}
	if jointSelected {
		for caseID := range cases.CaseIDsByTrack["joint"] {
			if jointCounts[caseID] != 1 {
				return fmt.Errorf("%w: joint requires exactly one routed broker record for case %q", ErrInvalid, caseID)
			}
		}
	}
	return nil
}

func validateBrokerRecord(
	entry executionAttestationEntry,
	record executionRecordEvidence,
	cases visibleCaseSet,
	grading gradingCaseEvidence,
	arms []ModelArm,
	poolOracleArmIDs map[string]struct{},
	generationSeed int64,
) error {
	if entry.TrackID != record.TrackID || entry.CaseID != record.CaseID || entry.AttemptID != record.AttemptID {
		return fmt.Errorf("broker evidence identity differs from the record")
	}
	expectedOperation := ""
	switch record.TrackID {
	case "routing":
		expectedOperation = workerBrokerRouterEvaluate
	case "model_pool":
		expectedOperation = workerBrokerArmChatCompletion
	case "joint", "multimodal", "capacity":
		expectedOperation = workerBrokerRoutedChatCompletion
	}
	if expectedOperation == "" || entry.Operation != expectedOperation {
		return fmt.Errorf("broker operation does not own the record track")
	}
	if err := validateBrokerCaseRequestBinding(entry, record, cases, generationSeed); err != nil {
		return err
	}
	if record.Success == nil || *record.Success != entry.Success ||
		(record.Status == "succeeded") != entry.Success || record.Status == "unavailable" {
		return fmt.Errorf("record outcome differs from the broker response")
	}
	expectedLatency := float64(entry.LatencyMicroseconds) / 1000
	if record.LatencyMS == nil || *record.LatencyMS != expectedLatency {
		return fmt.Errorf("record latency differs from the server observation")
	}
	quality := serverObservedQuality(entry, record.TrackID, grading, poolOracleArmIDs)
	if !sameOptionalFloat(record.Quality, quality) {
		return fmt.Errorf("record quality differs from server-side hidden-label grading")
	}
	switch record.TrackID {
	case "routing":
		if !sameOptionalString(record.SelectedArmID, entry.ArmID) ||
			!sameOptionalString(record.SelectionStatus, entry.SelectionStatus) ||
			!sameOptionalString(record.SelectionMethod, entry.SelectionMethod) ||
			!sameOptionalString(record.Recipe, entry.Recipe) ||
			!sameOptionalString(record.DecisionName, entry.DecisionName) ||
			!sameOptionalString(record.Algorithm, entry.Algorithm) {
			return fmt.Errorf("routing decision differs from the server response")
		}
	case "model_pool":
		if !sameOptionalString(record.ArmID, entry.ArmID) || record.SelectedArmID != nil {
			return fmt.Errorf("model_pool arm differs from the server-bound direct request")
		}
		if err := validateChatUsageAndCost(record, entry, arms); err != nil {
			return err
		}
	case "joint":
		if record.ArmID != nil || !sameOptionalString(record.SelectedArmID, entry.ArmID) ||
			!sameOptionalString(record.SelectionStatus, entry.SelectionStatus) ||
			!sameOptionalString(record.SelectionMethod, entry.SelectionMethod) ||
			!sameOptionalString(record.Recipe, entry.Recipe) ||
			!sameOptionalString(record.DecisionName, entry.DecisionName) ||
			!sameOptionalString(record.Algorithm, entry.Algorithm) {
			return fmt.Errorf("joint routed decision differs from the server response")
		}
		if err := validateChatUsageAndCost(record, entry, arms); err != nil {
			return err
		}
	case "multimodal":
		if cases.Modalities[record.CaseID] == "text" || record.Modality == nil ||
			*record.Modality != cases.Modalities[record.CaseID] {
			return fmt.Errorf("multimodal record does not match the visible case")
		}
		if err := validateChatUsageAndCost(record, entry, arms); err != nil {
			return err
		}
	case "capacity":
		if err := validateChatUsageAndCost(record, entry, arms); err != nil {
			return err
		}
	default:
		return fmt.Errorf("record track has no server broker attestation contract")
	}
	return nil
}

func validateBrokerCaseRequestBinding(
	entry executionAttestationEntry,
	record executionRecordEvidence,
	cases visibleCaseSet,
	generationSeed int64,
) error {
	messageDigest, planned := cases.MessageDigests[record.CaseID]
	if !planned || entry.RequestedModel == nil {
		return fmt.Errorf("broker request omits its server-sealed case input")
	}
	expected, err := brokerRequestDigestForMessages(entry.Operation, *entry.RequestedModel, messageDigest, generationSeed)
	if err != nil || entry.RequestDigest != expected {
		return fmt.Errorf("broker request payload differs from the server-sealed visible case")
	}
	return nil
}

func validateChatUsageAndCost(record executionRecordEvidence, entry executionAttestationEntry, arms []ModelArm) error {
	if !reflect.DeepEqual(record.InputTokens, entry.InputTokens) ||
		!reflect.DeepEqual(record.OutputTokens, entry.OutputTokens) {
		return fmt.Errorf("chat token accounting differs from the server response")
	}
	if !sameOptionalFloat(record.RuntimeCost, serverRuntimeCost(entry, arms)) {
		return fmt.Errorf("chat runtime cost differs from the server-owned frozen mixture pricing")
	}
	return nil
}
