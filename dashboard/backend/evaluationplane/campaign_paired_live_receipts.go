package evaluationplane

import "fmt"

// campaignAttestedObservations binds every private result to exactly one
// server-issued broker receipt. Controlled-pair metadata is read only from the
// sealed execution attestation; browser-supplied result fields cannot create it.
func campaignAttestedObservations(role string, evidence campaignRunEvidence) ([]campaignAttestedObservation, error) {
	if evidence.attestation == nil || evidence.anchor.ExecutionAttestationDigest == "" ||
		evidence.attestation.Digest != evidence.anchor.ExecutionAttestationDigest ||
		evidence.attestation.RunID != evidence.report.Run.ID ||
		evidence.attestation.ManifestDigest != evidence.anchor.ManifestSemanticDigest ||
		evidence.attestation.TargetID != evidence.report.Run.TargetID || evidence.attestation.Mode != ModeLive ||
		evidence.attestation.PolicySnapshotDigest != evidence.report.Provenance.PolicySnapshotDigest {
		return nil, fmt.Errorf("%w: %s lacks exact server-attested execution provenance", ErrInvalid, role)
	}
	entries := make(map[string]executionAttestationEntry)
	for _, entry := range evidence.attestation.Entries {
		if entry.Operation == workerBrokerListModels {
			continue
		}
		if _, duplicate := entries[entry.BrokerReceipt]; duplicate {
			return nil, fmt.Errorf("%w: %s execution attestation duplicates a broker receipt", ErrInvalid, role)
		}
		entries[entry.BrokerReceipt] = entry
	}
	if len(entries) != len(evidence.records) {
		return nil, fmt.Errorf("%w: %s private records do not exactly cover its execution attestation", ErrInvalid, role)
	}
	observations := make([]campaignAttestedObservation, 0, len(evidence.records))
	used := make(map[string]bool, len(entries))
	for _, record := range evidence.records {
		if record.BrokerReceipt == nil || used[*record.BrokerReceipt] {
			return nil, fmt.Errorf("%w: %s private record lacks a unique broker receipt", ErrInvalid, role)
		}
		entry, ok := entries[*record.BrokerReceipt]
		latency := float64(entry.LatencyMicroseconds) / 1000
		requireControlledPair := role == "g3_baseline" || role == "g3_candidate"
		expectedControlledRole := controlledPairRoleBaseline
		if role == "g3_candidate" {
			expectedControlledRole = controlledPairRoleCandidate
		}
		if requireControlledPair && (entry.ControlledPair == nil || entry.ControlledPair.Role != expectedControlledRole) {
			return nil, fmt.Errorf(
				"%w: %s observation lacks server-owned controlled pair provenance",
				ErrInvalid, role,
			)
		}
		if !ok || entry.TrackID != record.TrackID || entry.CaseID != record.CaseID || entry.AttemptID != record.AttemptID ||
			record.Success == nil || *record.Success != entry.Success || (record.Status == "succeeded") != entry.Success ||
			(record.Status != "succeeded" && record.Status != "failed") ||
			!campaignOperationOwnsTrack(entry.Operation, record.TrackID) ||
			!sameOptionalFloat(record.Quality, entry.Quality) || record.LatencyMS == nil || *record.LatencyMS != latency ||
			!campaignRecordMatchesAttestedArm(record, entry) {
			return nil, fmt.Errorf("%w: %s private record differs from its server attestation", ErrInvalid, role)
		}
		used[*record.BrokerReceipt] = true
		observations = append(observations, campaignAttestedObservation{
			trackID: record.TrackID, caseID: entry.CaseID, attemptID: entry.AttemptID,
			operation: entry.Operation, armID: stringValue(entry.ArmID), concurrency: record.Concurrency,
			modality: record.Modality, loadPhase: record.LoadPhase, loadRepeat: record.LoadRepetition,
			loadIndex: record.LoadRequestIndex, success: entry.Success, quality: entry.Quality, latencyMS: latency,
			controlledPair: entry.ControlledPair,
		})
	}
	for receipt := range entries {
		if !used[receipt] {
			return nil, fmt.Errorf("%w: %s execution attestation has an unbound operation", ErrInvalid, role)
		}
	}
	return observations, nil
}

func campaignRecordMatchesAttestedArm(record executionRecordEvidence, entry executionAttestationEntry) bool {
	switch record.TrackID {
	case "routing", "joint":
		return record.ArmID == nil && sameOptionalString(record.SelectedArmID, entry.ArmID)
	case "model_pool":
		return record.SelectedArmID == nil && sameOptionalString(record.ArmID, entry.ArmID)
	default:
		return true
	}
}

func campaignOperationOwnsTrack(operation string, trackID TrackID) bool {
	switch trackID {
	case "routing":
		return operation == workerBrokerRouterEvaluate
	case "model_pool":
		return operation == workerBrokerArmChatCompletion
	case "joint", "multimodal", "capacity":
		return operation == workerBrokerRoutedChatCompletion
	default:
		return false
	}
}
