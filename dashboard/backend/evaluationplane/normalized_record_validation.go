package evaluationplane

import (
	"fmt"
	"strings"
)

var normalizedReplayEvidenceCeilings = map[string]struct{}{
	"E0": {}, "E1": {}, "E2": {}, "E3": {}, "E4": {}, "E5": {},
}

// validateNormalizedReplayDiagnosticRecord mirrors the fixed worker's typed
// record contract. It protects server-reduced diagnostics; it does not turn
// those diagnostics into a complete G4, G6, or G9 release-gate verdict.
func validateNormalizedReplayDiagnosticRecord(record executionRecordEvidence, executor executorContract) error {
	if !executor.RecordedNormalizedSource {
		return nil
	}
	prefix := executor.ID + ";ceiling="
	if record.EvidenceKind == nil || !strings.HasPrefix(*record.EvidenceKind, prefix) {
		return fmt.Errorf("normalized replay record omits its registered evidence kind")
	}
	ceiling := strings.TrimPrefix(*record.EvidenceKind, prefix)
	if _, valid := normalizedReplayEvidenceCeilings[ceiling]; !valid {
		return fmt.Errorf("normalized replay evidence_kind has an invalid ceiling")
	}
	switch record.TrackID {
	case "routing", "agentic", "preference":
	default:
		return nil
	}
	if record.Status == "unavailable" {
		if record.Error == nil || strings.TrimSpace(*record.Error) == "" {
			return fmt.Errorf("unavailable normalized replay evidence requires a reason")
		}
		if normalizedDiagnosticValuesPresent(record) {
			return fmt.Errorf("unavailable normalized replay evidence cannot carry diagnostic values")
		}
		return nil
	}
	if record.Error != nil {
		return fmt.Errorf("evaluated normalized replay evidence cannot carry an error")
	}
	switch record.TrackID {
	case "routing":
		return validateNormalizedRoutingRecord(record, executor.ID)
	case "agentic":
		return validateNormalizedAgenticRecord(record)
	default:
		return validateNormalizedPreferenceRecord(record, ceiling)
	}
}

func normalizedDiagnosticValuesPresent(record executionRecordEvidence) bool {
	switch record.TrackID {
	case "routing":
		return record.SelectedArmID != nil || record.SelectionStatus != nil || record.SelectionMethod != nil ||
			record.Success != nil || record.Quality != nil || record.Fallback != nil || record.LatencyMS != nil
	case "agentic":
		return record.SelectedArmID != nil || record.Success != nil || record.Quality != nil ||
			record.TrajectorySteps != nil || record.ToolCalls != nil || record.InvalidToolCalls != nil ||
			record.PrivacyViolations != nil
	default:
		return record.SelectedArmID != nil || record.Success != nil || record.Quality != nil ||
			record.PreferenceMatch != nil || record.BehaviorPropensity != nil
	}
}

func validateNormalizedSuccessStatus(record executionRecordEvidence) error {
	if record.Success == nil {
		return fmt.Errorf("evaluated normalized replay evidence requires success")
	}
	expected := "failed"
	if *record.Success {
		expected = "succeeded"
	}
	if record.Status != expected {
		return fmt.Errorf("normalized replay status must agree with success")
	}
	return nil
}

func validateNormalizedRoutingRecord(record executionRecordEvidence, executorID string) error {
	if err := validateNormalizedSuccessStatus(record); err != nil {
		return err
	}
	if record.SelectionStatus == nil || record.SelectionMethod == nil ||
		*record.SelectionMethod != executorID || record.Fallback == nil {
		return fmt.Errorf("normalized routing evidence lacks typed decision facts")
	}
	fallback := *record.SelectionStatus == "fallback"
	if *record.Fallback != fallback {
		return fmt.Errorf("normalized routing fallback facts disagree")
	}
	if (*record.SelectionStatus == "selected" || fallback) && record.SelectedArmID == nil {
		return fmt.Errorf("normalized selected routing evidence lacks an action")
	}
	return nil
}

func validateNormalizedAgenticRecord(record executionRecordEvidence) error {
	if err := validateNormalizedSuccessStatus(record); err != nil {
		return err
	}
	if record.Quality == nil || record.TrajectorySteps == nil || record.ToolCalls == nil ||
		record.InvalidToolCalls == nil || record.PrivacyViolations == nil {
		return fmt.Errorf("normalized agentic evidence lacks typed trajectory facts")
	}
	if *record.InvalidToolCalls > *record.ToolCalls {
		return fmt.Errorf("invalid tool calls cannot exceed tool calls")
	}
	if *record.ToolCalls > *record.TrajectorySteps {
		return fmt.Errorf("tool-call steps cannot exceed trajectory steps")
	}
	return nil
}

func validateNormalizedPreferenceRecord(record executionRecordEvidence, ceiling string) error {
	if record.Status != "succeeded" || record.Success == nil || !*record.Success {
		return fmt.Errorf("normalized preference observations must be successful")
	}
	if record.SelectedArmID == nil || record.Quality == nil || record.PreferenceMatch == nil {
		return fmt.Errorf("normalized preference evidence lacks typed outcome facts")
	}
	if ceiling == "E5" && record.BehaviorPropensity == nil {
		return fmt.Errorf("E5 normalized preference evidence requires propensity")
	}
	return nil
}
