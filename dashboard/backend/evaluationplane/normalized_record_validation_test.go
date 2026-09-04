package evaluationplane

import (
	"strings"
	"testing"
)

func normalizedDiagnosticRecord(track TrackID, ceiling string) executionRecordEvidence {
	kind := normalizedSuiteExecutorID + ";ceiling=" + ceiling
	return executionRecordEvidence{
		SchemaVersion: SchemaVersion,
		ID:            string(track) + "-record",
		TrackID:       track,
		CaseID:        "case-1",
		AttemptID:     string(track) + "-attempt",
		Status:        "succeeded",
		EvidenceKind:  &kind,
	}
}

func TestNormalizedReplayDiagnosticRecordsRequireTypedFacts(t *testing.T) {
	executor := builtinExecutorContractForTest(t, normalizedSuiteExecutorID)
	t.Run("routing", func(t *testing.T) {
		record := normalizedDiagnosticRecord("routing", "E4")
		selectionStatus, method := "selected", normalizedSuiteExecutorID
		selected := "arm-a"
		record.SelectedArmID, record.SelectionStatus, record.SelectionMethod = &selected, &selectionStatus, &method
		record.Success, record.Fallback, record.Quality = boolPointer(true), boolPointer(false), floatPointer(1)
		if err := validateNormalizedReplayDiagnosticRecord(record, executor); err != nil {
			t.Fatalf("valid routing diagnostics rejected: %v", err)
		}
		record.Fallback = nil
		if err := validateNormalizedReplayDiagnosticRecord(record, executor); err == nil || !strings.Contains(err.Error(), "typed decision facts") {
			t.Fatalf("missing routing fact error=%v", err)
		}
	})

	t.Run("agentic", func(t *testing.T) {
		record := normalizedDiagnosticRecord("agentic", "E5")
		record.Success, record.Quality = boolPointer(true), floatPointer(1)
		record.TrajectorySteps, record.ToolCalls = int64Pointer(3), int64Pointer(2)
		record.InvalidToolCalls, record.PrivacyViolations = int64Pointer(1), int64Pointer(0)
		if err := validateNormalizedReplayDiagnosticRecord(record, executor); err != nil {
			t.Fatalf("valid agentic diagnostics rejected: %v", err)
		}
		record.InvalidToolCalls = int64Pointer(3)
		if err := validateNormalizedReplayDiagnosticRecord(record, executor); err == nil || !strings.Contains(err.Error(), "cannot exceed") {
			t.Fatalf("invalid tool counter error=%v", err)
		}
	})

	t.Run("preference", func(t *testing.T) {
		record := normalizedDiagnosticRecord("preference", "E5")
		selected := "action-a"
		record.SelectedArmID = &selected
		record.Success, record.Quality = boolPointer(true), floatPointer(1)
		record.PreferenceMatch = boolPointer(true)
		if err := validateNormalizedReplayDiagnosticRecord(record, executor); err == nil || !strings.Contains(err.Error(), "requires propensity") {
			t.Fatalf("missing E5 propensity error=%v", err)
		}
		record.BehaviorPropensity = floatPointer(0.25)
		if err := validateNormalizedReplayDiagnosticRecord(record, executor); err != nil {
			t.Fatalf("valid E5 preference diagnostics rejected: %v", err)
		}
	})
}

func TestUnavailableNormalizedReplayRecordRejectsDiagnosticValues(t *testing.T) {
	executor := builtinExecutorContractForTest(t, normalizedSuiteExecutorID)
	record := normalizedDiagnosticRecord("routing", "E4")
	reason := "decision evidence is missing"
	record.Status, record.Error = "unavailable", &reason
	if err := validateNormalizedReplayDiagnosticRecord(record, executor); err != nil {
		t.Fatalf("valid unavailable diagnostic rejected: %v", err)
	}
	record.Quality = floatPointer(1)
	if err := validateNormalizedReplayDiagnosticRecord(record, executor); err == nil || !strings.Contains(err.Error(), "cannot carry") {
		t.Fatalf("unavailable value error=%v", err)
	}
}
