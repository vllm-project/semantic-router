package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func validVisibleCaseRow(id string) map[string]any {
	return map[string]any{
		"schema_version": SchemaVersion,
		"id":             id,
		"messages": []map[string]any{
			{"role": "user", "content": "evaluate this case"},
		},
		"modality": "text",
		"tags":     []string{"test"},
	}
}

func validExecutionRecordRow(id, caseID string) map[string]any {
	return map[string]any{
		"schema_version": SchemaVersion,
		"id":             id,
		"track_id":       "routing",
		"case_id":        caseID,
		"attempt_id":     "attempt-" + caseID,
		"status":         "succeeded",
		"quality":        1.0,
		"latency_ms":     1.5,
	}
}

func validFailureSummaryRow() map[string]any {
	return map[string]any{
		"schema_version": SchemaVersion,
		"total_records":  1,
		"failed":         0,
		"unavailable":    0,
		"by_track": []map[string]any{
			{"track_id": "routing", "succeeded": 1, "failed": 0, "unavailable": 0},
		},
	}
}

func writeJSONLinesForTest(t *testing.T, path string, rows ...map[string]any) {
	t.Helper()
	var data []byte
	for _, row := range rows {
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatalf("marshal JSONL row: %v", err)
		}
		data = append(data, encoded...)
		data = append(data, '\n')
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write %s: %v", filepath.Base(path), err)
	}
}

func writeRecordValidationFixture(
	t *testing.T,
	cases []map[string]any,
	records []map[string]any,
	summary map[string]any,
) (string, RunManifest) {
	t.Helper()
	runDir := t.TempDir()
	writeJSONLinesForTest(t, filepath.Join(runDir, "cases.jsonl"), cases...)
	writeJSONLinesForTest(t, filepath.Join(runDir, "records.jsonl"), records...)
	if err := writeJSONAtomic(filepath.Join(runDir, "failure-summary.json"), summary); err != nil {
		t.Fatalf("write failure summary: %v", err)
	}
	return runDir, RunManifest{SampleLimit: len(cases), TrackIDs: []TrackID{"routing"}}
}

func TestValidateRecordsAndFailureSummaryAttestsStrictEvidence(t *testing.T) {
	runDir, manifest := writeRecordValidationFixture(
		t,
		[]map[string]any{validVisibleCaseRow("case-1")},
		[]map[string]any{validExecutionRecordRow("routing-case-1", "case-1")},
		validFailureSummaryRow(),
	)
	attestation, err := validateRecordsAndFailureSummary(runDir, manifest)
	if err != nil {
		t.Fatalf("validateRecordsAndFailureSummary: %v", err)
	}
	if !attestation.validated || attestation.Total != 1 || attestation.Succeeded != 1 || attestation.ByTrack["routing"].Succeeded != 1 {
		t.Fatalf("unexpected records attestation: %+v", attestation)
	}
}

type forgedEvidenceTest struct {
	name             string
	mutate           func([]map[string]any, []map[string]any, map[string]any)
	additionalRecord map[string]any
	wantError        string
}

func forgedVisibleEvidenceTests() []forgedEvidenceTest {
	return []forgedEvidenceTest{
		{
			name: "unknown visible message field",
			mutate: func(cases []map[string]any, _ []map[string]any, _ map[string]any) {
				messages := cases[0]["messages"].([]map[string]any)
				messages[0]["prompt"] = "untyped"
			},
			wantError: "unknown field",
		},
		{
			name: "unknown nested content field",
			mutate: func(cases []map[string]any, _ []map[string]any, _ map[string]any) {
				cases[0]["messages"] = []map[string]any{{
					"role":    "user",
					"content": []map[string]any{{"type": "text", "text": "ok", "prompt": "untyped"}},
				}}
			},
			wantError: "text contract",
		},
		{
			name: "empty content parts",
			mutate: func(cases []map[string]any, _ []map[string]any, _ map[string]any) {
				cases[0]["messages"] = []map[string]any{{"role": "user", "content": []map[string]any{}}}
			},
			wantError: "must be non-empty",
		},
	}
}

func forgedRecordEvidenceTests() []forgedEvidenceTest {
	return []forgedEvidenceTest{
		{
			name: "unknown record field",
			mutate: func(_ []map[string]any, records []map[string]any, _ map[string]any) {
				records[0]["provider_secret"] = "must-not-be-accepted"
			},
			wantError: "unknown field",
		},
		{
			name: "unknown case",
			mutate: func(_ []map[string]any, records []map[string]any, _ map[string]any) {
				records[0]["case_id"] = "invented-case"
			},
			wantError: "absent from the validated case set",
		},
		{
			name: "unselected track",
			mutate: func(_ []map[string]any, records []map[string]any, _ map[string]any) {
				records[0]["track_id"] = "safety"
			},
			wantError: "not selected",
		},
		{
			name: "invalid status",
			mutate: func(_ []map[string]any, records []map[string]any, _ map[string]any) {
				records[0]["status"] = "passed"
			},
			wantError: "status",
		},
		{
			name: "invalid stable identity",
			mutate: func(_ []map[string]any, records []map[string]any, _ map[string]any) {
				records[0]["id"] = ""
			},
			wantError: "portable non-empty identities",
		},
		{
			name: "duplicate record id",
			mutate: func(_ []map[string]any, _ []map[string]any, summary map[string]any) {
				summary["total_records"] = 2
			},
			additionalRecord: validExecutionRecordRow("routing-case-1", "case-1"),
			wantError:        "duplicate record id",
		},
		{
			name: "duplicate semantic attempt with changed id",
			mutate: func(_ []map[string]any, _ []map[string]any, summary map[string]any) {
				summary["total_records"] = 2
			},
			additionalRecord: validExecutionRecordRow("routing-case-1-copy", "case-1"),
			wantError:        "duplicates semantic attempt",
		},
		{
			name: "non-finite numeric envelope",
			mutate: func(_ []map[string]any, records []map[string]any, _ map[string]any) {
				records[0]["quality"] = -0.1
			},
			wantError: "finite fraction",
		},
		{
			name: "failure summary mismatch",
			mutate: func(_ []map[string]any, _ []map[string]any, summary map[string]any) {
				summary["failed"] = 1
			},
			wantError: "does not match validated records",
		},
	}
}

func TestValidateRecordsAndFailureSummaryRejectsForgedEvidence(t *testing.T) {
	tests := append(forgedVisibleEvidenceTests(), forgedRecordEvidenceTests()...)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cases := []map[string]any{validVisibleCaseRow("case-1")}
			records := []map[string]any{validExecutionRecordRow("routing-case-1", "case-1")}
			summary := validFailureSummaryRow()
			test.mutate(cases, records, summary)
			if test.additionalRecord != nil {
				records = append(records, test.additionalRecord)
			}
			runDir, manifest := writeRecordValidationFixture(t, cases, records, summary)
			_, err := validateRecordsAndFailureSummary(runDir, manifest)
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("error=%v, want ErrInvalid containing %q", err, test.wantError)
			}
		})
	}
}

func TestEvidenceJSONLScannerEnforcesLineCountBytesAndNewline(t *testing.T) {
	tests := []struct {
		name         string
		data         string
		maxBytes     int64
		maxLineBytes int
		maxLines     int
		wantError    string
	}{
		{name: "line count", data: "{}\n{}\n", maxBytes: 32, maxLineBytes: 8, maxLines: 1, wantError: "line-count"},
		{name: "total bytes", data: "{}\n", maxBytes: 2, maxLineBytes: 8, maxLines: 1, wantError: "total-byte"},
		{name: "line bytes", data: "{}\n", maxBytes: 32, maxLineBytes: 1, maxLines: 1, wantError: "scan evidence"},
		{name: "trailing newline", data: "{}", maxBytes: 32, maxLineBytes: 8, maxLines: 1, wantError: "end with a newline"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "evidence.jsonl")
			if err := os.WriteFile(path, []byte(test.data), 0o600); err != nil {
				t.Fatal(err)
			}
			err := scanEvidenceJSONLines(path, test.maxBytes, test.maxLineBytes, test.maxLines, func([]byte, int) error { return nil })
			if !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("error=%v, want ErrInvalid containing %q", err, test.wantError)
			}
		})
	}
}

func TestReportBundleRejectsSelfConsistentWorkerClaimsOverForgedRecords(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	now := time.Now().UTC()
	run.Status = StatusRunning
	run.StartedAt = &now
	if err := service.store.UpdateRun(run); err != nil {
		t.Fatalf("stage running run: %v", err)
	}
	spec := ProcessSpec{ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName), StorePath: root}
	if err := writeProcessReport(spec); err != nil {
		t.Fatalf("writeProcessReport: %v", err)
	}
	runDir := filepath.Dir(spec.ManifestPath)
	forged := validExecutionRecordRow("routing-invented-case", "invented-case")
	writeJSONLinesForTest(t, filepath.Join(runDir, "records.jsonl"), forged)
	if err := writeTestPrivateReceiptWithoutTesting(runDir); err != nil {
		t.Fatalf("rewrite private receipt: %v", err)
	}
	if err := service.validateAndAnchorReport(run.ID); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "absent from the validated case set") {
		t.Fatalf("forged records error=%v, want case attestation ErrInvalid", err)
	}
}
