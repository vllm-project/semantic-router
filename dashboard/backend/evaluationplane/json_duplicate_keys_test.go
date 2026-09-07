package evaluationplane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func injectDuplicateJSONKey(
	t *testing.T,
	payload []byte,
	marker []byte,
	replacement []byte,
) []byte {
	t.Helper()
	if !bytes.Contains(payload, marker) {
		t.Fatalf("JSON payload does not contain duplicate-key injection marker %q", marker)
	}
	return bytes.Replace(payload, marker, replacement, 1)
}

func TestWorkerReportRejectsNestedDuplicateJSONKeys(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	payload, err := json.Marshal(workerReportFromReport(reportForRun(run, nil)))
	if err != nil {
		t.Fatalf("encode worker report: %v", err)
	}
	payload = injectDuplicateJSONKey(
		t,
		payload,
		[]byte(`"run":{"schema_version":`),
		[]byte(`"run":{"schema_version":"forged","schema_version":`),
	)

	_, decodeErr := decodeWorkerReportStrict(run.ID, payload)
	if !errors.Is(decodeErr, ErrInvalid) || !strings.Contains(decodeErr.Error(), `duplicate JSON object key "schema_version"`) {
		t.Fatalf("nested duplicate worker-report key error=%v, want strict rejection", decodeErr)
	}
}

func TestPublishedReportRejectsNestedDuplicateJSONKeys(t *testing.T) {
	service, _ := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	report := reportForRun(run, nil)
	report.MethodReports = []CompoundModelBudgetReport{}
	payload, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("encode published report: %v", err)
	}
	payload = injectDuplicateJSONKey(
		t,
		payload,
		[]byte(`"run":{"schema_version":`),
		[]byte(`"run":{"schema_version":"forged","schema_version":`),
	)

	_, decodeErr := decodeReportStrict(run.ID, payload)
	if !errors.Is(decodeErr, ErrInvalid) || !strings.Contains(decodeErr.Error(), `duplicate JSON object key "schema_version"`) {
		t.Fatalf("nested duplicate published-report key error=%v, want strict rejection", decodeErr)
	}
}

func TestMethodDeclarationRejectsDuplicateJSONKeysInsideArrays(t *testing.T) {
	payload, err := json.Marshal(R2CompoundModelBudgetMethod())
	if err != nil {
		t.Fatalf("encode method declaration: %v", err)
	}
	payload = injectDuplicateJSONKey(
		t,
		payload,
		[]byte(`"slices":[{"schema_version":`),
		[]byte(`"slices":[{"schema_version":"forged","schema_version":`),
	)

	var method EvaluationMethodDefinition
	decodeErr := json.Unmarshal(payload, &method)
	if decodeErr == nil || !strings.Contains(decodeErr.Error(), `duplicate JSON object key "schema_version"`) {
		t.Fatalf("nested duplicate method-declaration key error=%v, want strict rejection", decodeErr)
	}
}

func TestLineageDocumentRejectsNestedDuplicateJSONKeys(t *testing.T) {
	payload := []byte(`{"schema_version":"evaluation.v1","manifest_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","workload":{"schema_version":"forged","schema_version":"evaluation.v1"}}`)
	_, decodeErr := decodeLineageDocument(payload)
	if decodeErr == nil || !strings.Contains(decodeErr.Error(), `duplicate JSON object key "schema_version"`) {
		t.Fatalf("nested duplicate lineage key error=%v, want strict rejection", decodeErr)
	}
}
