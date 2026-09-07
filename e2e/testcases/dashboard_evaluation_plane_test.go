package testcases

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/uuid"
)

func TestCreateEvaluationRunSendsDistinctCandidateIdentity(t *testing.T) {
	t.Parallel()

	const (
		baselineID      = "1f5c9ac3-054b-43d4-8902-cc3d63e3ea2d"
		clientRequestID = "7295515a-4c6c-40a3-8436-75edc691369c"
	)
	var requestPayload map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if err := json.NewDecoder(request.Body).Decode(&requestPayload); err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(writer).Encode(dashboardEvaluationRun{
			ID:              clientRequestID,
			ClientRequestID: clientRequestID,
			Status:          "pending",
		})
	}))
	defer server.Close()

	run, err := createEvaluationRun(
		context.Background(), server.Client(), server.URL, "token", clientRequestID, "candidate", 41, baselineID,
	)
	if err != nil {
		t.Fatalf("create evaluation run: %v", err)
	}
	if got := requestPayload["client_request_id"]; got != clientRequestID {
		t.Fatalf("client_request_id = %v, want %s", got, clientRequestID)
	}
	if got := requestPayload["baseline_run_id"]; got != baselineID {
		t.Fatalf("baseline_run_id = %v, want %s", got, baselineID)
	}
	if requestPayload["client_request_id"] == requestPayload["baseline_run_id"] {
		t.Fatalf("candidate client_request_id must not reuse baseline_run_id")
	}
	if run.ID != clientRequestID || run.ClientRequestID != clientRequestID {
		t.Fatalf("created run identity = %q/%q, want %q", run.ID, run.ClientRequestID, clientRequestID)
	}
}

func TestSameRevisionCandidateIsRejectedAtAdmissionWithFreshIdentity(t *testing.T) {
	t.Parallel()

	const baselineID = "1f5c9ac3-054b-43d4-8902-cc3d63e3ea2d"
	var (
		requestMethod  string
		requestPath    string
		requestPayload map[string]interface{}
	)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requestMethod = request.Method
		requestPath = request.URL.Path
		if err := json.NewDecoder(request.Body).Decode(&requestPayload); err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(writer).Encode(map[string]interface{}{
			"error": map[string]string{
				"message": `invalid evaluation request: change_profile "schema_adapter" requires the code treatment factor to change`,
			},
		})
	}))
	defer server.Close()

	if err := verifySameRevisionCandidateAdmissionGuard(
		context.Background(), server.Client(), server.URL, "token", baselineID,
	); err != nil {
		t.Fatalf("verify same-revision candidate admission: %v", err)
	}
	if requestMethod != http.MethodPost || requestPath != "/api/evaluation/v1/runs" {
		t.Fatalf("candidate admission request = %s %s, want POST /api/evaluation/v1/runs", requestMethod, requestPath)
	}
	clientRequestID, ok := requestPayload["client_request_id"].(string)
	if !ok || clientRequestID == "" || clientRequestID == baselineID {
		t.Fatalf("candidate client_request_id = %v, want a fresh identity", requestPayload["client_request_id"])
	}
	if got := requestPayload["baseline_run_id"]; got != baselineID {
		t.Fatalf("baseline_run_id = %v, want %s", got, baselineID)
	}
}

func TestNewEvaluationClientRequestIDIsCanonicalAndCollisionSafe(t *testing.T) {
	t.Parallel()

	const generatedIDs = 128
	seen := make(map[string]struct{}, generatedIDs)
	for index := 0; index < generatedIDs; index++ {
		requestID := newEvaluationClientRequestID()
		parsed, err := uuid.Parse(requestID)
		if err != nil || parsed.String() != requestID || parsed.Version() != uuid.Version(4) {
			t.Fatalf("generated client_request_id %q is not a canonical UUIDv4", requestID)
		}
		if _, duplicate := seen[requestID]; duplicate {
			t.Fatalf("generated duplicate client_request_id %q", requestID)
		}
		seen[requestID] = struct{}{}
	}
}
