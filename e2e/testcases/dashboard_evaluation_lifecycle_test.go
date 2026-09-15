package testcases

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/google/uuid"
)

type pendingCancellationGuardFixture struct {
	requests        []string
	clientRequestID string
}

func (fixture *pendingCancellationGuardFixture) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	fixture.requests = append(fixture.requests, request.Method+" "+request.URL.Path)
	writer.Header().Set("Content-Type", "application/json")
	switch len(fixture.requests) {
	case 1:
		fixture.serveCreate(writer, request)
	case 2:
		fixture.serveCancelConflict(writer, request)
	case 3:
		fixture.servePendingRun(writer, request)
	case 4:
		fixture.serveReportConflict(writer, request)
	default:
		http.Error(writer, "unexpected request", http.StatusInternalServerError)
	}
}

func (fixture *pendingCancellationGuardFixture) serveCreate(writer http.ResponseWriter, request *http.Request) {
	var payload map[string]interface{}
	if request.Method != http.MethodPost || request.URL.Path != "/api/evaluation/v1/runs" {
		http.Error(writer, "invalid create request", http.StatusInternalServerError)
		return
	}
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		http.Error(writer, "invalid create payload", http.StatusInternalServerError)
		return
	}
	fixture.clientRequestID, _ = payload["client_request_id"].(string)
	writer.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(writer).Encode(fixture.pendingRun())
}

func (fixture *pendingCancellationGuardFixture) serveCancelConflict(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost || request.URL.Path != "/api/evaluation/v1/runs/"+fixture.clientRequestID+"/cancel" {
		http.Error(writer, "invalid cancel request", http.StatusInternalServerError)
		return
	}
	writer.WriteHeader(http.StatusConflict)
	_ = json.NewEncoder(writer).Encode(map[string]interface{}{
		"error": map[string]string{"message": pendingCancelConflict},
	})
}

func (fixture *pendingCancellationGuardFixture) servePendingRun(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet || request.URL.Path != "/api/evaluation/v1/runs/"+fixture.clientRequestID {
		http.Error(writer, "invalid read request", http.StatusInternalServerError)
		return
	}
	writer.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(writer).Encode(fixture.pendingRun())
}

func (fixture *pendingCancellationGuardFixture) serveReportConflict(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet || request.URL.Path != "/api/evaluation/v1/runs/"+fixture.clientRequestID+"/report" {
		http.Error(writer, "invalid report request", http.StatusInternalServerError)
		return
	}
	writer.WriteHeader(http.StatusConflict)
	_ = json.NewEncoder(writer).Encode(map[string]interface{}{
		"error": map[string]string{"message": pendingReportConflict},
	})
}

func (fixture *pendingCancellationGuardFixture) pendingRun() dashboardEvaluationRun {
	return dashboardEvaluationRun{
		ID: fixture.clientRequestID, ClientRequestID: fixture.clientRequestID, Status: "pending",
	}
}

func TestPendingEvaluationCancellationGuardIsDeterministicAndNonMutating(t *testing.T) {
	t.Parallel()

	fixture := &pendingCancellationGuardFixture{}
	server := httptest.NewServer(fixture)
	defer server.Close()

	if err := verifyPendingEvaluationCancellationGuard(
		context.Background(), server.Client(), server.URL, "token",
	); err != nil {
		t.Fatalf("verify pending cancellation guard: %v", err)
	}
	parsed, err := uuid.Parse(fixture.clientRequestID)
	if err != nil || parsed.String() != fixture.clientRequestID || parsed.Version() != uuid.Version(4) {
		t.Fatalf("cancellation client_request_id %q is not a fresh canonical UUIDv4", fixture.clientRequestID)
	}
	wantRequests := []string{
		"POST /api/evaluation/v1/runs",
		"POST /api/evaluation/v1/runs/" + fixture.clientRequestID + "/cancel",
		"GET /api/evaluation/v1/runs/" + fixture.clientRequestID,
		"GET /api/evaluation/v1/runs/" + fixture.clientRequestID + "/report",
	}
	if !slices.Equal(fixture.requests, wantRequests) {
		t.Fatalf("request sequence = %v, want %v", fixture.requests, wantRequests)
	}
}
