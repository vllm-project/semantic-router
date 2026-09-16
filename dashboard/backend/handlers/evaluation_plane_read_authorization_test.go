package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/google/uuid"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

type unreadableEvaluationRequestBody struct{}

func (unreadableEvaluationRequestBody) Read([]byte) (int, error) {
	panic("unauthenticated evaluation request body was read")
}

func (unreadableEvaluationRequestBody) Close() error { return nil }

func evaluationReadRequest(
	method string,
	target string,
	authContext *dashboardauth.AuthContext,
) *http.Request {
	request := httptest.NewRequest(method, target, nil)
	if authContext != nil {
		request = request.WithContext(dashboardauth.WithAuthContext(request.Context(), *authContext))
	}
	return request
}

func TestEvaluationCreateRoutesAuthenticateBeforeReadingBody(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	t.Cleanup(func() { _ = service.Close() })
	handler := NewEvaluationPlaneHandler(service, false)
	tests := []struct {
		name   string
		path   string
		direct func(http.ResponseWriter, *http.Request)
	}{
		{name: "run", path: evaluationAPIBase + "/runs", direct: handler.Runs},
		{name: "controlled pair", path: evaluationAPIBase + "/controlled-pairs", direct: handler.ControlledPairs},
		{name: "campaign", path: evaluationAPIBase + "/campaigns", direct: handler.Campaigns},
		{name: "campaign readiness", path: evaluationAPIBase + "/campaign-readiness", direct: handler.CampaignReadiness},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, test.path, nil)
			request.Body = unreadableEvaluationRequestBody{}
			request.ContentLength = maxEvaluationRequestBytes + 1
			response := httptest.NewRecorder()
			test.direct(response, request)
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("unauthenticated create status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestEvaluationMutationRoutesAuthenticateBeforeReadonlyAndBody(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	t.Cleanup(func() { _ = service.Close() })
	handler := NewEvaluationPlaneHandler(service, true)
	tests := []struct {
		name   string
		method string
		path   string
		direct func(http.ResponseWriter, *http.Request)
	}{
		{
			name: "run delete", method: http.MethodDelete,
			path:   evaluationAPIBase + "/runs/00000000-0000-4000-8000-000000000001",
			direct: handler.RunRoute,
		},
		{
			name: "run start", method: http.MethodPost,
			path:   evaluationAPIBase + "/runs/00000000-0000-4000-8000-000000000001/start",
			direct: handler.RunRoute,
		},
		{
			name: "controlled pair cancel", method: http.MethodPost,
			path:   evaluationAPIBase + "/controlled-pairs/pair-id/cancel",
			direct: handler.ControlledPairLifecycle,
		},
		{
			name: "campaign delete", method: http.MethodDelete,
			path:   evaluationAPIBase + "/campaigns/campaign-id",
			direct: handler.CampaignRoute,
		},
		{
			name: "lifecycle collection", method: http.MethodPost,
			path:   evaluationAPIBase + "/lifecycle/collection",
			direct: handler.LifecycleCollection,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(test.method, test.path, nil)
			request.Body = unreadableEvaluationRequestBody{}
			request.ContentLength = maxEvaluationRequestBytes + 1
			response := httptest.NewRecorder()
			test.direct(response, request)
			if response.Code != http.StatusUnauthorized {
				t.Fatalf("unauthenticated mutation status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func createHandlerReadRun(
	t *testing.T,
	service *evaluationplane.Service,
	actor evaluationplane.Actor,
	name string,
) evaluationplane.Run {
	t.Helper()
	request := evaluationplane.CreateRunRequest{
		ClientRequestID: uuid.NewString(),
		Name:            name, Description: "actor-aware handler read",
		SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []evaluationplane.TrackID{"routing"},
		Mode: evaluationplane.ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	run, err := service.CreateRunAs(context.Background(), actor, request)
	if err != nil {
		t.Fatalf("create handler read run: %v", err)
	}
	return run
}

func assertEvaluationRunAndEvidenceReadsAreActorScoped(
	t *testing.T, handler *EvaluationPlaneHandler, run evaluationplane.Run,
	owner, other, administrator dashboardauth.AuthContext,
) {
	t.Helper()
	for _, test := range []struct {
		name        string
		authContext dashboardauth.AuthContext
		wantStatus  int
	}{
		{name: "owner", authContext: owner, wantStatus: http.StatusOK},
		{name: "other", authContext: other, wantStatus: http.StatusForbidden},
		{name: "administrator", authContext: administrator, wantStatus: http.StatusOK},
	} {
		t.Run("run/"+test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.RunRoute(response, evaluationReadRequest(
				http.MethodGet, evaluationAPIBase+"/runs/"+run.ID, &test.authContext,
			))
			if response.Code != test.wantStatus {
				t.Fatalf("run status=%d want=%d body=%s", response.Code, test.wantStatus, response.Body.String())
			}
		})
	}

	for _, endpoint := range []string{
		evaluationAPIBase + "/runs/" + run.ID + "/report",
		evaluationAPIBase + "/runs/" + run.ID + "/artifacts/metrics",
	} {
		for _, test := range []struct {
			name        string
			authContext dashboardauth.AuthContext
			wantStatus  int
		}{
			{name: "owner", authContext: owner, wantStatus: http.StatusConflict},
			{name: "other", authContext: other, wantStatus: http.StatusForbidden},
			{name: "administrator", authContext: administrator, wantStatus: http.StatusConflict},
		} {
			t.Run("evidence/"+test.name+"/"+endpoint, func(t *testing.T) {
				response := httptest.NewRecorder()
				handler.RunRoute(response, evaluationReadRequest(http.MethodGet, endpoint, &test.authContext))
				if response.Code != test.wantStatus {
					t.Fatalf("evidence status=%d want=%d body=%s", response.Code, test.wantStatus, response.Body.String())
				}
			})
		}
	}
}

func assertEvaluationComparisonAndEventReadsAreActorScoped(
	t *testing.T, handler *EvaluationPlaneHandler, first, second evaluationplane.Run,
	owner, other, administrator dashboardauth.AuthContext,
) {
	t.Helper()
	comparisonTarget := fmt.Sprintf(
		"%s/compare?baseline_run_id=%s&candidate_run_id=%s",
		evaluationAPIBase, first.ID, second.ID,
	)
	for _, test := range []struct {
		name        string
		authContext dashboardauth.AuthContext
		wantStatus  int
	}{
		{name: "owner", authContext: owner, wantStatus: http.StatusConflict},
		{name: "other", authContext: other, wantStatus: http.StatusForbidden},
		{name: "administrator", authContext: administrator, wantStatus: http.StatusConflict},
	} {
		t.Run("compare/"+test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.Compare(response, evaluationReadRequest(http.MethodGet, comparisonTarget, &test.authContext))
			if response.Code != test.wantStatus {
				t.Fatalf("compare status=%d want=%d body=%s", response.Code, test.wantStatus, response.Body.String())
			}
		})
	}

	for _, test := range []struct {
		name        string
		authContext dashboardauth.AuthContext
		wantStatus  int
	}{
		{name: "owner", authContext: owner, wantStatus: http.StatusOK},
		{name: "other", authContext: other, wantStatus: http.StatusForbidden},
		{name: "administrator", authContext: administrator, wantStatus: http.StatusOK},
	} {
		t.Run("events/"+test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			request := evaluationReadRequest(
				http.MethodGet, evaluationAPIBase+"/runs/"+first.ID+"/events", &test.authContext,
			)
			cancelled, cancel := context.WithCancel(request.Context())
			cancel()
			request = request.WithContext(cancelled)
			handler.RunRoute(response, request)
			if response.Code != test.wantStatus {
				t.Fatalf("events status=%d want=%d body=%s", response.Code, test.wantStatus, response.Body.String())
			}
		})
	}
}

func TestEvaluationRunReadsAreActorScopedWhileCatalogIsShared(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := NewEvaluationPlaneHandler(service, false)
	owner := dashboardauth.AuthContext{UserID: "evaluation-read-owner", Role: dashboardauth.RoleWrite}
	other := dashboardauth.AuthContext{UserID: "evaluation-read-other", Role: dashboardauth.RoleWrite}
	administrator := dashboardauth.AuthContext{UserID: "evaluation-read-admin", Role: dashboardauth.RoleAdmin}
	ownerActor, err := evaluationplane.NewActor(owner.UserID, false)
	if err != nil {
		t.Fatalf("create owner actor: %v", err)
	}
	first := createHandlerReadRun(t, service, ownerActor, "first owner run")
	second := createHandlerReadRun(t, service, ownerActor, "second owner run")

	catalogResponse := httptest.NewRecorder()
	handler.Catalog(catalogResponse, evaluationReadRequest(http.MethodGet, evaluationAPIBase+"/catalog", nil))
	if catalogResponse.Code != http.StatusOK {
		t.Fatalf("shared catalog status=%d body=%s", catalogResponse.Code, catalogResponse.Body.String())
	}

	unauthenticatedList := httptest.NewRecorder()
	handler.Runs(unauthenticatedList, evaluationReadRequest(http.MethodGet, evaluationAPIBase+"/runs", nil))
	if unauthenticatedList.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated list status=%d body=%s", unauthenticatedList.Code, unauthenticatedList.Body.String())
	}

	for _, test := range []struct {
		name        string
		authContext dashboardauth.AuthContext
		wantRuns    int
	}{
		{name: "owner", authContext: owner, wantRuns: 2},
		{name: "other", authContext: other, wantRuns: 0},
		{name: "administrator", authContext: administrator, wantRuns: 2},
	} {
		t.Run("list/"+test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.Runs(response, evaluationReadRequest(http.MethodGet, evaluationAPIBase+"/runs?limit=10", &test.authContext))
			if response.Code != http.StatusOK {
				t.Fatalf("list status=%d body=%s", response.Code, response.Body.String())
			}
			var ledger evaluationplane.RunLedger
			if err := json.NewDecoder(response.Body).Decode(&ledger); err != nil {
				t.Fatalf("decode ledger: %v", err)
			}
			if ledger.TotalRuns != test.wantRuns || len(ledger.Runs) != test.wantRuns {
				t.Fatalf("ledger=%+v, want %d actor-visible runs", ledger, test.wantRuns)
			}
		})
	}

	assertEvaluationRunAndEvidenceReadsAreActorScoped(t, handler, first, owner, other, administrator)
	assertEvaluationComparisonAndEventReadsAreActorScoped(t, handler, first, second, owner, other, administrator)
}
