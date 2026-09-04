package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

var evaluationHandlerTestAdmin = dashboardauth.AuthContext{
	UserID: "evaluation-handler-test-admin",
	Role:   dashboardauth.RoleAdmin,
}

type authenticatedEvaluationTestHandler struct {
	*EvaluationPlaneHandler
}

func newAuthenticatedEvaluationTestHandler(
	service *evaluationplane.Service,
	readonly bool,
) *authenticatedEvaluationTestHandler {
	return &authenticatedEvaluationTestHandler{NewEvaluationPlaneHandler(service, readonly)}
}

func authenticatedEvaluationTestRequest(request *http.Request) *http.Request {
	return request.WithContext(dashboardauth.WithAuthContext(request.Context(), evaluationHandlerTestAdmin))
}

func (h *authenticatedEvaluationTestHandler) Runs(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.Runs(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) RunRoute(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.RunRoute(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) Compare(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.Compare(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) ControlledPairs(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.ControlledPairs(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) ControlledPairLifecycle(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.ControlledPairLifecycle(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) Campaigns(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.Campaigns(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) CampaignRoute(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.CampaignRoute(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) LifecycleUsage(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.LifecycleUsage(w, authenticatedEvaluationTestRequest(r))
}

func (h *authenticatedEvaluationTestHandler) LifecycleCollection(w http.ResponseWriter, r *http.Request) {
	h.EvaluationPlaneHandler.LifecycleCollection(w, authenticatedEvaluationTestRequest(r))
}

func newEvaluationHandlerService(t *testing.T, dataDir string) *evaluationplane.Service {
	return newEvaluationHandlerServiceWithProcess(t, dataDir, nil)
}

const defaultEvaluationHandlerCodeRevision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

const evaluationHandlerMOMConfig = `version: v0.3
global:
  router:
    auto_model_names: [test-mom]
providers:
  defaults:
    default_model: model-fast
  models:
    - name: model-fast
      backend_refs: [{endpoint: fast.models.test:8000}]
    - name: model-strong
      backend_refs: [{endpoint: strong.models.test:8000}]
routing:
  modelCards:
    - {name: model-fast, modality: text}
    - {name: model-strong, modality: text}
  decisions:
    - name: route
      rules: {}
      modelRefs: [{model: model-fast}, {model: model-strong}]
`

func newEvaluationHandlerServiceWithProcess(
	t *testing.T,
	dataDir string,
	process evaluationplane.Process,
) *evaluationplane.Service {
	return newEvaluationHandlerServiceAtRevision(t, dataDir, process, defaultEvaluationHandlerCodeRevision)
}

func newEvaluationHandlerServiceAtRevision(
	t *testing.T,
	dataDir string,
	process evaluationplane.Process,
	codeRevision string,
) *evaluationplane.Service {
	t.Helper()
	if dataDir == "" {
		dataDir = t.TempDir()
	}
	if err := os.Chmod(dataDir, 0o700); err != nil {
		t.Fatalf("protect evaluation data dir: %v", err)
	}
	configPath := filepath.Join(dataDir, "config.yaml")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		if err := os.WriteFile(configPath, []byte(evaluationHandlerMOMConfig), 0o600); err != nil {
			t.Fatalf("write config: %v", err)
		}
	}
	service, err := evaluationplane.NewService(evaluationplane.Options{
		DataDir: dataDir, PythonPath: "python3", ConfigPath: configPath,
		RouterAPIURL: "http://router.internal", EnvoyURL: "http://envoy.internal",
		CodeRevision: codeRevision, Process: process,
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	return service
}

func requireLiveMOMTargetID(t *testing.T, service *evaluationplane.Service) string {
	t.Helper()
	catalog, err := service.Catalog()
	if err != nil {
		t.Fatalf("read evaluation catalog: %v", err)
	}
	for _, target := range catalog.Targets {
		if target.Kind == "mixture-of-models" {
			return target.ID
		}
	}
	t.Fatal("evaluation catalog has no Mixture-of-Models target")
	return ""
}

type blockingEvaluationProcess struct{}

func (blockingEvaluationProcess) Run(
	ctx context.Context,
	_ evaluationplane.ProcessSpec,
	_ func(evaluationplane.WorkerEvent) error,
) (evaluationplane.ProcessResult, error) {
	<-ctx.Done()
	return evaluationplane.ProcessResult{}, ctx.Err()
}

func validCreateRunJSON() string {
	return `{
		"client_request_id":"5b49edbb-2008-4dc3-a245-b7cc78b839b1",
        "name":"fixture",
        "description":"handler test",
        "suite_ids":["evaluation-smoke"],
        "track_ids":["routing"],
        "mode":"replay",
        "target_id":"fixture",
		"change_profile":"schema_adapter",
        "sample_limit":4,
        "concurrency":1,
		"seed":17
    }`
}

func TestEvaluationPlaneDomainErrorIsPublicBadRequest(t *testing.T) {
	response := httptest.NewRecorder()
	writeEvaluationError(
		response,
		fmt.Errorf(
			"%w: distinct deployment targets require a server-owned controlled pair with exact manifest bindings",
			evaluationplane.ErrInvalid,
		),
	)
	if response.Code != http.StatusBadRequest ||
		!strings.Contains(response.Body.String(), "server-owned controlled pair") ||
		strings.Contains(response.Body.String(), "Evaluation service failed") {
		t.Fatalf("domain error status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestEvaluationPlaneCreateIsStrictAndServerTargetAllowlisted(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	tests := []struct {
		name string
		body string
	}{
		{name: "unknown endpoint", body: strings.Replace(validCreateRunJSON(), `"seed":17`, `"seed":17,"endpoint":"https://attacker.invalid"`, 1)},
		{name: "unknown field", body: strings.Replace(validCreateRunJSON(), `"seed":17`, `"seed":17,"command":["sh"]`, 1)},
		{name: "unknown target", body: strings.Replace(validCreateRunJSON(), `"target_id":"fixture"`, `"target_id":"attacker"`, 1)},
		{name: "unknown suite", body: strings.Replace(validCreateRunJSON(), `"evaluation-smoke"`, `"unknown-suite"`, 1)},
		{name: "unknown change profile", body: strings.Replace(validCreateRunJSON(), `"change_profile":"schema_adapter"`, `"change_profile":"unknown"`, 1)},
		{name: "workflow field is not part of create contract", body: strings.Replace(validCreateRunJSON(), `"seed":17`, `"seed":17,"auto_start":false`, 1)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/runs", strings.NewReader(test.body))
			handler.Runs(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusBadRequest, response.Body.String())
			}
		})
	}
	ledger, err := service.ListRunLedgerPageAs(evaluationplane.SystemActor(), evaluationplane.RunListQuery{Limit: 10})
	if err != nil || len(ledger.Runs) != 0 {
		t.Fatalf("rejected creates changed store: runs=%d err=%v", len(ledger.Runs), err)
	}

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/runs", strings.NewReader(validCreateRunJSON()))
	handler.Runs(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("valid create status=%d body=%s", response.Code, response.Body.String())
	}
	var run evaluationplane.Run
	if err := json.NewDecoder(response.Body).Decode(&run); err != nil {
		t.Fatalf("decode run: %v", err)
	}
	if run.Status != evaluationplane.StatusPending || run.TargetID != "fixture" {
		t.Fatalf("unexpected created run: %+v", run)
	}
}

func TestEvaluationPlaneRunLedgerSurfacesQuarantineAndBlocksDecisions(t *testing.T) {
	root := t.TempDir()
	service := newEvaluationHandlerService(t, root)
	intact, err := service.CreateRunAs(context.Background(), evaluationplane.SystemActor(), evaluationplane.CreateRunRequest{
		ClientRequestID: "3d99aa02-bdf2-4cf3-bb9e-8a973b43bba1",
		Name:            "intact", SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []evaluationplane.TrackID{"routing"},
		Mode: evaluationplane.ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if err != nil {
		t.Fatalf("create intact run: %v", err)
	}
	startedAt := intact.CreatedAt.Add(time.Microsecond)
	completedAt := startedAt.Add(time.Microsecond)
	intact.Status = evaluationplane.StatusCompleted
	intact.StartedAt = &startedAt
	intact.CompletedAt = &completedAt
	intact.Progress = evaluationplane.RunProgress{
		Percent: 100, Completed: len(intact.TrackIDs), Total: len(intact.TrackIDs), Message: "Evaluation completed",
	}
	intactStatus, err := json.MarshalIndent(intact, "", "  ")
	if err != nil {
		t.Fatalf("encode completed baseline fixture: %v", err)
	}
	if writeErr := os.WriteFile(
		filepath.Join(root, "runs", intact.ID, "status.json"),
		append(intactStatus, '\n'),
		0o600,
	); writeErr != nil {
		t.Fatalf("complete baseline fixture: %v", writeErr)
	}
	corruptRequest := evaluationplane.CreateRunRequest{
		ClientRequestID: "cb9dd730-7424-4050-a73f-52378e634b15",
		Name:            "corrupt", SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []evaluationplane.TrackID{"routing"},
		Mode: evaluationplane.ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 18,
	}
	corrupt, err := service.CreateRunAs(context.Background(), evaluationplane.SystemActor(), corruptRequest)
	if err != nil {
		t.Fatalf("create corrupt candidate: %v", err)
	}
	comparisonCandidateRequest := corruptRequest
	comparisonCandidateRequest.ClientRequestID = "84a7890f-ad05-4374-aa0b-babb71d4429c"
	comparisonCandidateRequest.Name = "comparison candidate"
	comparisonCandidate, err := service.CreateRunAs(
		context.Background(),
		evaluationplane.SystemActor(),
		comparisonCandidateRequest,
	)
	if err != nil {
		t.Fatalf("create comparison candidate: %v", err)
	}
	if err := os.WriteFile(
		filepath.Join(root, "runs", corrupt.ID, "status.json"),
		[]byte("{not-json\n"),
		0o600,
	); err != nil {
		t.Fatalf("corrupt run status: %v", err)
	}
	if err := service.Close(); err != nil {
		t.Fatalf("close service before index rebuild: %v", err)
	}
	service = newEvaluationHandlerServiceAtRevision(
		t, root, nil, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
	)

	handler := newAuthenticatedEvaluationTestHandler(service, false)
	listResponse := httptest.NewRecorder()
	handler.Runs(listResponse, httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs", nil))
	if listResponse.Code != http.StatusOK {
		t.Fatalf("list status=%d body=%s", listResponse.Code, listResponse.Body.String())
	}
	var ledger evaluationplane.RunLedger
	if err := json.NewDecoder(listResponse.Body).Decode(&ledger); err != nil {
		t.Fatalf("decode run ledger: %v", err)
	}
	healthyRunIDs := make(map[string]bool, len(ledger.Runs))
	for _, run := range ledger.Runs {
		healthyRunIDs[run.ID] = true
	}
	if ledger.LedgerComplete || len(ledger.Runs) != 2 ||
		!healthyRunIDs[intact.ID] || !healthyRunIDs[comparisonCandidate.ID] || len(ledger.Warnings) != 1 {
		t.Fatalf("run ledger did not retain quarantine evidence: %+v", ledger)
	}
	warning := ledger.Warnings[0]
	if warning.EvidenceID != corrupt.ID || warning.EvidenceFile != "status.json" ||
		strings.Contains(warning.Message, root) || strings.Contains(warning.Message, "not-json") {
		t.Fatalf("unsafe or incomplete public warning: %+v", warning)
	}
	assertIncompleteEvaluationLedgerBlocksDecisions(t, handler, intact.ID, comparisonCandidate.ID)
}

func assertIncompleteEvaluationLedgerBlocksDecisions(
	t *testing.T,
	handler *authenticatedEvaluationTestHandler,
	baselineRunID string,
	candidateRunID string,
) {
	t.Helper()
	comparisonResponse := httptest.NewRecorder()
	comparisonRequest := httptest.NewRequest(
		http.MethodGet,
		evaluationAPIBase+"/compare?baseline_run_id="+baselineRunID+"&candidate_run_id="+candidateRunID,
		nil,
	)
	handler.Compare(comparisonResponse, comparisonRequest)
	if comparisonResponse.Code != http.StatusConflict || !strings.Contains(comparisonResponse.Body.String(), "ledger is incomplete") {
		t.Fatalf("comparison status=%d body=%s", comparisonResponse.Code, comparisonResponse.Body.String())
	}

	baselineBody := strings.Replace(
		validCreateRunJSON(),
		`"seed":17`,
		fmt.Sprintf(`"seed":17,"baseline_run_id":%q`, baselineRunID),
		1,
	)
	baselineResponse := httptest.NewRecorder()
	handler.Runs(
		baselineResponse,
		httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/runs", strings.NewReader(baselineBody)),
	)
	if baselineResponse.Code != http.StatusConflict || !strings.Contains(baselineResponse.Body.String(), "ledger is incomplete") {
		t.Fatalf("baseline create status=%d body=%s", baselineResponse.Code, baselineResponse.Body.String())
	}
}

func TestEvaluationPlaneRunListQueryIsStrictAndBounded(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	for _, path := range []string{
		evaluationAPIBase + "/runs?limit=201",
		evaluationAPIBase + "/runs?limit=not-an-integer",
		evaluationAPIBase + "/runs?limit=1&limit=2",
		evaluationAPIBase + "/runs?cursor=a&cursor=b",
		evaluationAPIBase + "/runs?cursor=" + strings.Repeat("a", 1025),
		evaluationAPIBase + "/runs?offset=1",
	} {
		response := httptest.NewRecorder()
		handler.Runs(response, httptest.NewRequest(http.MethodGet, path, nil))
		if response.Code != http.StatusBadRequest {
			t.Fatalf("GET %s status=%d body=%s", path, response.Code, response.Body.String())
		}
	}
}

func TestEvaluationPlaneReadonlyDeniesEveryMutation(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	run, createErr := service.CreateRunAs(context.Background(), evaluationplane.SystemActor(), evaluationplane.CreateRunRequest{
		ClientRequestID: "fd8a1e1d-36f2-4eb2-ac50-c9722b568c5c",
		Name:            "fixture", SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []evaluationplane.TrackID{"routing"},
		Mode: evaluationplane.ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	handler := newAuthenticatedEvaluationTestHandler(service, true)
	tests := []struct {
		method string
		path   string
		body   string
		direct func(http.ResponseWriter, *http.Request)
	}{
		{method: http.MethodPost, path: evaluationAPIBase + "/runs", body: validCreateRunJSON(), direct: handler.Runs},
		{method: http.MethodPost, path: evaluationAPIBase + "/runs/" + run.ID + "/start", direct: handler.RunRoute},
		{method: http.MethodPost, path: evaluationAPIBase + "/runs/" + run.ID + "/cancel", direct: handler.RunRoute},
		{method: http.MethodDelete, path: evaluationAPIBase + "/runs/" + run.ID, direct: handler.RunRoute},
	}
	for _, test := range tests {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(test.method, test.path, strings.NewReader(test.body))
		test.direct(response, request)
		if response.Code != http.StatusForbidden {
			t.Fatalf("%s %s status=%d want=%d body=%s", test.method, test.path, response.Code, http.StatusForbidden, response.Body.String())
		}
	}
	response := httptest.NewRecorder()
	handler.RunRoute(response, httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID, nil))
	if response.Code != http.StatusOK {
		t.Fatalf("readonly GET status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestEvaluationPlaneSSEReplaysPersistedEventsAfterRestart(t *testing.T) {
	root := t.TempDir()
	service := newEvaluationHandlerServiceWithProcess(t, root, blockingEvaluationProcess{})
	run, createErr := service.CreateRunAs(context.Background(), evaluationplane.SystemActor(), evaluationplane.CreateRunRequest{
		ClientRequestID: "d30b5492-849f-4680-8e3e-e7c23d4a2f09",
		Name:            "fixture", SuiteIDs: []string{"evaluation-smoke"}, TrackIDs: []evaluationplane.TrackID{"routing"},
		Mode: evaluationplane.ModeReplay, TargetID: "fixture", ChangeProfile: "schema_adapter",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	})
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	if _, startErr := service.StartRunAs(context.Background(), evaluationplane.SystemActor(), run.ID); startErr != nil {
		t.Fatalf("StartRun: %v", startErr)
	}
	if _, cancelErr := service.CancelRunAs(evaluationplane.SystemActor(), run.ID); cancelErr != nil {
		t.Fatalf("CancelRun: %v", cancelErr)
	}
	if closeErr := service.Close(); closeErr != nil {
		t.Fatalf("Close cancelled service: %v", closeErr)
	}
	restarted := newEvaluationHandlerService(t, root)
	handler := newAuthenticatedEvaluationTestHandler(restarted, false)
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/events", nil)
	request.Header.Set("Last-Event-ID", "1")
	handler.RunRoute(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("SSE status=%d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	if !strings.Contains(body, "id: 2\n") || !strings.Contains(body, "event: cancelled\n") || strings.Contains(body, "id: 1\n") {
		t.Fatalf("unexpected SSE replay body:\n%s", body)
	}
	if contentType := response.Header().Get("Content-Type"); contentType != "text/event-stream" {
		t.Fatalf("SSE Content-Type=%q", contentType)
	}

	badResponse := httptest.NewRecorder()
	badRequest := httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/events", bytes.NewReader(nil))
	badRequest.Header.Set("Last-Event-ID", "not-numeric")
	handler.RunRoute(badResponse, badRequest)
	if badResponse.Code != http.StatusBadRequest {
		t.Fatalf("invalid Last-Event-ID status=%d body=%s", badResponse.Code, badResponse.Body.String())
	}

	events, err := restarted.EventsAfterAs(evaluationplane.SystemActor(), run.ID, "")
	if err != nil || len(events) == 0 || events[len(events)-1].Type != "cancelled" {
		t.Fatalf("read cancelled terminal event: events=%+v err=%v", events, err)
	}
	for _, cursor := range []string{events[len(events)-1].ID, "999999"} {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, evaluationAPIBase+"/runs/"+run.ID+"/events", nil)
		request.Header.Set("Last-Event-ID", cursor)
		done := make(chan struct{})
		go func() {
			handler.RunRoute(response, request)
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatalf("terminal SSE stream did not close for Last-Event-ID=%s", cursor)
		}
		if response.Code != http.StatusOK || strings.Contains(response.Body.String(), "event:") {
			t.Fatalf("terminal SSE cursor=%s status=%d body=%s", cursor, response.Code, response.Body.String())
		}
	}
}

func TestEvaluationPlaneCORSRejectsHostileOriginsBeforeEvidenceAccess(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	paths := []string{
		evaluationAPIBase + "/runs/run-id/report",
		evaluationAPIBase + "/runs/run-id/artifacts/metrics-json",
		evaluationAPIBase + "/runs/run-id/events",
	}
	for _, path := range paths {
		response := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, "https://dashboard.example.test"+path, nil)
		request.Host = "dashboard.example.test"
		request.Header.Set("Origin", "https://sibling.example.test")
		handler.RunRoute(response, request)
		if response.Code != http.StatusForbidden {
			t.Fatalf("hostile Origin %s status=%d body=%s", path, response.Code, response.Body.String())
		}
		if response.Header().Get("Access-Control-Allow-Origin") != "" || response.Header().Get("Access-Control-Allow-Credentials") != "" {
			t.Fatalf("hostile Origin received credentialed CORS headers: %v", response.Header())
		}
	}

	allowed := httptest.NewRecorder()
	allowedRequest := httptest.NewRequest(http.MethodGet, "https://dashboard.example.test"+evaluationAPIBase+"/catalog", nil)
	allowedRequest.Host = "dashboard.example.test"
	allowedRequest.Header.Set("Origin", "https://dashboard.example.test")
	handler.Catalog(allowed, allowedRequest)
	if allowed.Code != http.StatusOK || allowed.Header().Get("Access-Control-Allow-Origin") != "https://dashboard.example.test" {
		t.Fatalf("same-origin catalog status=%d headers=%v", allowed.Code, allowed.Header())
	}

	downgrade := httptest.NewRecorder()
	downgradeRequest := httptest.NewRequest(http.MethodGet, "https://dashboard.example.test"+evaluationAPIBase+"/catalog", nil)
	downgradeRequest.Host = "dashboard.example.test"
	downgradeRequest.Header.Set("Origin", "http://dashboard.example.test")
	handler.Catalog(downgrade, downgradeRequest)
	if downgrade.Code != http.StatusForbidden || downgrade.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("mixed-scheme Origin status=%d headers=%v", downgrade.Code, downgrade.Header())
	}

	proxied := httptest.NewRecorder()
	proxiedRequest := httptest.NewRequest(http.MethodGet, "http://dashboard.example.test"+evaluationAPIBase+"/catalog", nil)
	proxiedRequest.Host = "dashboard.example.test"
	proxiedRequest.Header.Set("Origin", "https://dashboard.example.test")
	proxiedRequest.Header.Set("X-Forwarded-Proto", "https")
	handler.Catalog(proxied, proxiedRequest)
	if proxied.Code != http.StatusOK || proxied.Header().Get("Access-Control-Allow-Origin") != "https://dashboard.example.test" {
		t.Fatalf("same-origin proxied request status=%d headers=%v", proxied.Code, proxied.Header())
	}

	preflight := httptest.NewRecorder()
	preflightRequest := httptest.NewRequest(http.MethodOptions, "https://dashboard.example.test"+evaluationAPIBase+"/runs", nil)
	preflightRequest.Host = "dashboard.example.test"
	preflightRequest.Header.Set("Origin", "https://dashboard.example.test")
	preflightRequest.Header.Set("Access-Control-Request-Private-Network", "true")
	handler.Runs(preflight, preflightRequest)
	if preflight.Code != http.StatusNoContent || preflight.Header().Get("Access-Control-Allow-Private-Network") != "true" {
		t.Fatalf("same-origin preflight status=%d headers=%v", preflight.Code, preflight.Header())
	}
}
