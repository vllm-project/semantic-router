package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

func mutateValidCreateRunJSON(t *testing.T, field string, value *json.RawMessage) string {
	return mutateCreateRunJSON(t, validCreateRunJSON(), field, value)
}

func mutateCreateRunJSON(t *testing.T, body string, field string, value *json.RawMessage) string {
	t.Helper()
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("decode valid create fixture: %v", err)
	}
	if value == nil {
		delete(payload, field)
	} else {
		payload[field] = *value
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("encode create fixture: %v", err)
	}
	return string(encoded)
}

func liveCapacityCreateRunJSON(targetID string) string {
	return strings.Replace(`{
		"client_request_id":"3fa10d7e-fad8-4784-a221-b7265c204a52",
		"name":"live capacity",
		"description":"typed SLO handler test",
		"suite_ids":["live-capacity"],
		"track_ids":["capacity"],
		"mode":"live",
		"target_id":"__MOM_TARGET__",
		"change_profile":"runtime_capacity",
		"sample_limit":4,
		"concurrency":2,
		"capacity_slo":{
			"schema_version":"evaluation.v1",
			"required_concurrency":2,
			"max_latency_p95_ms":750,
			"max_error_rate":0.01,
			"min_throughput_rps":10,
			"min_throughput_scaling_efficiency":0.7
		},
		"capacity_load_protocol":{
			"schema_version":"evaluation.v1",
			"kind":"closed-loop",
			"concurrency_levels":[1,2],
			"warmup_request_multiplier":2,
			"measurement_requests_per_repetition":100,
			"repetitions_per_level":3,
			"minimum_measurement_clusters_per_level":3,
			"confidence_level":0.95,
			"max_error_rate_cluster_range":0.05,
			"max_throughput_cv":0.2,
			"max_latency_p95_cv":0.2
		},
		"seed":17
	}`, "__MOM_TARGET__", targetID, 1)
}

func TestEvaluationPlaneCreateRequiresExactLiveCapacitySLO(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	targetID := requireLiveMOMTargetID(t, service)
	null := json.RawMessage("null")
	tests := []struct {
		name string
		body string
	}{
		{
			name: "missing",
			body: mutateCreateRunJSON(t, liveCapacityCreateRunJSON(targetID), "capacity_slo", nil),
		},
		{
			name: "null",
			body: mutateCreateRunJSON(t, liveCapacityCreateRunJSON(targetID), "capacity_slo", &null),
		},
		{
			name: "missing load protocol",
			body: mutateCreateRunJSON(t, liveCapacityCreateRunJSON(targetID), "capacity_load_protocol", nil),
		},
		{
			name: "null load protocol",
			body: mutateCreateRunJSON(t, liveCapacityCreateRunJSON(targetID), "capacity_load_protocol", &null),
		},
		{
			name: "unknown nested field",
			body: strings.Replace(
				liveCapacityCreateRunJSON(targetID),
				`"min_throughput_scaling_efficiency":0.7`,
				`"min_throughput_scaling_efficiency":0.7,"proxy_pass":true`,
				1,
			),
		},
		{
			name: "single load level",
			body: strings.NewReplacer(
				`"concurrency":2`, `"concurrency":1`,
				`"required_concurrency":2`, `"required_concurrency":1`,
			).Replace(liveCapacityCreateRunJSON(targetID)),
		},
		{
			name: "tiny measurement window",
			body: strings.Replace(
				liveCapacityCreateRunJSON(targetID),
				`"measurement_requests_per_repetition":100`,
				`"measurement_requests_per_repetition":2`,
				1,
			),
		},
		{
			name: "unknown load field",
			body: strings.Replace(
				liveCapacityCreateRunJSON(targetID),
				`"max_latency_p95_cv":0.2`,
				`"max_latency_p95_cv":0.2,"legacy_window":true`,
				1,
			),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.Runs(
				response,
				httptest.NewRequest(
					http.MethodPost,
					evaluationAPIBase+"/runs",
					strings.NewReader(test.body),
				),
			)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusBadRequest, response.Body.String())
			}
		})
	}

	assertLiveCapacityCreateSuccess(t, handler, targetID)
}

func assertLiveCapacityCreateSuccess(t *testing.T, handler *authenticatedEvaluationTestHandler, targetID string) {
	t.Helper()
	response := httptest.NewRecorder()
	handler.Runs(response, httptest.NewRequest(
		http.MethodPost,
		evaluationAPIBase+"/runs",
		strings.NewReader(liveCapacityCreateRunJSON(targetID)),
	))
	if response.Code != http.StatusCreated {
		t.Fatalf("valid live capacity status=%d body=%s", response.Code, response.Body.String())
	}
	var run evaluationplane.Run
	if err := json.NewDecoder(response.Body).Decode(&run); err != nil {
		t.Fatalf("decode created capacity run: %v", err)
	}
	if run.CapacitySLO == nil || run.CapacitySLO.RequiredConcurrency != 2 ||
		run.CapacitySLO.MinThroughputScalingEfficiency != 0.7 {
		t.Fatalf("created run lost frozen Capacity SLO: %+v", run.CapacitySLO)
	}
	if run.CapacityLoadProtocol == nil ||
		!reflect.DeepEqual(run.CapacityLoadProtocol.ConcurrencyLevels, []int64{1, 2}) ||
		run.CapacityLoadProtocol.MeasurementRequestsPerRepetition != 100 ||
		run.CapacityLoadProtocol.RepetitionsPerLevel != 3 {
		t.Fatalf("created run lost frozen capacity load protocol: %+v", run.CapacityLoadProtocol)
	}
}

func TestEvaluationPlaneCreateRequiresEveryWireField(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	requiredFields := []string{
		"client_request_id",
		"name",
		"description",
		"suite_ids",
		"track_ids",
		"mode",
		"target_id",
		"change_profile",
		"sample_limit",
		"concurrency",
		"seed",
	}
	null := json.RawMessage("null")

	for _, field := range requiredFields {
		for _, test := range []struct {
			name  string
			value *json.RawMessage
		}{
			{name: "missing"},
			{name: "null", value: &null},
		} {
			t.Run(field+"/"+test.name, func(t *testing.T) {
				response := httptest.NewRecorder()
				request := httptest.NewRequest(
					http.MethodPost,
					evaluationAPIBase+"/runs",
					strings.NewReader(mutateValidCreateRunJSON(t, field, test.value)),
				)
				handler.Runs(response, request)
				if response.Code != http.StatusBadRequest {
					t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusBadRequest, response.Body.String())
				}
				if !strings.Contains(response.Body.String(), field) {
					t.Fatalf("response does not identify field %q: %s", field, response.Body.String())
				}
			})
		}
	}

	ledger, err := service.ListRunLedgerPageAs(evaluationplane.SystemActor(), evaluationplane.RunListQuery{Limit: 10})
	if err != nil {
		t.Fatalf("ListRunLedgerPage: %v", err)
	}
	if len(ledger.Runs) != 0 {
		t.Fatalf("rejected wire requests created %d runs", len(ledger.Runs))
	}
}

func TestEvaluationPlaneCreateWireOptionalAndZeroValueSemantics(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)

	for _, test := range []struct {
		name  string
		value json.RawMessage
	}{
		{name: "null", value: json.RawMessage("null")},
		{name: "empty string", value: json.RawMessage(`""`)},
		{name: "non UUID", value: json.RawMessage(`"baseline"`)},
		{name: "non canonical UUID", value: json.RawMessage(`"5B49EDBB-2008-4DC3-A245-B7CC78B839B1"`)},
	} {
		t.Run("baseline_run_id/"+test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.Runs(
				response,
				httptest.NewRequest(
					http.MethodPost,
					evaluationAPIBase+"/runs",
					strings.NewReader(mutateValidCreateRunJSON(t, "baseline_run_id", &test.value)),
				),
			)
			if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "baseline_run_id") {
				t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
			}
		})
	}

	emptyDescription := json.RawMessage(`""`)
	seedZero := json.RawMessage("0")
	body := mutateValidCreateRunJSON(t, "description", &emptyDescription)
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("decode empty-description request: %v", err)
	}
	payload["seed"] = seedZero
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("encode empty-description request: %v", err)
	}

	response := httptest.NewRecorder()
	handler.Runs(
		response,
		httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/runs", strings.NewReader(string(encoded))),
	)
	if response.Code != http.StatusCreated {
		t.Fatalf("empty description with explicit zero seed status=%d body=%s", response.Code, response.Body.String())
	}
	var run evaluationplane.Run
	if err := json.NewDecoder(response.Body).Decode(&run); err != nil {
		t.Fatalf("decode created run: %v", err)
	}
	if run.Description != "" || run.Seed != 0 {
		t.Fatalf("wire zero values were not preserved: description=%q seed=%d", run.Description, run.Seed)
	}
}

func TestEvaluationPlaneCreateRejectsDuplicateKeysAndTrailingJSON(t *testing.T) {
	service := newEvaluationHandlerService(t, "")
	handler := newAuthenticatedEvaluationTestHandler(service, false)
	tests := []struct {
		name        string
		body        string
		wantMessage string
	}{
		{
			name:        "duplicate key",
			body:        strings.Replace(validCreateRunJSON(), `"seed":17`, `"seed":17,"seed":18`, 1),
			wantMessage: `duplicate JSON object key \"seed\"`,
		},
		{
			name:        "trailing document",
			body:        validCreateRunJSON() + `{}`,
			wantMessage: "trailing JSON",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			handler.Runs(
				response,
				httptest.NewRequest(http.MethodPost, evaluationAPIBase+"/runs", strings.NewReader(test.body)),
			)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status=%d want=%d body=%s", response.Code, http.StatusBadRequest, response.Body.String())
			}
			if !strings.Contains(response.Body.String(), test.wantMessage) {
				t.Fatalf("response missing %q: %s", test.wantMessage, response.Body.String())
			}
		})
	}
}
