package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestHandleRouterReplayAPIListTrimsQueryAndReturnsRecords(t *testing.T) {
	router, recordID := newReplayAPITestRouter(t)

	body := mustReplayListBody(t, router, "/v1/router_replay?limit=10")
	assertReplayPage(t, body, 1, 1, 10, 0)

	record := mustSingleReplayRecord(t, body)
	assertStringField(t, record, "id", recordID)
	assertIntField(t, record, "prompt_tokens", 1200)
	assertStringField(t, record, "currency", "USD")
	assertStringField(t, record, "baseline_model", "expensive-model")
}

func TestHandleRouterReplayAPIRecordLookup(t *testing.T) {
	router, recordID := newReplayAPITestRouter(t)

	body := mustReplayLookupBody(t, router, recordID)
	assertStringField(t, body, "id", recordID)
	assertStringField(t, body, "decision", "decision-a")
	assertIntField(t, body, "decision_tier", 2)
	assertIntField(t, body, "decision_priority", 100)
	assertIntField(t, body, "completion_tokens", 300)
	assertFloat64Field(t, body, "actual_cost", 0.0042)
	assertFloat64Field(t, body, "cost_savings", 0.0058)
	assertStringSliceField(t, body, "projections", []string{"balance_reasoning"})
}

func TestHandleRouterReplayAPIListDoesNotDuplicateSharedStorageRecords(t *testing.T) {
	sharedStore := store.NewMemoryStore(10, 0)
	recorderA := routerreplay.NewRecorder(sharedStore)
	recorderB := routerreplay.NewRecorder(sharedStore)
	if _, err := recorderA.AddRecord(routerreplay.RoutingRecord{
		ID:        "replay-shared-1",
		Decision:  "decision-a",
		RequestID: "req-shared-1",
		Timestamp: time.Unix(1, 0).UTC(),
	}); err != nil {
		t.Fatalf("failed to add shared replay record: %v", err)
	}

	router := &OpenAIRouter{
		ReplayRecorder:    recorderA,
		ReplayStoreShared: true,
		ReplayRecorders: map[string]*routerreplay.Recorder{
			"decision-a": recorderA,
			"decision-b": recorderB,
		},
	}

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay?limit=10")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate replay list response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != 1 {
		t.Fatalf("expected shared count=1, got %d", got)
	}
}

func TestHandleRouterReplayAPIListDefaultsToBoundedPage(t *testing.T) {
	router := newReplayAPITestRouterWithRecords(t, 25, "")

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate replay list response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != 20 {
		t.Fatalf("expected count=20, got %d", got)
	}
	if got := int(body["total"].(float64)); got != 25 {
		t.Fatalf("expected total=25, got %d", got)
	}
	if got := int(body["limit"].(float64)); got != routerReplayDefaultListLimit {
		t.Fatalf("expected default limit=%d, got %d", routerReplayDefaultListLimit, got)
	}
	if hasMore, ok := body["has_more"].(bool); !ok || !hasMore {
		t.Fatalf("expected has_more=true, got %#v", body["has_more"])
	}
	if got := int(body["next_offset"].(float64)); got != 20 {
		t.Fatalf("expected next_offset=20, got %d", got)
	}

	data := mustReplayData(t, body["data"])
	if got := data[0]["id"]; got != "replay-025" {
		t.Fatalf("expected newest record first, got %#v", got)
	}
	if got := data[len(data)-1]["id"]; got != "replay-006" {
		t.Fatalf("expected 20th newest record, got %#v", got)
	}
}

func TestHandleRouterReplayAPIListRespectsOffset(t *testing.T) {
	router := newReplayAPITestRouterWithRecords(t, 6, "")

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay?limit=2&offset=2")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate replay list response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != 2 {
		t.Fatalf("expected count=2, got %d", got)
	}
	if got := int(body["offset"].(float64)); got != 2 {
		t.Fatalf("expected offset=2, got %d", got)
	}
	data := mustReplayData(t, body["data"])
	if got := data[0]["id"]; got != "replay-004" {
		t.Fatalf("expected first offset record replay-004, got %#v", got)
	}
	if got := data[1]["id"]; got != "replay-003" {
		t.Fatalf("expected second offset record replay-003, got %#v", got)
	}
}

func TestHandleRouterReplayAPIListRejectsInvalidQuery(t *testing.T) {
	router, _ := newReplayAPITestRouter(t)

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay?limit=abc")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate error response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_BadRequest {
		t.Fatalf("expected bad request status, got %v", got)
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	errorBody, ok := body["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error payload, got %#v", body)
	}
	if got := errorBody["message"]; got != "limit must be an integer" {
		t.Fatalf("expected invalid limit message, got %#v", got)
	}
}

func TestHandleRouterReplayAPIListReturnsPayloadTooLargeForOversizedPage(t *testing.T) {
	oversizedBody := strings.Repeat("x", 5*1024*1024)
	router := newReplayAPITestRouterWithRecords(t, 1, oversizedBody)

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay?limit=1")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate error response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_PayloadTooLarge {
		t.Fatalf("expected payload too large status, got %v", got)
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	errorBody, ok := body["error"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected error payload, got %#v", body)
	}
	if got := int(errorBody["code"].(float64)); got != 413 {
		t.Fatalf("expected error code 413, got %d", got)
	}
	if got := errorBody["message"]; got == nil || !strings.Contains(got.(string), "ext-proc message size limit") {
		t.Fatalf("expected oversized response guidance, got %#v", got)
	}
}

func TestHandleRouterReplayAPIReturnsErrorsForInvalidRequests(t *testing.T) {
	router, _ := newReplayAPITestRouter(t)

	tests := []struct {
		name string
		resp *ext_proc.ProcessingResponse
	}{
		{
			name: "method not allowed on list",
			resp: router.handleRouterReplayAPI("POST", "/v1/router_replay"),
		},
		{
			name: "missing replay record returns 404",
			resp: router.handleRouterReplayAPI("GET", "/v1/router_replay/missing"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.resp == nil || tt.resp.GetImmediateResponse() == nil {
				t.Fatal("expected immediate error response")
			}
			if got := tt.resp.GetImmediateResponse().GetStatus().GetCode(); got == typev3.StatusCode_OK {
				t.Fatalf("expected non-OK error status, got %v", got)
			}
			body := decodeJSONBody(t, tt.resp.GetImmediateResponse().Body)
			errorBody, ok := body["error"].(map[string]interface{})
			if !ok {
				t.Fatalf("expected error payload, got %#v", body)
			}
			if errorBody["message"] == "" {
				t.Fatalf("expected error message, got %#v", errorBody)
			}
		})
	}
}

func newReplayAPITestRouter(t *testing.T) (*OpenAIRouter, string) {
	t.Helper()
	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	promptTokens := 1200
	completionTokens := 300
	totalTokens := 1500
	actualCost := 0.0042
	baselineCost := 0.01
	costSavings := 0.0058
	currency := "USD"
	baselineModel := "expensive-model"
	recordID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:                "replay-1",
		Decision:          "decision-a",
		DecisionTier:      2,
		DecisionPriority:  100,
		RequestID:         "req-1",
		Projections:       []string{"balance_reasoning"},
		ProjectionScores:  map[string]float64{"reasoning_pressure": 0.82},
		SignalConfidences: map[string]float64{"projection:balance_reasoning": 0.82},
		PromptTokens:      &promptTokens,
		CompletionTokens:  &completionTokens,
		TotalTokens:       &totalTokens,
		ActualCost:        &actualCost,
		BaselineCost:      &baselineCost,
		CostSavings:       &costSavings,
		Currency:          &currency,
		BaselineModel:     &baselineModel,
	})
	if err != nil {
		t.Fatalf("failed to add replay record: %v", err)
	}

	return &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{
			"decision-a": recorder,
		},
	}, recordID
}

func newReplayAPITestRouterWithRecords(t *testing.T, count int, requestBody string) *OpenAIRouter {
	t.Helper()

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(count+1, 0))
	for i := 1; i <= count; i++ {
		recordID := fmt.Sprintf("replay-%03d", i)
		timestamp := time.Unix(int64(i), 0).UTC()
		if _, err := recorder.AddRecord(routerreplay.RoutingRecord{
			ID:          recordID,
			Decision:    "decision-a",
			RequestID:   fmt.Sprintf("req-%03d", i),
			RequestBody: requestBody,
			Timestamp:   timestamp,
		}); err != nil {
			t.Fatalf("failed to add replay record %s: %v", recordID, err)
		}
	}

	return &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{
			"decision-a": recorder,
		},
	}
}

func mustReplayListBody(t *testing.T, router *OpenAIRouter, path string) map[string]interface{} {
	t.Helper()
	return mustReplayResponseBody(t, router.handleRouterReplayAPI("GET", path), "expected immediate replay list response")
}

func mustReplayLookupBody(t *testing.T, router *OpenAIRouter, recordID string) map[string]interface{} {
	t.Helper()
	return mustReplayResponseBody(t, router.handleRouterReplayAPI("GET", "/v1/router_replay/"+recordID), "expected immediate replay lookup response")
}

func mustReplayResponseBody(
	t *testing.T,
	response *ext_proc.ProcessingResponse,
	failureMessage string,
) map[string]interface{} {
	t.Helper()
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal(failureMessage)
	}
	return decodeJSONBody(t, response.GetImmediateResponse().Body)
}

func assertReplayPage(t *testing.T, body map[string]interface{}, count int, total int, limit int, offset int) {
	t.Helper()
	assertIntField(t, body, "count", count)
	assertIntField(t, body, "total", total)
	assertIntField(t, body, "limit", limit)
	assertIntField(t, body, "offset", offset)
}

func mustSingleReplayRecord(t *testing.T, body map[string]interface{}) map[string]interface{} {
	t.Helper()
	data := mustReplayData(t, body["data"])
	if len(data) != 1 {
		t.Fatalf("expected one replay record, got %#v", body["data"])
	}
	return data[0]
}

func assertIntField(t *testing.T, body map[string]interface{}, field string, want int) {
	t.Helper()
	got, ok := body[field].(float64)
	if !ok || int(got) != want {
		t.Fatalf("expected %s=%d, got %#v", field, want, body[field])
	}
}

func assertFloat64Field(t *testing.T, body map[string]interface{}, field string, want float64) {
	t.Helper()
	got, ok := body[field].(float64)
	if !ok || got != want {
		t.Fatalf("expected %s=%v, got %#v", field, want, body[field])
	}
}

func assertStringField(t *testing.T, body map[string]interface{}, field string, want string) {
	t.Helper()
	got, ok := body[field].(string)
	if !ok || got != want {
		t.Fatalf("expected %s=%q, got %#v", field, want, body[field])
	}
}

func assertStringSliceField(t *testing.T, body map[string]interface{}, field string, want []string) {
	t.Helper()
	values, ok := body[field].([]interface{})
	if !ok || len(values) != len(want) {
		t.Fatalf("expected %s=%v, got %#v", field, want, body[field])
	}
	for index, expected := range want {
		value, ok := values[index].(string)
		if !ok || value != expected {
			t.Fatalf("expected %s[%d]=%q, got %#v", field, index, expected, values[index])
		}
	}
}

func mustReplayData(t *testing.T, value interface{}) []map[string]interface{} {
	t.Helper()

	rawData, ok := value.([]interface{})
	if !ok {
		t.Fatalf("expected replay data array, got %#v", value)
	}
	data := make([]map[string]interface{}, 0, len(rawData))
	for _, item := range rawData {
		record, ok := item.(map[string]interface{})
		if !ok {
			t.Fatalf("expected replay record object, got %#v", item)
		}
		data = append(data, record)
	}
	return data
}

func decodeJSONBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("failed to decode JSON body: %v", err)
	}
	return decoded
}
