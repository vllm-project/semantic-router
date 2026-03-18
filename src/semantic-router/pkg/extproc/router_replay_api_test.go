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

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay?limit=10")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate replay list response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != 1 {
		t.Fatalf("expected count=1, got %d", got)
	}
	if got := int(body["total"].(float64)); got != 1 {
		t.Fatalf("expected total=1, got %d", got)
	}
	if got := int(body["limit"].(float64)); got != 10 {
		t.Fatalf("expected limit=10, got %d", got)
	}
	if got := int(body["offset"].(float64)); got != 0 {
		t.Fatalf("expected offset=0, got %d", got)
	}
	data, ok := body["data"].([]interface{})
	if !ok || len(data) != 1 {
		t.Fatalf("expected one replay record, got %#v", body["data"])
	}
	record, ok := data[0].(map[string]interface{})
	if !ok {
		t.Fatalf("expected replay record object, got %#v", data[0])
	}
	if got := record["id"]; got != recordID {
		t.Fatalf("expected replay id %q, got %#v", recordID, got)
	}
	if got := int(record["prompt_tokens"].(float64)); got != 1200 {
		t.Fatalf("expected prompt_tokens=1200, got %d", got)
	}
	if got := record["currency"]; got != "USD" {
		t.Fatalf("expected currency=USD, got %#v", got)
	}
	if got := record["baseline_model"]; got != "expensive-model" {
		t.Fatalf("expected baseline_model=expensive-model, got %#v", got)
	}
}

func TestHandleRouterReplayAPIRecordLookup(t *testing.T) {
	router, recordID := newReplayAPITestRouter(t)

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay/"+recordID)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate replay lookup response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := body["id"]; got != recordID {
		t.Fatalf("expected replay id %q, got %#v", recordID, got)
	}
	if got := body["decision"]; got != "decision-a" {
		t.Fatalf("expected decision-a, got %#v", got)
	}
	if got := int(body["completion_tokens"].(float64)); got != 300 {
		t.Fatalf("expected completion_tokens=300, got %d", got)
	}
	if got := body["actual_cost"].(float64); got != 0.0042 {
		t.Fatalf("expected actual_cost=0.0042, got %#v", got)
	}
	if got := body["cost_savings"].(float64); got != 0.0058 {
		t.Fatalf("expected cost_savings=0.0058, got %#v", got)
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
		ID:               "replay-1",
		Decision:         "decision-a",
		RequestID:        "req-1",
		PromptTokens:     &promptTokens,
		CompletionTokens: &completionTokens,
		TotalTokens:      &totalTokens,
		ActualCost:       &actualCost,
		BaselineCost:     &baselineCost,
		CostSavings:      &costSavings,
		Currency:         &currency,
		BaselineModel:    &baselineModel,
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
