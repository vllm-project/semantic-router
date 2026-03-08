package extproc

import (
	"encoding/json"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

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
	recordID, err := recorder.AddRecord(routerreplay.RoutingRecord{
		ID:        "replay-1",
		Decision:  "decision-a",
		RequestID: "req-1",
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

func decodeJSONBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("failed to decode JSON body: %v", err)
	}
	return decoded
}
