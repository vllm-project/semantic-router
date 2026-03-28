package extproc

import (
	"testing"
	"time"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestHandleRouterReplayAPIListAppliesFilters(t *testing.T) {
	router := newReplayAggregateTestRouter(t)

	response := router.handleRouterReplayAPI(
		"GET",
		"/v1/router_replay?decision=decision-b&cache_status=streamed&limit=10",
	)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate replay list response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["count"].(float64)); got != 1 {
		t.Fatalf("expected count=1, got %d", got)
	}
	data := mustReplayData(t, body["data"])
	if got := data[0]["id"]; got != "replay-2" {
		t.Fatalf("expected filtered replay-2, got %#v", got)
	}
}

func TestHandleRouterReplayAggregateAPIReturnsChartsAndSummary(t *testing.T) {
	router := newReplayAggregateTestRouter(t)

	response := router.handleRouterReplayAPI("GET", "/v1/router_replay/aggregate")
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate aggregate response")
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := body["object"]; got != "router_replay.aggregate" {
		t.Fatalf("expected aggregate object, got %#v", got)
	}
	if got := int(body["record_count"].(float64)); got != 2 {
		t.Fatalf("expected record_count=2, got %d", got)
	}

	summary := body["summary"].(map[string]interface{})
	if got := summary["total_saved"].(float64); got != 0.003 {
		t.Fatalf("expected total_saved=0.003, got %#v", got)
	}
	if got := int(summary["cost_record_count"].(float64)); got != 1 {
		t.Fatalf("expected cost_record_count=1, got %d", got)
	}
	if got := int(summary["excluded_record_count"].(float64)); got != 1 {
		t.Fatalf("expected excluded_record_count=1, got %d", got)
	}

	modelSelection := body["model_selection"].([]interface{})
	if got := modelSelection[0].(map[string]interface{})["name"]; got != "gpt-4o" {
		t.Fatalf("expected alphabetical tie-breaker for model_selection, got %#v", got)
	}

	tokenVolume := body["token_volume"].(map[string]interface{})
	if got := int(tokenVolume["total_tokens"].(float64)); got != 375 {
		t.Fatalf("expected total_tokens=375, got %d", got)
	}

	tokenBreakdown := body["token_breakdown"].(map[string]interface{})
	byDecision := tokenBreakdown["by_decision"].([]interface{})
	if got := byDecision[0].(map[string]interface{})["name"]; got != "decision-b" {
		t.Fatalf("expected highest token decision first, got %#v", got)
	}

	availableModels := body["available_models"].([]interface{})
	if len(availableModels) != 3 {
		t.Fatalf("expected three available models, got %#v", availableModels)
	}
}

func TestHandleRouterReplayAggregateAPIAppliesFilters(t *testing.T) {
	router := newReplayAggregateTestRouter(t)

	response := router.handleRouterReplayAPI(
		"GET",
		"/v1/router_replay/aggregate?cache_status=cached&search=alpha",
	)
	if response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("expected immediate aggregate response")
	}
	if got := response.GetImmediateResponse().GetStatus().GetCode(); got != typev3.StatusCode_OK {
		t.Fatalf("expected OK status, got %v", got)
	}

	body := decodeJSONBody(t, response.GetImmediateResponse().Body)
	if got := int(body["record_count"].(float64)); got != 1 {
		t.Fatalf("expected record_count=1, got %d", got)
	}
	summary := body["summary"].(map[string]interface{})
	if got := summary["total_saved"].(float64); got != 0.003 {
		t.Fatalf("expected filtered total_saved=0.003, got %#v", got)
	}
}

func newReplayAggregateTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()

	recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
	actualCost := 0.002
	baselineCost := 0.005
	costSavings := 0.003
	currency := "USD"
	baselineModel := "gpt-4"
	promptA := 100
	completionA := 50
	totalA := 150
	promptB := 200
	completionB := 25

	records := []routerreplay.RoutingRecord{
		{
			ID:               "replay-1",
			Timestamp:        time.Unix(1, 0).UTC(),
			RequestID:        "req-alpha",
			Decision:         "decision-a",
			OriginalModel:    "gpt-4",
			SelectedModel:    "gpt-4o-mini",
			FromCache:        true,
			PromptTokens:     &promptA,
			CompletionTokens: &completionA,
			TotalTokens:      &totalA,
			ActualCost:       &actualCost,
			BaselineCost:     &baselineCost,
			CostSavings:      &costSavings,
			Currency:         &currency,
			BaselineModel:    &baselineModel,
			Signals: store.Signal{
				Domain:  []string{"business", "finance"},
				Keyword: []string{"pricing"},
			},
		},
		{
			ID:               "replay-2",
			Timestamp:        time.Unix(2, 0).UTC(),
			RequestID:        "req-beta",
			Decision:         "decision-b",
			OriginalModel:    "gpt-4",
			SelectedModel:    "gpt-4o",
			Streaming:        true,
			PromptTokens:     &promptB,
			CompletionTokens: &completionB,
			Signals: store.Signal{
				Domain:     []string{"business"},
				Preference: []string{"latency"},
			},
		},
	}

	for _, record := range records {
		if _, err := recorder.AddRecord(record); err != nil {
			t.Fatalf("failed to add replay record %s: %v", record.ID, err)
		}
	}

	return &OpenAIRouter{
		ReplayRecorders: map[string]*routerreplay.Recorder{
			"decision-a": recorder,
		},
	}
}
