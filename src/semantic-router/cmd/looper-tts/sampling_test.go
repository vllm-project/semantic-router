package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestNativePerModelSamplingOnOutboundCalls(t *testing.T) {
	models := map[string]manifestModel{
		"a": {ID: "a", Model: "model-a", Sampling: manifestSampling{Temperature: 0.1, TopP: 0.2}},
		"b": {ID: "b", Model: "model-b", Sampling: manifestSampling{Temperature: 0.8, TopP: 0.9}},
	}
	arms := []manifestArm{
		{ID: "direct", Algorithm: "direct", ModelIDs: []string{"b"}},
		{ID: "confidence", Algorithm: "confidence", ModelIDs: []string{"a", "b"}, Parameters: map[string]interface{}{"threshold": 0.8}},
		{ID: "remom", Algorithm: "remom", ModelIDs: []string{"a", "b"}, Parameters: map[string]interface{}{"breadth": []interface{}{float64(2)}}},
		{ID: "fusion", Algorithm: "fusion", ModelIDs: []string{"a", "b"}, Parameters: map[string]interface{}{"panel_model_ids": []interface{}{"a", "b"}, "judge_model_id": "b", "synthesis_model_id": "b"}},
	}
	for _, arm := range arms {
		t.Run(arm.ID, func(t *testing.T) {
			var mu sync.Mutex
			counts := map[string]int{}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Model       string   `json:"model"`
					Temperature *float64 `json:"temperature"`
					TopP        *float64 `json:"top_p"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
					w.WriteHeader(400)
					return
				}
				expected := models["a"].Sampling
				if request.Model == "model-b" {
					expected = models["b"].Sampling
				}
				if request.Temperature == nil || request.TopP == nil || *request.Temperature != expected.Temperature || *request.TopP != expected.TopP {
					t.Errorf("outbound sampling for %s = %+v, want %+v", request.Model, request, expected)
				}
				mu.Lock()
				counts[request.Model]++
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				// A low verifier score exercises both models and both verification calls.
				_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"{\"confidence\":0.1,\"reason\":\"fixture\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
			}))
			defer server.Close()
			cfg := &config.LooperConfig{Endpoint: server.URL, TimeoutSeconds: 5}
			client, err := looper.NewConnectorClient(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			_, err = executeAlgorithm(context.Background(), client, cfg, arm, manifestBudget{MaxCalls: 20, MaxTotalTokens: 100000}, "fixture question", 7, models)
			if err != nil {
				t.Fatal(err)
			}
			mu.Lock()
			defer mu.Unlock()
			expected := map[string]map[string]int{
				"direct":     {"model-b": 1},
				"confidence": {"model-a": 2, "model-b": 2},
				"remom":      {"model-a": 2, "model-b": 1},
				"fusion":     {"model-a": 1, "model-b": 3},
			}[arm.ID]
			for model, want := range expected {
				if counts[model] != want {
					t.Errorf("%s calls = %d, want %d", model, counts[model], want)
				}
			}
		})
	}
}
